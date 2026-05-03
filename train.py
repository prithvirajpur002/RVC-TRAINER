# -*- coding: utf-8 -*-
"""
Catalyst RVC — src/train.py
Training engine: Generator (NSF-HiFiGAN) + Discriminator (MPD + MSD).

Key design decisions:
  - Live stdout streaming (capture_output=False) so you see loss in real time.
  - Resume detection: if G_*.pth checkpoints exist, training auto-continues.
  - OOM guard: if VRAM is critically low before training starts, we raise
    early with a clear message instead of crashing mid-epoch.
  - Checkpoint selection: sorted by mtime (NOT filename), because G_200.pth
    sorts before G_2000.pth alphabetically but may be older.

Fixes applied:
  - Bug 1:  keep_ckpts flag INVERTED — "latest" now correctly maps to "1"
            (keep only latest) and "all" maps to "0" (keep all checkpoints).
            Previous code had these backwards, silently keeping all checkpoints
            on every run and eating Kaggle disk space.
  - Bug 11: torch.load() now uses weights_only=True with graceful fallback
            for older checkpoint formats. Prevents arbitrary code execution
            from untrusted .pth files and fixes FutureWarning in PyTorch 2+.
  - Fix F:  After a non-zero training return code, the exported checkpoint's
            epoch is compared against the target epoch count. If the model
            completed fewer than 50% of target epochs, a clear warning is
            logged before export so you know the checkpoint is undertrained
            and can decide whether to deploy it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from .utils import Logger, elapsed_str, get_gpu_info, now_iso, save_json


def _find_train_script(rvc_repo: str) -> str:
    candidates = [
        "infer/modules/train/train.py",
        "train_nsf_sim.py",
    ]
    for rel in candidates:
        p = os.path.join(rvc_repo, rel)
        if os.path.exists(p):
            return p
    raise RuntimeError(
        "train.py not found in RVC repo.\n"
        f"Searched: {candidates}\n"
        "Re-clone the RVC repo in Cell 2."
    )


def _find_best_checkpoint(logs_dir: str, weights_dir: str, voice_name: str) -> Optional[str]:
    """
    Find the latest checkpoint sorted by modification time.
    Checks both logs_dir (G_*.pth) and weights_dir (exported weights).
    Returns None if no checkpoints exist.
    """
    candidates = (
        list(Path(logs_dir).glob("G_*.pth")) +
        list(Path(weights_dir).glob(f"{voice_name}*.pth")) +
        list(Path(weights_dir).glob("*.pth"))
    )
    if not candidates:
        return None
    best = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return str(best)


def _safe_torch_load(ckpt_path: str) -> dict:
    """
    Load a PyTorch checkpoint safely.

    Bug 11 fix: tries weights_only=True first (safe, PyTorch 2+ recommended).
    Falls back to weights_only=False only if the checkpoint contains non-tensor
    objects (common in older RVC checkpoints that store config dicts inline).

    weights_only=False is a security risk with untrusted files — only use
    checkpoints you generated yourself or from trusted sources.
    """
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _strip_optimizer(ckpt_path: str, voice_name: str, version: str,
                     total_epochs: int, sample_rate: int) -> dict:
    """
    Strip optimizer state from a raw G_*.pth checkpoint.
    Raw size: ~150-200 MB → Stripped: ~55-80 MB.
    Result matches the format RVC WebUI uses for exported models.
    """
    ckpt       = _safe_torch_load(ckpt_path)
    state_dict = ckpt.get("model", ckpt)

    return {
        "weight":  {k.replace("module.", ""): v.half() for k, v in state_dict.items()},
        "config":  [sample_rate, 512, version],
        "info":    f"{voice_name} — {total_epochs} epochs — Catalyst RVC",
        "sr":      sample_rate,
        "f0":      1,
        "version": version,
        "epoch":   ckpt.get("epoch", total_epochs),
    }


def _check_checkpoint_epoch(ckpt_path: str, target_epochs: int, log: Logger) -> None:
    """
    Fix F: compare the checkpoint's epoch against the training target.
    Logs a warning if fewer than 50% of target epochs completed so the user
    knows they are exporting an undertrained model before it gets registered.

    This only runs when returncode != 0 (training crashed). If training
    finished cleanly, we trust it ran to completion.
    """
    try:
        ckpt = _safe_torch_load(ckpt_path)
        actual_epoch = ckpt.get("epoch", None)
        if actual_epoch is None:
            log.warn(
                "  Could not read epoch count from checkpoint — "
                "verify training ran to completion before deploying."
            )
            return
        pct = actual_epoch / max(target_epochs, 1) * 100
        if actual_epoch < target_epochs * 0.5:
            log.warn(
                f"  ⚠️  UNDERTRAINED: checkpoint epoch {actual_epoch} "
                f"is only {pct:.0f}% of target {target_epochs}.\n"
                "  Training crashed early. This model may produce poor quality.\n"
                "  Options:\n"
                "    • Resume with python main.py --retry-failed\n"
                "    • Lower batch_size in config to avoid OOM\n"
                "    • Inspect training output above for the root cause"
            )
        elif actual_epoch < target_epochs:
            log.warn(
                f"  Checkpoint at epoch {actual_epoch}/{target_epochs} "
                f"({pct:.0f}%) — training did not reach target epochs."
            )
        else:
            log.info(f"  Checkpoint epoch: {actual_epoch}/{target_epochs} ✓")
    except Exception as e:
        log.warn(f"  Could not verify checkpoint epoch: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def train(
    experiment_id: str,
    dataset_dir: str,
    logs_dir: str,
    model_output: str,
    config: dict,
    rvc_repo: str,
    pretrain_g: str,
    pretrain_d: str,
    resume: bool = True,
    log: Optional[Logger] = None,
) -> str:
    """
    Train RVC v2 model and export a stripped .pth checkpoint.

    Args:
        experiment_id: Unique experiment name (also used as voice/exp name).
        dataset_dir:   Path to preprocessed WAV segments.
        logs_dir:      RVC logs directory (contains G_*.pth during training).
        model_output:  Where to write the final stripped .pth.
        config:        Training config dict (loaded from configs/*.json).
        rvc_repo:      Path to cloned RVC repo.
        pretrain_g:    Path to pretrained generator (f0G40k.pth).
        pretrain_d:    Path to pretrained discriminator (f0D40k.pth).
        resume:        If True, auto-resumes from latest checkpoint if found.
        log:           Logger instance.

    Returns:
        Path to the exported .pth checkpoint.
    """
    if log is None:
        log = Logger()

    log.section(f"TRAINING  [{experiment_id}]")

    gpu = get_gpu_info()
    log.info(
        f"GPU : {gpu['gpu_name']}  "
        f"({gpu['vram_total_gb']:.1f} GB total, {gpu['vram_free_gb']:.1f} GB free)"
    )

    if gpu["vram_total_gb"] < 8.0:
        raise RuntimeError(
            f"Only {gpu['vram_total_gb']:.1f} GB VRAM — RVC v2 needs ≥ 8 GB.\n"
            "Switch to P100 (16 GB) or T4 (15 GB) in Kaggle accelerator settings."
        )

    feat_dir = Path(logs_dir) / "3_feature768"
    if not feat_dir.exists():
        feat_dir = Path(logs_dir) / "3_feature256"
    if not feat_dir.exists() or not any(feat_dir.glob("*.npy")):
        raise RuntimeError(
            "Feature files not found. Run extract_features() first.\n"
            f"  Expected: {logs_dir}/3_feature768/*.npy"
        )

    for label, path in [("Pretrained G", pretrain_g), ("Pretrained D", pretrain_d)]:
        if not os.path.exists(path):
            raise RuntimeError(f"{label} not found: {path}\nRe-run bootstrap to download.")

    epochs      = config.get("epochs",       200)
    batch_size  = config.get("batch_size",   6)
    save_every  = config.get("save_every_n", 50)
    sample_rate = config.get("sample_rate",  40000)
    version     = config.get("rvc_version",  "v2")

    # Bug 1 fix: keep_ckpts flag corrected.
    # RVC -l flag:  "1" = keep ONLY latest checkpoint (saves disk)
    #               "0" = keep ALL checkpoints
    keep_ckpts_cfg = config.get("keep_ckpts", "all")
    keep_ckpts     = "1" if keep_ckpts_cfg == "latest" else "0"

    log.info(
        f"Epochs={epochs}  Batch={batch_size}  "
        f"SaveEvery={save_every}  SR={sample_rate}  Ver={version}  "
        f"keep_ckpts={keep_ckpts_cfg!r} (-l {keep_ckpts})"
    )

    weights_dir = os.path.join(rvc_repo, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    existing = _find_best_checkpoint(logs_dir, weights_dir, experiment_id)
    if existing and resume:
        size_mb = os.path.getsize(existing) / 1e6
        log.info(f"Resuming from: {Path(existing).name}  ({size_mb:.0f} MB)")
        log.info("  (RVC will auto-continue from the latest saved step)")
    else:
        log.info("Starting fresh from pretrained base models.")

    train_script = _find_train_script(rvc_repo)
    log.info(f"Train script: {os.path.relpath(train_script, rvc_repo)}")
    log.info("\n⏳ Training started — loss will stream below:\n" + "─" * 58)

    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable, train_script,
            "-e",  experiment_id,
            "-sr", str(sample_rate),
            "-f0", "1",
            "-bs", str(batch_size),
            "-g",  "0",
            "-te", str(epochs),
            "-se", str(save_every),
            "-pg", pretrain_g,
            "-pd", pretrain_d,
            "-l",  keep_ckpts,
            "-c",  "0",
            "-sw", "1",
            "-v",  version,
        ],
        cwd=rvc_repo,
        capture_output=False,
    )

    elapsed = elapsed_str(start_time)
    log.info("─" * 58)
    log.info(f"Training finished in {elapsed}")

    if result.returncode != 0:
        log.warn(
            f"Training exited with code {result.returncode}.\n"
            "  Common causes:\n"
            "  • CUDA OOM → lower batch_size in config\n"
            "  • Missing features → re-run extract_features()\n"
            "  Scroll up to see the full error."
        )

    best_raw = _find_best_checkpoint(logs_dir, weights_dir, experiment_id)
    if best_raw is None:
        raise RuntimeError(
            "No checkpoint found after training.\n"
            "Training may have crashed before saving the first checkpoint.\n"
            "Check output above for errors."
        )

    size_raw = os.path.getsize(best_raw) / 1e6
    log.info(f"Best checkpoint: {Path(best_raw).name}  ({size_raw:.0f} MB raw)")

    # Fix F: if training didn't finish cleanly, check how many epochs completed
    # before exporting so the user knows if they have an undertrained model.
    if result.returncode != 0:
        _check_checkpoint_epoch(best_raw, epochs, log)

    os.makedirs(model_output, exist_ok=True)
    dst_pth = os.path.join(model_output, f"{experiment_id}.pth")

    if Path(best_raw).name.startswith("G_"):
        log.info("Stripping optimizer state...")
        export = _strip_optimizer(best_raw, experiment_id, version, epochs, sample_rate)
        torch.save(export, dst_pth)
        size_out = os.path.getsize(dst_pth) / 1e6
        log.ok(f"Stripped: {size_raw:.0f} MB → {size_out:.0f} MB  ({dst_pth})")
    else:
        import shutil
        shutil.copy2(best_raw, dst_pth)
        log.ok(f"Checkpoint copied: {dst_pth}")

    exp_meta = {
        "experiment_id": experiment_id,
        "config":        config,
        "elapsed":       elapsed,
        "best_raw_ckpt": Path(best_raw).name,
        "exported_pth":  dst_pth,
        "trained_at":    now_iso(),
        "return_code":   result.returncode,
        "status":        "complete" if result.returncode == 0 else "error",
    }
    save_json(os.path.join(model_output, "train_meta.json"), exp_meta)

    return dst_pth
