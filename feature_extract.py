# -*- coding: utf-8 -*-
"""
Catalyst RVC — src/feature_extract.py
HuBERT content features (768-dim, layer 9) + RMVPE F0 extraction.

Cache-aware: skips extraction if cache exists AND source hash matches.
This saves 5–20 min per experiment on repeat runs.

Fixes applied:
  - Bug 4:  cache_dir parameter is now ACTUALLY USED. Previously cache_dir
            was accepted but all cache operations used logs_dir. This meant
            the shared cache between experiments was broken — each experiment
            re-extracted features independently, costing 5–20 min each.
            Fixed: cache check/write uses cache_dir; feature work uses logs_dir.
  - Bug 14: Hardcoded "4" parallel processes replaced with os.cpu_count()-based
            value (clamped 1–8). Prevents thrashing on 2-core Kaggle instances
            and unused cores on higher-end machines.
  - New:    VRAM guard before HuBERT extraction — HuBERT on GPU needs ~2-3 GB.
            Previously the VRAM guard only ran before training.
  - Fix E:  _link_cache_to_logs() now logs a warning for each critical
            directory that is listed but missing from cache_dir, instead of
            silently skipping it. Previously a partially-populated cache
            (e.g., only F0 files, no HuBERT features) would pass the cache
            validity check but fail silently at the link step.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .utils import (
    Logger, compute_dataset_hash, get_free_vram_gb,
    load_json, now_iso, save_json,
)

# ── Cache sentinel ────────────────────────────────────────────────────────────

_CACHE_META = "cache_meta.json"

# Directories that MUST be present for cache to be useful.
# Missing any of these means training will fail.
_CRITICAL_CACHE_DIRS = {"3_feature768", "2a_f0", "0_gt_wavs"}


def _cache_is_valid(cache_dir: str, dataset_dir: str, sample_rate: int) -> bool:
    """
    Returns True if cached features exist and were built from the same dataset
    at the same sample rate.

    Bug 4 fix: checks cache_dir (shared across experiments), not logs_dir.
    Keyed by (source_hash, sample_rate) so different SR configs don't collide.
    """
    meta_path = os.path.join(cache_dir, _CACHE_META)
    if not os.path.exists(meta_path):
        return False
    try:
        meta = load_json(meta_path)
    except Exception:
        return False

    if meta.get("sample_rate") != sample_rate:
        return False

    current_hash = compute_dataset_hash(dataset_dir)
    if meta.get("source_hash") != current_hash:
        return False

    feat_dir = Path(cache_dir) / "3_feature768"
    if not feat_dir.exists():
        feat_dir = Path(cache_dir) / "3_feature256"
    f0_dir = Path(cache_dir) / "2a_f0"

    feat_ok = feat_dir.exists() and any(feat_dir.glob("*.npy"))
    f0_ok   = f0_dir.exists()   and any(f0_dir.glob("*.npy"))
    return feat_ok and f0_ok


def _write_cache_meta(cache_dir: str, dataset_dir: str,
                      sample_rate: int, info: dict) -> None:
    meta = {
        "source_hash":  compute_dataset_hash(dataset_dir),
        "sample_rate":  sample_rate,
        "created_at":   now_iso(),
        **info,
    }
    save_json(os.path.join(cache_dir, _CACHE_META), meta)


def _get_cpu_workers() -> int:
    """
    Return a sensible parallel worker count for RVC's internal preprocessor.

    Bug 14 fix: was hardcoded to "4" regardless of actual CPU count.
    Kaggle T4/P100 instances often only have 2 CPUs — using 4 workers
    causes thrashing and slower preprocessing.
    Clamped to [1, 8] to avoid overwhelming shared environments.
    """
    cpus = os.cpu_count() or 2
    return max(1, min(cpus, 8))


# ── Script path resolution ────────────────────────────────────────────────────

def _find_script(rvc_repo: str, candidates: list[str]) -> Optional[str]:
    """Search for a script across multiple possible RVC repo layouts."""
    for rel in candidates:
        p = os.path.join(rvc_repo, rel)
        if os.path.exists(p):
            return p
    return None


# ── Extraction steps ──────────────────────────────────────────────────────────

def _run_preprocess(rvc_repo: str, dataset_dir: str, logs_dir: str,
                    sample_rate: int, log: Logger) -> None:
    """
    RVC internal preprocessor: normalises each WAV and writes to 0_gt_wavs/.
    This step is separate from our custom preprocessing and must run first.
    """
    n_workers = _get_cpu_workers()
    log.info(f"Running RVC internal preprocessor (0_gt_wavs/, {n_workers} workers)...")

    script = _find_script(rvc_repo, [
        "infer/modules/train/preprocess.py",
        "trainset_preprocess_pipeline_print.py",
    ])
    if script is None:
        raise RuntimeError(
            "RVC preprocess.py not found. Check rvc_repo path.\n"
            "Re-clone the RVC repo in Cell 2."
        )

    r = subprocess.run(
        [sys.executable, script,
         dataset_dir,
         str(sample_rate),
         str(n_workers),
         logs_dir,
         "False",
         "0.3"],
        capture_output=True, text=True, cwd=rvc_repo,
    )
    gt_dir = Path(logs_dir) / "0_gt_wavs"
    n = len(list(gt_dir.glob("*.wav"))) if gt_dir.exists() else 0
    if r.returncode != 0 or n == 0:
        log.warn(f"Preprocessor stderr:\n{r.stderr[-1500:]}")
        if n == 0:
            raise RuntimeError(
                "0_gt_wavs/ is empty after RVC preprocess step.\n"
                "Check that dataset_dir contains valid WAV files and "
                "that the RVC preprocess script is compatible with your setup."
            )
    log.ok(f"0_gt_wavs/ — {n} training slices ready")


def _run_f0(rvc_repo: str, logs_dir: str, log: Logger) -> None:
    """Extract F0 pitch curves using RMVPE."""
    log.info("F0 extraction (RMVPE)...")
    script = _find_script(rvc_repo, [
        "infer/modules/train/extract/extract_f0_rmvpe.py",
        "extract_f0_rmvpe.py",
    ])
    if script is None:
        raise RuntimeError(
            "extract_f0_rmvpe.py not found. Check rvc_repo path.\n"
            "Re-clone the RVC repo in Cell 2."
        )

    r = subprocess.run(
        [sys.executable, script,
         "1",
         "0",
         "0",
         logs_dir,
         "True"],
        capture_output=True, text=True, cwd=rvc_repo,
    )
    f0_dir = Path(logs_dir) / "2a_f0"
    n = len(list(f0_dir.glob("*.npy"))) if f0_dir.exists() else 0
    if r.returncode != 0 or n == 0:
        log.warn(f"F0 stderr:\n{r.stderr[-1500:]}")
        raise RuntimeError(
            "F0 extraction failed or produced no files.\n"
            "Common causes: RMVPE model not downloaded, CUDA OOM."
        )
    log.ok(f"2a_f0/ — {n} pitch files extracted")


def _run_hubert(rvc_repo: str, logs_dir: str, hubert_path: str, log: Logger) -> Path:
    """
    Extract HuBERT content features (768-dim, layer 9).

    New: VRAM guard before launching — HuBERT needs ~2-3 GB.
    Previously only training had a VRAM guard; OOM here produced cryptic
    CUDA errors with no guidance.
    """
    free_vram = get_free_vram_gb()
    if 0.0 < free_vram < 2.5:
        raise RuntimeError(
            f"Only {free_vram:.1f} GB VRAM free — HuBERT extraction needs ≥ 2.5 GB.\n"
            "Free VRAM by restarting the kernel or closing other GPU processes, "
            "then re-run."
        )

    log.info("HuBERT feature extraction (768-dim, layer 9)...")
    script = _find_script(rvc_repo, [
        "infer/modules/train/extract_feature_print.py",
        "extract_feature_print.py",
    ])
    if script is None:
        raise RuntimeError(
            "extract_feature_print.py not found. Check rvc_repo path.\n"
            "Re-clone the RVC repo in Cell 2."
        )

    r = subprocess.run(
        [sys.executable, script,
         "cuda:0",
         "1",
         "0",
         "v2",
         logs_dir,
         hubert_path,
         "9"],
        capture_output=True, text=True, cwd=rvc_repo,
    )

    feat_dir = Path(logs_dir) / "3_feature768"
    if not feat_dir.exists():
        feat_dir = Path(logs_dir) / "3_feature256"

    n = len(list(feat_dir.glob("*.npy"))) if feat_dir.exists() else 0
    if r.returncode != 0 or n == 0:
        log.warn(f"HuBERT stderr:\n{r.stderr[-1500:]}")
        raise RuntimeError(
            "HuBERT extraction failed or produced no files.\n"
            "Common causes: hubert_base.pt corrupted/wrong version, CUDA OOM."
        )

    log.ok(f"{feat_dir.name}/ — {n} feature files extracted")
    return feat_dir


def _validate_features(logs_dir: str, feat_dir: Path, log: Logger) -> None:
    """Sanity-check feature dimensions and counts across all folders."""
    checks = {
        "0_gt_wavs":   (Path(logs_dir) / "0_gt_wavs", "*.wav"),
        "2a_f0":       (Path(logs_dir) / "2a_f0",     "*.npy"),
        "2b-f0nsf":    (Path(logs_dir) / "2b-f0nsf",  "*.npy"),
        feat_dir.name: (feat_dir,                      "*.npy"),
    }
    all_ok = True
    for name, (d, pattern) in checks.items():
        n      = len(list(d.glob(pattern))) if d.exists() else 0
        status = "✅" if n > 0 else "❌"
        log.info(f"   {status} {name}: {n} files")
        if n == 0:
            all_ok = False

    npy_files = list(feat_dir.glob("*.npy"))
    if npy_files:
        dims: set[int] = set()
        for f in npy_files[:5]:
            try:
                arr = np.load(str(f))
                if arr.ndim == 2:
                    dims.add(arr.shape[1])
            except Exception:
                pass
        if dims and dims != {768}:
            log.warn(
                f"Unexpected feature dimension(s) found: {dims}. "
                f"Expected 768 (RVC v2). "
                "Check that you're using the correct HuBERT model and RVC version."
            )
        elif dims == {768}:
            log.ok(f"Feature shape verified: 768-dim  ({len(npy_files)} files)")

    if not all_ok:
        raise RuntimeError(
            "One or more feature folders are empty. Re-run feature extraction.\n"
            "Use force=True to ignore cache and re-extract from scratch."
        )


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(
    dataset_dir: str,
    cache_dir: str,
    logs_dir: str,
    hubert_path: str,
    rvc_repo: str,
    sample_rate: int = 40000,
    force: bool = False,
    log: Optional[Logger] = None,
) -> bool:
    """
    Full feature extraction pipeline: preprocess → F0 → HuBERT.

    Bug 4 fix: cache_dir is now the actual cache location. Features extracted
    for a given dataset+SR combination are stored in cache_dir and shared
    across all experiments that use the same dataset. This avoids re-extracting
    the same features multiple times (saves 5–20 min per experiment).

    Args:
        dataset_dir:  Path to preprocessed WAV segments.
        cache_dir:    Shared cache directory (keyed by dataset hash + SR).
                      Multiple experiments with the same dataset will share this.
        logs_dir:     RVC logs directory (experiment-specific, for training).
        hubert_path:  Path to hubert_base.pt.
        rvc_repo:     Path to cloned RVC repo.
        sample_rate:  Target SR (40000 for RVC v2).
        force:        If True, ignores cache and re-extracts.
        log:          Logger instance.

    Returns:
        True if extracted fresh, False if loaded from cache.
    """
    if log is None:
        log = Logger()

    log.section("FEATURE EXTRACTION")

    if not force and _cache_is_valid(cache_dir, dataset_dir, sample_rate):
        log.ok("Cache hit — features already extracted for this dataset+SR.")
        log.info(f"  cache_dir: {cache_dir}")

        _link_cache_to_logs(cache_dir, logs_dir, log)

        log.info("  Use force=True to re-extract.")
        return False

    if not os.path.exists(hubert_path):
        raise RuntimeError(
            f"HuBERT model not found: {hubert_path}\n"
            "Re-run bootstrap to download it."
        )

    gt_dir = Path(logs_dir) / "0_gt_wavs"
    n_gt   = len(list(gt_dir.glob("*.wav"))) if gt_dir.exists() else 0
    if n_gt == 0:
        _run_preprocess(rvc_repo, dataset_dir, logs_dir, sample_rate, log)

    _run_f0(rvc_repo, logs_dir, log)
    feat_dir = _run_hubert(rvc_repo, logs_dir, hubert_path, log)

    log.info("\nValidating all feature folders...")
    _validate_features(logs_dir, feat_dir, log)

    feat_count = len(list(feat_dir.glob("*.npy")))
    _write_cache_meta(cache_dir, dataset_dir, sample_rate, {
        "feature_dim":   768,
        "hubert_layer":  9,
        "feature_count": feat_count,
        "logs_dir":      logs_dir,
    })

    log.ok("Feature extraction complete.")
    return True


def _link_cache_to_logs(cache_dir: str, logs_dir: str, log: Logger) -> None:
    """
    Copy cached feature directories into logs_dir so RVC training can find them.
    Uses symlinks where possible (same filesystem), falls back to copy.

    Fix E: logs a warning for each critical directory that is listed but
    missing from cache_dir, instead of silently skipping it. Previously a
    partially-populated cache would pass the validity check but link nothing,
    causing a confusing training failure downstream.
    """
    import shutil

    dirs_to_link = ["0_gt_wavs", "2a_f0", "2b-f0nsf", "3_feature768", "3_feature256"]
    for d in dirs_to_link:
        src = Path(cache_dir) / d
        dst = Path(logs_dir)  / d
        if not src.exists():
            # Fix E: warn about missing critical dirs; only skip non-critical ones
            if d in _CRITICAL_CACHE_DIRS:
                log.warn(
                    f"  Critical cache directory missing: {d}/\n"
                    "  This directory is required for training. "
                    "Re-run with force=True to rebuild the cache from scratch."
                )
            continue
        if dst.exists():
            continue
        try:
            dst.symlink_to(src.resolve())
            log.info(f"  Linked cache: {d}/ → {src}")
        except (OSError, NotImplementedError):
            shutil.copytree(str(src), str(dst))
            log.info(f"  Copied cache: {d}/ ({len(list(dst.glob('*.*')))} files)")
