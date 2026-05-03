# -*- coding: utf-8 -*-
"""
Catalyst RVC — src/registry.py
Model registry: track all trained models with metadata and quality scores.

Registry file lives at models/registry.json.
Each entry records the experiment, config, dataset, scores, and file paths.

Fixes applied:
  - Bug 8:  Non-existent model paths are no longer registered. If model_path
            does not exist, the function raises RuntimeError instead of
            storing a broken path that makes get_best_model() return a model
            that can't be deployed.
  - Fix D:  register_model() now logs an explicit warning when it is about
            to overwrite an existing registry entry. Previously the upsert
            was silent — if you accidentally re-ran exp_001 with different
            parameters, the old scores and paths were gone with no indication.

New features (production additions):
  - mark_champion(): manually mark a model as champion based on your own
    listening test, overriding the automated composite score ranking.
    This is essential because heuristic metrics can't replace your ears.
  - validate_registry(): scan all registry entries and report broken paths,
    so you catch drift (e.g., files moved) before a run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from .utils import Logger, load_json, now_iso, save_json

_REGISTRY_FILE = "models/registry.json"


def _load_registry(base_dir: str) -> dict:
    path = os.path.join(base_dir, _REGISTRY_FILE)
    if os.path.exists(path):
        try:
            return load_json(path)
        except Exception:
            pass
    return {"models": [], "champion": None, "updated_at": now_iso()}


def _save_registry(base_dir: str, data: dict) -> None:
    data["updated_at"] = now_iso()
    save_json(os.path.join(base_dir, _REGISTRY_FILE), data)


# ── Public API ────────────────────────────────────────────────────────────────

def register_model(
    experiment_id: str,
    model_path: str,
    index_path: str,
    config: dict,
    scores: dict,
    dataset_name: str = "",
    base_dir: str = ".",
    log: Optional[Logger] = None,
) -> None:
    """
    Register a trained model in the project registry.

    Copies .pth and .index into models/<experiment_id>/ and records
    all metadata in models/registry.json.

    Bug 8 fix: if model_path does not exist, raises RuntimeError instead
    of registering a broken path. A registry entry with a missing model
    file is worse than no entry — it causes silent deployment failures.

    Fix D: logs an explicit warning when overwriting an existing entry so
    accidental re-runs of the same experiment ID are immediately visible.

    Args:
        experiment_id: Unique experiment name.
        model_path:    Path to the stripped .pth checkpoint.
        index_path:    Path to the FAISS .index file (may be empty string).
        config:        Training config dict used for this experiment.
        scores:        Quality scores dict {clarity, naturalness, identity}.
        dataset_name:  Name of the dataset used (for display).
        base_dir:      Project root (where models/ directory lives).
        log:           Logger instance.

    Raises:
        RuntimeError: If model_path does not exist.
    """
    if log is None:
        log = Logger()

    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(
            f"Cannot register {experiment_id}: model file not found at:\n"
            f"  {model_path}\n"
            "Training may have failed to produce a checkpoint. "
            "Check training output above."
        )

    dest_dir = os.path.join(base_dir, "models", experiment_id)
    os.makedirs(dest_dir, exist_ok=True)

    dst_pth = os.path.join(dest_dir, Path(model_path).name)
    if os.path.abspath(model_path) != os.path.abspath(dst_pth):
        shutil.copy2(model_path, dst_pth)
    model_size_mb = round(os.path.getsize(dst_pth) / 1e6, 1)

    has_index     = bool(index_path and os.path.exists(index_path))
    dst_idx       = ""
    index_size_mb = 0.0
    if has_index:
        dst_idx = os.path.join(dest_dir, Path(index_path).name)
        if os.path.abspath(index_path) != os.path.abspath(dst_idx):
            shutil.copy2(index_path, dst_idx)
        index_size_mb = round(os.path.getsize(dst_idx) / 1e6, 1)
    else:
        log.info("  No FAISS index provided (inference will run without index)")

    composite = round(
        scores.get("naturalness", 0) * 0.45 +
        scores.get("clarity",     0) * 0.35 +
        scores.get("identity",    0) * 0.20,
        4,
    )

    entry = {
        "experiment_id":  experiment_id,
        "dataset":        dataset_name,
        "config_name":    config.get("name", "custom"),
        "registered_at":  now_iso(),
        "model_path":     dst_pth,
        "index_path":     dst_idx,
        "model_size_mb":  model_size_mb,
        "index_size_mb":  index_size_mb,
        "champion":       False,
        "scores": {
            **scores,
            "composite": composite,
        },
        "config": config,
    }

    registry = _load_registry(base_dir)

    # Fix D: warn before overwriting an existing entry so accidental re-runs
    # are visible. The old entry's scores and paths are about to be replaced.
    existing_ids = [m["experiment_id"] for m in registry["models"]]
    if experiment_id in existing_ids:
        log.warn(
            f"  Overwriting existing registry entry for '{experiment_id}'.\n"
            "  The previous scores and model paths will be replaced.\n"
            "  If this was unintentional, use a new experiment ID to "
            "preserve both runs (e.g. exp_001a vs exp_001b)."
        )

    registry["models"] = [
        m for m in registry["models"] if m["experiment_id"] != experiment_id
    ]
    registry["models"].append(entry)
    _save_registry(base_dir, registry)

    log.ok(
        f"Registered: {experiment_id}  "
        f"(naturalness={scores.get('naturalness',0):.3f}  "
        f"composite={composite:.3f})"
    )
    log.info(f"   .pth  : {dst_pth}  ({model_size_mb} MB)")
    if dst_idx:
        log.info(f"   .index: {dst_idx}  ({index_size_mb} MB)")


def mark_champion(
    experiment_id: str,
    base_dir: str = ".",
    reason: str = "",
    log: Optional[Logger] = None,
) -> None:
    """
    Manually mark an experiment as the champion model based on listening test.

    This overrides the automated composite score ranking. The champion is
    always returned first by get_best_model() when it exists.

    Use this when your ears tell you a model sounds better than the highest-
    scoring model — which happens regularly because heuristic metrics are
    imperfect proxies for perceived quality.

    Args:
        experiment_id: The experiment to mark as champion.
        base_dir:      Project root.
        reason:        Optional note on why this model was chosen.
        log:           Logger instance.

    Raises:
        ValueError: If experiment_id is not found in the registry.
    """
    if log is None:
        log = Logger()

    registry = _load_registry(base_dir)
    found = False

    for m in registry["models"]:
        was_champion = m.get("champion", False)
        if m["experiment_id"] == experiment_id:
            m["champion"]          = True
            m["champion_reason"]   = reason or "manually selected"
            m["champion_set_at"]   = now_iso()
            found = True
            if not was_champion:
                log.ok(f"Champion set: {experiment_id}")
                if reason:
                    log.info(f"  Reason: {reason}")
        else:
            m["champion"] = False

    if not found:
        raise ValueError(
            f"Experiment '{experiment_id}' not found in registry.\n"
            f"Registered experiments: "
            f"{[m['experiment_id'] for m in registry['models']]}"
        )

    registry["champion"] = experiment_id
    _save_registry(base_dir, registry)


def clear_champion(base_dir: str = ".", log: Optional[Logger] = None) -> None:
    """Remove champion designation — fall back to composite score ranking."""
    if log is None:
        log = Logger()

    registry = _load_registry(base_dir)
    for m in registry["models"]:
        m["champion"] = False
    registry["champion"] = None
    _save_registry(base_dir, registry)
    log.info("Champion cleared — ranking by composite score.")


def list_models(base_dir: str = ".") -> list[dict]:
    """
    Return all registered models sorted by:
    1. Champion (manually selected by listening test) — always first
    2. Composite score (descending)
    """
    registry = _load_registry(base_dir)
    return sorted(
        registry.get("models", []),
        key=lambda m: (
            int(m.get("champion", False)),
            m.get("scores", {}).get("composite", 0.0),
        ),
        reverse=True,
    )


def get_best_model(base_dir: str = ".") -> Optional[dict]:
    """
    Return the best registered model, or None if registry is empty.

    If a champion has been manually set (via mark_champion()), it is
    returned first. Otherwise the highest composite score wins.
    """
    models = list_models(base_dir)
    return models[0] if models else None


def validate_registry(base_dir: str = ".", log: Optional[Logger] = None) -> list[str]:
    """
    Scan all registry entries for broken file paths.
    Returns list of experiment_ids with broken paths (empty = all good).

    Run this before a training session to catch drift (moved/deleted files).
    """
    if log is None:
        log = Logger()

    models  = list_models(base_dir)
    broken: list[str] = []

    log.section("REGISTRY VALIDATION")
    for m in models:
        exp_id     = m["experiment_id"]
        pth_ok     = os.path.exists(m.get("model_path", ""))
        idx_ok     = not m.get("index_path") or os.path.exists(m["index_path"])
        champ_mark = " [CHAMPION]" if m.get("champion") else ""

        if pth_ok and idx_ok:
            log.ok(f"  {exp_id}{champ_mark}")
        else:
            log.error(f"  {exp_id}{champ_mark} — BROKEN PATH(S)")
            if not pth_ok:
                log.warn(f"    .pth missing: {m.get('model_path')}")
            if not idx_ok:
                log.warn(f"    .index missing: {m.get('index_path')}")
            broken.append(exp_id)

    if not broken:
        log.ok(f"All {len(models)} registered models have valid paths.")
    else:
        log.warn(
            f"{len(broken)}/{len(models)} models have broken paths: {broken}\n"
            "  Re-run those experiments or update paths manually in registry.json"
        )

    return broken


def print_registry(base_dir: str = ".", log: Optional[Logger] = None) -> None:
    """Pretty-print all registered models in a table."""
    if log is None:
        log = Logger()

    models = list_models(base_dir)
    if not models:
        log.warn("Registry is empty — no models registered yet.")
        return

    log.section("MODEL REGISTRY")
    header = (
        f"{'Experiment':<20} {'Dataset':<16} {'Config':<14} "
        f"{'Natural':>8} {'Composite':>10} {'Flag'}"
    )
    log.info(header)
    log.info("─" * len(header))

    for m in models:
        sc    = m.get("scores", {})
        flag  = "★ CHAMPION" if m.get("champion") else ""
        log.info(
            f"  {m['experiment_id']:<18}  "
            f"{m.get('dataset','?'):<14}  "
            f"{m.get('config_name','?'):<12}  "
            f"{sc.get('naturalness',0):.3f}    "
            f"{sc.get('composite',0):.3f}    "
            f"{flag}"
        )

    best = models[0]
    sc   = best.get("scores", {})
    tag  = "Champion" if best.get("champion") else "Best by score"
    log.ok(
        f"{tag}: {best['experiment_id']}  "
        f"(composite={sc.get('composite',0):.3f})"
    )
    log.info(f"   Model: {best['model_path']}")
    if best.get("index_path"):
        log.info(f"   Index: {best['index_path']}")
