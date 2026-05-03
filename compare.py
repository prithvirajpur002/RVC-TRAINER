# -*- coding: utf-8 -*-
"""
Catalyst RVC — src/compare.py
Compare multiple experiments and pick the best model.

Scoring philosophy:
  naturalness  — penalises over-processing (flat dynamics = compressed = bad)
  clarity      — rewards adequate loudness without clipping
  identity     — proxy via silence_ratio: excessive gaps = bad conversion

These are HEURISTIC scores — use them to spot broken experiments quickly,
not as absolute ground truth. Always do a listening test for final selection.

Fixes applied:
  - Bug 3:  Identity penalty was catastrophically aggressive. Natural speech
            has 30–50% silence (pauses, breaths), but the old formula gave
            any silence > 20% a score of 0.0. Fixed: only penalises silence
            ABOVE a 35% natural baseline so real speech scores fairly.
  - Bug 11: Crest factor target corrected from 13 dB to 9 dB. RVC output
            typically has 6–10 dB crest factor. The old target of 13 dB
            systematically penalised all good RVC output as "unnatural".
  - Fix C:  Clarity RMS floor widened from -20 dBFS to -28 dBFS.
            The old window [-20, -8] dBFS gave clarity=0 for any speech
            at -21 dBFS or quieter — a completely audible and usable level.
            New window [-28, -8] dBFS (range = 20 dB) scores natural
            speech levels correctly. Voice recorded at -24 dBFS now
            receives clarity=0.2 instead of 0.0.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .utils import Logger, load_json, now_iso, save_json


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_from_metrics(metrics: dict) -> dict[str, float]:
    """
    Derive heuristic quality scores from audio metrics.
    All scores are in [0, 1] — higher is better.

    After Bug 2 fix in evaluate.py, metrics are in correct dBFS range:
      rms_db:    typically -24 to -8 dBFS for voice
      peak_db:   typically -3 to -0.5 dBFS
      crest_db:  typically 6–14 dB for natural voice
      silence_ratio: typically 0.30–0.50 for natural speech
    """
    rms_db    = metrics.get("rms_db",       -20.0)
    peak_db   = metrics.get("peak_db",       -3.0)
    crest_db  = metrics.get("crest_db",       9.0)
    silence_r = metrics.get("silence_ratio",  0.35)

    # ── Clarity: reward -28 to -8 dBFS RMS, penalise silence and clipping ────
    # Fix C: floor widened from -20 to -28 dBFS (range = 20 dB).
    # Old: (rms_db + 20) / 12 → clarity=0 for anything below -20 dBFS.
    # New: (rms_db + 28) / 20 → clarity=0 only below -28 dBFS.
    # Voice at -24 dBFS (very common) now scores 0.2 instead of 0.0.
    rms_norm     = max(0.0, min(1.0, (rms_db + 28.0) / 20.0))
    clip_penalty = max(0.0, (peak_db + 0.5) / 0.5) if peak_db > -0.5 else 0.0
    clarity      = max(0.0, rms_norm - clip_penalty * 0.5)

    # ── Naturalness: reward RVC-typical crest factor (6–12 dB) ───────────────
    # Bug 11 fix: target changed from 13 dB to 9 dB.
    # RVC output typically has 6–10 dB crest. Old target of 13 dB scored all
    # good RVC output as unnatural. New window: [1, 17] dB centred on 9 dB.
    crest_norm  = max(0.0, min(1.0, 1.0 - abs(crest_db - 9.0) / 8.0))
    naturalness = crest_norm

    # ── Identity proxy: penalise silence ABOVE natural baseline ───────────────
    # Bug 3 fix: natural speech has 30–50% silence. Old formula penalised
    # anything over 20%, meaning all real speech got identity = 0.
    # New formula: only penalise silence ABOVE 35% baseline.
    #   silence_r = 0.40 → penalty = 0.05 * 10 = 0.5 → identity = 0.5 (mild)
    #   silence_r = 0.60 → penalty = 0.25 * 10 = 2.5 → identity = 0.0 (bad)
    #   silence_r = 0.30 → no penalty → identity = 1.0 (good)
    excess_silence = max(0.0, silence_r - 0.35)
    identity       = max(0.0, 1.0 - excess_silence * 10.0)

    return {
        "clarity":     round(clarity,     3),
        "naturalness": round(naturalness, 3),
        "identity":    round(identity,    3),
    }


def _aggregate_scores(eval_results: dict) -> dict[str, float]:
    """Average scores across all successful test clips."""
    all_scores: list[dict] = []
    for test_data in eval_results.values():
        if not test_data.get("success"):
            continue
        metrics = test_data.get("metrics", {})
        if not metrics or "error" in metrics:
            continue
        all_scores.append(_score_from_metrics(metrics))

    if not all_scores:
        return {"clarity": 0.0, "naturalness": 0.0, "identity": 0.0}

    keys = ["clarity", "naturalness", "identity"]
    return {
        k: round(sum(s[k] for s in all_scores) / len(all_scores), 3)
        for k in keys
    }


def _composite(scores: dict) -> float:
    """
    Weighted composite score.
    Naturalness weighted highest — the most common failure mode in RVC is
    over-processing, which destroys naturalness while clarity stays high.
    """
    return (
        scores.get("naturalness", 0.0) * 0.45 +
        scores.get("clarity",     0.0) * 0.35 +
        scores.get("identity",    0.0) * 0.20
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compare_experiments(
    results: dict[str, dict],
    log: Optional[Logger] = None,
) -> str:
    """
    Compare experiment scores and return the best experiment_id.

    Args:
        results: {experiment_id: {"clarity": float, "naturalness": float,
                                   "identity": float}}

    Returns:
        experiment_id of the best-scoring experiment.
    """
    if log is None:
        log = Logger()

    if not results:
        raise ValueError("results dict is empty — no experiments to compare.")

    log.section("EXPERIMENT COMPARISON")

    header = (
        f"{'Experiment':<20} {'Clarity':>9} {'Natural':>9} "
        f"{'Identity':>9} {'Composite':>10}"
    )
    log.info(header)
    log.info("─" * len(header))

    best_id    = None
    best_score = -1.0

    for exp_id, scores in sorted(results.items()):
        c = _composite(scores)
        marker = ""
        if c > best_score:
            best_score = c
            best_id    = exp_id
            marker     = "  ◀ best so far"

        log.info(
            f"  {exp_id:<18}  "
            f"{scores.get('clarity',0):.3f}     "
            f"{scores.get('naturalness',0):.3f}     "
            f"{scores.get('identity',0):.3f}    "
            f"{c:.3f}"
            f"{marker}"
        )

    log.info("─" * len(header))
    log.ok(f"Best experiment: {best_id}  (composite={best_score:.3f})")
    log.info(
        "  ↳ Always do a listening test — these scores are proxy metrics only."
    )

    return best_id


def save_comparison(
    results: dict[str, dict],
    output_path: str,
    log: Optional[Logger] = None,
) -> None:
    """
    Save comparison results to JSON and generate a bar chart if matplotlib is available.
    """
    if log is None:
        log = Logger()

    best_id = max(results, key=lambda k: _composite(results[k]))

    data = {
        "experiments": {
            exp_id: {
                **scores,
                "composite": round(_composite(scores), 4),
            }
            for exp_id, scores in results.items()
        },
        "best_experiment": best_id,
        "generated_at":    now_iso(),
    }
    save_json(output_path, data)
    log.ok(f"Comparison report: {output_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        exp_ids = sorted(results.keys())
        metrics = ["clarity", "naturalness", "identity"]
        colors  = ["#4ec9b0", "#569cd6", "#dcdcaa"]
        x       = np.arange(len(exp_ids))
        width   = 0.25

        fig, ax = plt.subplots(
            figsize=(max(6, len(exp_ids) * 1.8), 5),
            facecolor="#1e1e1e",
        )
        ax.set_facecolor("#252526")

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            vals = [results[eid].get(metric, 0.0) for eid in exp_ids]
            bars = ax.bar(x + i * width, vals, width,
                          label=metric.capitalize(), color=color)
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center", va="bottom", color="white", fontsize=8,
                )

        if best_id in exp_ids:
            best_idx = exp_ids.index(best_id)
            ax.axvspan(
                best_idx - 0.1, best_idx + 3 * width + 0.1,
                alpha=0.12, color="gold", label=f"Best: {best_id}",
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels(exp_ids, color="white", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", color="white")
        ax.set_title(
            "Catalyst RVC — Experiment Comparison\n"
            "(scores are proxy metrics — listening test required)",
            color="white", fontweight="bold", fontsize=10,
        )
        ax.tick_params(colors="white")
        ax.legend(
            facecolor="#333", edgecolor="#444",
            labelcolor="white", fontsize=9,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        chart_path = str(Path(output_path).with_suffix(".png"))
        plt.tight_layout()
        plt.savefig(chart_path, dpi=130, bbox_inches="tight", facecolor="#1e1e1e")
        plt.close(fig)
        log.ok(f"Comparison chart: {chart_path}")

    except ImportError:
        log.info("matplotlib not installed — chart skipped (pip install matplotlib)")
    except Exception as e:
        log.warn(f"Chart generation failed: {e}")


def score_experiment(eval_results: dict) -> dict[str, float]:
    """
    Convert raw evaluate() output → {clarity, naturalness, identity} scores.
    Call this after evaluate() to get scores for compare_experiments().
    """
    return _aggregate_scores(eval_results)
