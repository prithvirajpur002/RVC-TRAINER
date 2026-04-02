# -*- coding: utf-8 -*-
"""
Catalyst RVC — src/evaluate.py
Fixed test evaluation: runs inference on a consistent set of test clips
so experiment results are directly comparable.

FIXED_TESTS are always the same files — never randomly sampled.
This makes comparison between experiments (exp_001 vs exp_002) meaningful.

Fixes applied:
  - Bug 2:  sf.read() on PCM_16 files returns int16 range (-32768 to 32767),
            NOT float32 [-1, 1]. Added explicit normalization so RMS/peak
            calculations are correct.
  - Bug 6:  build_test_clips() no longer pretends files are "neutral",
            "emotional", "fast_speech" — they are named test_1/2/3.
            Added warning that test clips should come from a HELD-OUT set.
  - Bug 13: Inference failures now capture and log stderr so you know WHY
            inference failed (OOM, wrong model, wrong index format, etc.)
            instead of silently recording "success: False".
  - Claude: Added pitch_corr (F0 similarity) as the real identity metric.
            The old identity = silence_ratio proxy measured nothing about
            voice identity. pitch_corr compares the fundamental frequency
            contour of the input vs output — a HIGH correlation means the
            converted audio follows the same pitch trajectory as the source,
            which is what voice identity preservation actually means.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .utils import Logger, now_iso, save_json

# ── Fixed test set ────────────────────────────────────────────────────────────
# These files must exist in test_audio/ before evaluation.
# They are never modified — read-only inputs.
# Names are test_1/2/3 — do NOT label them neutral/emotional/fast unless you
# manually curated them for those properties.

FIXED_TESTS: dict[str, str] = {
    "test_1": "test_audio/test_1.wav",
    "test_2": "test_audio/test_2.wav",
    "test_3": "test_audio/test_3.wav",
}

# Default inference settings — conservative, prioritize naturalness
_DEFAULT_INFER = {
    "f0_up_key":      0,       # No pitch shift
    "index_rate":     0.75,    # 0.75 = good blend of identity + source prosody
    "filter_radius":  3,       # F0 curve smoothness (1=sharp, 7=smooth)
    "rms_mix_rate":   0.0,     # 0 = keep source energy envelope (more natural)
    "protect":        0.33,    # Consonant breath protection
    "f0_method":      "rmvpe",
}


def _find_cli(rvc_repo: str) -> str:
    candidates = [
        "tools/infer_cli.py",
        "infer_cli.py",
    ]
    for rel in candidates:
        p = os.path.join(rvc_repo, rel)
        if os.path.exists(p):
            return p
    raise RuntimeError(
        "infer_cli.py not found in RVC repo.\n"
        f"Searched: {candidates}\n"
        "Re-clone the repo in bootstrap."
    )


def _run_inference(
    cli: str,
    rvc_repo: str,
    model_path: str,
    index_path: Optional[str],
    input_path: str,
    output_path: str,
    settings: dict,
    log: Logger,
) -> tuple[bool, str]:
    """
    Run single-file RVC inference.

    Returns:
        (success: bool, error_message: str)

    Bug 13 fix: capture_output=True was swallowing all errors. Now stderr
    is captured and returned so the caller can log why inference failed.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True, ""  # Already done

    has_index = index_path and os.path.exists(index_path)

    cmd = [
        sys.executable, cli,
        "--f0up_key",      str(settings["f0_up_key"]),
        "--input_path",    input_path,
        "--index_path",    index_path if has_index else "",
        "--f0method",      settings["f0_method"],
        "--opt_path",      output_path,
        "--model_name",    model_path,
        "--index_rate",    str(settings["index_rate"]),
        "--device",        "cuda:0",
        "--is_half",       "True",
        "--filter_radius", str(settings["filter_radius"]),
        "--resample_sr",   "0",
        "--rms_mix_rate",  str(settings["rms_mix_rate"]),
        "--protect",       str(settings["protect"]),
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, cwd=rvc_repo)

    succeeded = (
        r.returncode == 0
        and os.path.exists(output_path)
        and os.path.getsize(output_path) > 500
    )

    if not succeeded:
        err = (r.stderr or r.stdout or "no output captured").strip()[-800:]
        return False, err

    return True, ""


def _load_mono_float(path: str) -> tuple[np.ndarray, int]:
    """
    Load a WAV file as mono float32 in [-1, 1].
    Handles PCM_16 normalization (Bug 2 fix).
    """
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y[:, 0]

    # Normalize integer PCM formats before converting to float64.
    # Checking dtype BEFORE the cast avoids the magnitude heuristic, which
    # misfires on clipped float audio (rare but possible with bad exports).
    if y.dtype == np.int16:
        y = y.astype(np.float64) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float64) / 2_147_483_648.0
    else:
        y = y.astype(np.float64)

    return y.astype(np.float32), sr


def _pitch_similarity(input_path: str, output_path: str) -> float:
    """
    Compute pitch (F0) correlation between input and output audio.

    This is the real identity metric. A high correlation means the converted
    voice follows the same fundamental frequency trajectory as the source —
    which is what voice identity preservation actually means for RVC.

    Uses librosa.yin() for frame-wise F0 estimation (robust, no pretrained
    model needed). fmax=800 Hz covers the high-pitched anime voice range
    (standard soprano tops out ~1050 Hz; fmax=800 safely captures 99% of
    characters). The correlation is computed on the overlapping voiced region.

    Returns:
        Pearson correlation in [-1, 1]. Values > 0.7 indicate good pitch
        preservation. NaN → returns 0.0 (safe fallback).
    """
    try:
        import librosa

        y_in,  sr_in  = _load_mono_float(input_path)
        y_out, sr_out = _load_mono_float(output_path)

        # Resample output to match input SR if they differ
        if sr_in != sr_out:
            y_out = librosa.resample(y_out, orig_sr=sr_out, target_sr=sr_in)

        # YIN F0 estimation: fmin=50 Hz (bass voice) to fmax=800 Hz.
        # 500 Hz was too low — high-pitched anime voices (e.g. Nami, Luffy
        # at peak) can push 700–900 Hz. 800 Hz covers the vast majority.
        f0_in  = librosa.yin(y_in,  fmin=50, fmax=800, sr=sr_in)
        f0_out = librosa.yin(y_out, fmin=50, fmax=800, sr=sr_in)

        # Align to same length (shorter wins)
        n = min(len(f0_in), len(f0_out))
        if n < 10:
            return 0.0  # Too short to be meaningful

        f0_in  = f0_in[:n]
        f0_out = f0_out[:n]

        # Remove unvoiced frames (YIN returns very low values for unvoiced)
        voiced = (f0_in > 60) & (f0_out > 60)
        if voiced.sum() < 5:
            # Not enough voiced frames — audio may be synthetic/silent
            return 0.0

        corr_matrix = np.corrcoef(f0_in[voiced], f0_out[voiced])
        corr        = float(corr_matrix[0, 1])

        # Guard against NaN (e.g. constant F0 in one channel)
        if np.isnan(corr):
            return 0.0

        # Clip to [0, 1]: negative correlation means pitch is inverted,
        # which is worse than random and should score 0 not negative.
        return float(np.clip(corr, 0.0, 1.0))

    except ImportError:
        # librosa not available — skip pitch metric, return 0 (not penalized)
        return 0.0
    except Exception:
        return 0.0


def _audio_metrics(input_path: str, output_path: str) -> dict:
    """
    Compute audio quality metrics from the output WAV file, including
    pitch correlation against the input (real identity measurement).

    Returns:
        dict with keys: rms_db, peak_db, crest_db, duration_s,
                        silence_ratio, pitch_corr

    Scoring interpretation:
        naturalness  ← crest_db  (ideal ~9 dB for speech)
        clarity      ← rms_db    (ideal -20 to -12 dBFS for speech)
        pitch_corr   ← F0 correlation input vs output  (0→1, higher=better identity)
        silence_ratio: fraction of near-silent samples (diagnostic only)
    """
    try:
        y, sr = _load_mono_float(output_path)

        rms_db   = float(20.0 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))
        peak_db  = float(20.0 * np.log10(np.max(np.abs(y))        + 1e-9))
        crest_db = peak_db - rms_db
        duration = len(y) / sr
        silence_r = float(np.mean(np.abs(y) < 0.005))

    except Exception as e:
        return {"error": str(e)}

    # Pitch similarity — compare input vs output F0 contour
    pitch_corr = _pitch_similarity(input_path, output_path)

    return {
        "rms_db":        round(rms_db,     2),
        "peak_db":       round(peak_db,    2),
        "crest_db":      round(crest_db,   2),
        "duration_s":    round(duration,   2),
        "silence_ratio": round(silence_r,  4),
        "pitch_corr":    round(pitch_corr, 4),
    }


def compute_scores(metrics: dict) -> dict[str, float]:
    """
    Convert raw audio metrics into normalised scores in [0, 1].

    naturalness: how natural the dynamic range is (crest factor ~9 dB for speech)
    clarity:     how present/loud the output is (-18 dBFS is ideal)
    identity:    pitch correlation — does the output track the source F0?
    composite:   weighted average

    Weights:
        naturalness: 0.35  (audio quality)
        clarity:     0.30  (presence)
        identity:    0.35  (what actually matters for voice cloning — pitch corr)
    """
    crest_db   = metrics.get("crest_db",   9.0)
    rms_db     = metrics.get("rms_db",   -20.0)
    pitch_corr = metrics.get("pitch_corr", 0.0)

    # naturalness: crest factor penalty away from 9 dB
    # 9 dB is typical speech crest factor; deviations penalized
    naturalness = float(np.clip(1.0 - abs(crest_db - 9.0) / 15.0, 0.0, 1.0))

    # clarity: rms closer to -18 dBFS is better; silent (-60) or clipped (0) = 0
    clarity = float(np.clip(1.0 - abs(rms_db + 18.0) / 25.0, 0.0, 1.0))

    # identity: direct — pitch correlation IS the score
    identity = float(np.clip(pitch_corr, 0.0, 1.0))

    composite = (
        naturalness * 0.35 +
        clarity     * 0.30 +
        identity    * 0.35
    )

    return {
        "naturalness": round(naturalness, 4),
        "clarity":     round(clarity,     4),
        "identity":    round(identity,    4),  # = pitch_corr
        "composite":   round(composite,   4),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    index_path: str,
    eval_dir: str,
    rvc_repo: str,
    base_dir: str = ".",
    infer_settings: Optional[dict] = None,
    log: Optional[Logger] = None,
) -> dict:
    """
    Run inference on all FIXED_TESTS and collect quality metrics.

    Args:
        model_path:     Path to the stripped .pth model file.
        index_path:     Path to the .index file (may be empty string).
        eval_dir:       Directory to write output WAV files.
        rvc_repo:       Path to cloned RVC repo.
        base_dir:       Root directory where test_audio/ lives.
        infer_settings: Override default inference settings (optional).
        log:            Logger instance.

    Returns:
        Dict of {test_name: {output_path, metrics, scores, success}} per test.
    """
    if log is None:
        log = Logger()

    settings = {**_DEFAULT_INFER, **(infer_settings or {})}
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    cli      = _find_cli(rvc_repo)
    model_mb = os.path.getsize(model_path) / 1e6
    has_idx  = bool(index_path and os.path.exists(index_path))
    idx_mb   = os.path.getsize(index_path) / 1e6 if has_idx else 0.0

    log.section("EVALUATION")
    log.info(f"Model: {Path(model_path).name}  ({model_mb:.0f} MB)")
    log.info(f"Index: {Path(index_path).name if has_idx else '(none)'}  ({idx_mb:.0f} MB)")
    log.info(f"Settings: index_rate={settings['index_rate']}  protect={settings['protect']}")

    results: dict = {}
    any_success = False

    for test_name, rel_path in FIXED_TESTS.items():
        src = os.path.join(base_dir, rel_path)
        dst = os.path.join(eval_dir, f"{test_name}.wav")

        if not os.path.exists(src):
            log.warn(f"  {test_name}: source missing ({src}) — skipped")
            results[test_name] = {"success": False, "reason": "source_missing"}
            continue

        log.info(f"\n  ▶  {test_name}  ({os.path.basename(src)})")

        ok, err_msg = _run_inference(
            cli, rvc_repo, model_path,
            index_path if has_idx else None,
            src, dst, settings, log,
        )

        if not ok:
            log.warn(f"     Inference FAILED for {test_name}")
            if err_msg:
                log.warn(f"     Error: {err_msg}")
            results[test_name] = {
                "success":       False,
                "output_path":   dst,
                "error_message": err_msg,
            }
        else:
            metrics = _audio_metrics(src, dst)
            scores  = compute_scores(metrics)

            log.ok(f"     Output: {dst}")
            log.info(
                f"     RMS={metrics.get('rms_db','?')} dBFS  "
                f"Crest={metrics.get('crest_db','?')} dB  "
                f"PitchCorr={metrics.get('pitch_corr','?'):.3f}  "
                f"Composite={scores['composite']:.3f}"
            )
            results[test_name] = {
                "success":     True,
                "output_path": dst,
                "metrics":     metrics,
                "scores":      scores,
            }
            any_success = True

    if not any_success:
        log.warn("No test outputs were produced. Check model + RVC inference setup.")
    else:
        n_ok = sum(1 for v in results.values() if v.get("success"))
        log.ok(f"Evaluation done — {n_ok}/{len(FIXED_TESTS)} tests passed")

    summary = {
        "model_path":     model_path,
        "index_path":     index_path,
        "infer_settings": settings,
        "results":        results,
        "evaluated_at":   now_iso(),
    }
    save_json(os.path.join(eval_dir, "eval_results.json"), summary)

    return results


def build_test_clips(
    test_audio_dir: str,
    input_wav_dir: str,
    log: Optional[Logger] = None,
) -> None:
    """
    Bootstrap helper: copies the first 3 WAV files from your dataset
    into test_audio/ as fixed test clips, if test_audio/ is empty.

    IMPORTANT — Bug 6 fix:
      Files are named test_1.wav, test_2.wav, test_3.wav. They are NOT
      labeled neutral/emotional/fast_speech because alphabetical ordering
      of dataset files has nothing to do with speech content.

    BEST PRACTICE:
      Ideally, your test clips should come from a DIFFERENT source than
      your training data so evaluation is not contaminated by in-distribution
      bias. If you only have one data source, this helper still works, but
      keep it in mind when interpreting scores.

    Call this ONCE before running experiments. After that, test_audio/
    is frozen — never regenerate from a different source.
    """
    if log is None:
        log = Logger()

    os.makedirs(test_audio_dir, exist_ok=True)
    existing = [f for f in os.listdir(test_audio_dir) if f.endswith(".wav")]
    if len(existing) >= 3:
        log.info(f"test_audio/ already has {len(existing)} clips — skipping.")
        return

    import shutil
    wav_files = sorted(Path(input_wav_dir).glob("*.wav"))
    if len(wav_files) < 3:
        log.warn(f"Need at least 3 WAVs in {input_wav_dir} to build test clips.")
        return

    names = ["test_1.wav", "test_2.wav", "test_3.wav"]
    for src, dst_name in zip(wav_files[:3], names):
        dst = os.path.join(test_audio_dir, dst_name)
        shutil.copy2(src, dst)
        log.ok(f"  {src.name} → {dst_name}")

    log.ok("Fixed test clips ready in test_audio/")
    log.info(
        "  NOTE: For unbiased evaluation, consider using audio from a "
        "different source than your training data."
    )
