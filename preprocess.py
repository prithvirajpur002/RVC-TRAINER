# -*- coding: utf-8 -*-
"""
Catalyst RVC — src/preprocess.py
Audio preprocessing with three modes tuned for different source qualities.

NATURAL TONE PHILOSOPHY
-----------------------
Over-processing is the enemy of natural voice.  Less is more:

  raw     → No demucs, no deepfilter. Only resample + gentle HPF.
              Use when your recordings are already clean studio audio.

  natural → Skip Demucs (keeps breath transients), apply DeepFilterNet
              only for hiss. Lower SNR bar = more segments kept.
              Best for clean but not studio recordings.

  clean   → Full stack for noisy source (anime rip, live recording, mic
              bleed). Demucs + DeepFilter both on. Highest SNR threshold.

HPF NOTE: filtfilt() applies the filter twice (forward + backward), making
a butter(order=1) an effective 2nd-order filter. We use order=1 to achieve
the intended gentle 2nd-order rolloff that keeps male voice fundamentals
(85–180 Hz) intact. Using order=2 with filtfilt gives effective 4th-order
which over-attenuates low fundamentals.

Fixes applied:
  - Bug 9:  HPF order corrected — order=1 with filtfilt = effective 2nd-order
            (was order=2 which silently became 4th-order via filtfilt)
  - Bug 7:  SNR fallback retry — if all segments fail SNR filter, retries
            once with snr_min reduced by 5 dB before raising RuntimeError
  - Bug 12: _segment() fixed — large silence gaps now force a buffer flush
            to prevent mid-file silent sections inflating segment size
  - Bug 17: Deepfilter and Demucs failures are now explicitly logged with
            reason — no more silent fallback to unfiltered audio
  - Fix A:  _run_deepfilter() now explicitly deletes model and df_state
            in a finally block after each call. Previously the model stayed
            loaded in VRAM for the entire preprocessing loop, reducing
            available memory for training.
  - Fix B:  _run_demucs() now cleans up the full per-stem directory tree
            rather than only the leaf folder, preventing orphaned temp files
            from accumulating across multi-file runs.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

from .utils import Logger, compute_dataset_hash, now_iso, save_json

# ── Mode presets ──────────────────────────────────────────────────────────────

MODES: dict[str, dict] = {
    "clean": {
        "demucs":      True,
        "deepfilter":  True,
        "snr_min":     18.0,    # Strict: noisy source needs higher bar
        "hpf_hz":      80.0,
        "hpf_order":   1,       # order=1 + filtfilt = effective 2nd-order rolloff
        "norm_db":     -3.0,
        "min_dur_s":   3.0,
        "max_dur_s":   15.0,
        "top_db_trim": 25,
        "max_silence_s": 0.5,   # Max gap before forcing buffer flush in _segment
    },
    "natural": {
        "demucs":      False,   # Skip: keeps breath, room, natural transients
        "deepfilter":  True,    # Only: remove hiss without killing texture
        "snr_min":     15.0,    # Relaxed: accept more natural imperfections
        "hpf_hz":      80.0,
        "hpf_order":   1,
        "norm_db":     -3.0,
        "min_dur_s":   3.0,
        "max_dur_s":   15.0,
        "top_db_trim": 28,      # Less aggressive silence trim
        "max_silence_s": 0.7,
    },
    "raw": {
        "demucs":      False,
        "deepfilter":  False,   # No processing — trust the source
        "snr_min":     12.0,
        "hpf_hz":      80.0,
        "hpf_order":   1,
        "norm_db":     -3.0,
        "min_dur_s":   3.0,
        "max_dur_s":   15.0,
        "top_db_trim": 30,
        "max_silence_s": 1.0,
    },
}


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _hpf(y: np.ndarray, sr: int, cutoff_hz: float = 80.0, order: int = 1) -> np.ndarray:
    """
    Gentle Butterworth high-pass filter via filtfilt.

    filtfilt applies the filter forward and backward, so the effective order
    is 2× the butter() order. We use order=1 to achieve effective 2nd-order
    rolloff — preserving male voice fundamentals (85–180 Hz).

    Do NOT increase order here — it will over-attenuate low fundamentals.
    """
    nyq = 0.5 * sr
    normal_cutoff = cutoff_hz / nyq
    normal_cutoff = max(1e-4, min(normal_cutoff, 0.9999))
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, y).astype(np.float32)


def _peak_norm(y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Peak-normalize only if clip is loud — don't amplify quiet recordings."""
    peak = float(np.max(np.abs(y)))
    if peak < 1e-6:
        return y
    target_amp = 10 ** (target_db / 20.0)
    if peak > target_amp:
        return (y * (target_amp / peak)).astype(np.float32)
    return y


def _estimate_snr(y: np.ndarray, sr: int) -> float:
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    noise  = float(np.percentile(rms, 10)) + 1e-12
    signal = float(np.percentile(rms, 90)) + 1e-12
    return float(20.0 * np.log10(signal / noise))


def _fade(seg: np.ndarray, sr: int, ms: float = 40.0) -> np.ndarray:
    """40 ms micro-fades — just enough to prevent click artifacts, no more."""
    n = min(int(ms * sr / 1000), len(seg) // 4)
    if n < 1:
        return seg
    out = seg.copy()
    out[:n]  *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def _segment(
    y: np.ndarray,
    sr: int,
    min_s: float,
    max_s: float,
    top_db: int = 28,
    max_silence_s: float = 0.5,
) -> list[np.ndarray]:
    """
    Split audio into segments using silence detection.

    Greedy buffer accumulation: keeps building until max_s reached, then
    flushes. Large silent gaps (> max_silence_s) force an immediate flush
    so we don't create segments that are mostly silence.

    Falls back to time-chunking if no silence markers are found.

    Fix (Bug 12): Added max_silence_gap check — previously a long silence
    in the middle of a file could silently inflate the buffer past max_samp.
    """
    min_samp         = int(min_s * sr)
    max_samp         = int(max_s * sr)
    max_silence_samp = int(max_silence_s * sr)

    if len(y) < min_samp:
        return []

    ivs = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    if len(ivs) == 0:
        ivs = np.array([[0, len(y)]])

    segs: list[np.ndarray] = []
    buf_s = buf_e = None

    for s, e in ivs:
        if buf_s is None:
            buf_s, buf_e = s, e
            continue

        gap_samp    = s - buf_e
        buffer_full = (e - buf_s) > max_samp
        large_gap   = gap_samp > max_silence_samp

        if buffer_full or large_gap:
            seg_len = buf_e - buf_s
            if seg_len >= min_samp:
                segs.append(y[buf_s : buf_s + min(seg_len, max_samp)].copy())
            buf_s, buf_e = s, e
        else:
            buf_e = e

    if buf_s is not None and (buf_e - buf_s) >= min_samp:
        seg_len = buf_e - buf_s
        segs.append(y[buf_s : buf_s + min(seg_len, max_samp)].copy())

    if not segs:
        for off in range(0, len(y), max_samp):
            chunk = y[off : off + max_samp]
            if len(chunk) >= min_samp:
                segs.append(chunk.copy())

    return segs


def _run_demucs(src: str, tmp_dir: str, log: Logger) -> str:
    """
    Isolate vocals via Demucs htdemucs.
    Returns path to vocals.wav on success, or original src on failure.
    Logs the reason for failure explicitly (Bug 17 fix).

    Fix B: removes the entire per-stem directory tree after use, not just
    the leaf folder. Previously htdemucs model-level directories were left
    behind on each run, accumulating orphaned files across multi-file batches.
    """
    stem = Path(src).stem
    try:
        r = subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
             "-o", tmp_dir, src],
            capture_output=True, text=True, timeout=600,
        )
        cands = list(Path(tmp_dir).glob(f"*/{stem}/vocals.wav"))
        if cands and r.returncode == 0:
            return str(cands[0])
        reason = r.stderr[-500:].strip() if r.stderr else f"exit code {r.returncode}"
        log.warn(f"     Demucs failed ({reason}) — using original audio")
        return src
    except FileNotFoundError:
        log.warn("     Demucs not installed (pip install demucs) — using original audio")
        return src
    except subprocess.TimeoutExpired:
        log.warn("     Demucs timed out (>10 min) — using original audio")
        return src
    except Exception as exc:
        log.warn(f"     Demucs exception ({exc}) — using original audio")
        return src
    finally:
        # Fix B: clean up all directories containing this stem, not just the leaf
        import shutil
        for parent_dir in Path(tmp_dir).glob("*"):
            stem_dir = parent_dir / stem
            if stem_dir.exists():
                shutil.rmtree(stem_dir, ignore_errors=True)
            # Remove the parent model dir if it's now empty
            try:
                if parent_dir.is_dir() and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
            except OSError:
                pass


def _run_deepfilter(y: np.ndarray, sr: int, log: Logger) -> np.ndarray:
    """
    Apply DeepFilterNet noise reduction.
    Returns filtered audio on success, original y on failure.
    Logs the reason for failure explicitly (Bug 17 fix).

    Fix A: model and df_state are now explicitly deleted in a finally block
    after each call. Previously the model stayed loaded in VRAM through the
    entire preprocessing loop, reducing available memory for training.
    """
    model = None
    df_state = None
    try:
        import torch
        import torchaudio
        from df.enhance import enhance, init_df
        model, df_state, _ = init_df()
        y_t = torch.from_numpy(y).float().unsqueeze(0)
        if sr != df_state.sr():
            y_t = torchaudio.functional.resample(y_t, sr, df_state.sr())
        out = enhance(model, df_state, y_t).squeeze().cpu().numpy()
        if sr != df_state.sr():
            out = librosa.resample(out, orig_sr=df_state.sr(), target_sr=sr)
        return out.astype(np.float32)
    except ImportError:
        log.warn("     DeepFilterNet not installed (pip install deepfilternet) — skipping")
        return y
    except Exception as exc:
        log.warn(f"     DeepFilterNet failed ({exc}) — using unfiltered audio")
        return y
    finally:
        # Fix A: release model from VRAM immediately after each file is processed
        try:
            del model, df_state
        except NameError:
            pass
        gc.collect()
        try:
            import torch as _tc
            if _tc.cuda.is_available():
                _tc.cuda.empty_cache()
        except Exception:
            pass


# ── SNR-aware segment filter with fallback ───────────────────────────────────

def _filter_segments_by_snr(
    segs: list[np.ndarray],
    sr: int,
    snr_min: float,
    log: Logger,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Filter segments by SNR threshold. Returns (passing_segs, snr_values).
    """
    passing: list[np.ndarray] = []
    snr_vals: list[float]     = []

    for seg in segs:
        snr = _estimate_snr(seg, sr)
        if snr < snr_min:
            log.info(f"     ↷  SNR {snr:.1f} dB < {snr_min:.1f} — rejected")
        else:
            passing.append(seg)
            snr_vals.append(snr)

    return passing, snr_vals


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    mode: str = "natural",
    target_sr: int = 40000,
    log: Optional[Logger] = None,
) -> dict:
    """
    Process all WAV files in input_dir → clean segments in output_dir.

    Args:
        input_dir:  Directory containing source WAV files.
        output_dir: Destination for processed segments (RVC dataset_dir).
        mode:       One of "clean", "natural", "raw".
        target_sr:  Output sample rate (40000 for RVC v2).
        log:        Logger instance. A default one is created if None.

    Returns:
        metadata dict (also saved as metadata.json in output_dir).

    Raises:
        RuntimeError: If no segments survive even after SNR fallback retry.
        ValueError:   If mode is not one of the known modes.
    """
    if log is None:
        log = Logger()

    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODES)}")

    cfg = dict(MODES[mode])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tmp_demucs = "/tmp/catalyst_demucs"
    os.makedirs(tmp_demucs, exist_ok=True)

    log.section(f"PREPROCESSING  [{mode.upper()} mode]")
    log.info(f"Input  : {input_dir}")
    log.info(f"Output : {output_dir}")
    log.info(
        f"Demucs={cfg['demucs']}  DeepFilter={cfg['deepfilter']}  "
        f"SNR≥{cfg['snr_min']}dB  SR={target_sr}"
    )

    wav_files = sorted(
        f for f in Path(input_dir).glob("**/*.wav") if f.is_file()
    )
    if not wav_files:
        raise RuntimeError(f"No WAV files found in {input_dir}")

    log.info(f"Found {len(wav_files)} source file(s)")

    all_raw_segs: list[np.ndarray] = []
    file_sr: int = target_sr

    for src_path in wav_files:
        log.info(f"\n  ▶  {src_path.name}")
        audio_path = str(src_path)

        if cfg["demucs"]:
            log.info("     Demucs vocal isolation...")
            audio_path = _run_demucs(audio_path, tmp_demucs, log)

        try:
            y, file_sr = librosa.load(audio_path, sr=target_sr, mono=True)
        except Exception as e:
            log.warn(f"     Load failed ({e}) — skipping")
            continue

        if cfg["deepfilter"]:
            y = _run_deepfilter(y, file_sr, log)

        y = _hpf(y, file_sr, cfg["hpf_hz"], cfg["hpf_order"])
        y, _ = librosa.effects.trim(y, top_db=cfg["top_db_trim"])
        y = _peak_norm(y, cfg["norm_db"])

        segs = _segment(
            y, file_sr,
            cfg["min_dur_s"], cfg["max_dur_s"],
            cfg["top_db_trim"],
            cfg.get("max_silence_s", 0.5),
        )
        del y
        gc.collect()

        if not segs:
            log.warn("     Too short to segment — skipped")
            continue

        all_raw_segs.extend(segs)

    if not all_raw_segs:
        raise RuntimeError(
            f"No audio could be loaded or segmented from {input_dir}.\n"
            "Check that the input files are valid WAV files."
        )

    passing_segs, snr_vals = _filter_segments_by_snr(
        all_raw_segs, file_sr, cfg["snr_min"], log
    )

    if not passing_segs:
        fallback_snr = cfg["snr_min"] - 5.0
        log.warn(
            f"No segments passed SNR ≥ {cfg['snr_min']:.1f} dB.\n"
            f"  Retrying with relaxed threshold: SNR ≥ {fallback_snr:.1f} dB..."
        )
        passing_segs, snr_vals = _filter_segments_by_snr(
            all_raw_segs, file_sr, fallback_snr, log
        )

    if not passing_segs:
        raise RuntimeError(
            f"No segments survived the quality filter even after relaxing threshold.\n"
            f"  Mode '{mode}' tried SNR ≥ {cfg['snr_min']:.1f} dB, "
            f"then ≥ {cfg['snr_min'] - 5.0:.1f} dB.\n"
            "  Options:\n"
            "  • Switch to mode='raw' to skip SNR filtering entirely\n"
            "  • Add cleaner recordings\n"
            "  • Reduce snr_min in MODES config"
        )

    saved_count  = 0
    reject_count = len(all_raw_segs) - len(passing_segs)
    total_dur    = 0.0

    for seg_index, seg in enumerate(passing_segs, start=1):
        seg  = _fade(seg, file_sr, ms=40.0)
        dur  = len(seg) / file_sr
        name = f"v{seg_index:05d}.wav"
        path = os.path.join(output_dir, name)
        sf.write(path, seg, file_sr, subtype="PCM_16")
        total_dur += dur
        log.info(f"     ✔  {name}  {dur:.2f}s  SNR {snr_vals[seg_index-1]:.1f} dB")
        saved_count += 1

    metadata = {
        "name":               Path(output_dir).name,
        "mode":               mode,
        "total_segments":     saved_count,
        "rejected_segments":  reject_count,
        "total_duration_min": round(total_dur / 60, 2),
        "snr_range":          [round(min(snr_vals), 1), round(max(snr_vals), 1)],
        "avg_snr_db":         round(float(np.mean(snr_vals)), 2),
        "sample_rate":        target_sr,
        "processing":         cfg,
        "source_hash":        compute_dataset_hash(input_dir),
        "created_at":         now_iso(),
    }

    save_json(os.path.join(output_dir, "metadata.json"), metadata)
    log.ok(
        f"Preprocessing done — {saved_count} segments "
        f"({total_dur/60:.1f} min, {reject_count} rejected) → {output_dir}"
    )
    return metadata
