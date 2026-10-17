"""Audio-Analyse: Block-Merkmale (0,5 s), Segmentierung und Trigger-Klassifikation.

Die Klassen sind bewusst *heuristisch* (relativ zum Rauschboden und zur Datei) und erklaerbar,
kein trainiertes Klassifikationsmodell: silence, speech, tapping, brushing, breath, texture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .config import Config
from . import media

LABELS = ["silence", "speech", "tapping", "brushing", "breath", "texture"]
FEATURE_NAMES = ["rms_db", "peak_db", "crest_db", "centroid", "flatness", "onset_rate", "low", "mid", "high", "pan", "width", "corr"]
EPS = 1e-9
N_FFT, HOP = 1024, 256


def _db(x: float | np.ndarray) -> np.ndarray: return 20 * np.log10(np.maximum(x, EPS))


def block_features(block: np.ndarray, sr: int) -> list[float]:
    left, right = block[:, 0].astype(np.float64), block[:, 1].astype(np.float64)
    mid, side = (left + right) / 2, (left - right) / 2
    rms_l, rms_r = np.sqrt(np.mean(left ** 2)), np.sqrt(np.mean(right ** 2))
    rms = np.sqrt((rms_l ** 2 + rms_r ** 2) / 2)
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    rms_db, peak_db = float(_db(rms)), float(_db(peak))
    pan = float((rms_r - rms_l) / (rms_r + rms_l + EPS))
    width = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + EPS))
    corr = float(np.corrcoef(left, right)[0, 1]) if rms > 1e-6 else 1.0
    if len(mid) < N_FFT * 2 or rms < 1e-6:
        return [rms_db, peak_db, peak_db - rms_db, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, pan, width, corr if np.isfinite(corr) else 0.0]
    count = 1 + (len(mid) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(mid, (count, N_FFT), (mid.strides[0] * HOP, mid.strides[0]))
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1)) + EPS          # (T, F)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    power = spectrum ** 2
    energy = power.sum(axis=1)
    centroid = float(np.sum((spectrum * freqs).sum(axis=1)) / np.sum(spectrum.sum(axis=1)))
    band = (freqs > 100) & (freqs < 10000)
    flat = np.exp(np.mean(np.log(power[:, band]), axis=1)) / np.mean(power[:, band], axis=1)
    flatness = float(np.average(flat, weights=energy / energy.sum()))
    total = energy.sum()
    low = float(power[:, freqs < 500].sum() / total); high = float(power[:, freqs > 4000].sum() / total)
    # Onsets: Anstieg der Frame-Energie um >=8 dB innerhalb von ~35 ms, Refraktaerzeit 60 ms.
    e_db = 10 * np.log10(energy + EPS)
    onsets, last = 0, -100
    for t in range(3, count):
        if e_db[t] - e_db[t - 3:t].min() >= 8 and e_db[t] > e_db.max() - 25 and t - last >= 5:
            onsets += 1; last = t
    onset_rate = onsets / (len(mid) / sr)
    corr = corr if np.isfinite(corr) else 0.0
    return [rms_db, peak_db, peak_db - rms_db, centroid, flatness, onset_rate, low, 1 - low - high, high, pan, width, corr]


def extract_features(config: Config, path: Path, on_progress: Callable[[float], None] | None = None, duration: float = 0) -> np.ndarray:
    sr = int(config.get("analysis", "analysis_sample_rate", 22050))
    block_len = int(float(config.get("analysis", "block_seconds", 0.5)) * sr)
    rows, carry = [], np.zeros((0, 2), dtype=np.float32)
    done = 0.0
    for chunk in media.stream_audio(config, path, sr, 60.0):
        data = np.concatenate([carry, chunk]) if len(carry) else chunk
        whole = len(data) // block_len * block_len
        for start in range(0, whole, block_len):
            rows.append(block_features(data[start:start + block_len], sr))
        carry = data[whole:]
        done += len(chunk) / sr
        if on_progress and duration: on_progress(min(0.99, done / duration))
    if len(carry) > block_len // 2: rows.append(block_features(carry, sr))
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))


def classify_blocks(feat: np.ndarray, speech_mask: np.ndarray | None = None) -> list[str]:
    """Ordnet jedem Block eine Klasse zu (relativ zum Rauschboden der Datei)."""
    n = len(feat)
    if n == 0: return []
    col = {name: feat[:, i] for i, name in enumerate(FEATURE_NAMES)}
    floor = np.percentile(col["rms_db"], 5)
    peak_floor = np.percentile(col["peak_db"], 5)
    # Still = weder dauerhaft lauter als der Rauschboden noch einzelne Transienten (Tapping hat kaum RMS, aber klare Spitzen).
    silent = (col["rms_db"] < max(floor + 3.5, -62)) & (col["peak_db"] < peak_floor + 8)
    labels = []
    for i in range(n):
        if speech_mask is not None and i < len(speech_mask) and speech_mask[i] and not silent[i]:
            labels.append("speech"); continue
        if silent[i]: labels.append("silence"); continue
        onset, crest, flat, cent, low = col["onset_rate"][i], col["crest_db"][i], col["flatness"][i], col["centroid"][i], col["low"][i]
        if onset >= 2 and crest >= 15: labels.append("tapping")
        elif low > 0.9 and cent < 1400 and onset < 2: labels.append("breath")
        elif flat > 0.1 and onset < 4: labels.append("brushing")
        else: labels.append("texture")
    return labels


def smooth_labels(labels: list[str], window: int = 2) -> list[str]:
    """Mehrheitsfilter, damit einzelne Ausreisser-Bloecke keine Mini-Segmente erzeugen."""
    out = []
    for i in range(len(labels)):
        part = labels[max(0, i - window): i + window + 1]
        out.append(max(set(part), key=lambda value: (part.count(value), value == labels[i])))
    return out


def build_segments(labels: list[str], feat: np.ndarray, block_s: float, min_blocks: int = 2) -> list[dict]:
    """Fasst gleiche Labels zusammen; Mini-Segmente (<min_blocks) werden dem Vorgaenger zugeschlagen."""
    runs: list[list] = []
    for i, label in enumerate(labels):
        if runs and runs[-1][0] == label: runs[-1][2] = i + 1
        else: runs.append([label, i, i + 1])
    merged: list[list] = []
    for run in runs:
        if merged and (run[2] - run[1]) < min_blocks: merged[-1][2] = run[2]
        elif merged and merged[-1][0] == run[0]: merged[-1][2] = run[2]
        else: merged.append(run)
    segments = []
    for label, a, b in merged:
        part = feat[a:b]
        stats = {name: round(float(part[:, i].mean()), 3) for i, name in enumerate(FEATURE_NAMES)}
        segments.append({"start": round(a * block_s, 2), "end": round(b * block_s, 2), "label": label, "features": stats})
    return segments


def speech_mask_from_transcript(segments: list[dict], n_blocks: int, block_s: float) -> np.ndarray:
    mask = np.zeros(n_blocks, dtype=bool)
    for seg in segments:
        if seg.get("no_speech_prob", 0) > 0.7 or not seg.get("text"): continue
        mask[int(seg["start"] / block_s): min(n_blocks, int(np.ceil(seg["end"] / block_s)))] = True
    return mask
