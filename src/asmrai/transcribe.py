"""faster-whisper-Transkription (lokal, GPU nur wenn genug VRAM frei ist)."""
from __future__ import annotations

import json
import logging
import threading
import subprocess
from pathlib import Path

import numpy as np

from .config import Config
from . import media
from .jobs import check_cancel
from .media import NO_WINDOW

log = logging.getLogger("asmrai.analysis")
_lock = threading.Lock()
_models: dict[tuple, object] = {}


def available_models(config: Config) -> list[dict]:
    """Lokal vorhandene faster-whisper-Modelle (Cache-Ordner mit model.bin)."""
    cache = config.path("whisper", "cache", "../SeriesAI/models/whisper"); found = []
    for folder in sorted(cache.glob("models--*--faster-whisper-*")) if cache and cache.exists() else []:
        weights = list(folder.glob("snapshots/*/model.bin"))
        if weights: found.append({"name": folder.name.split("faster-whisper-")[-1], "size_gb": round(weights[0].stat().st_size / 1024 ** 3, 2)})
    order = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    return sorted(found, key=lambda m: order.index(m["name"]) if m["name"] in order else 99)


def free_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available(): return torch.cuda.mem_get_info()[0] / 1024 ** 3
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _load(config: Config, model_name: str, device: str):
    from faster_whisper import WhisperModel
    key = (model_name, device)
    if key not in _models:
        cache = config.path("whisper", "cache", "../SeriesAI/models/whisper")
        compute = config.get("whisper", "compute_type", "auto")
        if compute == "auto": compute = "float16" if device == "cuda" else "int8"
        _models.clear()   # nur ein Modell gleichzeitig im Speicher
        _models[key] = WhisperModel(model_name, device=device, compute_type=compute, download_root=str(cache))
    return _models[key]


def decode_for_asr(config: Config, path: Path) -> np.ndarray:
    """16-kHz-Mono mit lokaler Pegelanhebung: ASMR-Fluestern liegt oft bei -50 dB und wird von der VAD sonst ueberhoert."""
    cmd = [config.ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
           "-af", "highpass=f=70,dynaudnorm=f=300:g=21:p=0.9:m=40:s=8", "-f", "f32le", "-"]
    result = media.run(cmd)                        # abbrechbar
    if result.returncode != 0: raise RuntimeError("Audio-Dekodierung fehlgeschlagen: " + result.stderr.decode(errors="replace")[-200:])
    return np.frombuffer(result.stdout, dtype=np.float32)


HALLUCINATIONS = {"thank you", "thanks for watching", "you", "bye", "thank you for watching", "thanks", "okay", "so"}


def drop_hallucinations(segments: list[dict]) -> list[dict]:
    """Whisper ohne VAD erfindet auf Tapping/Stille gern 'Thank you' o.ae. - das filtern wir heraus."""
    kept, previous, repeats = [], None, 0
    for seg in segments:
        text = seg["text"].strip().lower().strip(" .!?,")
        repeats = repeats + 1 if text == previous else 0; previous = text
        if repeats >= 2 or not text: continue
        if seg["avg_logprob"] < -1.2 or (seg["no_speech_prob"] > 0.6 and seg["avg_logprob"] < -0.7): continue
        if text in HALLUCINATIONS and seg["end"] - seg["start"] < 4 and seg["no_speech_prob"] > 0.2: continue
        kept.append(seg)
    return kept


def transcribe(config: Config, path: Path, model_name: str | None = None, language: str | None = None, on_progress=None, duration: float = 0) -> dict:
    pooled = Path(__file__).resolve().parents[2] / "pool" / "results" / f"{path.stem}.transcript.json"
    if config.get("pool", "use_results", True) and pooled.is_file():   # vom Power Pool berechnet (Browser-Whisper, doppelt geprueft)
        data = json.loads(pooled.read_text(encoding="utf-8"))
        segments = [{"start": g["start"], "end": g["end"], "text": g["text"].strip(), "avg_logprob": 0.0, "no_speech_prob": 0.0} for g in data["segments"] if g["text"].strip()]
        return {"language": language or data.get("language") or config.get("whisper", "language") or "und", "language_probability": 1.0,
                "device": "pool", "model": data.get("model", "pool"), "segments": segments}
    model_name = model_name or config.get("whisper", "model", "medium")
    language = language or config.get("whisper", "language")
    names = [m["name"] for m in available_models(config)]
    if model_name not in names and not config.get("whisper", "allow_download", False):
        raise RuntimeError(f"Whisper-Modell '{model_name}' ist nicht lokal vorhanden (vorhanden: {', '.join(names) or 'keine'}). "
                           "Anderes Modell waehlen oder whisper.allow_download: true setzen.")
    device = config.get("whisper", "device", "auto")
    if device == "auto": device = "cuda" if free_vram_gb() > 3.0 else "cpu"
    with _lock:
        try:
            model = _load(config, model_name, device)
        except Exception as error:  # noqa: BLE001 - CUDA-Bibliotheken fehlen o.ae. -> CPU
            if device == "cpu": raise
            log.warning("Whisper '%s' laesst sich nicht auf der GPU laden (%s) - weiche auf CPU aus (viel langsamer).", model_name, error)
            device = "cpu"; model = _load(config, model_name, device)
        segments_iter, info = model.transcribe(decode_for_asr(config, path), language=language, beam_size=1, vad_filter=False,   # Silero-VAD ueberhoert Fluestern komplett (an Poki-Material gemessen)
                                               no_speech_threshold=0.6, log_prob_threshold=-1.0, compression_ratio_threshold=2.2,
                                               condition_on_previous_text=False, word_timestamps=False)
        segments = []
        for item in segments_iter:
            check_cancel()
            segments.append({"start": round(item.start, 2), "end": round(item.end, 2), "text": item.text.strip(),
                             "avg_logprob": round(item.avg_logprob, 3), "no_speech_prob": round(item.no_speech_prob, 3)})
            if on_progress and duration: on_progress(min(0.99, item.end / duration))
    segments = drop_hallucinations(segments)
    return {"language": info.language, "language_probability": round(info.language_probability, 3), "device": device,
            "model": model_name, "segments": segments}
