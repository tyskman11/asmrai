"""Stimmen: nur eingebaute XTTS-Sprecher oder vom Nutzer registrierte *eigene* Stimmen mit Einwilligung.

Audio aus den Creator-Archiven kann hier nicht als Stimmvorlage verwendet werden - der Pfad liegt
absichtlich ausserhalb von ``data/voices`` und wird von ``register_voice`` nicht akzeptiert.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..db import Database

XTTS_SR = 24000
XTTS_LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"}
_lock = threading.Lock()
_model = None
_cpu_only = False       # nach einem GPU-Speicherfehler bleibt XTTS im RAM (langsamer, aber es laeuft)
log = logging.getLogger("asmrai.voice")


def builtin_speakers(config: Config) -> list[str]:
    """Namen der in XTTS eingebauten, synthetischen Sprecher (ohne echte Person als Vorlage)."""
    model_dir = config.path("xtts", "model_dir", "../SeriesAI/models/tts/xtts_v2")
    try:
        import torch
        data = torch.load(model_dir / "speakers_xtts.pth", map_location="cpu", weights_only=False)
        return sorted(data.keys())
    except Exception:  # noqa: BLE001
        return [config.get("xtts", "default_speaker", "Claribel Dervla")]


def list_voices(config: Config, db: Database) -> list[dict]:
    default = config.get("xtts", "default_speaker", "Claribel Dervla")
    names = builtin_speakers(config); names = ([default] if default in names else []) + [n for n in names if n != default]
    voices = [{"id": f"builtin:{name}", "name": name, "kind": "builtin", "owner": None, "consent": True} for name in names[:12]]
    from . import blend
    voices += blend.list_blends(config, db)
    for row in db.query("SELECT * FROM voices ORDER BY name"):
        folder = config.voices / row["slug"]
        voices.append({"id": f"own:{row['slug']}", "name": row["name"], "kind": "own", "owner": row["owner"], "consent": bool(row["consent"]),
                       "samples": len(list(folder.glob("ref_*.wav")))})
    return voices


def register_voice(config: Config, db: Database, name: str, owner: str, consent: bool, files: list[tuple[str, bytes]]) -> dict:
    if not consent: raise ValueError("Ohne Einwilligung der sprechenden Person wird keine Stimme registriert.")
    if not owner.strip(): raise ValueError("Bitte den Namen der Person angeben, die die Stimme besitzt/freigibt.")
    if not files: raise ValueError("Mindestens eine Audiodatei (>= 6 s, sauber) hochladen.")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "voice"
    folder = config.voices / slug; folder.mkdir(parents=True, exist_ok=True)
    from ..media import run
    for index, (filename, content) in enumerate(files):
        raw = folder / f"upload_{index}{Path(filename).suffix or '.bin'}"; raw.write_bytes(content)
        target = folder / f"ref_{index:02d}.wav"
        result = run([config.ffmpeg, "-v", "error", "-y", "-i", str(raw), "-ac", "1", "-ar", str(XTTS_SR), "-t", "30", str(target)], 120)
        raw.unlink(missing_ok=True)
        if result.returncode != 0: raise ValueError(f"Audio nicht lesbar: {filename}")
    db.execute("INSERT INTO voices(slug,name,kind,owner,consent,created) VALUES(?,?,?,?,?,?) "
               "ON CONFLICT(slug) DO UPDATE SET name=excluded.name, owner=excluded.owner, consent=excluded.consent",
               (slug, name, "own", owner.strip(), 1, time.time()))
    return {"slug": slug}


def delete_voice(config: Config, db: Database, slug: str) -> None:
    db.execute("DELETE FROM voices WHERE slug=?", (slug,))
    shutil.rmtree(config.voices / slug, ignore_errors=True)


def _load(config: Config):
    global _model
    if _model is None:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        model_dir = config.path("xtts", "model_dir", "../SeriesAI/models/tts/xtts_v2")
        if not (model_dir / "model.pth").exists(): raise FileNotFoundError(f"XTTS-v2 nicht gefunden: {model_dir}")
        cfg = XttsConfig(); cfg.load_json(str(model_dir / "config.json"))
        model = Xtts.init_from_config(cfg); model.load_checkpoint(cfg, checkpoint_dir=str(model_dir), eval=True)
        if torch.cuda.is_available() and not _cpu_only and torch.cuda.mem_get_info()[0] > 4 * 1024 ** 3: model.cuda()
        else: log.info("XTTS laeuft im RAM (CPU) - weniger als 4 GB VRAM frei oder GPU nicht verfuegbar; das dauert laenger.")
        _model = model
    return _model


def _move_to_cpu() -> None:
    global _cpu_only
    import torch
    _cpu_only = True
    if _model is not None: _model.cpu()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def with_oom_fallback(fn):
    """Reicht der GPU-Speicher nicht, wird XTTS automatisch in den RAM verlagert und der Schritt wiederholt (dauert laenger, laeuft aber)."""
    import torch
    try: return fn()
    except torch.cuda.OutOfMemoryError:
        log.warning("GPU-Speicher reicht nicht - XTTS wird in den RAM (CPU) verlagert und der Schritt wiederholt. Das ist deutlich langsamer.")
        _move_to_cpu(); return fn()


def default_relax(config: Config, voice_id: str) -> float:
    from . import blend
    return blend.default_relax(config, voice_id)


def latents_for(config: Config, voice_id: str):
    """(gpt_cond_latent, speaker_embedding) einer Stimme auf der CPU - Grundlage fuer das Mischen."""
    from . import blend
    kind, _, name = (voice_id or "").partition(":")
    with _lock:
        model = _load(config)

        def run():
            if kind == "builtin":
                entry = model.speaker_manager.speakers[name]; return entry["gpt_cond_latent"].cpu(), entry["speaker_embedding"].cpu()
            if kind == "own":
                refs = sorted((config.voices / name).glob("ref_*.wav"))
                if not refs: raise FileNotFoundError(f"Keine Referenzaufnahmen fuer Stimme '{name}'")
                gpt, emb = model.get_conditioning_latents(audio_path=[str(p) for p in refs], max_ref_length=20, gpt_cond_len=12, gpt_cond_chunk_len=4)
                return gpt.cpu(), emb.cpu()
            if kind == "blend": return blend.load(config, name)
            raise ValueError(f"Unbekannte Stimme: {voice_id}")
        return with_oom_fallback(run)


def split_text(text: str, limit: int = 220) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip()); chunks, current = [], ""
    for part in parts:
        if len(current) + len(part) + 1 > limit and current: chunks.append(current); current = part
        else: current = f"{current} {part}".strip()
    return chunks + ([current] if current else [])


def synthesize(config: Config, text: str, voice_id: str, language: str, speed: float = 0.92) -> np.ndarray:
    """Liefert Mono-float32 bei 24 kHz. voice_id: 'builtin:<Name>' oder 'own:<slug>'."""
    language = language if language in XTTS_LANGS else "en"
    kind, _, name = (voice_id or "").partition(":")
    kwargs: dict = {}
    if kind == "blend":
        from . import blend
        pass
    elif kind == "own":
        refs = sorted((config.voices / name).glob("ref_*.wav"))
        if not refs: raise FileNotFoundError(f"Keine Referenzaufnahmen fuer Stimme '{name}'")
        kwargs["speaker_wav"] = [str(path) for path in refs]
    else:
        kwargs["speaker"] = name or config.get("xtts", "default_speaker", "Claribel Dervla")
    with _lock:
        model = _load(config)

        def run():
            pieces = []
            if kind == "blend":
                device = next(model.parameters()).device; gpt, emb = (t.to(device) for t in blend.load(config, name))
            for chunk in split_text(text):
                result = model.inference(chunk, language, gpt, emb, speed=speed) if kind == "blend" else model.synthesize(chunk, language=language, speed=speed, **kwargs)
                pieces.append(np.asarray(result["wav"], dtype=np.float32)); pieces.append(np.zeros(int(0.35 * XTTS_SR), dtype=np.float32))
            return pieces
        pieces = with_oom_fallback(run)
    return np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype=np.float32)
