"""Stimmen mischen: Die XTTS-Konditionierungs-Latents mehrerer Stimmen werden gewichtet gemittelt - das ergibt eine neue,
gemeinsame Stimme (z. B. aus allen freigegebenen Stimmen plus einer weichen, ruhigen Grundstimme) ohne Netz-Training.

Ablage je AI-Modell:  ai_models/<modell>/_data/models/voice/<mix>/{latents.pt, blend.json, preview.wav}
Jede Quelle muss freigegeben sein (eigene Stimme mit Einwilligung oder eingebaute synthetische Stimme); blend.json haelt die Quellen fest.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from .. import looks
from ..config import Config
from ..db import Database

log = logging.getLogger("asmrai.voice")
SAMPLE_TEXT = {"en": "Hello, and welcome. Let's slow down together. Take a deep breath, and let everything feel soft and quiet.",
               "de": "Hallo und willkommen. Lass uns gemeinsam ganz langsam werden. Atme tief ein, und lass alles weich und ruhig werden."}


def mix_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "mix"


def blend_dir(config: Config, model_slug: str, name: str, create: bool = False) -> Path:
    path = looks.person_dir(config, model_slug) / "models" / "voice" / mix_slug(name)
    if create: path.mkdir(parents=True, exist_ok=True)
    return path


def combine(latents: list[tuple], weights: list[float]) -> tuple:
    """Gewichteter Mittelwert von (gpt_cond_latent, speaker_embedding)-Paaren; Formen muessen uebereinstimmen."""
    import torch
    if not latents or len(latents) != len(weights): raise ValueError("Keine Stimmen zum Mischen.")
    if any(w <= 0 for w in weights): raise ValueError("Gewichte muessen groesser als 0 sein.")
    for gpt, emb in latents[1:]:
        if gpt.shape != latents[0][0].shape or emb.shape != latents[0][1].shape: raise ValueError("Die Latents der Stimmen passen nicht zusammen.")
    w = torch.tensor(weights, dtype=torch.float32); w = w / w.sum()
    return (sum(float(wi) * g.float() for wi, (g, _) in zip(w, latents)), sum(float(wi) * e.float() for wi, (_, e) in zip(w, latents)))


def load(config: Config, ref: str) -> tuple:
    """ref = '<modell>/<mix>' -> (gpt_cond_latent, speaker_embedding) auf der CPU."""
    import torch
    model_slug, _, name = ref.partition("/")
    file = blend_dir(config, model_slug, name) / "latents.pt"
    if not file.exists(): raise FileNotFoundError(f"Gemischte Stimme nicht gefunden: {ref}")
    data = torch.load(file, map_location="cpu")
    return data["gpt_cond_latent"], data["speaker_embedding"]


def default_relax(config: Config, voice_id: str) -> float:
    kind, _, ref = (voice_id or "").partition(":")
    if kind == "blend":
        meta = blend_dir(config, *ref.split("/", 1)) / "blend.json" if "/" in ref else None
        if meta and meta.exists(): return float(json.loads(meta.read_text(encoding="utf-8")).get("relax", 0.4))
    return 0.3


def list_blends(config: Config, db: Database) -> list[dict]:
    out = []
    for persona in db.query("SELECT slug,name FROM personas"):
        base = looks.person_dir(config, persona["slug"]) / "models" / "voice"
        for meta_file in sorted(base.glob("*/blend.json")) if base.is_dir() else []:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            out.append({"id": f"blend:{persona['slug']}/{meta_file.parent.name}", "name": f"{meta.get('name', meta_file.parent.name)} (Mix, {persona['name']})", "kind": "blend",
                        "owner": persona["name"], "consent": True, "model": persona["slug"], "mix": meta_file.parent.name, "relax": meta.get("relax"),
                        "sources": len(meta.get("sources", [])), "preview": (meta_file.parent / "preview.wav").exists()})
    return out


def validate_sources(db: Database, sources: list[dict]) -> list[dict]:
    """Nur freigegebene Quellen: eingebaute synthetische Stimmen, eigene Stimmen mit dokumentierter Einwilligung, bereits gemischte Stimmen."""
    checked = []
    for item in sources:
        voice, weight = str(item.get("voice", "")), float(item.get("weight", 1.0))
        kind, _, name = voice.partition(":")
        if weight <= 0: raise ValueError(f"Gewicht fuer '{voice}' muss groesser als 0 sein.")
        if kind == "own":
            row = db.one("SELECT consent,owner FROM voices WHERE slug=?", (name,))
            if not row or not row["consent"]: raise ValueError(f"Stimme '{name}' hat keine dokumentierte Einwilligung - sie wird nicht gemischt.")
            checked.append({"voice": voice, "weight": weight, "owner": row["owner"]})
        elif kind in ("builtin", "blend"): checked.append({"voice": voice, "weight": weight, "owner": None})
        else: raise ValueError(f"Unbekannte Stimme: {voice}")
    if not checked: raise ValueError("Keine Stimmen gewaehlt.")
    return checked


def create_blend(config: Config, db: Database, model_slug: str, name: str, sources: list[dict], relax: float = 0.5, set_default: bool = True, ctx=None, language: str = "en") -> dict:
    import soundfile
    import torch
    from . import dsp, voice as voices
    if not db.one("SELECT 1 AS x FROM personas WHERE slug=?", (model_slug,)): raise ValueError("AI-Modell nicht gefunden.")
    name = (name or "").strip()
    if not name: raise ValueError("Name der gemischten Stimme fehlt.")
    checked = validate_sources(db, sources); relax = max(0.0, min(1.0, float(relax)))
    latents = []
    for index, source in enumerate(checked):
        if ctx: ctx.check(); ctx.progress(index / (len(checked) + 1)); ctx.log(f"Stimme {index + 1}/{len(checked)}: {source['voice']} (Gewicht {source['weight']})")
        latents.append(voices.latents_for(config, source["voice"]))
    gpt, emb = combine(latents, [s["weight"] for s in checked])
    folder = blend_dir(config, model_slug, name, create=True)
    torch.save({"gpt_cond_latent": gpt.cpu(), "speaker_embedding": emb.cpu()}, folder / "latents.pt")
    meta = {"name": name, "model": model_slug, "sources": checked, "relax": relax, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Gewichteter Mittelwert der XTTS-Konditionierungs-Latents; alle Quellen waren freigegeben (eingebaut oder mit dokumentierter Einwilligung)."}
    (folder / "blend.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    voice_id = f"blend:{model_slug}/{folder.name}"
    if ctx: ctx.log("Hoerprobe wird erzeugt ...")
    wav = voices.synthesize(config, SAMPLE_TEXT.get(language, SAMPLE_TEXT["en"]), voice_id, language, speed=round(0.95 - 0.12 * relax, 3))
    soundfile.write(str(folder / "preview.wav"), dsp.relax(wav, voices.XTTS_SR, relax), voices.XTTS_SR)
    db.execute("UPDATE personas SET voice_model=?" + (", voice=?" if set_default else "") + " WHERE slug=?",
               (voice_id, voice_id, model_slug) if set_default else (voice_id, model_slug))
    looks.export_person(config, db, model_slug)
    log.info("Gemischte Stimme '%s' fuer %s aus %d Stimme(n) gespeichert", name, model_slug, len(checked))
    return {"voice": voice_id, "sources": len(checked), "folder": str(folder)}
