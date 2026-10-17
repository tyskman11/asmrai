"""Visuelle Trigger-Analyse: ein lokales Vision-Modell (llama.cpp, GGUF + mmproj) beschriftet Keyframes mit Triggern, Requisiten und Kamera.

Es wird nur gelabelt, *was zu sehen ist* (Handlung, Objekte, Einstellung, Licht). Die Person selbst wird nicht
identifiziert oder beschrieben; aus dem Katalog entstehen allgemeine Bild-Prompts fuer eine eigene Figur.
"""
from __future__ import annotations

import base64
import json
import random
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

from . import media
from .config import Config

TRIGGERS = ["tapping", "brushing", "mic_brushing", "scratching", "ear_massage", "ear_cleaning", "cupping", "eating", "whispering_to_camera",
            "crinkling", "hair_play", "makeup_roleplay", "unboxing", "hand_movements", "gaming", "none"]
FRAMING = ["close_up", "medium", "wide"]
LIGHTING = ["dark", "dim", "warm", "bright", "colorful"]

PROMPT = ("You label one frame of an ASMR video for a style catalog. Do NOT identify the person and do NOT describe their face or body. "
          "Only describe what is happening and what objects are visible. Answer with JSON only, exactly this shape: "
          '{"triggers": [zero to three items from ' + json.dumps(TRIGGERS) + '], "props": [up to four short object nouns], '
          '"framing": one of ' + json.dumps(FRAMING) + ', "lighting": one of ' + json.dumps(LIGHTING) + ', "setting": "two to five words"}')

PHRASES = {
    "tapping": "hands tapping rhythmically on {prop}", "brushing": "a soft brush slowly sweeping across {prop}",
    "mic_brushing": "a fluffy brush gently stroking a microphone", "scratching": "fingernails lightly scratching {prop}",
    "ear_massage": "hands softly massaging near a binaural microphone", "ear_cleaning": "a small soft tool gently moving near a binaural microphone",
    "cupping": "cupped hands moving close to a binaural microphone", "eating": "a calm person quietly eating with {prop}",
    "whispering_to_camera": "a calm person whispering softly toward the camera", "crinkling": "hands slowly crinkling {prop}",
    "hair_play": "fingers gently combing through long hair", "makeup_roleplay": "a calm makeup roleplay with soft brushes and {prop}",
    "unboxing": "hands slowly unboxing {prop}", "hand_movements": "slow, gentle hand movements near the camera", "gaming": "a cozy gaming setup"}
FOR_KIND = {"tapping": ["tapping"], "brushing": ["brushing", "mic_brushing"], "scratching": ["scratching"], "breath": ["whispering_to_camera", "hand_movements"],
            "crinkle": ["crinkling", "unboxing"], "speech": ["whispering_to_camera"], "texture": ["hand_movements", "ear_massage", "cupping", "ear_cleaning"]}


def _post(config: Config, model: str, image: bytes) -> dict:
    base = config.get("vision", "base_url") or config.get("llm", "base_url")
    data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
    body = json.dumps({"model": model, "stream": False, "temperature": 0.1, "response_format": {"type": "json_object"},
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": data_url}}]}]}).encode()
    request = Request(base.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=float(config.get("vision", "timeout_seconds", 300))) as response:
        return json.loads(json.loads(response.read())["choices"][0]["message"]["content"])


def clean_label(raw: dict) -> dict:
    """Begrenzt Modellausgaben auf das kontrollierte Vokabular."""
    triggers = [t for t in raw.get("triggers", []) if isinstance(t, str) and t in TRIGGERS and t != "none"][:3]
    props = [p.strip().lower()[:30] for p in raw.get("props", []) if isinstance(p, str) and p.strip()][:4]
    framing = raw.get("framing") if raw.get("framing") in FRAMING else None
    lighting = raw.get("lighting") if raw.get("lighting") in LIGHTING else None
    setting = str(raw.get("setting", "")).strip().lower()[:40]
    return {"triggers": triggers, "props": props, "framing": framing, "lighting": lighting, "setting": setting}


def analyze_video(config: Config, path: Path, duration: float, work: Path, model: str, frames: int | None = None, ctx=None) -> dict:
    frames = frames or int(config.get("vision", "frames_per_video", 12))
    labels, failures = [], 0
    for k in range(frames):
        if ctx: ctx.check(); ctx.progress(k / frames)
        image = work / "vision" / f"{k}.jpg"
        if not media.extract_frame(config, path, duration * (k + 0.5) / frames, image, width=512): continue
        try: labels.append({"t": round(duration * (k + 0.5) / frames, 1), **clean_label(_post(config, model, image.read_bytes()))})
        except Exception as error:  # noqa: BLE001 - einzelne Frames duerfen scheitern
            failures += 1
            if failures >= 3 and not labels: raise RuntimeError(f"Vision-Modell '{model}' liefert keine Antworten: {error}") from error
    return {"model": model, "frames": labels}


def aggregate(videos: list[dict]) -> dict:
    """Katalog ueber alle Frames aller Videos eines Creators (relative Haeufigkeiten)."""
    frames = [f for v in videos for f in v.get("frames", [])]
    if not frames: return {}
    def share(counter: Counter, top: int = 10) -> dict:
        total = sum(counter.values()) or 1
        return {k: round(n / total, 3) for k, n in counter.most_common(top)}
    return {"frames": len(frames), "videos": len(videos),
            "triggers": share(Counter(t for f in frames for t in f["triggers"]), 12), "props": share(Counter(p for f in frames for p in f["props"]), 15),
            "framing": share(Counter(f["framing"] for f in frames if f["framing"])), "lighting": share(Counter(f["lighting"] for f in frames if f["lighting"])),
            "settings": share(Counter(f["setting"] for f in frames if f["setting"]), 6)}


def build_prompt(kind: str, catalog: dict | None, rng: random.Random, theme: str = "", subject: str = "") -> str:
    """Bild-Prompt fuer eine eigene Figur aus dem Katalog - ohne Personenbeschreibung des Creators."""
    catalog = catalog or {}
    wanted = FOR_KIND.get(kind, ["hand_movements"]); seen = catalog.get("triggers", {})
    options = [t for t in wanted if t in seen] or wanted
    trigger = rng.choices(options, weights=[seen.get(t, 0.05) for t in options])[0]
    props = list(catalog.get("props", {})) or ["a wooden surface"]
    pick = lambda table, default: rng.choices(list(table), weights=list(table.values()))[0] if table else default
    framing = {"close_up": "extreme close-up", "medium": "medium shot", "wide": "wide shot"}[pick(catalog.get("framing", {}), "close_up")]
    lighting = {"dark": "dark moody", "dim": "dim", "warm": "warm soft", "bright": "bright soft", "colorful": "soft colorful"}[pick(catalog.get("lighting", {}), "warm")]
    phrase = PHRASES[trigger].format(prop=rng.choice(props))
    scene = theme.strip() or pick(catalog.get("settings", {}), "cozy room")
    return (f"{framing}, {phrase}, {scene}, {lighting} lighting, {subject.strip() or 'an original calm character'}, shallow depth of field, "
            "very slow gentle movement, relaxing ASMR atmosphere, camera nearly static")
