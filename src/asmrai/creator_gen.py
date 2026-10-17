"""Generator fuer eine eigene, frei erfundene virtuelle ASMR-Person (Name, Aussehen, Persoenlichkeit, Repertoire, Setting).

Die Person ist erwachsen und ein Original - sie darf keiner realen Person nachempfunden sein. Ein lokales LLM (llama.cpp) schreibt das
Profil; ohne LLM entsteht es regelbasiert aus Bausteinen. Aussehen wird als Text gespeichert (Bild-Prompt) und spaeter als Porträt erzeugt.
"""
from __future__ import annotations

import json
import random
import re
from urllib.request import Request, urlopen

from .config import Config

TRIGGERS = ["tapping", "brushing", "scratching", "breath", "crinkle", "speech"]
FIRST = ["Lio", "Mira", "Noa", "Sela", "Ive", "Runa", "Tavi", "Ella", "Kiri", "Nell", "Odie", "Vera", "Juno", "Lumi", "Asha", "Wren"]
LAST = ["Hale", "Moss", "Vale", "Quill", "Ash", "Fern", "Lark", "Sable", "Rue", "Thorne", "Wisp", "Dove"]
HAIR = ["short silver bob", "long wavy chestnut hair", "soft lavender pixie cut", "dark curly shoulder-length hair", "honey-blonde braids", "teal-dyed straight hair with bangs"]
EYES = ["warm brown eyes", "pale green eyes", "grey eyes", "hazel eyes"]
STYLE = ["oversized cream knit sweater", "soft pastel hoodie", "dark green velvet cardigan", "simple white linen shirt", "cozy mustard sweater"]
EXTRA = ["round glasses", "small freckles", "tiny star earrings", "a thin silver necklace", "no accessories"]
SETTING = ["a dim bedroom with fairy lights and a wooden desk", "a cozy attic studio with warm lamps and plants", "a quiet library corner with old books",
           "a candle-lit room with soft blankets and a window with rain", "a tidy studio with a large binaural microphone and pastel curtains"]
PERSONALITY = ["gentle, patient and slightly shy", "warm, playful and softly teasing", "calm, caring and a little sleepy", "curious, thoughtful and soothing"]
PHRASES = ["Come a little closer.", "Let's take our time, there is no hurry.", "You can just listen now.", "Everything is quiet and safe here.", "Good, just like that.",
           "I'm right here with you.", "Let your shoulders drop."]


def _fallback(rng: random.Random, language: str) -> dict:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    return {"name": name, "age": rng.randint(24, 34), "personality": rng.choice(PERSONALITY),
            "appearance": f"an adult woman in her late twenties with {rng.choice(HAIR)}, {rng.choice(EYES)}, {rng.choice(STYLE)}, {rng.choice(EXTRA)}",
            "setting": rng.choice(SETTING), "catchphrases": rng.sample(PHRASES, 4),
            "triggers": rng.sample(["tapping", "brushing", "scratching", "breath", "crinkle"], 4) + ["speech"],
            "voice": "soft, warm, low and slow whisper", "backstory": "A fictional night-time ASMR creator who makes calm, slow sessions for people who struggle to sleep."}


def _llm(config: Config, rng: random.Random, hints: str, language: str) -> dict | None:
    base = config.get("llm", "base_url")
    if not (base and config.get("llm", "enabled", True)): return None
    prompt = ("Invent ONE completely original, fictional adult ASMR creator (NOT based on any real person, celebrity or streamer). Return ONLY JSON with keys: "
              '"name" (fictional stage name, two words), "age" (integer 24-40), "personality" (one short phrase), '
              '"appearance" (one English sentence for an image prompt: hair, eyes, clothing, accessories; adult; no real names), '
              '"setting" (one English sentence: the room she records in), "catchphrases" (4 short calm whispered English phrases), '
              f'"triggers" (3-5 items from {TRIGGERS}), "voice" (short description of a natural soft voice), "backstory" (two sentences). '
              f"Creative hints: {hints or 'none'}. Variation seed: {rng.randint(0, 99999)}.")
    try:
        body = json.dumps({"model": "local", "messages": [{"role": "user", "content": prompt}], "stream": False, "temperature": 0.9,
                           "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False}}).encode()
        with urlopen(Request(base.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=150) as response:
            return json.loads(json.loads(response.read())["choices"][0]["message"]["content"])
    except Exception:  # noqa: BLE001 - LLM optional
        return None


def clean(raw: dict, fallback: dict) -> dict:
    """Begrenzt Laengen/Vokabular und stellt sicher, dass die Person erwachsen und ohne reale Bezuege beschrieben ist."""
    def text(key: str, limit: int) -> str:
        value = raw.get(key); return re.sub(r"\s+", " ", str(value)).strip()[:limit] if isinstance(value, (str, int)) and str(value).strip() else fallback[key]
    appearance = text("appearance", 300)
    appearance = re.sub(r"^(she|he|they)\s+(has|have|is|are|wears|wear)\s+", "", appearance, flags=re.I)   # 'She has ...' -> reine Beschreibung
    if not re.search(r"\badult\b|\bwoman\b|\bman\b|\bperson\b", appearance, re.I): appearance = "an adult person with " + appearance
    triggers = [t for t in raw.get("triggers", []) if t in TRIGGERS] if isinstance(raw.get("triggers"), list) else []
    phrases = [re.sub(r"\s+", " ", str(p)).strip()[:80] for p in raw.get("catchphrases", []) if str(p).strip()][:5] if isinstance(raw.get("catchphrases"), list) else []
    try: age = max(21, min(int(raw.get("age", fallback["age"])), 45))
    except (TypeError, ValueError): age = fallback["age"]
    return {"name": text("name", 40), "age": age, "personality": text("personality", 80), "appearance": appearance, "setting": text("setting", 200),
            "catchphrases": phrases or fallback["catchphrases"], "triggers": (triggers + ["speech"] if "speech" not in triggers else triggers) if len(triggers) >= 2 else fallback["triggers"],
            "voice": text("voice", 120), "backstory": text("backstory", 400)}


def generate(config: Config, seed: int, hints: str = "", language: str = "en", taken: set[str] | None = None) -> dict:
    rng = random.Random(seed); fallback = _fallback(rng, language)
    raw = _llm(config, rng, hints, language); source = "llm" if raw else "fallback"
    persona = clean(raw or {}, fallback)
    taken = {t.lower() for t in (taken or set())}
    attempts = 0
    while persona["name"].lower() in taken and attempts < 20:           # eindeutiger Name
        persona["name"] = f"{rng.choice(FIRST)} {rng.choice(LAST)}"; attempts += 1
    persona["source"] = source
    return persona
