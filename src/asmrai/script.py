"""Fluester-Skripte: lokales LLM (llama.cpp) falls erreichbar, sonst Textbausteine."""
from __future__ import annotations

import json
import random
from urllib.request import Request, urlopen

from .config import Config

FALLBACK = {
    "en": ["Hello, and welcome. Let's slow down together and get comfortable.", "Take a deep breath in, and let it out slowly.",
           "I'm going to tap softly, and you can just listen.", "Nothing you have to do right now. Just relax.",
           "Let the sounds wash over you, one at a time.", "You're doing so well. Stay with the sound.",
           "Now a gentle brushing, slowly from one side to the other.", "Feel your shoulders drop. Everything is quiet and calm.",
           "A little closer now. Just soft sounds and soft words.", "Rest here for a moment. There is no hurry at all."],
    "de": ["Hallo und willkommen. Lass uns gemeinsam ganz langsam werden.", "Atme tief ein und ganz langsam wieder aus.",
           "Ich klopfe jetzt leise, und du hoerst einfach zu.", "Du musst gerade nichts tun. Entspann dich einfach.",
           "Lass die Geraeusche nacheinander an dir vorbeiziehen.", "Das machst du ganz toll. Bleib bei den Klaengen.",
           "Jetzt ein sanftes Buersten, langsam von einer Seite zur anderen.", "Spuer, wie deine Schultern sinken. Alles ist ruhig.",
           "Ein kleines Stueck naeher. Nur leise Klaenge und leise Worte.", "Ruh dich einen Moment aus. Es eilt gar nichts."],
}


def write_lines(config: Config, count: int, language: str, theme: str, words: int, style_words: list[str], seed: int, persona: dict | None = None) -> tuple[list[str], str]:
    """-> (Zeilen, quelle) mit quelle in {'llm','fallback'}"""
    base = config.get("llm", "base_url")
    if base and config.get("llm", "enabled", True):
        prompt = (f"Write {count} calm, whispered ASMR lines in language '{language}' for a relaxation video. "
                  f"Theme: {theme or 'relaxing, sleepy triggers'}. Each line about {words} words (2-4 sentences), soothing, no emojis, no stage directions. "
                  f"Style hints (frequent words from the genre, use sparingly): {', '.join(style_words[:12]) or 'none'}. "
                  + (f"Speaker personality: {persona.get('personality')}. Occasionally use phrases like: {'; '.join(persona.get('catchphrases', [])[:4])}. " if persona and persona.get("personality") else "")
                  + "Return ONLY a JSON array of strings.")
        try:
            body = json.dumps({"model": "local", "messages": [{"role": "user", "content": prompt}], "stream": False, "seed": seed, "temperature": 0.8,
                               "chat_template_kwargs": {"enable_thinking": False}}).encode()
            with urlopen(Request(base.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}),
                         timeout=float(config.get("llm", "timeout_seconds", 300))) as response:
                text = json.loads(response.read())["choices"][0]["message"]["content"]
            start, end = text.find("["), text.rfind("]")
            lines = [str(x).strip() for x in json.loads(text[start:end + 1]) if str(x).strip()]
            if len(lines) >= max(2, count // 2): return lines[:count], "llm"
        except Exception:  # noqa: BLE001 - LLM optional
            pass
    pool = FALLBACK.get(language, FALLBACK["en"]); rng = random.Random(seed)
    return [rng.choice(pool) for _ in range(count)], "fallback"
