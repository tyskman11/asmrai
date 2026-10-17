"""Session-Rendering: Plan -> Skript -> Sprache/Trigger -> binaurale Mischung -> WAV (+ optional Video)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile

from . import planner, script
from .config import Config
from .db import Database, dumps
from .jobs import JobContext
from .profile import get_profile, style_profile
from .synth import dsp, voice as voices

CROSSFADE = 0.12


def session_dir(config: Config, session_id: int) -> Path:
    path = config.sessions / str(session_id); path.mkdir(parents=True, exist_ok=True); return path


def make_plan(config: Config, db: Database, session: dict) -> list[dict]:
    params = json.loads(session["params"])
    profile = style_profile(db, session)
    return planner.plan_session(profile, float(params.get("duration_min", 10)) * 60, int(params.get("seed", 1)), params)


def render_audio(config: Config, db: Database, session_id: int, ctx: JobContext) -> None:
    session = db.one("SELECT * FROM sessions WHERE id=?", (session_id,))
    params = json.loads(session["params"]); seed = int(params.get("seed", 1))
    profile = style_profile(db, session) or planner.DEFAULT_PROFILE
    db.execute("UPDATE sessions SET status='rendering', error=NULL, updated=? WHERE id=?", (time.time(), session_id))
    try:
        plan = json.loads(session["plan"]) if session["plan"] else make_plan(config, db, session)
        language = params.get("language", "en"); voice_id = params.get("voice") or f"builtin:{config.get('xtts', 'default_speaker')}"
        if voice_id.startswith("own:"):
            row = db.one("SELECT consent FROM voices WHERE slug=?", (voice_id[4:],))
            if not row or not row["consent"]: raise ValueError("Diese Stimme hat keine dokumentierte Einwilligung.")
        relax = float(params["relax"]) if params.get("relax") is not None else voices.default_relax(config, voice_id)
        speed = round(0.95 - 0.12 * relax, 3)                                  # entspannter = etwas langsamer
        speech_items = [p for p in plan if p["kind"] == "speech"]
        lines: list[str] = []
        if speech_items:
            ctx.log(f"Skript fuer {len(speech_items)} Sprechpassagen ...")
            words = int(sum(p.get("words", 20) for p in speech_items) / len(speech_items))
            lines, source = script.write_lines(config, len(speech_items), language, params.get("theme", ""), words,
                                               (profile.get("speech") or {}).get("top_words", []), seed,
                                               {"personality": params.get("personality"), "catchphrases": params.get("catchphrases", [])})
            ctx.log(f"Skript-Quelle: {source}")
        speech_cursor = 0
        sr = dsp.SR; audio_parts: list[tuple[float, np.ndarray]] = []; cursor = 0.0
        target = float(params.get("duration_min", 10)) * 60; index = -1; extensions = 0
        while True:
            index += 1
            if index >= len(plan):
                # Gesprochene Passagen sind oft kuerzer als geplant -> Restzeit mit weiteren Klangsegmenten fuellen.
                if cursor >= target - 3 or extensions >= 20: break
                extensions += 1
                extra = planner.plan_session(profile, target - cursor + 3, seed + 7919 * extensions, {**params, "speech": False})
                plan.extend(extra)
            item = plan[index]
            ctx.check(); ctx.progress(0.05 + 0.85 * min(cursor / target, 0.999))
            rng = np.random.default_rng(seed * 1000 + index)
            if item["kind"] == "pause":
                item["start"] = round(cursor, 2); cursor += item["duration"]; continue
            if item["kind"] == "speech":
                text = lines[speech_cursor] if speech_cursor < len(lines) else ""; speech_cursor += 1
                item["text"] = text
                try:
                    wav = voices.synthesize(config, text, voice_id, language, speed)
                    wav = dsp.relax(dsp.resample(wav, voices.XTTS_SR, sr), sr, relax); mono = dsp.whisperize(wav, sr, float(params.get("whisper", 0.85)), seed + index)
                    item["duration"] = round(len(mono) / sr, 2)
                except Exception as error:  # noqa: BLE001 - Session soll trotzdem fertig werden
                    ctx.log(f"Sprachsynthese fehlgeschlagen ({error}); Passage wird durch Atem ersetzt."); item["kind"] = "breath"
                    mono = dsp.make_trigger("breath", item["duration"], item["params"], seed + index)
            else:
                mono = dsp.make_trigger(item["kind"], item["duration"], item["params"], seed * 1000 + index)
            if len(mono) == 0: continue
            mono = dsp.normalize_rms(mono, -20.0) * 10 ** (min(item["level_db"], 6) / 20)
            azimuth = dsp.azimuth_path(len(mono) / sr, item["spatial"], rng, start=float(rng.uniform(-40, 40)))
            stereo = dsp.spatialize(mono, azimuth, item["distance"]).astype(np.float32)
            item["start"] = round(cursor, 2); audio_parts.append((cursor, stereo)); cursor += len(mono) / sr - CROSSFADE
            item["duration"] = round(len(mono) / sr, 2)
        length = int((cursor + 2.0) * sr)
        mix = np.zeros((length, 2), dtype=np.float32); fade = int(CROSSFADE * sr)
        for start, stereo in audio_parts:
            stereo = stereo.copy(); n = min(fade, len(stereo) // 2)
            if n > 0:
                ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]; stereo[:n] *= ramp; stereo[-n:] *= ramp[::-1]
            a = int(start * sr); b = min(length, a + len(stereo)); mix[a:b] += stereo[:b - a]
        tone = np.resize(dsp.room_tone(length / sr + 0.1, -48, seed), length)[:, None].astype(np.float32); mix += np.repeat(tone, 2, axis=1) * 0.5
        mix = dsp.normalize_rms(mix, float(params.get("target_db", -26)), -1.0)
        out = session_dir(config, session_id) / "audio.wav"
        soundfile.write(str(out), mix, sr, subtype="PCM_16")
        db.execute("UPDATE sessions SET status='audio_ready', audio_path=?, plan=?, updated=? WHERE id=?",
                   (str(out), dumps(plan), time.time(), session_id))
        ctx.log(f"Audio fertig: {length / sr / 60:.1f} min -> {out}")
    except Exception as error:
        db.execute("UPDATE sessions SET status='error', error=?, updated=? WHERE id=?", (str(error)[:500], time.time(), session_id)); raise
