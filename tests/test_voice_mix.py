import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmrai import looks, video  # noqa: E402
from asmrai.app import App  # noqa: E402
from asmrai.config import Config  # noqa: E402
from asmrai.synth import blend, dsp, voice  # noqa: E402


def make(tmp_path):
    cfg = Config({"paths": {"data": str(tmp_path / "data"), "ai_models": str(tmp_path / "ai_models"), "seriesai": str(tmp_path / "none")}})
    return cfg, App(cfg)


def f0(x, sr):
    ac = np.correlate(x[:8000], x[:8000], "full")[7999:]; lo, hi = int(sr / 400), int(sr / 100)
    return sr / (lo + int(np.argmax(ac[lo:hi])))


def high_share(x, sr, split=5000):
    power = np.abs(np.fft.rfft(x)) ** 2; freqs = np.fft.rfftfreq(len(x), 1 / sr); return float(power[freqs > split].sum() / power.sum())


def test_relax_lowers_pitch_softens_highs_and_keeps_level():
    sr = 24000; t = np.arange(sr) / sr
    sibilance = 0.04 * np.random.default_rng(0).standard_normal(sr)                                  # Zischlaute/Hoehen wie in echter Sprache
    voiced = (0.3 * sum(np.sin(2 * np.pi * 200 * k * t) / k for k in range(1, 12)) + sibilance).astype(np.float32)
    out = dsp.relax(voiced, sr, 1.0)
    assert f0(out, sr) < f0(voiced, sr) * 0.95 and high_share(out, sr) < high_share(voiced, sr) * 0.9 and np.isfinite(out).all()
    assert np.abs(out).max() <= np.abs(voiced).max() * 1.01 and dsp.relax(voiced, sr, 0.0) is voiced


def test_combine_is_weighted_mean_and_validates():
    import torch
    a = (torch.ones(1, 4, 6), torch.ones(1, 3, 1)); b = (torch.zeros(1, 4, 6), torch.zeros(1, 3, 1))
    gpt, emb = blend.combine([a, b], [3, 1]); assert torch.allclose(gpt, torch.full((1, 4, 6), 0.75)) and torch.allclose(emb, torch.full((1, 3, 1), 0.75))
    with pytest.raises(ValueError): blend.combine([a, (torch.zeros(1, 5, 6), torch.zeros(1, 3, 1))], [1, 1])
    with pytest.raises(ValueError): blend.combine([a, b], [1, 0])


def test_only_consented_voices_can_be_mixed(tmp_path):
    cfg, app = make(tmp_path)
    app.db.execute("INSERT INTO voices(slug,name,kind,owner,consent,created) VALUES('ok','Ok','own','Anna',1,?)", (time.time(),))
    app.db.execute("INSERT INTO voices(slug,name,kind,owner,consent,created) VALUES('no','No','own','Bob',0,?)", (time.time(),))
    assert [s["voice"] for s in blend.validate_sources(app.db, [{"voice": "own:ok"}, {"voice": "builtin:Claribel Dervla", "weight": 0.25}])] == ["own:ok", "builtin:Claribel Dervla"]
    with pytest.raises(ValueError, match="Einwilligung"): blend.validate_sources(app.db, [{"voice": "own:no"}])
    with pytest.raises(ValueError): blend.validate_sources(app.db, [{"voice": "own:ghost"}])
    with pytest.raises(ValueError): blend.validate_sources(app.db, [{"voice": "web:x"}]);
    app.create_model("Mia")
    with pytest.raises(ValueError, match="Einwilligung"): app.start_voice_blend("mia", {"sources": [{"voice": "own:no"}]})          # sofort abgelehnt, kein Job
    assert not app.db.query("SELECT 1 FROM jobs WHERE kind='voice_blend'")


def test_create_blend_saves_files_sets_voice_and_lists_it(tmp_path, monkeypatch):
    import torch
    cfg, app = make(tmp_path); app.create_model("Mia")
    app.db.execute("INSERT INTO voices(slug,name,kind,owner,consent,created) VALUES('ok','Ok','own','Anna',1,?)", (time.time(),))
    monkeypatch.setattr(voice, "latents_for", lambda c, v: (torch.ones(1, 4, 6) * (1 if v.startswith("own") else 3), torch.ones(1, 3, 1)))
    monkeypatch.setattr(voice, "synthesize", lambda c, text, vid, lang, speed=0.92: (0.1 * np.sin(np.arange(24000) / 10)).astype(np.float32))
    result = blend.create_blend(cfg, app.db, "mia", "Ruhiger Mix", [{"voice": "own:ok", "weight": 1}, {"voice": "builtin:Claribel Dervla", "weight": 1}], relax=0.7)
    folder = tmp_path / "ai_models" / "mia" / "_data" / "models" / "voice" / "ruhiger-mix"
    assert result["voice"] == "blend:mia/ruhiger-mix" and {p.name for p in folder.iterdir()} == {"latents.pt", "blend.json", "preview.wav"}
    gpt, emb = blend.load(cfg, "mia/ruhiger-mix"); assert torch.allclose(gpt, torch.full((1, 4, 6), 2.0))
    persona = app.db.one("SELECT voice,voice_model FROM personas WHERE slug='mia'"); assert persona["voice"] == persona["voice_model"] == "blend:mia/ruhiger-mix"
    listed = [v for v in voice.list_voices(cfg, app.db) if v["kind"] == "blend"]
    assert len(listed) == 1 and listed[0]["id"] == "blend:mia/ruhiger-mix" and listed[0]["preview"] and voice.default_relax(cfg, "blend:mia/ruhiger-mix") == 0.7
    assert voice.default_relax(cfg, "builtin:x") == 0.3 and app.models_overview()["models"][0]["blends"] == ["ruhiger-mix"] and app.models_overview()["models"][0]["voice_model"]
    import json
    assert json.loads((folder / "blend.json").read_text(encoding="utf-8"))["sources"][0]["owner"] == "Anna"                    # Herkunft ist dokumentiert


def test_use_all_collects_consented_voices_plus_soft_base(tmp_path):
    cfg, app = make(tmp_path); app.create_model("Mia")
    for slug, consent in (("a", 1), ("b", 1), ("c", 0)): app.db.execute("INSERT INTO voices(slug,name,kind,owner,consent,created) VALUES(?,?,?,?,?,?)", (slug, slug, "own", "X", consent, time.time()))
    job = app.start_voice_blend("mia", {"use_all": True, "name": "Alle", "relax": 0.5}); import json
    params = json.loads(app.db.one("SELECT params FROM jobs WHERE id=?", (job,))["params"])
    assert [s["voice"] for s in params["sources"]] == ["own:a", "own:b", "builtin:Claribel Dervla"] and params["sources"][-1]["weight"] == 0.25


def test_oom_falls_back_to_ram_and_retries(monkeypatch):
    import torch
    moved, calls = [], []
    monkeypatch.setattr(voice, "_move_to_cpu", lambda: moved.append(1))
    def flaky():
        calls.append(1)
        if len(calls) == 1: raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "ok"
    assert voice.with_oom_fallback(flaky) == "ok" and moved == [1] and len(calls) == 2
    assert voice.with_oom_fallback(lambda: "direkt") == "direkt" and moved == [1]


def test_generator_choices_voice_style_face_and_relax(tmp_path):
    cfg, app = make(tmp_path); app.create_model("Mia"); (cfg.characters / "creator_face_00.jpg").write_bytes(b"x")
    app.db.execute("UPDATE personas SET voice='blend:mia/x' WHERE slug='mia'")
    default = app.create_session({"persona": "mia", "duration_min": 1, "seed": 1, "speech": False})
    chosen = app.create_session({"persona": "mia", "duration_min": 1, "seed": 1, "speech": False, "voice": "builtin:Claribel Dervla", "relax": 0.8,
                                 "style": "bun", "hair": "bob", "face_ref": "creator_face_00.jpg"})
    unknown = app.create_session({"persona": "mia", "duration_min": 1, "seed": 1, "speech": False, "face_ref": "../../etc/passwd"})
    assert default["params"]["voice"] == "blend:mia/x" and default["params"]["relax"] is None                     # Modell-Stimme als Standard
    assert chosen["params"]["voice"] == "builtin:Claribel Dervla" and chosen["params"]["relax"] == 0.8             # ausdrueckliche Wahl hat Vorrang
    assert (chosen["params"]["style"], chosen["params"]["hair"], chosen["params"]["face_ref"]) == ("bun", "bob", "creator_face_00.jpg")
    assert unknown["params"]["face_ref"] is None                                                                  # nur vorhandene Referenzen


def test_reference_choice_face_ref_then_look_then_default(tmp_path):
    import random
    cfg, app = make(tmp_path); app.create_model("Mia"); rng = random.Random(1); default = tmp_path / "default.jpg"
    (cfg.characters / "c_face_00.jpg").write_bytes(b"x")
    ref, subject, info = video.choose_reference(cfg, app.db, {"face_ref": "c_face_00.jpg", "appearance": "A"}, rng, "mia", default)
    assert ref == cfg.characters / "c_face_00.jpg" and subject == video.SAME_PERSON and "Gesichts-Referenz" in info
    ref, subject, _ = video.choose_reference(cfg, app.db, {"appearance": "A"}, rng, "mia", default)               # kein Look vorhanden -> Standardbild + Text
    assert ref == default and subject == "A"
    app.db.execute("INSERT INTO looks(group_name,persona_slug,file,angle,hair,style,has_face,source,created) VALUES('bun','mia','bun/001','front','bun','bun',1,'s',?)", (time.time(),))
    frame = looks.looks_dir(cfg, "mia", True) / "bun" / "001_frame.jpg"; frame.parent.mkdir(parents=True, exist_ok=True); frame.write_bytes(b"x")
    ref, subject, info = video.choose_reference(cfg, app.db, {"style": "bun"}, rng, "mia", default)
    assert ref == frame and subject == video.SAME_PERSON and "Look bun/001" in info
