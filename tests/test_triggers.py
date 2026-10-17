import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmrai import planner, triggers  # noqa: E402
from asmrai.app import App  # noqa: E402
from asmrai.config import Config  # noqa: E402
from asmrai.synth import dsp  # noqa: E402

SR = 22050


def tap_audio(material, rate=3.0, seconds=6.0, seed=5, hum=True):
    x = dsp.resample(dsp.tapping(seconds, rate, material, 1.0, seed), dsp.SR, SR); x = x / (np.abs(x).max() + 1e-9) * 0.3
    if hum: x = x + 0.02 * np.sin(2 * np.pi * 120 * np.arange(len(x)) / SR) + 0.004 * np.random.default_rng(0).standard_normal(len(x))   # Brummen + Rauschen
    return x.astype(np.float32)


def test_material_is_recovered_from_the_sound_even_with_hum():
    found = {}
    for material in ("wood", "plastic", "glass", "nails"):
        d = triggers.descriptors(tap_audio(material), SR); found[material] = triggers.material_for(d["resonance_hz"], d["decay_ms"], d["centroid"])
    assert found == {"wood": "wood", "plastic": "plastic", "glass": "glass", "nails": "nails"}


def test_decay_ordering_and_rate_follow_the_source():
    decay = {m: triggers.descriptors(tap_audio(m), SR)["decay_ms"] for m in ("nails", "wood", "glass", "metal")}
    assert decay["nails"] < decay["wood"] < decay["glass"] < decay["metal"]                              # kurze Schlaege -> langes Nachklingen
    rates = [triggers.descriptors(tap_audio("wood", rate, 8.0, 3, hum=False), SR)["onset_rate"] for rate in (1.5, 3.0, 5.0)]
    assert rates[0] < rates[1] < rates[2] and 0.8 < rates[0] < 2.2
    regular = triggers.descriptors(np.tile(np.r_[np.hanning(200) * np.sin(np.arange(200) / 3), np.zeros(SR // 3 - 200)], 18).astype(np.float32), SR)
    assert regular["onset_cv"] < 0.3                                                                      # exakt gleichmaessige Schlaege


def test_cluster_naming_rules():
    base = {"onset_rate": 2.0, "centroid": 2000, "rolloff": 4000, "onset_cv": 0.4, "resonance_hz": 420, "decay_ms": 50}
    assert triggers.classify_cluster("tapping", base)[:2] == ("tapping", "tapping_wood")
    assert triggers.classify_cluster("tapping", {**base, "onset_rate": 4.0})[1] == "tapping_wood_fast"
    assert triggers.classify_cluster("brushing", {**base, "centroid": 1200})[1] == "brushing_soft" and triggers.classify_cluster("brushing", {**base, "centroid": 4000})[1] == "brushing_bright"
    assert triggers.classify_cluster("brushing", {**base, "rolloff": 9000, "onset_rate": 4.0})[:2] == ("scratching", "scratching")
    assert triggers.classify_cluster("breath", {**base, "centroid": 700})[:2] == ("breath", "breath")
    assert triggers.classify_cluster("texture", {**base, "onset_cv": 1.2, "rolloff": 7000})[:2] == ("crinkle", "crinkle")
    assert triggers.classify_cluster("texture", {**base, "onset_cv": 0.3, "onset_rate": 3.0})[0] == "tapping"      # regelmaessige 'Textur' = leises Tapping


def make(tmp_path):
    cfg = Config({"paths": {"data": str(tmp_path / "data"), "ai_models": str(tmp_path / "ai_models"), "seriesai": str(tmp_path / "none")}}); return cfg, App(cfg)


def add_events(app, creator_id, video_id, material, count, label="tapping", start=0.0, **audio):
    for i in range(count):
        desc = triggers.descriptors(tap_audio(material, seed=i + 1, seconds=3.0, **audio), SR); desc.update(level_db=-35.0, width=1.0, pan_motion=0.2)
        app.db.execute("INSERT INTO trigger_events(video_id,start,end,label,desc) VALUES(?,?,?,?,?)", (video_id, start + i * 4, start + i * 4 + 3, label, json.dumps(desc)))


def test_library_separates_materials_and_gives_each_a_recipe(tmp_path):
    cfg, app = make(tmp_path); src = tmp_path / "v"; src.mkdir(); (src / "a.mp4").write_bytes(b"x")
    creator = app.add_creator("C", str(src)); video = app.db.one("SELECT id FROM videos")["id"]
    add_events(app, creator["id"], video, "wood", 14); add_events(app, creator["id"], video, "nails", 14, start=100)
    library = triggers.build_library(app.db, creator["id"])
    names = {t["name"].split("_fast")[0].split("_slow")[0] for t in library}
    assert {"tapping_wood", "tapping_nails"} <= names and abs(sum(t["share"] for t in library) - 1) < 0.01
    wood = next(t for t in library if t["name"].startswith("tapping_wood")); nails = next(t for t in library if t["name"].startswith("tapping_nails"))
    assert wood["recipe"]["material"] == "wood" and nails["recipe"]["material"] == "nails" and wood["recipe"]["decay_ms"] > nails["recipe"]["decay_ms"]
    assert wood["examples"] and set(wood["recipe"]) >= {"kind", "material", "centroid", "onset_rate", "rate_cv", "decay_ms", "level_db"}
    assert triggers.build_library(app.db, 999) == []


def test_same_material_clusters_get_distinguishing_names(tmp_path):
    cfg, app = make(tmp_path); src = tmp_path / "v"; src.mkdir(); (src / "a.mp4").write_bytes(b"x")
    creator = app.add_creator("C", str(src)); video = app.db.one("SELECT id FROM videos")["id"]
    add_events(app, creator["id"], video, "wood", 14, rate=1.2); add_events(app, creator["id"], video, "wood", 14, rate=5.0, start=100)       # gleiches Material, anderes Tempo
    names = [t["name"] for t in triggers.build_library(app.db, creator["id"])]
    assert len(names) == len(set(names))                                                                  # nie doppelte Namen


def test_merge_libraries_weights_and_dedupes():
    a = [{"name": "tapping_wood", "kind": "tapping", "label": "tapping", "share": 0.6, "segments": 6, "examples": [1], "recipe": {"kind": "tapping", "material": "wood", "onset_rate": 2.0, "centroid": 1000}}]
    b = [{"name": "tapping_wood", "kind": "tapping", "label": "tapping", "share": 0.5, "segments": 4, "examples": [2], "recipe": {"kind": "tapping", "material": "wood", "onset_rate": 4.0, "centroid": 3000}},
         {"name": "breath", "kind": "breath", "label": "breath", "share": 0.5, "segments": 2, "examples": [3], "recipe": {"kind": "breath", "material": None, "onset_rate": 1.0, "centroid": 500}}]
    merged = triggers.merge_libraries([(a, 3), (b, 1)])
    wood = next(t for t in merged if t["name"] == "tapping_wood"); assert wood["segments"] == 10 and abs(sum(t["share"] for t in merged) - 1) < 0.01
    assert 2.0 < wood["recipe"]["onset_rate"] < 3.0 and wood["recipe"]["onset_rate"] < 2.6          # naeher an der staerker gewichteten Quelle


def library_profile():
    def trig(name, kind, label, share, material=None, rate=2.0, cent=2000):
        return {"name": name, "kind": kind, "label": label, "share": share, "segments": 5, "examples": [], "recipe": {"kind": kind, "material": material, "centroid": cent, "onset_rate": rate, "rate_cv": 0.4, "decay_ms": 60.0, "level_db": -30.0}}
    p = json.loads(json.dumps(planner.DEFAULT_PROFILE))
    p["trigger_library"] = [trig("tapping_glass", "tapping", "tapping", 0.5, "glass", 2.5), trig("tapping_wood_slow", "tapping", "tapping", 0.3, "wood", 1.2),
                            trig("brushing_soft", "brushing", "brushing", 0.1, None, 0.8, 1200), trig("breath", "breath", "breath", 0.1, None, 0.3, 700)]
    return p


def test_planner_builds_sessions_from_detected_triggers():
    profile = library_profile(); plan = planner.plan_session(profile, 400, 3, {"speech": False})
    named = [p for p in plan if p["kind"] != "pause"]; assert named and all(p.get("trigger") for p in named)
    assert {p["trigger"] for p in named} <= {"tapping_glass", "tapping_wood_slow", "brushing_soft", "breath"}
    glass = [p for p in named if p["trigger"] == "tapping_glass"]
    assert glass and all(p["kind"] == "tapping" and p["params"]["material"] == "glass" and p["params"]["onset_rate"] == 2.5 and p["params"]["decay_ms"] == 60.0 for p in glass)
    assert plan == planner.plan_session(profile, 400, 3, {"speech": False})                                     # deterministisch
    only_wood = planner.plan_session(profile, 200, 3, {"speech": False, "triggers": ["tapping"], "materials": ["wood"]})
    assert {p["trigger"] for p in only_wood if p["kind"] != "pause"} == {"tapping_wood_slow"}                    # Materialwunsch filtert die Bibliothek
    legacy = planner.plan_session(None, 100, 3, {"speech": False}); assert all("trigger" not in p for p in legacy)   # ohne Bibliothek wie bisher


def test_recipe_parameters_change_the_synthesized_sound():
    def energy(decay_ms):
        x = dsp.make_trigger("tapping", 6.0, {"material": "wood", "onset_rate": 2.0, "centroid": 2200, "decay_ms": decay_ms, "rate_cv": 0.3}, 4); return float(np.sum(x ** 2))
    assert energy(200.0) > 2 * energy(25.0)                                                                       # jeder Schlag ist auf gleiche Spitze normiert -> laengerer Nachklang = mehr Energie
    steady = dsp.make_trigger("tapping", 8.0, {"material": "wood", "onset_rate": 3.0, "rate_cv": 0.1}, 2); loose = dsp.make_trigger("tapping", 8.0, {"material": "wood", "onset_rate": 3.0, "rate_cv": 1.5}, 2)
    assert not np.array_equal(steady, loose)


def test_demo_and_original_sample_are_playable_wavs(tmp_path):
    import soundfile, io
    wav = triggers.demo_wav({"kind": "brushing", "centroid": 3000, "onset_rate": 1.0}, 2.0, 1); data, rate = soundfile.read(io.BytesIO(wav))
    assert wav[:4] == b"RIFF" and data.shape == (2 * dsp.SR, 2) and rate == dsp.SR and np.abs(data).max() > 0.01
    cfg = Config({"paths": {"data": str(tmp_path / "d")}}); source = tmp_path / "x.wav"; soundfile.write(str(source), tap_audio("wood", seconds=5.0), SR)
    sample = triggers.sample_wav(cfg, str(source), 1.0, 3.0); assert sample and sample[:4] == b"RIFF"


def test_extract_video_measures_segments_and_replaces_old_rows(tmp_path):
    import soundfile
    cfg, app = make(tmp_path); src = tmp_path / "v"; src.mkdir(); (src / "a.mp4").write_bytes(b"x"); audio = tmp_path / "a.wav"; soundfile.write(str(audio), tap_audio("glass", seconds=12.0), SR)
    creator = app.add_creator("C", str(src)); video = app.db.one("SELECT * FROM videos"); app.db.execute("UPDATE videos SET path=? WHERE id=?", (str(audio), video["id"]))
    video = app.db.one("SELECT * FROM videos WHERE id=?", (video["id"],))
    app.db.execute("INSERT INTO segments(video_id,start,end,label,features) VALUES(?,?,?,?,?)", (video["id"], 0.0, 6.0, "tapping", "{}"))
    app.db.execute("INSERT INTO segments(video_id,start,end,label,features) VALUES(?,?,?,?,?)", (video["id"], 6.0, 11.0, "tapping", "{}"))
    app.db.execute("INSERT INTO segments(video_id,start,end,label,features) VALUES(?,?,?,?,?)", (video["id"], 11.0, 11.5, "tapping", "{}"))          # zu kurz
    app.db.execute("INSERT INTO segments(video_id,start,end,label,features) VALUES(?,?,?,?,?)", (video["id"], 0.0, 5.0, "speech", "{}"))              # Sprache wird nicht vermessen
    assert triggers.extract_video(cfg, app.db, video) == 2 and triggers.extract_video(cfg, app.db, video) == 2                                         # zweiter Lauf ersetzt, dupliziert nicht
    rows = app.db.query("SELECT * FROM trigger_events"); assert len(rows) == 2 and json.loads(rows[0]["desc"])["onset_rate"] > 0
    app.delete_creator(creator["id"]); assert app.db.query("SELECT 1 FROM trigger_events") == []              # wird mit dem Creator geloescht
