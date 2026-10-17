import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmrai import planner  # noqa: E402
from asmrai.app import App  # noqa: E402
from asmrai.config import Config  # noqa: E402
from asmrai.profile import fuse_profiles, get_fusion, style_profile  # noqa: E402
from asmrai.synth import blend  # noqa: E402


def profile(tapping, speech, hours, wpm=40, cent=2000, pairs=None):
    def label(share, cent_):
        return {"share": share, "segments": 10, "duration": {"n": 10, "mean": 5.0, "median": 4.0, "p10": 2.0, "p90": 9.0}, "rms_db": -40.0, "peak_db": -20.0, "crest_db": 20.0,
                "centroid": cent_, "flatness": 0.05, "onset_rate": 2.0, "low": 0.7, "high": 0.05, "pan_abs": 0.2, "width": 1.0, "corr": 0.1}
    return {"videos": 10, "analyzed": 10, "hours": hours, "labels": {"tapping": label(tapping, cent), "speech": label(speech, cent)},
            "transitions": {"tapping": {"speech": 1.0}, "speech": {"tapping": 1.0}}, "loudness_db": -40.0, "noise_floor_db": -55.0,
            "speech": {"transcribed": 5, "languages": {"en": 5}, "words_per_min": wpm, "top_words": ["sleep", "soft"]},
            "visual": {"cut_rate_per_min": 0.2, "brightness": 100.0, "resolution": "1280x720", "fps": 30}}


def test_fusion_blends_shares_levels_and_sources_by_weight():
    a = profile(0.8, 0.2, hours=3, wpm=40, cent=1000); b = profile(0.2, 0.8, hours=1, wpm=100, cent=3000)
    fused = fuse_profiles([("A", a, 3), ("B", b, 1)])
    assert fused["fused"] and fused["hours"] == 4 and fused["analyzed"] == 20
    assert round(fused["labels"]["tapping"]["share"], 3) == 0.65 and round(fused["labels"]["speech"]["share"], 3) == 0.35 and round(sum(v["share"] for v in fused["labels"].values()), 3) == 1.0
    assert fused["speech"]["words_per_min"] == 55.0 and fused["speech"]["languages"] == {"en": 10}
    assert [f["creator"] for f in fused["fused_from"]] == ["A", "B"] and fused["fused_from"][0]["weight"] == 0.75
    assert 1000 < fused["labels"]["tapping"]["centroid"] < 3000 and abs(sum(fused["transitions"]["tapping"].values()) - 1) < 1e-6
    with pytest.raises(ValueError): fuse_profiles([("A", None, 1)]);
    with pytest.raises(ValueError): fuse_profiles([("A", a, 0)])


def test_fused_profile_drives_the_planner_like_a_normal_profile():
    fused = fuse_profiles([("A", profile(0.9, 0.1, 2), 1), ("B", profile(0.1, 0.9, 2), 1)])
    plan = planner.plan_session(fused, 300, 4, {"speech": True}); again = planner.plan_session(fused, 300, 4, {"speech": True})
    assert plan == again and sum(p["duration"] for p in plan) >= 300 and {p["kind"] for p in plan} >= {"tapping", "speech"}


def make(tmp_path):
    cfg = Config({"paths": {"data": str(tmp_path / "data"), "ai_models": str(tmp_path / "ai_models"), "seriesai": str(tmp_path / "none")}}); return cfg, App(cfg)


def add_creator(app, tmp_path, name, prof, hours):
    src = tmp_path / name; src.mkdir(exist_ok=True); (src / "a.mp4").write_bytes(b"x"); creator = app.add_creator(name, str(src))
    app.db.execute("INSERT OR REPLACE INTO profiles(creator_id,data,updated) VALUES(?,?,?)", (creator["id"], json.dumps(prof), time.time())); return creator


def test_fusion_job_builds_one_style_adapts_triggers_and_blends_voices(tmp_path, monkeypatch):
    cfg, app = make(tmp_path); app.create_model("Mia")
    a = add_creator(app, tmp_path, "Alpha", profile(0.8, 0.2, 3), 3); b = add_creator(app, tmp_path, "Beta", profile(0.2, 0.8, 1), 1)
    app.db.execute("INSERT INTO voices(slug,name,kind,owner,consent,created) VALUES('v1','V1','own','Alpha',1,?)", (time.time(),))
    called = {}
    monkeypatch.setattr(blend, "create_blend", lambda c, d, slug, name, sources, relax, set_default, ctx: called.update(slug=slug, name=name, sources=[s["voice"] for s in sources]) or {"voice": "blend:mia/x", "sources": len(sources), "folder": ""})
    job = app.start_fusion("mia", {})                                                                              # Standard: Gewicht = Stunden
    for _ in range(100):
        if app.db.one("SELECT status FROM jobs WHERE id=?", (job,))["status"] in ("done", "error"): break
        time.sleep(0.1)
    time.sleep(0.2); assert app.db.one("SELECT status FROM jobs WHERE id=?", (job,))["status"] == "done"
    fused = get_fusion(app.db, "mia"); assert fused and round(fused["labels"]["tapping"]["share"], 2) == 0.65 and [f["creator"] for f in fused["fused_from"]] == ["Alpha", "Beta"]
    assert called["slug"] == "mia" and called["sources"] == ["own:v1", "builtin:Claribel Dervla"]                       # nur freigegebene Stimmen + weiche Grundstimme
    assert set(json.loads(app.db.one("SELECT triggers FROM personas WHERE slug='mia'")["triggers"])) == {"tapping", "speech"}                            # nur Klassen, die im Mix vorkommen
    summary = app.models_overview()["models"][0]; assert summary["fusion"]["labels"]["speech"] > 0.3 and {c["creator"] for c in summary["fusion"]["creators"]} == {"Alpha", "Beta"}


def test_sessions_use_the_fusion_automatically_unless_overridden(tmp_path):
    from asmrai.profile import save_fusion
    cfg, app = make(tmp_path); app.create_model("Mia"); creator = add_creator(app, tmp_path, "Alpha", profile(0.8, 0.2, 3), 3)
    plain = app.create_session({"persona": "mia", "duration_min": 1, "seed": 1, "speech": False}); assert plain["params"]["fusion"] is None
    with pytest.raises(ValueError, match="verschmolzen"): app.create_session({"persona": "mia", "duration_min": 1, "style_source": "fusion"})
    save_fusion(app.db, "mia", {"1": 1}, fuse_profiles([("Alpha", profile(0.8, 0.2, 3), 1)]))
    auto = app.create_session({"persona": "mia", "duration_min": 1, "seed": 1, "speech": False})
    assert auto["params"]["fusion"] == "mia" and auto["creator_id"] is None
    assert style_profile(app.db, app.db.one("SELECT * FROM sessions WHERE id=?", (auto["id"],)))["fused"] is True
    single = app.create_session({"persona": "mia", "duration_min": 1, "seed": 1, "speech": False, "style_source": str(creator["id"])})
    assert single["params"]["fusion"] is None and single["creator_id"] == creator["id"]
    forced = app.create_session({"persona": "mia", "duration_min": 1, "seed": 1, "speech": False, "style_source": "default"}); assert forced["params"]["fusion"] is None and forced["creator_id"] is None


def test_fusion_needs_an_analyzed_creator(tmp_path):
    cfg, app = make(tmp_path); app.create_model("Mia")
    with pytest.raises(ValueError, match="analysiert"): app.start_fusion("mia", {})
    with pytest.raises(ValueError): app.start_fusion("nobody", {})
