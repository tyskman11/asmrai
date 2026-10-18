import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmrai import analysis, planner  # noqa: E402
from asmrai.app import App  # noqa: E402
from asmrai.config import Config  # noqa: E402
from asmrai.synth import dsp, voice  # noqa: E402
from asmrai.web import make_handler  # noqa: E402

SR = 22050


def stereo(mono):
    return np.stack([mono, mono], axis=1).astype(np.float32)


def test_block_features_and_classes():
    rng = np.random.default_rng(1)
    quiet = rng.standard_normal(SR * 10) * 0.0005                       # Rauschboden
    taps = np.zeros(SR * 10)
    for t in np.arange(5.2, 10, 0.4): taps[int(t * SR):int(t * SR) + 300] = rng.standard_normal(300) * np.exp(-np.arange(300) / 40)   # erst Stille, dann Taps
    signal_ = quiet + taps * 0.3
    rows = [analysis.block_features(stereo(signal_)[i:i + SR // 2], SR) for i in range(0, len(signal_) - SR // 2, SR // 2)]
    feat = np.asarray(rows, dtype=np.float32)
    labels = analysis.classify_blocks(feat)
    assert labels[:8].count("silence") >= 7 and labels[11:18].count("tapping") >= 6
    assert feat[11:18, analysis.FEATURE_NAMES.index("onset_rate")].mean() > 1.5


def test_segments_merge_short_runs():
    labels = ["tapping"] * 6 + ["texture"] + ["tapping"] * 5 + ["silence"] * 4
    feat = np.zeros((len(labels), len(analysis.FEATURE_NAMES)), dtype=np.float32)
    segments = analysis.build_segments(analysis.smooth_labels(labels, 1), feat, 0.5)
    assert [s["label"] for s in segments] == ["tapping", "silence"]
    assert segments[0]["end"] == segments[1]["start"]


def test_spatialize_itd_and_side():
    rng = np.random.default_rng(0); mono = rng.standard_normal(44100) * 0.1
    right = dsp.spatialize(mono, 80.0, 0.3); left = dsp.spatialize(mono, -80.0, 0.3)
    assert np.sqrt(np.mean(right[:, 1] ** 2)) > np.sqrt(np.mean(right[:, 0] ** 2))      # rechte Quelle: rechtes Ohr lauter
    assert np.sqrt(np.mean(left[:, 0] ** 2)) > np.sqrt(np.mean(left[:, 1] ** 2))
    xcorr = np.correlate(right[:, 0], right[:, 1], "full"); lag = np.argmax(xcorr) - (len(mono) - 1)
    assert lag > 10                                                                       # fernes (linkes) Ohr ist verzoegert


def test_triggers_are_finite_and_nonsilent():
    for label in ("tapping", "brushing", "scratching", "breath", "crinkle"):
        x = dsp.make_trigger(label, 3.0, {"centroid": 2200, "onset_rate": 2.5, "material": "glass"}, 5)
        assert len(x) == 3 * dsp.SR and np.isfinite(x).all() and np.abs(x).max() > 0
        assert np.array_equal(x, dsp.make_trigger(label, 3.0, {"centroid": 2200, "onset_rate": 2.5, "material": "glass"}, 5))   # deterministisch


def test_whisperize_keeps_length_and_energy():
    t = np.arange(24000) / 24000; voiced = 0.3 * np.sin(2 * np.pi * 140 * t) * (1 + np.sin(2 * np.pi * 3 * t))
    out = dsp.whisperize(voiced.astype(np.float32), 24000, 1.0, 1)
    assert len(out) == len(voiced) and abs(dsp.rms_db(out) - dsp.rms_db(voiced)) < 2


def test_planner_is_deterministic_and_respects_options():
    a = planner.plan_session(None, 300, 3, {"speech": True, "speech_share": 0.3})
    b = planner.plan_session(None, 300, 3, {"speech": True, "speech_share": 0.3})
    assert a == b
    assert sum(p["duration"] for p in a) >= 300
    assert any(p["kind"] == "speech" for p in a)
    no_speech = planner.plan_session(None, 120, 3, {"speech": False})
    assert all(p["kind"] != "speech" for p in no_speech)
    only = planner.plan_session(None, 120, 3, {"triggers": ["tapping"], "speech": False})
    assert {p["kind"] for p in only if p["kind"] != "pause"} == {"tapping"}
    with pytest.raises(ValueError): planner.plan_session(None, 60, 1, {"triggers": ["nothing"], "speech": False})


@pytest.fixture()
def app(tmp_path):
    cfg = Config({"paths": {"data": str(tmp_path / "data")}, "web": {}}); return App(cfg)


def test_voice_requires_consent_and_owner(app):
    with pytest.raises(ValueError): voice.register_voice(app.config, app.db, "x", "Me", False, [("a.wav", b"x")])
    with pytest.raises(ValueError): voice.register_voice(app.config, app.db, "x", " ", True, [("a.wav", b"x")])


def test_creator_roundtrip_never_touches_sources(app, tmp_path):
    src = tmp_path / "videos"; src.mkdir(); (src / "a.mp4").write_bytes(b"not a real video")
    creator = app.add_creator("Test Creator", str(src))
    assert creator["videos"] == 1 and creator["slug"] == "test-creator"
    with pytest.raises(ValueError): app.add_creator("Test Creator", str(src))
    app.delete_creator(creator["id"])
    assert (src / "a.mp4").exists() and app.creators() == []


def test_creator_source_is_limited_to_configured_nas_root(tmp_path):
    nas = tmp_path / "nas"; allowed = nas / "videos"; outside = tmp_path / "outside"
    allowed.mkdir(parents=True); outside.mkdir()
    cfg = Config({"paths": {"data": str(tmp_path / "data"), "nas_root": str(nas)}}); guarded = App(cfg)
    assert guarded.add_creator("Allowed", str(allowed))["source_dir"] == str(allowed.resolve())
    with pytest.raises(ValueError, match="lokalen NAS"):
        guarded.add_creator("Outside", str(outside))


def test_session_plan_and_web_guards(app):
    session = app.create_session({"duration_min": 2, "seed": 4, "speech": False})
    assert session["plan"] and session["status"] == "planned"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, "127.0.0.1", 0)); port = server.server_address[1]
    server.RequestHandlerClass = make_handler(app, "127.0.0.1", port); threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert json.loads(urllib.request.urlopen(base + "/api/sessions").read())[0]["id"] == session["id"]
        assert b"ASMRAI" in urllib.request.urlopen(base + "/").read()
        bad = urllib.request.Request(base + "/api/sessions", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as err: urllib.request.urlopen(bad)
        assert err.value.code == 403
        cross = urllib.request.Request(base + "/api/creators", data=b"{}", method="POST", headers={"Origin": "http://evil.example", "Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as err: urllib.request.urlopen(cross)
        assert err.value.code == 403
    finally:
        server.shutdown()


def test_job_manager_runs_and_cancels(app):
    done = threading.Event()
    app.jobs.register("t_ok", lambda ctx, p: (ctx.log("hi"), done.set()))
    def slow(ctx, p):
        for _ in range(100): ctx.check(); time.sleep(0.05)
    app.jobs.register("t_slow", slow)
    job = app.jobs.submit("t_ok", "ok"); assert done.wait(5)
    slow_job = app.jobs.submit("t_slow", "slow"); time.sleep(0.3); assert app.jobs.cancel(slow_job)
    for _ in range(50):
        if app.db.one("SELECT status FROM jobs WHERE id=?", (slow_job,))["status"] == "canceled": break
        time.sleep(0.1)
    assert app.db.one("SELECT status FROM jobs WHERE id=?", (slow_job,))["status"] == "canceled"
    time.sleep(0.2); assert app.db.one("SELECT status FROM jobs WHERE id=?", (job,))["status"] == "done"


def test_vision_label_cleaning_and_catalog():
    from asmrai import vision
    label = vision.clean_label({"triggers": ["tapping", "none", "dancing", 5], "props": [" Wooden Box ", "", "brush"], "framing": "close_up", "lighting": "nope", "setting": "Dark Room"})
    assert label == {"triggers": ["tapping"], "props": ["wooden box", "brush"], "framing": "close_up", "lighting": None, "setting": "dark room"}
    catalog = vision.aggregate([{"frames": [label, {**label, "triggers": ["brushing"]}]}])
    assert catalog["frames"] == 2 and catalog["triggers"] == {"tapping": 0.5, "brushing": 0.5} and catalog["framing"] == {"close_up": 1.0}


def test_vision_prompts_use_catalog_without_person_description():
    import random
    from asmrai import vision
    catalog = {"triggers": {"mic_brushing": 1.0}, "props": {"fluffy mic cover": 1.0}, "framing": {"close_up": 1.0}, "lighting": {"dim": 1.0}, "settings": {"dark bedroom": 1.0}}
    prompt = vision.build_prompt("brushing", catalog, random.Random(1))
    assert "microphone" in prompt and "extreme close-up" in prompt and "dim" in prompt and "original calm character" in prompt
    assert vision.build_prompt("tapping", None, random.Random(1))   # ohne Katalog trotzdem sinnvoll


def test_vision_stage_without_model_reports_error(app, tmp_path, monkeypatch):
    from asmrai import pipeline
    src = tmp_path / "v"; src.mkdir(); (src / "a.mp4").write_bytes(b"x")
    creator = app.add_creator("V", str(src)); video = app.db.one("SELECT * FROM videos WHERE creator_id=?", (creator["id"],))
    app.db.execute("UPDATE videos SET duration=10 WHERE id=?", (video["id"],))
    for stage in ("probe", "features", "visual", "transcribe", "segment"): app.db.set_stage(video["id"], stage, "done")
    class Ctx:
        def log(self, m, level=20): pass
        def check(self): pass
        def progress(self, *a): pass
    video = app.db.one("SELECT * FROM videos WHERE id=?", (video["id"],))
    pipeline.process_video(app.config, app.db, video, Ctx(), {"vision": True})
    assert app.db.stages(video["id"])["vision"]["state"] == "error"


def test_persona_drives_session_and_checks_consent(app):
    with pytest.raises(ValueError): app.save_persona({"name": "Mia", "voice": "own:ghost"})
    with pytest.raises(ValueError): app.save_persona({"name": "Mia", "image": "missing.png"})
    persona = app.save_persona({"name": "Mia", "appearance": "a woman with a silver bob", "voice": "builtin:Claribel Dervla", "triggers": ["tapping", "speech", "bogus"]})
    assert persona["triggers"] == ["tapping", "speech"]
    session = app.create_session({"persona": "mia", "duration_min": 1, "seed": 2})
    assert session["params"]["voice"] == "builtin:Claribel Dervla" and session["params"]["appearance"] == "a woman with a silver bob"
    assert {p["kind"] for p in session["plan"]} <= {"tapping", "speech", "pause"}
    from asmrai import vision
    import random
    assert "a woman with a silver bob" in vision.build_prompt("tapping", None, random.Random(1), "", session["params"]["appearance"])
    app.delete_persona("mia"); assert app.personas() == []


def test_watcher_settles_files_and_reports_unassigned(tmp_path):
    import os
    from asmrai.watcher import Watcher
    inbox = tmp_path / "input"; (inbox / "known").mkdir(parents=True); (inbox / "newcomer").mkdir()
    cfg = Config({"paths": {"data": str(tmp_path / "data")}, "inbox": {"path": str(inbox), "auto_analyze": False, "settle_seconds": 60}})
    app = App(cfg); app.add_creator("Known", str(inbox / "known"))
    old = inbox / "known" / "old.mp4"; old.write_bytes(b"x"); os.utime(old, (time.time() - 600,) * 2)
    fresh = inbox / "known" / "fresh.mp4"; fresh.write_bytes(b"x")                    # eben erst geschrieben
    (inbox / "newcomer" / "a.mp4").write_bytes(b"x")
    watcher = Watcher(app); result = watcher.check()
    titles = {v["title"] for v in app.db.query("SELECT title FROM videos")}
    assert titles == {"old"} and result["added"] == 1                                  # fresh wartet noch
    assert [u["name"] for u in watcher.unassigned] == ["newcomer"]
    os.utime(fresh, (time.time() - 600,) * 2); assert watcher.check()["added"] == 1
    assert app.creators()[0]["videos"] == 2 and not any(app.db.query("SELECT 1 FROM jobs"))   # auto_analyze aus -> kein Job


def test_whisper_models_listed_from_cache_only(tmp_path):
    from asmrai import transcribe
    for name in ("base", "medium"):
        snap = tmp_path / "w" / f"models--Systran--faster-whisper-{name}" / "snapshots" / "abc"; snap.mkdir(parents=True); (snap / "model.bin").write_bytes(b"0" * 10)
    (tmp_path / "w" / "models--Systran--faster-whisper-small").mkdir()          # unvollstaendig -> nicht anbieten
    cfg = Config({"paths": {"data": str(tmp_path / "d")}, "whisper": {"cache": str(tmp_path / "w")}})
    assert [m["name"] for m in transcribe.available_models(cfg)] == ["base", "medium"]
    with pytest.raises(RuntimeError, match="nicht lokal vorhanden"): transcribe.transcribe(cfg, tmp_path / "x.mp4", "large-v3")


def test_profile_without_segments_and_path_healing(app, tmp_path):
    from asmrai import pipeline
    from asmrai.profile import build_profile
    src = tmp_path / "old"; src.mkdir(); (src / "a.mp4").write_bytes(b"1234")
    creator = app.add_creator("H", str(src)); assert build_profile(app.db, creator["id"])["analyzed"] == 0     # kein IndexError
    new = tmp_path / "renamed"; src.rename(new)
    app.db.execute("UPDATE creators SET source_dir=? WHERE id=?", (str(new), creator["id"]))
    added, total = pipeline.scan_creator(app.db, app.db.one("SELECT * FROM creators WHERE id=?", (creator["id"],)))
    assert (added, total) == (0, 1) and app.db.one("SELECT path FROM videos")["path"] == str(new / "a.mp4")


def test_assets_require_documented_consent(app, tmp_path):
    from asmrai import assets
    src = tmp_path / "v"; src.mkdir(); (src / "a.mp4").write_bytes(b"x")
    creator = app.add_creator("Anna", str(src))
    for action in ("voice_refs", "voice_dataset", "face_refs"):
        with pytest.raises(ValueError, match="Keine dokumentierte Freigabe"): app.start_assets(creator["id"], action)
    with pytest.raises(ValueError, match="Nachweis"): app.set_consent(creator["id"], {"voice": True, "note": "ja"})
    app.set_consent(creator["id"], {"voice": True, "face": False, "note": "E-Mail von Anna vom 01.01.2026"})
    with pytest.raises(ValueError, match="Gesicht"): app.start_assets(creator["id"], "face_refs")      # nur Stimme freigegeben
    assert app.start_assets(creator["id"], "voice_refs")                                                # darf starten
    revoked = app.set_consent(creator["id"], {"voice": False, "face": False, "note": ""})
    assert not revoked["consent_voice"]
    with pytest.raises(ValueError): app.start_assets(creator["id"], "voice_refs")


def test_job_delete_retry_and_resume(app, tmp_path):
    gate = threading.Event(); ran = []
    app.jobs.register("t_gate", lambda ctx, p: (gate.wait(5), ran.append(p.get("n"))))
    app.jobs.register("t_fail", lambda ctx, p: (_ for _ in ()).throw(RuntimeError("boom")))
    ok = app.jobs.submit("t_gate", "a", {"n": 1}); fail = app.jobs.submit("t_fail", "b"); time.sleep(0.3)
    with pytest.raises(ValueError): app.jobs.delete(job_id=ok)                         # laeuft -> nicht loeschbar
    gate.set()
    for _ in range(50):
        if app.db.one("SELECT status FROM jobs WHERE id=?", (ok,))["status"] == "done": break
        time.sleep(0.1)
    for _ in range(50):
        if app.db.one("SELECT status FROM jobs WHERE id=?", (fail,))["status"] == "error": break
        time.sleep(0.1)
    assert app.jobs.delete(statuses=["error"]) == 1 and app.db.one("SELECT 1 AS x FROM jobs WHERE id=?", (fail,)) is None
    assert app.db.one("SELECT 1 AS x FROM jobs WHERE id=?", (ok,))                      # 'done' blieb unberuehrt
    with pytest.raises(ValueError): app.jobs.delete(statuses=["running", "queued"])      # nie laufende/wartende
    again = app.jobs.retry(ok); time.sleep(0.5); assert ran.count(1) == 2 and again != ok
    assert app.jobs.delete(job_id=ok) == 1
    # Neustart: ein als 'running' hinterlassener Job wird wieder eingereiht und laeuft
    stale = app.db.execute("INSERT INTO jobs(kind,title,params,status,created) VALUES('t_gate','stale','{\"n\": 7}','running',?)", (time.time(),))
    assert app.jobs.resume() == [stale]; time.sleep(0.6); assert 7 in ran
    assert app.db.one("SELECT status FROM jobs WHERE id=?", (stale,))["status"] == "done"


def test_generated_persona_is_original_adult_and_unique(app):
    app.config.raw.pop("llm", None)                                   # ohne LLM: regelbasierter Fallback
    first = app.generate_persona({"seed": 5}); second = app.generate_persona({"seed": 5})
    assert first["name"] != second["name"] and first["origin"] == "generated"                 # eindeutige Namen trotz gleichem Seed
    assert "adult" in first["appearance"].lower() and 21 <= first["age"] <= 45
    assert set(first["triggers"]) <= {"tapping", "brushing", "scratching", "breath", "crinkle", "speech"} and first["catchphrases"]
    assert (app.config.data / "ai_models" / first["slug"] / "_data" / "models" / "voice").is_dir()   # Ablage fuer spaetere Modelle: im Modell-Ordner der Person
    assert not (app.config.data / "models").exists()
    session = app.create_session({"persona": first["slug"], "duration_min": 1, "seed": 3})
    assert session["params"]["personality"] == first["personality"] and session["params"]["setting"] == first["setting"]


def test_persona_cleaning_blocks_minor_and_junk():
    from asmrai import creator_gen
    fb = creator_gen._fallback(__import__("random").Random(1), "en")
    out = creator_gen.clean({"name": "  A  B ", "age": 12, "appearance": "long hair, blue eyes", "triggers": ["tapping", "evil", "speech"], "catchphrases": ["hi", "", 5]}, fb)
    assert out["age"] >= 21 and out["appearance"].startswith("an adult person with long hair") and out["triggers"] == ["tapping", "speech"] and out["name"] == "A B"


def test_portrait_waits_for_gpu_jobs(app):
    from asmrai import portrait
    app.jobs.register("process", lambda ctx, p: time.sleep(0.6))
    busy = app.jobs.submit("process", "x"); time.sleep(0.15)
    class Ctx:
        job_id = 999; logs = []
        def check(self): pass
        def log(self, m): self.logs.append(m)
    ctx = Ctx(); started = time.time(); portrait.POLL_SECONDS = 0.2
    portrait.wait_for_gpu(app, ctx, 999)
    assert ctx.logs and time.time() - started >= 0.3


def test_persona_appearance_drops_pronoun_lead():
    from asmrai import creator_gen
    fb = creator_gen._fallback(__import__("random").Random(1), "en")
    out = creator_gen.clean({"appearance": "She has wavy hair and round glasses."}, fb)
    assert out["appearance"] == "an adult person with wavy hair and round glasses."


def _fake_portrait(path, shade):
    import cv2
    image = np.full((800, 600, 3), 40, np.uint8); cv2.circle(image, (300, 300), 140, (shade, 160, 200), -1); cv2.imwrite(str(path), image)


def test_pick_look_prefers_front_and_hair(app):
    from asmrai import looks
    import random
    for i, (angle, hair, face) in enumerate([("profile", "bob", 1), ("front", "bob", 1), ("front", "long", 1), ("front", "bob", 0)]):
        app.db.execute("INSERT INTO looks(group_name,persona_slug,file,angle,hair,has_face,source,created) VALUES('g','p',?,?,?,?,?,?)", (f"g/{i:03d}", angle, hair, face, f"s{i}", time.time()))
    picks = {looks.pick_look(app.db, "p", random.Random(s), "bob")["file"] for s in range(30)}
    assert picks == {"g/001"}                                      # nur frontal + Frisur + Gesicht
    assert looks.pick_look(app.db, "nobody", random.Random(1)) is None


def test_rename_persona_keeps_links(app):
    persona = app.save_persona({"name": "Lumi"}); other = app.save_persona({"name": "Elara"})
    app.db.execute("INSERT INTO looks(group_name,persona_slug,file,angle,has_face,source,created) VALUES('g',?,?,?,1,'s1',?)", (persona["slug"], "g/001", "front", time.time()))
    renamed = app.rename_persona("lumi", "  Mira Sol ")
    assert renamed["name"] == "Mira Sol" and renamed["slug"] == "lumi"                         # Kennung bleibt -> Looks bleiben verbunden
    assert app.db.one("SELECT persona_slug FROM looks")["persona_slug"] == "lumi"
    with pytest.raises(ValueError, match="vergeben"): app.rename_persona("lumi", "elara")
    with pytest.raises(ValueError): app.rename_persona("lumi", "  ")
    with pytest.raises(ValueError): app.rename_persona("nobody", "X")


def test_dominant_identity_excludes_guest_faces():
    import numpy as np
    from asmrai import assets
    rng = np.random.default_rng(0); main = rng.standard_normal(128); guest = rng.standard_normal(128)
    embeddings = [main + 0.1 * rng.standard_normal(128) for _ in range(30)] + [guest + 0.1 * rng.standard_normal(128) for _ in range(4)]
    keep = assets.dominant_identity(embeddings)
    assert sum(keep) == 30 and all(keep[:30]) and not any(keep[30:])
