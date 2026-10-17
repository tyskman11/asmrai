import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmrai.app import App  # noqa: E402
from asmrai.config import Config  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    return App(Config({"paths": {"data": str(tmp_path / "data")}}))


def wait(predicate, tries=100):
    for _ in range(tries):
        if predicate(): return True
        time.sleep(0.05)
    return False


def test_higher_priority_runs_first_then_fifo(app):
    gate, order = threading.Event(), []
    def handler(ctx, p):
        if p["name"] == "A": gate.wait(5)
        order.append(p["name"])
    app.jobs.register("t_prio", handler)
    app.jobs.submit("t_prio", "A", {"name": "A"}); assert wait(lambda: app.jobs.contexts)      # A belegt den Worker
    app.jobs.submit("t_prio", "B", {"name": "B"}, 0); app.jobs.submit("t_prio", "C", {"name": "C"}, 0)
    app.jobs.submit("t_prio", "D", {"name": "D"}, 10); app.jobs.submit("t_prio", "E", {"name": "E"}, -5)
    gate.set(); assert wait(lambda: len(order) == 5)
    assert order == ["A", "D", "B", "C", "E"]


def test_prioritize_moves_waiting_job_to_front(app):
    gate, order = threading.Event(), []
    def handler(ctx, p):
        if p["name"] == "A": gate.wait(5)
        order.append(p["name"])
    app.jobs.register("t_prio", handler)
    app.jobs.submit("t_prio", "A", {"name": "A"}); assert wait(lambda: app.jobs.contexts)
    app.jobs.submit("t_prio", "B", {"name": "B"}); c = app.jobs.submit("t_prio", "C", {"name": "C"})
    app.jobs.prioritize(c); gate.set(); assert wait(lambda: len(order) == 3)
    assert order == ["A", "C", "B"]
    with pytest.raises(ValueError): app.jobs.prioritize(c)                                      # nur wartende Jobs


def test_running_job_pauses_for_higher_priority_and_resumes(app):
    state, order = {"i": 0}, []
    def handler(ctx, p):
        if p["name"] == "H": order.append("H"); return
        while state["i"] < 6:
            ctx.yield_to_higher_priority()                                                      # wie in der Analyse nach jedem Video
            time.sleep(0.15); order.append(f"L{state['i']}"); state["i"] += 1
    app.jobs.register("t_prio", handler)
    low = app.jobs.submit("t_prio", "low", {"name": "L"}); assert wait(lambda: "L1" in order)
    app.jobs.submit("t_prio", "high", {"name": "H"}, 10)
    assert wait(lambda: "L5" in order, 200)
    assert order.index("H") < order.index("L5") and order.index("H") > order.index("L1")        # H unterbrach L mittendrin
    assert order[:2] == ["L0", "L1"] and [x for x in order if x.startswith("L")] == [f"L{i}" for i in range(6)]   # nichts doppelt, nichts verloren
    assert wait(lambda: app.db.one("SELECT status FROM jobs WHERE id=?", (low,))["status"] == "done")


def test_no_automatic_priority_and_running_jobs_resume_first(app, tmp_path):
    src = tmp_path / "v"; src.mkdir(); (src / "a.mp4").write_bytes(b"x")
    creator = app.add_creator("P", str(src))
    video = app.db.one("SELECT id FROM videos"); app.db.set_stage(video["id"], "segment", "done")      # Creator hat schon Ergebnisse
    app.jobs.register("process", lambda ctx, p: None)
    job = app.start_process(creator["id"], {}); row = app.db.one("SELECT priority,title FROM jobs WHERE id=?", (job,))
    assert row["priority"] == 0 and "vorrangig" not in row["title"]
    queued = app.db.execute("INSERT INTO jobs(kind,title,params,status,created,priority) VALUES('process','Q','{}','queued',0,0)")
    running = app.db.execute("INSERT INTO jobs(kind,title,params,status,created,priority) VALUES('process','R','{}','running',0,0)")
    assert [r["id"] for r in app.db.query("SELECT id FROM jobs WHERE status IN ('running','queued') ORDER BY status='running' DESC, id")][0] == running


def test_retry_only_failed(app, tmp_path):
    src = tmp_path / "v"; src.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"): (src / name).write_bytes(b"x")
    creator = app.add_creator("P", str(src)); ids = [v["id"] for v in app.db.query("SELECT id FROM videos ORDER BY id")]
    app.db.set_stage(ids[0], "segment", "done"); app.db.set_stage(ids[1], "visual", "error", "boom")
    app.db.execute("UPDATE videos SET error='x' WHERE id=?", (ids[2],))
    assert app.failed_video_ids(creator["id"]) == [ids[1], ids[2]] and app.creator_summary(creator)["failed"] == 2
    seen = []
    app.jobs.register("process", lambda ctx, p: seen.append(p.get("video_ids")))
    job = app.start_process(creator["id"], {"only_failed": True})
    assert wait(lambda: seen) and seen[0] == [ids[1], ids[2]] and "nur fehlgeschlagene" in app.db.one("SELECT title FROM jobs WHERE id=?", (job,))["title"]
    app.db.execute("UPDATE videos SET error=NULL"); app.db.execute("DELETE FROM stages WHERE state='error'")
    with pytest.raises(ValueError): app.start_process(creator["id"], {"only_failed": True})
