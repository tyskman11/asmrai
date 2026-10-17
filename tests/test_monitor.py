import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmrai import monitor  # noqa: E402
from asmrai.app import App  # noqa: E402
from asmrai.config import Config  # noqa: E402


def make(tmp_path):
    return App(Config({"paths": {"data": str(tmp_path / "data"), "ai_models": str(tmp_path / "ai")}}))


def test_duration_text():
    assert monitor.duration_text(42) == "42 s" and monitor.duration_text(185) == "3 min" and monitor.duration_text(7 * 3600 + 14 * 60) == "7 h 14 min"


def test_idle_status_always_shows_urls(tmp_path):
    app = make(tmp_path)
    lines, busy = monitor.status_lines(app, "laeuft", "http://127.0.0.1:8780/", "http://127.0.0.1:8188/")
    assert not busy and "Jetzt: Leerlauf" in lines[1]
    assert "http://127.0.0.1:8780/" in lines[-1] and "http://127.0.0.1:8188/ (laeuft)" in lines[-1] and "Logs:" in lines[-1]
    assert "Videos analysiert 0/0" in lines[0] and "Jobs 0 laeuft, 0 warten" in lines[0]


def test_running_job_shows_current_step_percent_and_eta(tmp_path):
    app = make(tmp_path)
    ctx = None
    def long_job(c, params):
        c.log("  transcribe ..."); c.progress(0.25); time.sleep(1.5)
    app.jobs.register("t_long", long_job); app.jobs.submit("t_long", "Analyse: Test")
    for _ in range(30):
        if app.jobs.contexts: break
        time.sleep(0.1)
    time.sleep(0.3)
    lines, busy = monitor.status_lines(app, "nicht erreichbar", "http://127.0.0.1:8780/", "http://127.0.0.1:8188/")
    text = "\n".join(lines)
    assert busy and "Jobs 1 laeuft" in text and "Jetzt: Job 1 Analyse: Test 25 %" in text and "transcribe ..." in text and "ETA" in text
    assert "(nicht erreichbar)" in lines[-1] and "http://127.0.0.1:8780/" in lines[-1]
    time.sleep(1.5)
