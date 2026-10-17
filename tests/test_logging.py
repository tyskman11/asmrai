import logging
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmrai import logging_setup  # noqa: E402
from asmrai.app import App  # noqa: E402
from asmrai.config import Config  # noqa: E402


@pytest.fixture()
def logs(tmp_path):
    root = logging.getLogger("asmrai"); analysis = logging.getLogger("asmrai.analysis")
    saved = (list(root.handlers), list(analysis.handlers), logging_setup._configured, sys.excepthook)
    root.handlers.clear(); analysis.handlers.clear(); logging_setup._configured = False
    directory = logging_setup.setup(log_dir=tmp_path / "logs")
    yield directory
    for handler in root.handlers + analysis.handlers: handler.close()
    root.handlers[:], analysis.handlers[:], logging_setup._configured, sys.excepthook = saved[0], saved[1], saved[2], saved[3]


def flush():
    for name in ("asmrai", "asmrai.analysis"):
        for handler in logging.getLogger(name).handlers: handler.flush()


def wait_for(app, job, states, tries=100):
    for _ in range(tries):
        if app.db.one("SELECT status FROM jobs WHERE id=?", (job,))["status"] in states: break
        time.sleep(0.1)
    time.sleep(0.2)


def test_console_format_marks_warnings_and_errors(capsys, logs):
    log = logging.getLogger("asmrai.test")
    log.info("normal"); log.warning("achtung"); log.error("kaputt")
    out = capsys.readouterr().out
    assert "] normal" in out and "! WARNUNG: achtung" in out and "X FEHLER: kaputt" in out


def test_errors_file_has_traceback_and_ignores_info(logs):
    log = logging.getLogger("asmrai.test")
    log.info("nur info")
    try: 1 / 0
    except ZeroDivisionError: log.error("Division", exc_info=True)
    flush()
    errors = (logs / "errors.log").read_text(encoding="utf-8"); everything = (logs / "asmrai.log").read_text(encoding="utf-8")
    assert "ZeroDivisionError" in errors and "Traceback" in errors and "nur info" not in errors and "nur info" in everything
    assert "Division" in logging_setup.tail("errors", 50, logs) and logging_setup.tail("analysis", 5, logs) == ""
    with pytest.raises(ValueError): logging_setup.tail("../config", 5, logs)


def test_failed_job_is_logged_with_traceback_and_steps_are_visible(capsys, logs, tmp_path):
    app = App(Config({"paths": {"data": str(tmp_path / "data")}}))
    def broken(ctx, params):
        ctx.log("Schritt 1 laeuft"); raise RuntimeError("kaputter Schritt")
    app.jobs.register("t_broken", broken); job = app.jobs.submit("t_broken", "Testjob")
    wait_for(app, job, ("error",))
    out = capsys.readouterr().out
    assert f"Job {job} gestartet: Testjob" in out and f"Job {job} Schritt 1 laeuft" in out and "X FEHLER" in out and "fehlgeschlagen" in out
    flush(); errors = (logs / "errors.log").read_text(encoding="utf-8")
    assert "kaputter Schritt" in errors and "Traceback" in errors


def test_analysis_logs_each_stage_and_failure(capsys, logs, tmp_path):
    app = App(Config({"paths": {"data": str(tmp_path / "data")}}))
    src = tmp_path / "v"; src.mkdir(); (src / "kaputt.mp4").write_bytes(b"kein video")
    creator = app.add_creator("T", str(src))
    job = app.start_process(creator["id"], {"transcribe": False})
    wait_for(app, job, ("done", "error"))
    out = capsys.readouterr().out
    assert "probe ..." in out and "FEHLER in probe" in out and "uebersprungen wegen Fehler" in out
    flush()
    analysis_log = (logs / "analysis.log").read_text(encoding="utf-8"); errors = (logs / "errors.log").read_text(encoding="utf-8")
    assert "[Video 1] probe ..." in analysis_log and "ffprobe fehlgeschlagen" in errors and "Traceback" in errors
