"""Logging: Konsole (immer sichtbar im Terminal) + Dateien in logs/.

* logs/asmrai.log    - alles (Rotation 5 MB x 3)
* logs/errors.log    - nur WARNUNG/FEHLER mit vollem Traceback (Fehlererfassung)
* logs/analysis.log  - Schritte der Video-Analyse mit Zeiten pro Stufe

Die Konsole zeigt jede Zeile sofort mit Uhrzeit; Warnungen/Fehler sind deutlich markiert.
"""
from __future__ import annotations

import logging
import platform
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, "%H:%M:%S"); text = record.getMessage()
        if record.levelno >= logging.ERROR: return f"[{stamp}] X FEHLER: {text}"
        if record.levelno >= logging.WARNING: return f"[{stamp}] ! WARNUNG: {text}"
        return f"[{stamp}] {text}"


class _StdoutHandler(logging.StreamHandler):
    """Schreibt auf das *aktuelle* sys.stdout (auch wenn es spaeter umgeleitet wird)."""
    def __init__(self) -> None: logging.Handler.__init__(self)

    @property
    def stream(self): return sys.stdout


def setup(level: int = logging.INFO, log_dir: Path | None = None) -> Path:
    """Idempotent. Gibt den Log-Ordner zurueck."""
    global _configured
    directory = log_dir or LOG_DIR; directory.mkdir(parents=True, exist_ok=True)
    if _configured: return directory
    for stream in (sys.stdout, sys.stderr):
        try: stream.reconfigure(encoding="utf-8", errors="replace")      # kein Absturz durch Sonderzeichen in Videotiteln
        except Exception:  # noqa: BLE001
            pass
    root = logging.getLogger("asmrai"); root.setLevel(level); root.propagate = False
    console = _StdoutHandler(); console.setFormatter(ConsoleFormatter()); console.setLevel(level)
    everything = RotatingFileHandler(directory / "asmrai.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"); everything.setFormatter(logging.Formatter(FILE_FORMAT))
    errors = RotatingFileHandler(directory / "errors.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"); errors.setLevel(logging.WARNING); errors.setFormatter(logging.Formatter(FILE_FORMAT))
    root.addHandler(console); root.addHandler(everything); root.addHandler(errors)
    # Analyse: eigene Datei; Konsole/asmrai.log bekommen diese Zeilen ueber den Job-Logger (ctx.log), daher keine Weitergabe (keine Doppelausgabe)
    analysis = logging.getLogger("asmrai.analysis"); analysis.propagate = False; analysis.setLevel(level)
    file = RotatingFileHandler(directory / "analysis.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"); file.setFormatter(logging.Formatter(FILE_FORMAT))
    analysis.addHandler(file); analysis.addHandler(errors)

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt): sys.__excepthook__(exc_type, exc, tb); return
        root.critical("Unbehandelte Ausnahme", exc_info=(exc_type, exc, tb))
    sys.excepthook = hook
    threading.excepthook = lambda args: root.critical("Unbehandelte Ausnahme im Thread %s", getattr(args.thread, "name", "?"), exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    _configured = True
    # Ein eindeutiger Startdatensatz erleichtert die Zuordnung von Fehlern zu
    # Python-/Betriebssystem-Versionen, ohne private Konfigurationswerte zu loggen.
    from . import __version__
    root.info("ASMRAI %s gestartet | Python %s | %s | Logs: %s", __version__, platform.python_version(), platform.platform(), directory)
    return directory


def tail(name: str, lines: int = 200, log_dir: Path | None = None) -> str:
    """Letzte Zeilen einer Log-Datei (fuer die WebUI). Nur bekannte Namen."""
    if name not in ("asmrai", "errors", "analysis"): raise ValueError("Unbekanntes Log.")
    path = (log_dir or LOG_DIR) / f"{name}.log"
    if not path.exists(): return ""
    with open(path, "rb") as handle:
        handle.seek(0, 2); size = handle.tell(); handle.seek(max(0, size - 400_000))
        data = handle.read().decode("utf-8", errors="replace")
    return "\n".join(data.splitlines()[-max(1, min(lines, 2000)):])
