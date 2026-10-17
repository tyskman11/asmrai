"""Live-Statusblock im Terminal (wie bei SeriesAI): Kennzahlen, was gerade passiert, und immer die direkten Adressen.

[22:28:52] Videos analysiert 5/214 | Transkripte 5/214 | Looks 18 (offen 0) | Sessions 1 | Jobs 1 laeuft, 2 warten
        Jetzt: Job 19 Analyse: PokiASMR 3 % | Transkription (large-v3) 40 % | ETA 2 h 10 min
        WebUI: http://127.0.0.1:8780/  |  ComfyUI: http://127.0.0.1:8188/ (laeuft)  |  Logs: C:\\AI\\ASMRAI\\logs
"""
from __future__ import annotations

import sys
import threading
import time
import urllib.request

from . import looks
from .logging_setup import LOG_DIR

INDENT = " " * 8


def duration_text(seconds: float) -> str:
    seconds = int(max(seconds, 0)); hours, rest = divmod(seconds, 3600); minutes = rest // 60
    return f"{hours} h {minutes:02d} min" if hours else (f"{minutes} min" if minutes else f"{seconds} s")


def comfyui_state(url: str, timeout: float = 5.0) -> str:
    from .comfy import state
    return state(url, timeout)


def status_lines(app, comfy_state: str, web_url: str, comfy_url: str, now: float | None = None) -> tuple[list[str], bool]:
    """-> (Zeilen, busy). Reine Funktion ueber den App-Zustand (gut testbar)."""
    db = app.db; stamp = time.strftime("%H:%M:%S", time.localtime(now or time.time()))
    videos = db.one("SELECT COUNT(*) AS n FROM videos")["n"]
    segmented = db.one("SELECT COUNT(*) AS n FROM stages WHERE stage='segment' AND state='done'")["n"]
    transcribed = db.one("SELECT COUNT(*) AS n FROM stages WHERE stage='transcribe' AND state='done'")["n"]
    sessions = db.one("SELECT COUNT(*) AS n FROM sessions")["n"]
    active = db.query("SELECT status FROM jobs WHERE status IN ('queued','running')")
    running = sum(1 for j in active if j["status"] == "running"); waiting = len(active) - running
    try: pending_looks = looks.count_pending(app.config, db); looks_total = db.one("SELECT COUNT(*) AS n FROM looks")["n"]
    except Exception:  # noqa: BLE001
        pending_looks = looks_total = 0
    lines = [f"[{stamp}] Videos analysiert {segmented}/{videos} | Transkripte {transcribed}/{videos} | Looks {looks_total} (offen {pending_looks}) | Sessions {sessions} | "
             f"Jobs {running} laeuft, {waiting} warten"]
    contexts = list(app.jobs.contexts.values())
    if not contexts: lines.append(f"{INDENT}Jetzt: Leerlauf")
    for context in contexts[:3]:
        percent = round(context._progress * 100); elapsed = time.monotonic() - context.started
        eta = f" | ETA {duration_text(elapsed * (1 - context._progress) / context._progress)}" if context._progress >= 0.02 else ""
        step = f" | {context.current[:90]}" if context.current else ""
        lines.append(f"{INDENT}Jetzt: Job {context.job_id} {context.title} {percent} %{step}{eta}")
    lines.append(f"{INDENT}WebUI: {web_url}  |  ComfyUI: {comfy_url} ({comfy_state})  |  Logs: {LOG_DIR}")
    return lines, bool(contexts)


class Monitor(threading.Thread):
    """Druckt den Statusblock alle ``busy_seconds`` (waehrend Jobs laufen) bzw. ``idle_seconds`` (Leerlauf)."""
    def __init__(self, app, web_url: str, comfy_url: str, busy_seconds: float = 5.0, idle_seconds: float = 30.0) -> None:
        super().__init__(daemon=True, name="monitor")
        self.app, self.web_url, self.comfy_url, self.busy, self.idle = app, web_url, comfy_url, busy_seconds, idle_seconds

    def run(self) -> None:
        while True:
            try:
                lines, busy = status_lines(self.app, comfyui_state(self.comfy_url), self.web_url, self.comfy_url)
                print("\n".join(lines), flush=True)
            except Exception as error:  # noqa: BLE001 - der Monitor darf nie stoeren
                print(f"[Monitor] {error}", file=sys.stderr, flush=True); busy = False
            time.sleep(self.busy if busy else self.idle)
