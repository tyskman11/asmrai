"""Eingangsordner-Wächter: erkennt nach und nach eintreffende Videos und analysiert sie automatisch.

* Neue Dateien in einem Creator-Ordner werden erst registriert, wenn sie ``settle_seconds`` unveraendert sind
  (damit halb kopierte Dateien nicht analysiert werden).
* Unterordner des Eingangsordners, die noch keinem Creator gehoeren, werden nur *gemeldet* - angelegt wird erst auf Klick.
* Es wird nie etwas geloescht oder veraendert.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from . import pipeline
from .app import App


class Watcher:
    def __init__(self, app: App) -> None:
        cfg = app.config
        self.app = app
        self.inbox = cfg.path("inbox", "path", "input")
        self.auto = bool(cfg.get("inbox", "auto_analyze", False))
        self.interval = float(cfg.get("inbox", "interval_seconds", 60))
        self.settle = float(cfg.get("inbox", "settle_seconds", 45))
        self.options = {"transcribe": bool(cfg.get("inbox", "transcribe", True)), "whisper_model": cfg.get("inbox", "whisper_model", "medium")}
        self.unassigned: list[dict] = []
        self.last_check: float | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="inbox-watcher").start()

    def trigger(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while True:
            try: self.check()
            except Exception as error:  # noqa: BLE001 - der Waechter darf nie sterben
                print(f"[Waechter] Fehler: {error}", flush=True)
            self._wake.wait(self.interval); self._wake.clear()

    def check(self) -> dict:
        with self._lock:
            db = self.app.db; added_total = 0; submitted = []
            creators = db.query("SELECT * FROM creators")
            active = {json.loads(j["params"] or "{}").get("creator_id") for j in self.app.jobs.active("process")}
            for creator in creators:
                if not Path(creator["source_dir"]).is_dir(): continue
                added, _ = pipeline.scan_creator(db, creator, settle_seconds=self.settle); added_total += added
                if self.auto and creator["id"] not in active:
                    pending = [v["id"] for v in db.query(
                        "SELECT v.id FROM videos v WHERE v.creator_id=? AND v.error IS NULL AND NOT EXISTS "
                        "(SELECT 1 FROM stages s WHERE s.video_id=v.id AND s.stage='segment' AND s.state='done')", (creator["id"],))]
                    if pending:
                        submitted.append(self.app.jobs.submit("process", f"Auto-Analyse: {creator['name']} ({len(pending)} neu)",
                                                              {"creator_id": creator["id"], "video_ids": pending, **self.options}))
            self.unassigned = self._find_unassigned(creators)
            self.last_check = time.time()
            return {"added": added_total, "jobs": submitted, "unassigned": len(self.unassigned)}

    def _find_unassigned(self, creators: list[dict]) -> list[dict]:
        if not self.inbox or not self.inbox.is_dir(): return []
        owned = [Path(c["source_dir"]).resolve() for c in creators]
        found = []
        for folder in sorted(p for p in self.inbox.iterdir() if p.is_dir()):
            resolved = folder.resolve()
            if any(resolved == o or o in resolved.parents or resolved in o.parents for o in owned): continue
            videos = sum(1 for p in folder.rglob("*") if p.suffix.lower() in pipeline.VIDEO_EXT)
            if videos: found.append({"folder": str(folder), "name": folder.name, "videos": videos})
        return found

    def status(self) -> dict:
        return {"inbox": str(self.inbox), "auto_analyze": self.auto, "interval": self.interval, "settle_seconds": self.settle,
                "last_check": self.last_check, "unassigned": self.unassigned}
