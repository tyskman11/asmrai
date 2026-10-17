"""Kommandozeile: python -m asmrai <befehl>"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .app import App
from .web import serve


def wait(app: App, job_id: int) -> None:
    shown = 0
    while True:
        job = app.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        lines = (job["log"] or "").splitlines()
        for line in lines[shown:]: print(line, flush=True)
        shown = len(lines)
        if job["status"] in ("done", "error", "canceled", "interrupted"):
            print(f"-> {job['status']}"); return
        time.sleep(1.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asmrai", description="ASMRAI - Stil-Analyse und eigene ASMR-Sessions")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve", help="WebUI starten"); p.add_argument("--host"); p.add_argument("--port", type=int)
    sub.add_parser("doctor", help="Umgebung pruefen")
    sub.add_parser("creators", help="Creators auflisten")
    p = sub.add_parser("add-creator", help="Creator mit Videoordner anlegen"); p.add_argument("name"); p.add_argument("folder")
    p = sub.add_parser("analyze", help="Videos eines Creators analysieren"); p.add_argument("creator"); p.add_argument("--no-transcribe", action="store_true")
    p.add_argument("--model", default=None); p.add_argument("--force", action="store_true"); p.add_argument("--limit", type=int)
    p = sub.add_parser("session", help="Session planen und Audio rendern"); p.add_argument("--creator"); p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--theme", default=""); p.add_argument("--language", default="en"); p.add_argument("--seed", type=int); p.add_argument("--no-speech", action="store_true")
    p.add_argument("--video", action="store_true", help="zusaetzlich Ambient-Video rendern")
    args = parser.parse_args(argv)
    from . import logging_setup
    log_dir = logging_setup.setup()
    logging.getLogger("asmrai").info("Befehl gestartet: %s", args.cmd)
    print(f"Logs: {log_dir}  (asmrai.log, errors.log, analysis.log)", flush=True)
    app = App()

    def find(slug_or_name: str) -> dict:
        from .pipeline import slugify
        row = app.db.one("SELECT * FROM creators WHERE slug=?", (slugify(slug_or_name),))
        if not row: raise SystemExit(f"Creator nicht gefunden: {slug_or_name}")
        return row

    if args.cmd == "serve": serve(app, args.host, args.port)
    elif args.cmd == "doctor":
        for key, value in app.doctor().items(): print(f"{key:18} {value or 'FEHLT'}")
    elif args.cmd == "creators":
        for c in app.creators(): print(f"{c['slug']:20} {c['videos']:3} Videos {c['hours']:6.1f} h  analysiert {c['analyzed']}  Profil {'ja' if c['has_profile'] else 'nein'}")
    elif args.cmd == "add-creator":
        c = app.add_creator(args.name, args.folder); print(f"{c['name']}: {c['videos']} Videos")
    elif args.cmd == "analyze":
        creator = find(args.creator); ids = None
        if args.limit: ids = [v["id"] for v in app.db.query("SELECT id FROM videos WHERE creator_id=? ORDER BY id LIMIT ?", (creator["id"], args.limit))]
        wait(app, app.start_process(creator["id"], {"transcribe": not args.no_transcribe, "whisper_model": args.model, "force": args.force, "video_ids": ids}))
    elif args.cmd == "session":
        creator = find(args.creator) if args.creator else None
        session = app.create_session({"creator_id": creator["id"] if creator else None, "duration_min": args.minutes, "theme": args.theme,
                                      "language": args.language, "seed": args.seed, "speech": not args.no_speech})
        wait(app, app.jobs.submit("audio", f"Audio: Session {session['id']}", {"session_id": session["id"]}))
        if args.video: wait(app, app.jobs.submit("video", f"Video: Session {session['id']}", {"session_id": session["id"], "mode": "ambient"}))
        print(json.dumps({"session": session["id"], "folder": str(app.config.sessions / str(session["id"]))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
