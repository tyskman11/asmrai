"""WebUI-Server (nur Standardbibliothek): JSON-API + statische Single-Page-App, standardmaessig nur auf 127.0.0.1."""
from __future__ import annotations

import json
import threading
import logging
import mimetypes
import os
import re
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .app import App
from .profile import build_profile, get_profile
from .synth import voice as voices

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("asmrai.web")
ROUTES: list[tuple[str, re.Pattern, str]] = []
MAX_JSON_BODY = 5 * 1024 * 1024  # Uploads (z. B. Stimmreferenzen) bleiben moeglich, Speicherverbrauch aber begrenzt.


def route(method: str, pattern: str):
    def decorator(fn):
        ROUTES.append((method, re.compile("^" + pattern + "$"), fn.__name__)); return fn
    return decorator


class Api:
    """Alle Endpunkte. Rueckgabe: JSON-serialisierbar oder (status, dict)."""
    def __init__(self, app: App) -> None: self.app = app

    @route("GET", "/api/overview")
    def overview(self, q, body): return self.app.overview()

    @route("GET", "/api/doctor")
    def doctor(self, q, body): return self.app.doctor()

    @route("GET", "/api/fs")
    def fs(self, q, body):
        """Ordner-Auswahl: listet nur Unterverzeichnisse (lokale Bedienung)."""
        raw = (q.get("path") or [""])[0]
        if not raw:
            drives = [f"{c}:\\" for c in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{c}:\\")] if os.name == "nt" else ["/"]
            return {"path": "", "parent": None, "dirs": drives, "videos": 0}
        path = Path(raw)
        if not path.is_dir(): raise ValueError("Kein Ordner")
        dirs = sorted(str(p) for p in path.iterdir() if p.is_dir() and not p.name.startswith("$"))[:500] if os.access(path, os.R_OK) else []
        count = sum(1 for p in path.rglob("*") if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}) if len(dirs) < 500 else 0
        return {"path": str(path), "parent": str(path.parent) if path.parent != path else "", "dirs": dirs, "videos": count}

    @route("GET", "/api/logs")
    def logs(self, q, body):
        from .logging_setup import tail
        name = (q.get("name") or ["errors"])[0]; lines = int((q.get("lines") or ["200"])[0])
        return {"name": name, "text": tail(name, lines)}

    @route("GET", "/api/whisper-models")
    def whisper_models(self, q, body):
        from .transcribe import available_models
        models = available_models(self.app.config); default = self.app.config.get("whisper", "model", "medium")
        return {"models": models, "default": default if default in [m["name"] for m in models] else (models[-1]["name"] if models else None)}

    @route("GET", "/api/inbox")
    def inbox(self, q, body): return self.app.watcher.status() if self.app.watcher else {"unassigned": [], "auto_analyze": False}

    @route("POST", "/api/inbox/scan")
    def inbox_scan(self, q, body):
        if not self.app.watcher: raise ValueError("Waechter laeuft nicht")
        return self.app.watcher.check()

    # Creators ----------------------------------------------------------------------------------
    @route("GET", "/api/creators")
    def creators(self, q, body): return self.app.creators()

    @route("POST", "/api/creators")
    def add_creator(self, q, body): return self.app.add_creator(body.get("name", ""), body.get("source_dir", ""), body.get("notes", ""))

    @route("DELETE", r"/api/creators/(\d+)")
    def delete_creator(self, q, body, cid): self.app.delete_creator(int(cid)); return {"ok": True}

    @route("POST", r"/api/creators/(\d+)/process")
    def process(self, q, body, cid): return {"job": self.app.start_process(int(cid), body)}

    @route("POST", r"/api/creators/(\d+)/consent")
    def consent(self, q, body, cid): return self.app.set_consent(int(cid), body)

    @route("POST", r"/api/creators/(\d+)/assets")
    def assets(self, q, body, cid): return {"job": self.app.start_assets(int(cid), body.get("action", ""))}

    @route("GET", r"/api/creators/(\d+)/videos")
    def videos(self, q, body, cid): return self.app.videos(int(cid))

    @route("GET", r"/api/creators/(\d+)/profile")
    def profile(self, q, body, cid): return get_profile(self.app.db, int(cid)) or {}

    @route("POST", r"/api/creators/(\d+)/profile")
    def rebuild_profile(self, q, body, cid): return build_profile(self.app.db, int(cid))

    @route("GET", r"/api/videos/(\d+)")
    def video(self, q, body, vid): return self.app.video_detail(int(vid))

    @route("GET", r"/api/videos/(\d+)/series")
    def series(self, q, body, vid): return self.app.video_series(int(vid))

    # Stimmen / Figuren ----------------------------------------------------------------------------
    @route("GET", "/api/voices")
    def list_voices(self, q, body): return voices.list_voices(self.app.config, self.app.db)

    @route("POST", "/api/voices")
    def add_voice(self, q, body):
        import base64
        files = [(f["name"], base64.b64decode(f["data"])) for f in body.get("files", [])]
        return voices.register_voice(self.app.config, self.app.db, body.get("name", ""), body.get("owner", ""), bool(body.get("consent")), files)

    @route("DELETE", r"/api/voices/([a-z0-9-]+)")
    def delete_voice(self, q, body, slug): voices.delete_voice(self.app.config, self.app.db, slug); return {"ok": True}

    @route("GET", "/api/characters")
    def characters(self, q, body): return self.app.characters()

    @route("POST", "/api/characters")
    def add_character(self, q, body): return {"name": self.app.add_character(body.get("name", ""), body.get("data", ""), bool(body.get("consent")))}

    @route("GET", "/api/models")
    def models(self, q, body): return self.app.models_overview()

    @route("POST", "/api/models")
    def create_model(self, q, body): return self.app.create_model(body.get("name", ""))

    @route("POST", "/api/models/import")
    def models_import(self, q, body): return {"job": self.app.jobs.submit("looks", "Looks importieren (alle AI-Modelle)", {})}

    @route("GET", "/api/fusion/candidates")
    def fusion_candidates(self, q, body): return self.app.fusion_candidates()

    @route("POST", r"/api/models/([a-z0-9-]+)/fusion")
    def model_fusion(self, q, body, slug): return {"job": self.app.start_fusion(slug, body)}

    @route("GET", r"/api/models/([a-z0-9-]+)/style")
    def model_style(self, q, body, slug):
        from .profile import get_fusion
        return get_fusion(self.app.db, slug) or {}

    @route("POST", r"/api/models/([a-z0-9-]+)/voice-blend")
    def model_voice_blend(self, q, body, slug): return {"job": self.app.start_voice_blend(slug, body)}

    @route("POST", r"/api/models/([a-z0-9-]+)/import")
    def model_import(self, q, body, slug): return {"job": self.app.jobs.submit("looks", f"Looks importieren: {slug}", {"slug": slug})}

    @route("POST", r"/api/looks/(\d+)")
    def look_update(self, q, body, lid): return self.app.update_look(int(lid), body)

    @route("DELETE", r"/api/looks/(\d+)")
    def look_delete(self, q, body, lid):
        from . import looks as looks_mod
        looks_mod.delete_look(self.app.config, self.app.db, int(lid)); return {"ok": True}

    @route("GET", "/api/personas")
    def personas(self, q, body): return self.app.personas()

    @route("POST", "/api/personas")
    def save_persona(self, q, body): return self.app.save_persona(body)

    @route("POST", "/api/personas/generate")
    def generate_persona(self, q, body): return self.app.generate_persona(body)

    @route("POST", r"/api/personas/([a-z0-9-]+)/rename")
    def rename_persona(self, q, body, slug): return self.app.rename_persona(slug, body.get("name", ""))

    @route("POST", r"/api/personas/([a-z0-9-]+)/portrait")
    def portrait(self, q, body, slug): return {"job": self.app.start_portrait(slug, body)}

    @route("DELETE", r"/api/personas/([a-z0-9-]+)")
    def delete_persona(self, q, body, slug): self.app.delete_persona(slug); return {"ok": True}

    # Sessions -------------------------------------------------------------------------------------
    @route("GET", "/api/sessions")
    def sessions(self, q, body): return self.app.sessions()

    @route("POST", "/api/sessions")
    def create_session(self, q, body): return self.app.create_session(body)

    @route("GET", r"/api/sessions/(\d+)")
    def session(self, q, body, sid): return self.app.session(int(sid))

    @route("DELETE", r"/api/sessions/(\d+)")
    def delete_session(self, q, body, sid): self.app.delete_session(int(sid)); return {"ok": True}

    @route("POST", r"/api/sessions/(\d+)/replan")
    def replan(self, q, body, sid): return self.app.replan(int(sid), body.get("seed"))

    @route("POST", r"/api/sessions/(\d+)/audio")
    def render_audio(self, q, body, sid): return {"job": self.app.jobs.submit("audio", f"Audio: Session {sid}", {"session_id": int(sid)})}

    @route("POST", r"/api/sessions/(\d+)/video")
    def render_video(self, q, body, sid):
        return {"job": self.app.jobs.submit("video", f"Video: Session {sid} ({body.get('mode', 'ambient')})", {"session_id": int(sid), **body})}

    # Jobs -----------------------------------------------------------------------------------------
    @route("GET", "/api/jobs")
    def jobs(self, q, body):
        limit = max(1, min(int((q.get("limit") or ["60"])[0]), 1000))
        return [{k: v for k, v in row.items() if k != "log"} for row in self.app.db.query("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]

    @route("GET", r"/api/jobs/(\d+)")
    def job(self, q, body, jid): return self.app.db.one("SELECT * FROM jobs WHERE id=?", (int(jid),)) or (404, {"error": "nicht gefunden"})

    @route("DELETE", r"/api/jobs/(\d+)")
    def delete_job(self, q, body, jid): return {"deleted": self.app.jobs.delete(job_id=int(jid))}

    @route("POST", "/api/jobs/diagnose")
    def diagnose_jobs(self, q, body):
        """Nur fehlgeschlagene Jobs (Status 'error'): gruppiert, mit Ursache/Gegenmassnahme; mit {llm:true, group:id} zusaetzlich LLM-Erklaerung."""
        from . import diagnose
        groups = diagnose.analyze(self.app.db)
        if body.get("llm"):
            groups = [g for g in groups if g["id"] == body.get("group")][:1]
            for g in groups: g["explanation"] = diagnose.explain(self.app.config, g)
        return {"groups": groups}

    @route("POST", "/api/jobs/clear")
    def clear_jobs(self, q, body): return {"deleted": self.app.jobs.delete(statuses=body.get("statuses"))}

    @route("POST", r"/api/jobs/(\d+)/prioritize")
    def prioritize_job(self, q, body, jid): self.app.jobs.prioritize(int(jid), int(body.get("priority", 20))); return {"ok": True}

    @route("POST", "/api/jobs/retry-failed")
    def retry_failed_jobs(self, q, body): return {"jobs": self.app.jobs.retry_failed()}

    @route("POST", r"/api/jobs/(\d+)/retry")
    def retry_job(self, q, body, jid): return {"job": self.app.jobs.retry(int(jid))}

    @route("POST", "/api/restart")
    def restart(self, q, body):
        """Beendet den Server mit Code 75; start.py (im Terminal) startet dann sich selbst und alles neu - Code-Aenderungen gelten sofort."""
        if os.environ.get("ASMRAI_START_CHILD") != "1":
            return 409, {"error": "Nur moeglich, wenn ASMRAI ueber start.py gestartet wurde."}
        threading.Timer(0.5, lambda: os._exit(75)).start()
        return {"ok": True}

    @route("POST", r"/api/jobs/(\d+)/cancel")
    def cancel(self, q, body, jid): return {"ok": self.app.jobs.cancel(int(jid))}


def make_handler(app: App, host: str, port: int):
    api = Api(app)
    allowed_hosts = {f"{host}:{port}", f"localhost:{port}", f"127.0.0.1:{port}"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ASMRAI"

        def log_message(self, fmt, *args): pass

        def _send(self, status: int, payload: bytes, content_type: str, extra: dict | None = None) -> None:
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items(): self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD": self.wfile.write(payload)

        def _json(self, status: int, data) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _guard(self) -> bool:
            """Schuetzt gegen DNS-Rebinding und fremde Webseiten (CSRF) bei lokalem Betrieb."""
            if self.headers.get("Host", "") not in allowed_hosts: self._json(403, {"error": "Host nicht erlaubt"}); return False
            origin = self.headers.get("Origin")
            if self.command != "GET" and origin and urlparse(origin).netloc not in allowed_hosts: self._json(403, {"error": "Origin nicht erlaubt"}); return False
            return True

        _static_cache: dict = {}

        def _cached(self, path: Path) -> None:
            """Statische WebUI-Dateien kommen aus dem Arbeitsspeicher (Neuladen alle 10 s): Das Projekt liegt evtl. auf einem langsamen Netzlaufwerk."""
            now = time.monotonic(); hit = self._static_cache.get(path)
            if not hit or now - hit[0] > 10:
                try: hit = (now, path.read_bytes())
                except OSError: return self._json(404, {"error": "Datei nicht gefunden"})
                self._static_cache[path] = hit
            self._send(200, hit[1], mimetypes.guess_type(path.name)[0] or "application/octet-stream")

        def _media(self, path: Path) -> None:
            if not path.is_file(): self._json(404, {"error": "Datei nicht gefunden"}); return
            size = path.stat().st_size; start, end, status = 0, size - 1, 200
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if match and (match.group(1) or match.group(2)):
                if match.group(1): start = int(match.group(1))
                if match.group(2): end = min(int(match.group(2)), size - 1)
                if not match.group(1): start = max(0, size - int(match.group(2))); end = size - 1
                if start >= size or start > end:
                    self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"}); return
                status = 206
            length = end - start + 1
            self.send_response(status); self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(length))
            if status == 206: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD": return
            try:
                with open(path, "rb") as handle:
                    handle.seek(start); remaining = length
                    while remaining > 0:
                        chunk = handle.read(min(1 << 20, remaining))
                        if not chunk: break
                        self.wfile.write(chunk); remaining -= len(chunk)
            except (ConnectionError, OSError): pass

        def _dispatch(self) -> None:
            try: self._dispatch_inner()
            except (ConnectionError, BrokenPipeError, TimeoutError): pass     # Browser hat die Verbindung geschlossen (Tab/Seitenwechsel)

        def _dispatch_inner(self) -> None:
            if not self._guard(): return
            parsed = urlparse(self.path); path = parsed.path
            if self.command in ("GET", "HEAD") and not path.startswith("/api/"):
                return self._static(path)
            body = {}
            try: length = int(self.headers.get("Content-Length") or 0)
            except ValueError: return self._json(400, {"error": "Ungueltige Content-Length"})
            if length < 0 or length > MAX_JSON_BODY:
                return self._json(413, {"error": f"Anfrage zu gross (maximal {MAX_JSON_BODY // 1024 // 1024} MB)"})
            if length:
                try: body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError: return self._json(400, {"error": "Ungueltiges JSON"})
            method = "GET" if self.command == "HEAD" else self.command
            for route_method, pattern, name in ROUTES:
                match = pattern.match(path)
                if match and route_method == method:
                    try:
                        result = getattr(api, name)(parse_qs(parsed.query), body, *match.groups())
                        if isinstance(result, tuple): return self._json(result[0], result[1])
                        return self._json(200, result)
                    except (ValueError, KeyError, FileNotFoundError) as error: return self._json(400, {"error": str(error.args[0] if error.args else error)})
                    except ConnectionError: raise
                    except Exception as error:  # noqa: BLE001
                        log.error("%s %s -> %s: %s", self.command, path, type(error).__name__, error, exc_info=True); return self._json(500, {"error": f"{type(error).__name__}: {error}"})
            self._json(404, {"error": "Unbekannter Endpunkt"})

        def _static(self, path: str) -> None:
            config = app.config
            if (m := re.match(r"^/media/thumb/(\d+)/(\d)\.jpg$", path)): return self._media(config.work / m.group(1) / "thumbs" / f"{m.group(2)}.jpg")
            if (m := re.match(r"^/media/session/(\d+)/(audio|video)(?:\.\w+)?$", path)):
                row = app.db.one("SELECT audio_path, video_path FROM sessions WHERE id=?", (int(m.group(1)),))
                target = row and row[f"{m.group(2)}_path"]
                return self._media(Path(target)) if target else self._json(404, {"error": "noch nicht gerendert"})
            if (m := re.match(r"^/media/character/([A-Za-z0-9._-]+\.(?:jpg|jpeg|png))$", path)): return self._media(config.characters / m.group(1))
            if (m := re.match(r"^/media/look/(\d+)/(full|frame)\.jpg$", path)):
                from . import looks as looks_mod
                row = app.db.one("SELECT * FROM looks WHERE id=?", (int(m.group(1)),))
                return self._media(looks_mod.frame_path(config, row) if m.group(2) == "frame" else looks_mod.full_path(config, row)) if row else self._json(404, {"error": "Look nicht gefunden"})
            if (m := re.match(r"^/media/trigger-demo/(c\d+|f[a-z0-9-]+)/([a-z0-9_-]+)\.wav$", path)):
                from . import triggers as trig
                from .profile import get_fusion, get_profile
                who = m.group(1); profile = get_fusion(app.db, who[1:]) if who[0] == "f" else get_profile(app.db, int(who[1:]))
                entry = next((t for t in (profile or {}).get("trigger_library", []) if t["name"] == m.group(2)), None)
                if not entry: return self._json(404, {"error": "Trigger nicht gefunden"})
                return self._send(200, trig.demo_wav(entry["recipe"], 6.0, 1), "audio/wav")
            if (m := re.match(r"^/media/trigger-sample/(\d+)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\.wav$", path)):
                from . import triggers as trig
                video = app.db.one("SELECT path FROM videos WHERE id=?", (int(m.group(1)),))
                data = trig.sample_wav(config, video["path"], float(m.group(2)), float(m.group(3))) if video else None
                return self._send(200, data, "audio/wav") if data else self._json(404, {"error": "Ausschnitt nicht verfuegbar"})
            if (m := re.match(r"^/media/blend/([a-z0-9-]+)/([a-z0-9-]+)\.wav$", path)):
                from .synth import blend as blend_mod
                return self._media(blend_mod.blend_dir(config, m.group(1), m.group(2)) / "preview.wav")
            if (m := re.match(r"^/media/person/([a-z0-9-]+)/image\.jpg$", path)):
                image = app.person_image(m.group(1))
                return self._media(image) if image else self._json(404, {"error": "kein Bild"})
            if path in ("/", "/index.html"): return self._cached(STATIC / "index.html")
            file = (STATIC / path.lstrip("/")).resolve()
            if STATIC.resolve() in file.parents and file.is_file(): return self._cached(file)
            self._json(404, {"error": "Nicht gefunden"})

        do_GET = do_POST = do_DELETE = do_HEAD = _dispatch

    return Handler


def serve(app: App, host: str | None = None, port: int | None = None) -> None:
    host = host or app.config.get("web", "host", "127.0.0.1"); port = int(port or app.config.get("web", "port", 8780))
    from .watcher import Watcher
    app.watcher = Watcher(app); app.watcher.start()
    from .monitor import Monitor
    interval = float(app.config.get("monitor", "interval_seconds", 5))
    Monitor(app, f"http://{host}:{port}/", app.config.get("comfyui", "api_url", "http://127.0.0.1:8188").rstrip("/") + "/", interval,
            float(app.config.get("monitor", "idle_seconds", 30))).start()
    class QuietServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)): return      # Browser hat die Verbindung geschlossen
            super().handle_error(request, client_address)

    server = QuietServer((host, port), make_handler(app, host, port)); server.daemon_threads = True
    print(f"ASMRAI WebUI: http://{host}:{port}", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nBeendet.")
