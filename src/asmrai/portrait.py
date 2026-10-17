"""Porträt der virtuellen Person: kurzer LTX-Clip (Text -> Video), daraus das schaerfste frontale Gesicht als Referenzbild."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import looks, video
from .config import Config
from .db import Database
from .jobs import JobContext
from .media import run

BUSY_KINDS = ("process", "video", "assets")
POLL_SECONDS = 10


def wait_for_gpu(app, ctx: JobContext, own_id: int) -> None:
    """Die GPU gehoert zuerst der Analyse/dem Video-Rendering; das Porträt wartet, bis nichts davon laeuft."""
    announced = False
    while True:
        ctx.check()
        busy = [j for kind in BUSY_KINDS for j in app.jobs.active(kind) if j["id"] != own_id]
        if not busy: return
        if not announced: ctx.log(f"Warte auf freie GPU ({len(busy)} Job(s) aktiv: {busy[0]['title']}) ..."); announced = True
        time.sleep(POLL_SECONDS)


def best_face(config: Config, clip: Path, out: Path) -> float:
    import cv2
    model = config.path("paths", "seriesai", "../SeriesAI") / "models" / "opencv" / "face_detection_yunet_2023mar.onnx"
    detector = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.8, 0.3, 5000)
    capture = cv2.VideoCapture(str(clip)); best, best_score = None, -1.0
    while True:
        ok, frame = capture.read()
        if not ok: break
        h, w = frame.shape[:2]; detector.setInputSize((w, h)); _, faces = detector.detect(frame)
        if faces is None or len(faces) != 1: continue
        x, y, fw, fh = faces[0][:4]; nose, re_, le = faces[0][8:10], faces[0][4:6], faces[0][6:8]
        if abs(nose[0] - (re_[0] + le[0]) / 2) / max(abs(re_[0] - le[0]), 1) > 0.3: continue
        cx, cy, half = x + fw / 2, y + fh / 2, max(fw, fh) * 1.4
        x0, y0, x1, y1 = int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, w)), int(min(cy + half, h))
        crop = frame[y0:y1, x0:x1]
        score = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) * float(faces[0][14])
        if score > best_score: best, best_score = crop, score
    capture.release()
    if best is None: raise RuntimeError("Im Porträt-Clip wurde kein frontales Einzelgesicht gefunden - Seed aendern und erneut versuchen.")
    cv2.imwrite(str(out), cv2.resize(best, (768, 768), interpolation=cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return best_score


def render_portrait(app, ctx: JobContext, params: dict) -> None:
    config, db = app.config, app.db
    persona = db.one("SELECT * FROM personas WHERE slug=?", (params["slug"],))
    if not persona: raise ValueError("Person nicht gefunden.")
    if db.one("SELECT 1 AS x FROM looks WHERE persona_slug=? AND has_face=1", (persona["slug"],)) and not params.get("force"):
        ctx.log("Diese Person hat importierte Looks (ai_models) - Porträt-Generierung wird uebersprungen."); return
    wait_for_gpu(app, ctx, ctx.job_id)
    if db.one("SELECT 1 AS x FROM looks WHERE persona_slug=? AND has_face=1", (persona["slug"],)) and not params.get("force"):    # waehrend des Wartens importiert
        ctx.log("Looks wurden inzwischen importiert - Porträt-Generierung entfaellt."); return
    seed = int(params.get("seed") or int(time.time()) % 100000)
    prompt = (f"close-up portrait video of {persona['appearance']}, looking straight at the camera, calm soft expression, slight natural movement, "
              f"{persona.get('setting') or 'cozy room'}, warm soft lighting, shallow depth of field, high detail face, cinematic")
    ctx.log(f"Porträt-Prompt: {prompt}")
    base = looks.person_dir(config, persona["slug"], create=True)
    clip = base / "_portrait_clip.mp4"
    cfg = Config({**config.raw, "comfyui": {**config.raw.get("comfyui", {}), "width": 512, "height": 704}})
    result = _clip(cfg, prompt, seed, clip, ctx, int(params.get("frames", 33)))      # wenige Frames genuegen fuer ein gutes Standbild
    out = base / "portrait.jpg"
    score = best_face(config, result, out)
    db.execute("UPDATE personas SET image=? WHERE slug=?", (out.name, persona["slug"]))
    ctx.log(f"Porträt gespeichert: {out.name} (Schaerfe-Score {score:.0f})")


def _clip(config: Config, prompt: str, seed: int, out: Path, ctx: JobContext, frames: int) -> Path:
    """Wie video.comfy_clip, aber mit kurzer Framezahl (Standbild statt 5-s-Clip)."""
    import json
    template = json.loads(config.path("comfyui", "workflow_template").read_text(encoding="utf-8"))
    spec = {"prompt": prompt, "negative_prompt": video.NEGATIVE, "seed": seed, "width": int(config.get("comfyui", "width", 512)), "height": int(config.get("comfyui", "height", 704)),
            "fps": 24, "frame_count": frames, "steps": int(config.get("comfyui", "steps", 32)), "cfg": float(config.get("comfyui", "cfg", 4.5)),
            "output_prefix": f"asmrai/{out.stem}", "duration_seconds": 2}
    return video.run_prompt(config, video._replace(template, spec), out, ctx)
