"""Video fuer eine Session: ``ambient`` (ffmpeg, audio-reaktiv, ohne GPU) oder ``comfyui`` (LTX-Clips, geloopt)."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from . import comfy, looks
from .config import Config
from .db import Database
from .generate import session_dir
from .jobs import JobContext
from .media import NO_WINDOW

PALETTES = [("0x14102b", "0x3a2a6b"), ("0x0f2230", "0x1f5f6b"), ("0x2a1020", "0x6b2a4f"), ("0x101a12", "0x2f5a3a")]
NEGATIVE = "blurry, low quality, text, subtitles, watermark, logo, deformed, extra fingers, flicker"


def _run_ffmpeg(config: Config, args: list[str], ctx: JobContext, duration: float) -> None:
    cmd = [config.ffmpeg, "-v", "error", "-y", "-progress", "pipe:1", "-nostats", *args]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=NO_WINDOW)
    for line in process.stdout:
        if line.startswith("out_time_ms="):
            try: ctx.progress(min(0.98, int(line.split("=")[1]) / 1e6 / max(duration, 1)))
            except ValueError: pass
        if ctx.cancel_event.is_set(): process.kill(); ctx.check()
    err = process.stderr.read()
    if process.wait() != 0: raise RuntimeError(f"ffmpeg fehlgeschlagen: {err[-400:]}")


def render_ambient(config: Config, session: dict, ctx: JobContext, size: tuple[int, int] = (960, 540), fps: int = 24) -> Path:
    audio = Path(session["audio_path"]); out = audio.with_name("video_ambient.mp4")
    duration = float(subprocess.run([config.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio)],
                                    capture_output=True, text=True, creationflags=NO_WINDOW).stdout.strip() or 0)
    c0, c1 = PALETTES[session["id"] % len(PALETTES)]; w, h = size
    graph = (f"gradients=s={w}x{h}:c0={c0}:c1={c1}:x0=0:y0=0:x1={w}:y1={h}:speed=0.008:rate={fps}:d={duration:.2f},format=yuv420p[bg];"
             f"[0:a]avectorscope=s=360x360:zoom=1.6:rc=140:gc=120:bc=230:rf=3:gf=3:bf=6:draw=line:scale=cbrt,format=yuva420p,colorchannelmixer=aa=0.8[vs];"
             f"[0:a]showfreqs=s={w}x120:mode=bar:ascale=cbrt:fscale=log:colors=white|0xb9a8ff:win_size=2048,format=yuva420p,colorchannelmixer=aa=0.5[fq];"
             f"[bg][vs]overlay=(W-w)/2:(H-h)/2-30[a];[a][fq]overlay=0:H-h-20[v]")
    _run_ffmpeg(config, ["-i", str(audio), "-filter_complex", graph, "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)], ctx, duration)
    return out


# -- ComfyUI (LTX) ---------------------------------------------------------------------------------

def _api(config: Config, endpoint: str, payload: dict | None = None, timeout: float = 30):
    url = config.get("comfyui", "api_url", "http://127.0.0.1:8188").rstrip("/") + endpoint
    request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response: return json.loads(response.read())


def _replace(value, spec):
    if isinstance(value, dict): return {k: _replace(v, spec) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list): return [_replace(v, spec) for v in value]
    if isinstance(value, str) and value.startswith("${seriesai.") and value.endswith("}"):
        key = value[11:-1]
        if key not in spec: raise ValueError(f"Workflow verlangt unbekanntes Feld: {key}")
        return spec[key]
    return value


def _upload(config: Config, image: Path) -> str:
    boundary = f"asmrai{uuid.uuid4().hex}"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{image.name}"\r\nContent-Type: image/jpeg\r\n\r\n').encode() + image.read_bytes() + \
           f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{boundary}--\r\n'.encode()
    url = config.get("comfyui", "api_url").rstrip("/") + "/upload/image"
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}), timeout=60) as response:
        return json.loads(response.read())["name"]


def comfy_clip(config: Config, prompt: str, seed: int, out: Path, reference: Path | None, ctx: JobContext) -> Path:
    template_key = "img2video_workflow_template" if reference else "workflow_template"
    template = json.loads(config.path("comfyui", template_key).read_text(encoding="utf-8"))
    width, height = int(config.get("comfyui", "width", 768)), int(config.get("comfyui", "height", 448))
    spec = {"prompt": prompt, "negative_prompt": NEGATIVE, "seed": seed, "width": width, "height": height, "fps": 24, "frame_count": 121, "steps": int(config.get("comfyui", "steps", 32)), "cfg": float(config.get("comfyui", "cfg", 4.5)),
            "output_prefix": f"asmrai/{out.stem}", "duration_seconds": 5}
    if reference: spec["image_filename"] = _upload(config, reference)
    return run_prompt(config, _replace(template, spec), out, ctx)


def run_prompt(config: Config, workflow: dict, out: Path, ctx: JobContext) -> Path:
    """Schickt einen fertigen ComfyUI-API-Workflow ab, wartet auf das Ergebnis und kopiert es nach ``out``."""
    queued = _api(config, "/prompt", {"prompt": workflow, "client_id": f"asmrai-{uuid.uuid4()}"})
    prompt_id = queued["prompt_id"]; deadline = time.monotonic() + float(config.get("comfyui", "timeout_seconds", 7200))
    while time.monotonic() < deadline:
        ctx.check()
        try: record = _api(config, f"/history/{prompt_id}", timeout=20).get(prompt_id)
        except Exception: record = None   # einzelne verlorene Abfrage bricht keinen laufenden Render ab
        if record:
            if record.get("status", {}).get("status_str") == "error": raise RuntimeError("ComfyUI-Fehler: " + json.dumps(record["status"].get("messages", []))[-500:])
            for node in record.get("outputs", {}).values():
                for key in ("gifs", "videos", "images"):
                    if isinstance(node, dict) and node.get(key):
                        asset = node[key][0]; src = config.path("comfyui", "root") / "output" / asset.get("subfolder", "") / asset["filename"]
                        out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, out); return out
        time.sleep(float(config.get("comfyui", "poll_seconds", 3)))
    raise TimeoutError("ComfyUI hat den Clip nicht rechtzeitig fertiggestellt")


SAME_PERSON = "the same person as in the reference image, same face, hairstyle and outfit"


def choose_reference(config: Config, db: Database | None, params: dict, rng, persona_slug: str | None, default_ref: Path | None) -> tuple[Path | None, str, str]:
    """Startbild je Clip: 1) ausdruecklich gewaehlte Gesichts-Referenz, 2) passender Look des AI-Modells (Style/Frisur), 3) Standardbild. -> (Bild, Subjekt, Info)"""
    face_ref = params.get("face_ref")
    if face_ref:
        path = config.characters / face_ref
        if path.is_file(): return path, SAME_PERSON, f"Gesichts-Referenz {face_ref}"
    if db and persona_slug:
        look = looks.pick_look(db, persona_slug, rng, params.get("hair"), params.get("style"))
        if look and looks.frame_path(config, look).exists():
            return looks.frame_path(config, look), SAME_PERSON, f"Look {look['file']} ({look['angle']}, Frisur: {look['hair'] or '?'}, Style: {look['style'] or '-'})"
    return default_ref, params.get("appearance", ""), "Standardbild"


def render_comfyui(config: Config, session: dict, params: dict, ctx: JobContext, db: Database | None = None) -> Path:
    comfy.ensure_running(config, ctx.log)
    audio = Path(session["audio_path"]); folder = session_dir(config, session["id"]); clips = []
    reference, persona_slug = None, params.get("persona")
    if params.get("character"):
        candidate = looks.persona_image_path(config, persona_slug, params["character"])
        if candidate: reference = candidate
    count = max(1, int(params.get("clips", 3))); theme = params.get("theme") or "calm cozy room"
    from . import vision
    from .profile import style_profile
    catalog = ((style_profile(db, session) or {}).get("visual_triggers") if db else None) or {}
    plan = json.loads(session["plan"] or "[]"); rng = random.Random(int(params.get("seed", 1)))
    kinds = [p["kind"] for p in plan if p["kind"] != "pause" for _ in range(max(1, int(p["duration"] // 5)))] or ["tapping"]
    ctx.log("Bild-Prompts aus dem visuellen Trigger-Katalog" if catalog else "Kein visueller Katalog vorhanden - allgemeine Prompts")
    for index in range(count):
        ctx.progress(0.9 * index / count, f"ComfyUI-Clip {index + 1}/{count} ...")
        ref, subject, info = choose_reference(config, db, params, rng, persona_slug, reference)
        ctx.log(f"Startbild: {info}")
        prompt = vision.build_prompt(rng.choice(kinds), catalog, rng, theme if params.get("theme") else params.get("setting", ""), subject)
        ctx.log(f"Prompt: {prompt}")
        clips.append(comfy_clip(config, prompt, int(params.get("seed", 1)) + index, folder / f"clip_{index}.mp4", ref, ctx))
    listing = folder / "clips.txt"
    pingpong = []
    for clip in clips:   # Vor-/Rueckwaerts-Schleife ergibt nahtlose, ruhige Loops
        out = clip.with_name(clip.stem + "_pp.mp4")
        subprocess.run([config.ffmpeg, "-v", "error", "-y", "-i", str(clip), "-filter_complex", "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[v]",
                        "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "20", str(out)], check=True, creationflags=NO_WINDOW); pingpong.append(out)
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in pingpong), encoding="utf-8")
    out = audio.with_name("video_comfyui.mp4")
    duration = float(subprocess.run([config.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio)],
                                    capture_output=True, text=True, creationflags=NO_WINDOW).stdout.strip() or 0)
    _run_ffmpeg(config, ["-stream_loop", "-1", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio), "-map", "0:v", "-map", "1:a",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.2f}",
                         "-movflags", "+faststart", str(out)], ctx, duration)
    return out


def render_video(config: Config, db: Database, session_id: int, ctx: JobContext, mode: str = "ambient", options: dict | None = None) -> None:
    session = db.one("SELECT * FROM sessions WHERE id=?", (session_id,))
    if not session or not session["audio_path"] or not Path(session["audio_path"]).exists(): raise ValueError("Zuerst das Audio rendern.")
    params = {**json.loads(session["params"]), **(options or {})}
    ctx.log(f"Video-Modus: {mode}")
    out = render_comfyui(config, session, params, ctx, db) if mode == "comfyui" else render_ambient(config, session, ctx)
    db.execute("UPDATE sessions SET status='video_ready', video_path=?, updated=? WHERE id=?", (str(out), time.time(), session_id))
    ctx.log(f"Video fertig: {out}")
