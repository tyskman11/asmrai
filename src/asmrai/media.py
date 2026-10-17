"""ffmpeg/ffprobe-Wrapper."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config
from .jobs import check_cancel

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Wie subprocess.run, aber abbrechbar: der Prozess wird alle 0,5 s gegen den Abbruch-Knopf des laufenden Jobs geprueft und dann beendet."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    waited = 0.0
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.5)
            return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            waited += 0.5
            try: check_cancel()
            except BaseException:
                process.kill(); process.communicate(); raise
            if timeout is not None and waited >= timeout:
                process.kill(); process.communicate(); raise subprocess.TimeoutExpired(cmd, timeout)


def probe(config: Config, path: Path) -> dict:
    result = run([config.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], 120)
    if result.returncode != 0: raise RuntimeError(f"ffprobe fehlgeschlagen: {result.stderr.decode(errors='replace')[-300:]}")
    data = json.loads(result.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    rate = video.get("avg_frame_rate", "0/1").split("/")
    fps = float(rate[0]) / float(rate[1]) if len(rate) == 2 and float(rate[1]) else 0.0
    return {"duration": float(data.get("format", {}).get("duration") or 0), "width": video.get("width"), "height": video.get("height"),
            "fps": round(fps, 2), "channels": audio.get("channels") if audio else None,
            "sample_rate": int(audio["sample_rate"]) if audio else None, "has_audio": audio is not None}


def stream_audio(config: Config, path: Path, sample_rate: int, chunk_seconds: float = 60.0) -> Iterator[np.ndarray]:
    """Dekodiert Audio als Stereo-float32 und liefert (n, 2)-Bloecke - ohne Zwischendatei."""
    cmd = [config.ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(sample_rate), "-f", "f32le", "-"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    block = int(chunk_seconds * sample_rate) * 8
    leftover = b""
    try:
        while True:
            check_cancel()
            data = process.stdout.read(block)
            if not data: break
            data = leftover + data
            usable = len(data) // 8 * 8
            leftover = data[usable:]
            if usable: yield np.frombuffer(data[:usable], dtype=np.float32).reshape(-1, 2)
    finally:
        process.stdout.close()
        if process.poll() is None: process.kill()
        process.wait()


def extract_frame(config: Config, path: Path, time_s: float, out: Path, width: int = 320) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [config.ffmpeg, "-v", "error", "-y", "-ss", f"{time_s:.2f}", "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(out)]
    for timeout in (240, 480):                      # Netzlaufwerk kann zeitweise sehr langsam sein: ein zweiter, laengerer Versuch
        try:
            result = run(cmd, timeout)
            return result.returncode == 0 and out.exists()
        except subprocess.TimeoutExpired:
            continue
    return False                                    # nur dieses Vorschaubild fehlt - die Stufe scheitert deswegen nicht


def scene_cuts(config: Config, path: Path, threshold: float = 0.35) -> list[float]:
    """Schnittzeitpunkte ueber ffmpeg-Szenenerkennung (niedrige Aufloesung fuer Tempo)."""
    cmd = [config.ffmpeg, "-v", "info", "-i", str(path), "-an", "-vf", f"fps=4,scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    result = run(cmd)
    cuts = []
    for line in result.stderr.decode(errors="replace").splitlines():
        if "pts_time:" in line:
            try: cuts.append(round(float(line.split("pts_time:")[1].split()[0]), 2))
            except ValueError: pass
    return cuts
