"""ComfyUI bei Bedarf starten: ist die API nicht erreichbar, wird die portable Installation (comfyui.root) gestartet und abgewartet."""
from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .config import Config

_child: subprocess.Popen | None = None


def local_env() -> dict[str, str]:
    """Caches und Temp-Dateien im Projekt (runtime/cache) statt im Benutzerprofil - ComfyUI laeuft dadurch auch portabel."""
    base = Path(__file__).resolve().parents[2] / "runtime" / "cache"; env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    for name, sub in (("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"), ("XDG_CACHE_HOME", "xdg"), ("PIP_CACHE_DIR", "pip"), ("TMP", "tmp"), ("TEMP", "tmp"), ("TMPDIR", "tmp")):
        (base / sub).mkdir(parents=True, exist_ok=True); env[name] = str(base / sub)
    return env


def port_open(url: str, timeout: float = 1.0) -> bool:
    """Nimmt ComfyUI Verbindungen an? (auch wenn es unter Last gerade zu langsam antwortet)"""
    parts = urlsplit(url)
    try:
        with socket.create_connection((parts.hostname or "127.0.0.1", parts.port or 80), timeout=timeout): return True
    except OSError:
        return False


def state(url: str, timeout: float = 5.0) -> str:
    """'laeuft' (API antwortet), 'antwortet langsam' (Port offen, Antwort kommt spaeter - z. B. beim Laden/Rendern) oder 'startet (lädt noch)' (Prozess laeuft, Port noch zu) oder 'nicht erreichbar'."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=timeout): return "laeuft"
    except Exception:  # noqa: BLE001
        if port_open(url): return "antwortet langsam"
        return "startet (lädt noch)" if launch_in_progress() else "nicht erreichbar"


def is_running(config: Config, timeout: float = 5.0) -> bool:
    """Laeuft ComfyUI? Ein offener Port zaehlt auch dann, wenn die API gerade langsam antwortet - dann nie ein zweites starten."""
    return state(config.get("comfyui", "api_url", "http://127.0.0.1:8188"), timeout) in ("laeuft", "antwortet langsam")


def launch_in_progress() -> bool:
    """True, solange irgendein ComfyUI-main.py-Prozess existiert (z. B. von SeriesAI gestartet und noch am Hochfahren) - dann nie ein zweites starten."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*ComfyUI*main.py*' })).Count"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out.isdigit() and int(out) > 0
    except (OSError, subprocess.SubprocessError):
        return False


def command(config: Config) -> tuple[list[str], Path] | None:
    """(Kommando, Arbeitsordner) der portablen Installation oder None, wenn sie nicht gefunden wird."""
    root = config.path("comfyui", "root", "../00 - DO NOT LIST/ComfyUI_windows_portable/ComfyUI").parent       # Ordner mit python_embeded
    python = root / "python_embeded" / "python.exe"
    if not python.exists() or not (root / "ComfyUI" / "main.py").exists(): return None
    vram = config.get("comfyui", "vram_mode", "lowvram"); reserve = str(config.get("comfyui", "reserve_vram", 1))
    args = ["--windows-standalone-build", "--disable-auto-launch"]      # ohne Manager: der braucht pip/uv, das die portable Installation nicht hat (sonst Absturz beim Start)
    if vram in ("lowvram", "normalvram", "highvram"): args += [f"--{vram}", "--reserve-vram", reserve]
    return [str(python), "-s", "ComfyUI/main.py", *args], root


def ensure_running(config: Config, log=print, wait_seconds: float = 900) -> None:
    """Startet ComfyUI, falls nicht erreichbar, und wartet bis die API antwortet. Fehler -> RuntimeError."""
    global _child
    if is_running(config): return
    if (_child is not None and _child.poll() is None) or launch_in_progress(): log("ComfyUI startet bereits - warte auf die API ...")
    else:
        found = command(config)
        if found is None:
            raise RuntimeError("ComfyUI nicht erreichbar und die portable Installation wurde nicht gefunden (comfyui.root in config.yaml).")
        cmd, cwd = found
        log(f"ComfyUI nicht erreichbar - starte {cwd} ...")
        logfile = config.path("paths", "data", "data").parent / "logs" / "comfyui.log"; logfile.parent.mkdir(parents=True, exist_ok=True)
        _child = subprocess.Popen(cmd, cwd=cwd, stdout=open(logfile, "ab"), stderr=subprocess.STDOUT, env=local_env(),
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _child is not None and _child.poll() is not None:
            raise RuntimeError(f"ComfyUI wurde beendet (Code {_child.returncode}) - siehe logs/comfyui.log.")
        if is_running(config, 2): log("ComfyUI bereit."); return
        time.sleep(3)
    raise RuntimeError(f"ComfyUI antwortet nach {int(wait_seconds)} s nicht - siehe logs/comfyui.log.")
