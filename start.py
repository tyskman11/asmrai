"""ASMRAI starten - legt bei Bedarf selbst eine virtuelle Umgebung an und installiert die Abhaengigkeiten.

    python start.py              # WebUI starten (Adresse wird im Terminal angezeigt, es wird nie ein Tab geoeffnet)
    python start.py doctor       # Umgebung pruefen
    python start.py setup        # nur Umgebung anlegen/aktualisieren (--core: ohne KI-Pakete, --yes: ohne Rueckfrage)
    python start.py comfyui      # nur ComfyUI starten (falls es nicht schon laeuft) und im Vordergrund ueberwachen
    python start.py <befehl>     # beliebiger CLI-Befehl (analyze, session, ...)

Dieses Skript nutzt ausschliesslich die Standardbibliothek, laeuft also mit jedem Python 3.10+.
Reihenfolge der Interpreter: ./.venv, ../SeriesAI/.venv (teilt dessen Pakete) - nie ein globaler Python.
Gibt es keinen vollstaendigen, wird ./.venv ohne Rueckfrage erzeugt und aus requirements.txt (+ requirements-full.txt) befuellt;
ffmpeg wird bei Bedarf nach ./runtime/ffmpeg geladen. Caches/Temp liegen in ./runtime.
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Keine globalen Verzeichnisse: Caches und Temp-Dateien liegen im Projekt (werden an alle Unterprozesse vererbt).
_LOCAL = ROOT / "runtime" / "cache"
for _name, _sub in (("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"), ("TTS_HOME", "tts"), ("XDG_CACHE_HOME", "xdg"),
                    ("PIP_CACHE_DIR", "pip"), ("TMP", "tmp"), ("TEMP", "tmp"), ("TMPDIR", "tmp")):
    (_LOCAL / _sub).mkdir(parents=True, exist_ok=True)
    os.environ[_name] = str(_LOCAL / _sub)


# ---- portabler Python (python-build-standalone): liegt im Projekt bzw. im gemeinsamen runtime-Ordner, nie in einer globalen Installation ----
PORTABLE_PY_MINOR = "3.11"      # 3.10-3.12 haben fuer alle KI-Pakete (torch, coqui-tts ...) Wheels


def portable_python() -> Path | None:
    exe = "python.exe" if os.name == "nt" else "bin/python3"
    for base in (ROOT / "runtime" / "python", ROOT.parent.parent / "runtime" / "python"):   # projekteigen, sonst gemeinsam im Pool-Ordner
        if (base / exe).exists() and (base / ".install_complete").exists(): return base / exe   # Marker: nur vollstaendig entpackte Installationen
    return None


def ensure_portable_python() -> Path | None:
    """Gibt den portablen Python zurueck; fehlt er, wird er einmalig (mit SHA-256-Pruefung) geladen. None = offline/fehlgeschlagen."""
    import hashlib, json, tarfile, urllib.request
    found = portable_python()
    if found: return found
    platform_tag = "x86_64-pc-windows-msvc" if os.name == "nt" else ("aarch64-apple-darwin" if sys.platform == "darwin" else "x86_64-unknown-linux-gnu")
    target = ROOT.parent.parent / "runtime" / "python" if ROOT.parent.name == "projects" else ROOT / "runtime" / "python"
    try:
        request = urllib.request.Request("https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest", headers={"User-Agent": "portable-python"})
        with urllib.request.urlopen(request, timeout=30) as response: release = json.load(response)
        pattern = re.compile(rf"^cpython-{re.escape(PORTABLE_PY_MINOR)}\.\d+\+\d+-{platform_tag}-install_only\.tar\.gz$")
        asset = next(a for a in release["assets"] if pattern.match(a["name"]))
        digest = str(asset.get("digest", ""))
        if not digest.startswith("sha256:"): raise RuntimeError("Release ohne Pruefsumme")
        archive = ROOT / "runtime" / "downloads" / asset["name"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Python] Lade portables Python {PORTABLE_PY_MINOR} ({asset['size'] >> 20} MB) nach {target} ...", flush=True)
        sha = hashlib.sha256()
        with urllib.request.urlopen(asset["browser_download_url"], timeout=60) as response, archive.open("wb") as out:
            while chunk := response.read(1 << 20): out.write(chunk); sha.update(chunk)
        if sha.hexdigest() != digest[7:]: archive.unlink(missing_ok=True); raise RuntimeError("Pruefsumme stimmt nicht")
        staging = target.with_name("python.staging")
        shutil.rmtree(staging, ignore_errors=True)
        skipped = 0
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            for number, member in enumerate(members, 1):      # einzeln entpacken: auf Netzlaufwerken scheitert gelegentlich eine Datei (z. B. leere .lock-Dateien)
                if number % 250 == 0: print(f"[Python] Entpacke ... {number}/{len(members)} Dateien", flush=True)
                for attempt in range(3):
                    try:
                        bundle.extract(member, staging, filter="data") if hasattr(tarfile, "data_filter") else bundle.extract(member, staging)
                        break
                    except PermissionError:
                        if attempt == 2:
                            if not member.name.endswith(".lock"): skipped += 1
                            break
                        time.sleep(1)
        if skipped or not (staging / "python" / ("python.exe" if os.name == "nt" else "bin/python3")).exists():
            raise RuntimeError(f"Entpacken unvollstaendig ({skipped} Dateien nicht schreibbar; laeuft ein Virenscanner/anderer Prozess auf dem Ziel?)")
        archive.unlink(missing_ok=True)
        shutil.rmtree(target, ignore_errors=True)
        for attempt in range(5):   # Netzlaufwerke/Virenscanner halten frisch entpackte Ordner kurz fest
            try: (staging / "python").replace(target); break
            except PermissionError:
                if attempt == 4: shutil.move(str(staging / "python"), str(target))
                else: time.sleep(2)
        shutil.rmtree(staging, ignore_errors=True)
        if not portable_python():
            shutil.rmtree(target, ignore_errors=True); raise RuntimeError("python nach dem Verschieben nicht vorhanden")
        (target / ".install_complete").write_text("ok", encoding="utf-8")
        if not portable_python():
            shutil.rmtree(target, ignore_errors=True); raise RuntimeError("Marker konnte nicht geschrieben werden")
        print("[Python] Portabler Python bereit.", flush=True)
    except Exception as error:  # noqa: BLE001 - offline o. ae.: Aufrufer weicht auf den startenden Python aus
        print(f"[Python] Portabler Python nicht verfuegbar ({type(error).__name__}: {error}); verwende den startenden Python.", flush=True)
        return None
    return portable_python()
EXE = "Scripts/python.exe" if os.name == "nt" else "bin/python"
CORE = ("numpy", "scipy", "soundfile", "yaml", "cv2")                                   # Basis: Analyse-Pipeline, WebUI, Klang-Synthese
FULL = CORE + ("librosa", "faster_whisper", "TTS", "torch", "speechbrain")              # KI: Transkription, Stimme, Sprecher-Erkennung
ANSI = re.compile(r"\[[0-9;]*m")                                                           # Farbcodes aus der ComfyUI-Ausgabe entfernen
SUPPORTED = ((3, 10), (3, 11), (3, 12))                                                 # Versionen, fuer die alle KI-Pakete Wheels haben


def has_modules(python: str, modules: tuple[str, ...]) -> bool:
    code = "import importlib.util,sys;sys.exit(any(importlib.util.find_spec(m) is None for m in %r))" % (list(modules),)
    return subprocess.run([python, "-c", code], capture_output=True).returncode == 0


def candidates() -> list[Path]:
    return [ROOT / ".venv" / EXE, (ROOT / ".." / "SeriesAI" / ".venv" / EXE).resolve()]      # nie ein globaler Python mit globalen Paketen


def find_python(modules: tuple[str, ...] = CORE) -> str | None:
    for candidate in candidates():
        if candidate.exists() and has_modules(str(candidate), modules): return str(candidate)
    return None


def base_pythons() -> list[list[str]]:
    """Interpreter, aus denen ein venv gebaut werden kann: zuerst der portable Python im Projekt; global installierte nur als letzte Notloesung."""
    found: list[list[str]] = []
    portable = ensure_portable_python()
    if portable: found.append([str(portable)])
    else: found.append([sys.executable])                                                # offline: der startende Python
    return found


def working_base() -> list[str]:
    for cmd in base_pythons():
        try:
            out = subprocess.run([*cmd, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True, timeout=30)
            if out.returncode == 0 and out.stdout.strip(): return cmd
        except (OSError, subprocess.SubprocessError): continue
    raise SystemExit("Kein Python verfuegbar: der portable Python konnte nicht geladen werden (Internet?) und der startende Python ist unbrauchbar.")


def setup(core_only: bool = False, assume_yes: bool = False) -> str:
    venv = ROOT / ".venv"; python = venv / EXE
    base = working_base()
    version = subprocess.run([*base, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True).stdout.strip()
    supported = tuple(int(x) for x in version.split(".")) in SUPPORTED
    size = "ca. 300 MB" if core_only else "ca. 5-6 GB (inkl. PyTorch mit CUDA, Whisper, XTTS)"
    print(f"\nEs fehlt eine vollstaendige Python-Umgebung. ASMRAI legt jetzt {venv} an (Python {version}) und installiert Pakete: {size}.")
    if not supported and not core_only:
        print(f"WARNUNG: Python {version} wird von den KI-Paketen (coqui-tts/torch) nicht durchgehend unterstuetzt. Empfohlen: Python 3.10-3.12.")
    if not python.exists(): subprocess.check_call([*base, "-m", "venv", str(venv)])
    run = lambda *args: subprocess.check_call([str(python), "-m", "pip", *args])
    run("install", "--upgrade", "pip")
    run("install", "-r", str(ROOT / "requirements.txt"))
    if not core_only:
        run("install", "-r", str(ROOT / "requirements-full.txt"))
    print("Umgebung fertig.\n")
    return str(python)


FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"      # enthaelt ffmpeg und ffprobe


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds); h, rest = divmod(seconds, 3600); m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def download(url: str, dest: Path) -> None:
    """Download mit Fortschritt, Geschwindigkeit und ETA (Ausgabe etwa alle 2 s)."""
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0)); done = 0; started = last = time.time()
        while True:
            chunk = response.read(1024 * 256)
            if not chunk: break
            out.write(chunk); done += len(chunk)
            if time.time() - last >= 2:
                speed = done / max(time.time() - started, 1e-6); last = time.time()
                eta = fmt_duration((total - done) / speed) if total and speed > 0 else "?"
                share = f"{done / total * 100:.0f} %" if total else "?"
                print(f"  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB ({share})  {speed / 1e6:.1f} MB/s  ETA {eta}", flush=True)


def ensure_ffmpeg() -> None:
    """ffmpeg/ffprobe muessen im Projekt liegen (runtime/ffmpeg/bin) - fehlen sie dort und im PATH, werden sie (Windows) geladen."""
    target = ROOT / "runtime" / "ffmpeg"; exe = target / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if exe.exists(): return
    if os.name != "nt":
        if shutil.which("ffmpeg"): return
        raise SystemExit("ffmpeg fehlt. Bitte ffmpeg installieren oder nach runtime/ffmpeg/bin legen.")
    print(f"ffmpeg wird nach {target} geladen ({FFMPEG_URL}) ...", flush=True)
    archive = _LOCAL / "tmp" / "ffmpeg.zip"
    download(FFMPEG_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        top = zf.namelist()[0].split("/")[0]
        zf.extractall(_LOCAL / "tmp")
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(_LOCAL / "tmp" / top), str(target))
    archive.unlink(missing_ok=True)
    print("ffmpeg bereit.\n", flush=True)


def web_address() -> str:
    text = (ROOT / "config.yaml").read_text(encoding="utf-8") if (ROOT / "config.yaml").exists() else ""
    host = re.search(r'^\s*host:\s*"?([\w.\-]+)"?', text, re.M); port = re.search(r"^\s*port:\s*(\d+)", text, re.M)
    return f"http://{host.group(1) if host else '127.0.0.1'}:{port.group(1) if port else '8780'}"


def config_value(section: str, key: str, default: str) -> str:
    """Einfacher Zugriff auf 'section: / key: wert' in config.yaml ohne PyYAML (der Starter nutzt nur die Standardbibliothek)."""
    path = ROOT / "config.yaml"
    block = re.search(rf"(?ms)^{section}:[^\n]*\n(.*?)(?=^\S|\Z)", path.read_text(encoding="utf-8")) if path.exists() else None
    value = re.search(rf"^\s+{key}:\s*[\"']?([^\"'#\r\n]*?)[\"']?\s*(?:#.*)?$", block.group(1), re.M) if block else None
    return value.group(1) if value and value.group(1) != "" else default


def comfyui_url() -> str:
    return config_value("comfyui", "api_url", "http://127.0.0.1:8188").rstrip("/")


def comfyui_running(timeout: float = 5.0) -> bool:
    """Laeuft ComfyUI? Ein offener Port zaehlt auch, wenn die API unter Last gerade langsam antwortet."""
    try:
        with urllib.request.urlopen(comfyui_url() + "/system_stats", timeout=timeout): return True
    except Exception:  # noqa: BLE001
        parts = urllib.parse.urlsplit(comfyui_url())
        try:
            with socket.create_connection((parts.hostname or "127.0.0.1", parts.port or 80), timeout=1.0): return True
        except OSError:
            return False


def comfyui_command() -> tuple[list[str], Path] | None:
    """(Kommando, Arbeitsordner) der portablen ComfyUI-Installation oder None, wenn sie nicht gefunden wird."""
    root = (ROOT / config_value("comfyui", "root", "../00 - DO NOT LIST/ComfyUI_windows_portable/ComfyUI")).resolve().parent     # Ordner mit python_embeded
    python = root / "python_embeded" / "python.exe"
    if not python.exists() or not (root / "ComfyUI" / "main.py").exists(): return None
    vram = config_value("comfyui", "vram_mode", "lowvram"); reserve = config_value("comfyui", "reserve_vram", "1")
    args = ["--windows-standalone-build", "--disable-auto-launch"]      # ohne Manager: der braucht pip/uv, das die portable Installation nicht hat (sonst Absturz beim Start)      # --disable-auto-launch: kein Browser-Fenster
    if vram in ("lowvram", "normalvram", "highvram"): args += [f"--{vram}", "--reserve-vram", reserve]
    return [str(python), "-s", "ComfyUI/main.py", *args], root


def comfyui_launching() -> bool:
    """Laeuft irgendwo schon ein ComfyUI-main.py (noch beim Hochfahren)? Dann nie ein zweites starten."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*ComfyUI*main.py*' })).Count"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out.isdigit() and int(out) > 0
    except (OSError, subprocess.SubprocessError):
        return False


def ensure_comfyui() -> subprocess.Popen | None:
    """Laeuft ComfyUI schon -> nichts tun. Sonst als Kindprozess starten (Ausgabe mit [ComfyUI] im selben Terminal)."""
    if config_value("comfyui", "autostart", "true").lower() not in ("true", "yes", "1"):
        return None
    if comfyui_running():
        print(f"[ComfyUI] laeuft bereits ({comfyui_url()}) - wird nicht erneut gestartet.", flush=True); return None
    if comfyui_launching():
        print("[ComfyUI] startet bereits (anderer Prozess, z. B. SeriesAI) - ASMRAI klinkt sich ein, sobald die API antwortet.", flush=True); return None
    found = comfyui_command()
    if found is None:
        print("[ComfyUI] portable Installation nicht gefunden (comfyui.root in config.yaml) - bitte manuell starten.", flush=True); return None
    command, cwd = found
    print(f"[ComfyUI] wird gestartet ({' '.join(command[3:])}) ...", flush=True)
    child = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def pump() -> None:
        for line in child.stdout: print(f"[ComfyUI] {ANSI.sub('', line.rstrip())}", flush=True)
    threading.Thread(target=pump, daemon=True).start()

    def announce() -> None:
        for _ in range(450):
            if child.poll() is not None: print(f"[ComfyUI] beendet (Code {child.returncode}) - Ausgabe oben pruefen.", flush=True); return
            if comfyui_running(1.5): print(f"[ComfyUI] bereit: {comfyui_url()}/", flush=True); return
            time.sleep(2)
        print("[ComfyUI] antwortet nach 15 Minuten noch nicht.", flush=True)
    threading.Thread(target=announce, daemon=True).start()
    return child


def stop_child(child: subprocess.Popen | None) -> None:
    """Beendet nur ein von uns gestartetes ComfyUI (ein schon vorher laufendes bleibt unberuehrt)."""
    if child is None or child.poll() is not None: return
    print("[ComfyUI] wird beendet ...", flush=True)
    child.terminate()
    try: child.wait(timeout=15)
    except subprocess.TimeoutExpired: child.kill()


def banner(comfy_child: subprocess.Popen | None) -> None:
    """Direkte Adressen im Terminal. Der Starter oeffnet nie selbst einen Browser- oder Terminal-Tab."""
    comfy = "gestartet (laedt noch)" if comfy_child is not None else ("laeuft" if comfyui_running(2) else "nicht erreichbar")
    lines = [f"ASMRAI WebUI : {web_address()}/", f"ComfyUI      : {comfyui_url()}/   ({comfy})", f"Logs         : {ROOT / 'logs'}", "Beenden      : Strg+C"]
    width = max(len(line) for line in lines) + 4
    print("\n" + "=" * width + "\n" + "\n".join(f"  {line}" for line in lines) + "\n" + "=" * width + "\n", flush=True)


def main() -> int:
    args = sys.argv[1:] or ["serve"]
    flags = {a for a in args if a in ("--core", "--yes", "--no-browser")}      # --no-browser bleibt als wirkungsloses Altflag akzeptiert
    args = [a for a in args if a not in flags]
    if args[0] == "setup":
        setup(core_only="--core" in flags, assume_yes="--yes" in flags); return 0
    ensure_ffmpeg()
    python = find_python()
    if python is None: python = setup(core_only="--core" in flags, assume_yes="--yes" in flags)
    elif args[0] != "doctor" and not has_modules(python, FULL):
        print("Hinweis: Nicht alle KI-Pakete sind in diesem Python installiert - 'python start.py doctor' zeigt, was fehlt. "
              "Mit 'python start.py setup' wird eine vollstaendige Umgebung angelegt.", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONWARNINGS": "ignore"}
    if args[0] == "comfyui":
        child = ensure_comfyui()
        try:
            while child is not None and child.poll() is None: time.sleep(1)
        except KeyboardInterrupt: pass
        finally: stop_child(child)
        return 0
    child = ensure_comfyui() if args[0] == "serve" else None
    holder = {"child": child}
    if args[0] == "serve" and config_value("comfyui", "autostart", "true").lower() in ("true", "yes", "1"):
        def watchdog() -> None:
            """Wie in SeriesAI: ComfyUI neu starten, wenn es weder antwortet noch gerade hochfaehrt (nie ein zweites parallel)."""
            while True:
                time.sleep(30)
                own = holder["child"]
                if own is not None and own.poll() is None: continue
                if comfyui_running(2): continue
                print("[ComfyUI] laeuft nicht - starte es.", flush=True)
                started = ensure_comfyui()
                if started is not None: holder["child"] = started
        threading.Thread(target=watchdog, daemon=True).start()
    if args[0] == "serve":
        banner(child)         # nur Adressen anzeigen - es wird nie selbst ein Browser-Tab geoeffnet
    try:
        return subprocess.call([python, "-m", "asmrai", *args], env={**env, "ASMRAI_START_CHILD": os.environ.get("ASMRAI_START_CHILD", "")}, cwd=ROOT)
    except KeyboardInterrupt:
        return 0
    finally:
        stop_child(holder["child"])


RESTART_CODE = 75   # Neustart-Button im Dashboard: start.py beendet sich mit diesem Code, der aeussere Teil startet neu


def supervise() -> int:
    """Aeusserer Teil im Terminal: startet start.py als Kindprozess und neu, wenn das Dashboard einen Neustart anfordert."""
    if os.environ.get("ASMRAI_START_CHILD") == "1": return main()
    env = {**os.environ, "ASMRAI_START_CHILD": "1"}
    while True:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
        try: code = child.wait()
        except KeyboardInterrupt:      # das Kind bekommt Strg+C ebenfalls und raeumt auf (ComfyUI beenden ...)
            try: return child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill(); return 130
        if code != RESTART_CODE: return code
        print("\n[Neustart] ASMRAI wird neu gestartet ...\n", flush=True)


if __name__ == "__main__":
    sys.exit(supervise())
