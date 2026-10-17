import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("start_script", ROOT / "start.py"); start = importlib.util.module_from_spec(spec); spec.loader.exec_module(start)


def test_starter_is_stdlib_only():
    source = (ROOT / "start.py").read_text(encoding="utf-8")
    assert "import yaml" not in source.replace('"yaml"', "")          # sonst scheitert der Start mit einem Python ohne PyYAML (Bug mit Python 3.13)


def test_web_address_read_without_yaml():
    assert start.web_address() == "http://127.0.0.1:8780"


def test_find_python_checks_modules_and_order(tmp_path):
    assert start.has_modules(sys.executable, ("json", "sys")) and not start.has_modules(sys.executable, ("definitely_not_a_module_xyz",))
    names = [p.parts[-3] if p.parts[-3] in (".venv",) else p.parts[-4] for p in start.candidates()[:2]]
    assert start.candidates()[0].parts[-3] == ".venv" and "SeriesAI" in start.candidates()[1].parts


def test_requirements_cover_required_modules():
    core = (ROOT / "requirements.txt").read_text().lower(); full = (ROOT / "requirements-full.txt").read_text().lower()
    for package in ("numpy", "scipy", "soundfile", "pyyaml", "opencv-python"): assert package in core
    for package in ("torch", "faster-whisper", "librosa", "coqui-tts", "speechbrain"): assert package in full


def test_config_value_parsing():
    assert start.config_value("comfyui", "autostart", "x") == "true"
    assert start.config_value("comfyui", "api_url", "") == "http://127.0.0.1:8188"
    assert start.config_value("comfyui", "reserve_vram", "") == "1" and start.config_value("comfyui", "missing", "dflt") == "dflt"
    assert start.config_value("web", "port", "") == "8780" and start.config_value("nosection", "k", "d") == "d"


def test_comfyui_not_started_when_already_running(monkeypatch, capsys):
    monkeypatch.setattr(start, "comfyui_running", lambda timeout=3.0: True)
    spawned = []; monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: spawned.append(a) or None)
    assert start.ensure_comfyui() is None and not spawned and "laeuft bereits" in capsys.readouterr().out


def test_comfyui_started_when_not_running_and_stopped_by_us(monkeypatch, tmp_path):
    monkeypatch.setattr(start, "comfyui_running", lambda timeout=3.0: False)
    monkeypatch.setattr(start, "comfyui_command", lambda: ([sys.executable, "-c", "import time; print('hi'); time.sleep(30)"], tmp_path))
    child = start.ensure_comfyui(); assert child is not None and child.poll() is None
    start.stop_child(child); assert child.poll() is not None
    start.stop_child(None)                                                     # kein Fehler, wenn nichts gestartet wurde


def test_comfyui_command_uses_portable_install_without_browser():
    found = start.comfyui_command()
    if found is None: return                                                   # Rechner ohne ComfyUI-Nachbarordner
    command, cwd = found
    assert command[1:4] == ["-s", "ComfyUI/main.py", "--windows-standalone-build"] and "--disable-auto-launch" in command and (cwd / "ComfyUI" / "main.py").exists()
