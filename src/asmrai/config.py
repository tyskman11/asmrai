from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str | None) -> Path | None:
    if value is None: return None
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or ROOT / "config.yaml"
        return cls((yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {})

    def get(self, section: str, key: str, default: Any = None) -> Any:
        value = (self.raw.get(section) or {}).get(key, default)
        return default if value is None and default is not None else value

    def path(self, section: str, key: str, default: str | None = None) -> Path | None:
        return _resolve(self.get(section, key, default))

    @property
    def data(self) -> Path:
        path = self.path("paths", "data", "data"); path.mkdir(parents=True, exist_ok=True); return path

    @property
    def db_path(self) -> Path: return self.data / "asmrai.db"
    @property
    def work(self) -> Path: return self._sub("work")
    @property
    def sessions(self) -> Path: return self._sub("sessions")
    @property
    def voices(self) -> Path: return self._sub("voices")
    @property
    def characters(self) -> Path: return self._sub("characters")

    def _sub(self, name: str) -> Path:
        path = self.data / name; path.mkdir(parents=True, exist_ok=True); return path

    @property
    def ffmpeg(self) -> str:
        local = ROOT / "runtime" / "ffmpeg" / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")      # vom Starter ins Projekt geladen
        if local.exists(): return str(local)
        configured = self.get("paths", "ffmpeg", "ffmpeg")
        return shutil.which(configured) or configured

    @property
    def ffprobe(self) -> str:
        ffmpeg = Path(self.ffmpeg)
        sibling = ffmpeg.with_name("ffprobe" + ffmpeg.suffix)
        return str(sibling) if sibling.exists() else (shutil.which("ffprobe") or "ffprobe")
