"""SQLite-Persistenz. Eine kurze Verbindung pro Aufruf, Schreibzugriffe serialisiert."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS creators(
  id INTEGER PRIMARY KEY, slug TEXT UNIQUE NOT NULL, name TEXT NOT NULL, source_dir TEXT NOT NULL,
  notes TEXT DEFAULT '', created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS videos(
  id INTEGER PRIMARY KEY, creator_id INTEGER NOT NULL REFERENCES creators(id) ON DELETE CASCADE,
  path TEXT NOT NULL, title TEXT NOT NULL, size INTEGER, duration REAL, width INTEGER, height INTEGER,
  fps REAL, channels INTEGER, sample_rate INTEGER, mtime REAL, error TEXT, UNIQUE(creator_id, path));
CREATE TABLE IF NOT EXISTS stages(
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE, stage TEXT NOT NULL,
  state TEXT NOT NULL, detail TEXT, finished REAL, PRIMARY KEY(video_id, stage));
CREATE TABLE IF NOT EXISTS segments(
  id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  start REAL NOT NULL, end REAL NOT NULL, label TEXT NOT NULL, features TEXT);
CREATE INDEX IF NOT EXISTS seg_video ON segments(video_id);
CREATE TABLE IF NOT EXISTS transcripts(
  video_id INTEGER PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE, language TEXT, words INTEGER, data TEXT);
CREATE TABLE IF NOT EXISTS profiles(
  creator_id INTEGER PRIMARY KEY REFERENCES creators(id) ON DELETE CASCADE, data TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS voices(
  id INTEGER PRIMARY KEY, slug TEXT UNIQUE NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL,
  owner TEXT, consent INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, creator_id INTEGER REFERENCES creators(id) ON DELETE SET NULL, name TEXT NOT NULL,
  params TEXT NOT NULL, plan TEXT, status TEXT NOT NULL, audio_path TEXT, video_path TEXT, error TEXT,
  created REAL NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS personas(
  id INTEGER PRIMARY KEY, slug TEXT UNIQUE NOT NULL, name TEXT NOT NULL, appearance TEXT DEFAULT '', image TEXT, voice TEXT,
  style_creator_id INTEGER REFERENCES creators(id) ON DELETE SET NULL, triggers TEXT, notes TEXT DEFAULT '', created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS trigger_events(
  id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE, start REAL NOT NULL, end REAL NOT NULL, label TEXT NOT NULL, desc TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS trig_video ON trigger_events(video_id);
CREATE TABLE IF NOT EXISTS styles(
  persona_slug TEXT PRIMARY KEY, weights TEXT NOT NULL, profile TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS looks(
  id INTEGER PRIMARY KEY, group_name TEXT NOT NULL, persona_slug TEXT, file TEXT NOT NULL, angle TEXT, hair TEXT, yaw REAL,
  has_face INTEGER NOT NULL DEFAULT 0, source TEXT UNIQUE, mtime REAL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, params TEXT, status TEXT NOT NULL,
  progress REAL DEFAULT 0, log TEXT DEFAULT '', created REAL NOT NULL, started REAL, finished REAL);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path, self._lock, self._conn = path, threading.RLock(), None
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            have = {row["name"] for row in conn.execute("PRAGMA table_info(creators)")}
            have_j = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "priority" not in have_j: conn.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0")
            have_l = {row["name"] for row in conn.execute("PRAGMA table_info(looks)")}
            if "style" not in have_l: conn.execute("ALTER TABLE looks ADD COLUMN style TEXT")
            have_p = {row["name"] for row in conn.execute("PRAGMA table_info(personas)")}
            for column, ddl in (("personality", "TEXT DEFAULT ''"), ("setting", "TEXT DEFAULT ''"), ("catchphrases", "TEXT"), ("backstory", "TEXT DEFAULT ''"),
                                ("voice_note", "TEXT DEFAULT ''"), ("age", "INTEGER"), ("voice_model", "TEXT"), ("look_model", "TEXT"), ("origin", "TEXT DEFAULT 'manual'")):
                if column not in have_p: conn.execute(f"ALTER TABLE personas ADD COLUMN {column} {ddl}")
            for column, ddl in (("consent_voice", "INTEGER NOT NULL DEFAULT 0"), ("consent_face", "INTEGER NOT NULL DEFAULT 0"),
                                ("consent_note", "TEXT DEFAULT ''"), ("consent_date", "REAL")):
                if column not in have: conn.execute(f"ALTER TABLE creators ADD COLUMN {column} {ddl}")

    PATH_COLUMNS = (("creators", "source_dir"), ("videos", "path"), ("sessions", "audio_path"), ("sessions", "video_path"), ("looks", "source"))

    def rebase_paths(self, root: Path) -> int:
        """Absolute Pfade aus einem frueheren Speicherort des Projekts (z. B. C:\\AI\\ASMRAI\\...) auf den aktuellen Projektordner umschreiben,
        damit das Projekt verschoben werden kann. Nur Pfade, die hinter einem Ordner namens wie der Projektordner liegen, werden angefasst."""
        pattern = re.compile(r"^(?:[A-Za-z]:|\\\\|/)[^\r\n]*?[\\/]" + re.escape(root.name) + r"[\\/](.+)$")
        changed = 0
        with self._lock, self.connect() as conn:
            for table, column in self.PATH_COLUMNS:
                for row in conn.execute(f"SELECT rowid AS rid, {column} AS value FROM {table} WHERE {column} IS NOT NULL").fetchall():
                    match = pattern.match(row["value"])
                    if not match: continue
                    new = str(root.joinpath(*re.split(r"[\\/]", match.group(1))))
                    if new == row["value"]: continue
                    try: conn.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?", (new, row["rid"])); changed += 1
                    except sqlite3.IntegrityError: pass                    # Ziel existiert schon (Doppeleintrag) - alten Wert lassen
        return changed

    @contextmanager
    def connect(self):
        """Eine dauerhaft offene Verbindung fuer alle Threads (serialisiert). Auf einem Netzlaufwerk kostet jedes Oeffnen/Schliessen einer
        WAL-Datenbank 3-4 s (Checkpoint, -wal/-shm neu anlegen) - das machte die ganze WebUI traege."""
        with self._lock:
            if self._conn is None:
                conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys=ON"); conn.execute("PRAGMA journal_mode=WAL")
                self._conn = conn
            try:
                yield self._conn; self._conn.commit()
            except BaseException:
                self._conn.rollback(); raise

    def query(self, sql: str, args: Iterable = ()) -> list[dict[str, Any]]:
        with self.connect() as conn: return [dict(row) for row in conn.execute(sql, tuple(args)).fetchall()]

    def one(self, sql: str, args: Iterable = ()) -> dict[str, Any] | None:
        rows = self.query(sql, args); return rows[0] if rows else None

    def execute(self, sql: str, args: Iterable = ()) -> int:
        with self._lock, self.connect() as conn: return conn.execute(sql, tuple(args)).lastrowid or 0

    def many(self, sql: str, rows: Iterable[Iterable]) -> None:
        with self._lock, self.connect() as conn: conn.executemany(sql, [tuple(r) for r in rows])

    def setting(self, key: str, default: str | None = None) -> str | None:
        row = self.one("SELECT value FROM settings WHERE key=?", (key,)); return row["value"] if row else default

    def set_setting(self, key: str, value: str | None) -> None:
        if value is None: self.execute("DELETE FROM settings WHERE key=?", (key,))
        else: self.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def set_stage(self, video_id: int, stage: str, state: str, detail: str = "") -> None:
        self.execute("INSERT INTO stages(video_id,stage,state,detail,finished) VALUES(?,?,?,?,?) "
                     "ON CONFLICT(video_id,stage) DO UPDATE SET state=excluded.state, detail=excluded.detail, finished=excluded.finished",
                     (video_id, stage, state, detail, time.time()))

    def stages(self, video_id: int) -> dict[str, dict]:
        return {row["stage"]: row for row in self.query("SELECT * FROM stages WHERE video_id=?", (video_id,))}


def dumps(value: Any) -> str: return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
