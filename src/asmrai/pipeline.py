"""Ingest + resumable Analyse-Pipeline pro Video: probe -> features -> visual -> transcribe -> segment."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np

from . import analysis, media
from .config import Config
from .db import Database, dumps
from .jobs import Cancelled, JobContext

alog = logging.getLogger("asmrai.analysis")

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv"}
STAGES = ["probe", "features", "visual", "transcribe", "segment", "triggers"]   # "vision" (Vision-Modell) nur auf Wunsch


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "creator"


def clean_title(path: Path) -> str:
    title = re.sub(r"\s*\(\d+p[^)]*\)\s*$", "", path.stem)
    return re.sub(r"\s+", " ", title).strip()


def heal_paths(db: Database, creator: dict) -> int:
    """Videos, deren Datei verschoben/umbenannt wurde (z. B. Ordner umbenannt), werden ueber Dateiname + Groesse neu zugeordnet."""
    root = Path(creator["source_dir"])
    missing = [v for v in db.query("SELECT id,path,size FROM videos WHERE creator_id=?", (creator["id"],)) if not Path(v["path"]).exists()]
    if not missing or not root.is_dir(): return 0
    index: dict[tuple[str, int], Path] = {(p.name, p.stat().st_size): p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT and p.is_file()}
    healed = 0
    for video in missing:
        target = index.get((Path(video["path"]).name, video["size"]))
        if target:
            db.execute("UPDATE videos SET path=?, error=NULL WHERE id=?", (str(target), video["id"])); healed += 1
    return healed


def scan_creator(db: Database, creator: dict, settle_seconds: float = 0) -> tuple[int, int]:
    """Registriert neue Videos des Quellordners; bestehende bleiben unveraendert. -> (neu, gesamt)"""
    root = Path(creator["source_dir"])
    if not root.is_dir(): raise FileNotFoundError(f"Quellordner nicht gefunden: {root}")
    heal_paths(db, creator)
    known = {row["path"] for row in db.query("SELECT path FROM videos WHERE creator_id=?", (creator["id"],))}
    added = 0
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in VIDEO_EXT and path.is_file() and str(path) not in known:
            stat = path.stat()
            if settle_seconds and time.time() - stat.st_mtime < settle_seconds: continue   # wird evtl. noch kopiert
            db.execute("INSERT INTO videos(creator_id,path,title,size,mtime) VALUES(?,?,?,?,?)",
                       (creator["id"], str(path), clean_title(path), stat.st_size, stat.st_mtime)); added += 1
    total = db.one("SELECT COUNT(*) AS n FROM videos WHERE creator_id=?", (creator["id"],))["n"]
    return added, total


def work_dir(config: Config, video_id: int) -> Path:
    path = config.work / str(video_id); path.mkdir(parents=True, exist_ok=True); return path


def _done(db: Database, video_id: int, stage: str) -> bool:
    row = db.one("SELECT state FROM stages WHERE video_id=? AND stage=?", (video_id, stage))
    return bool(row and row["state"] == "done")


def process_video(config: Config, db: Database, video: dict, ctx: JobContext, options: dict) -> None:
    vid, path, force = video["id"], Path(video["path"]), bool(options.get("force"))
    work = work_dir(config, vid)
    want = set(options.get("stages") or STAGES)
    tag = f"Video {vid}"

    def note(message: str, level: int = logging.INFO) -> None:
        ctx.log(f"  {message}", level); alog.log(level, "[%s] %s", tag, message)

    def stage(name: str) -> bool:
        # Voraussetzungen (probe/features) werden bei Bedarf immer nachgeholt, auch bei eingeschraenkter Stufenwahl.
        wanted = (force and name in want) or not _done(db, vid, name) if name in ("probe", "features") else name in want and (force or not _done(db, vid, name))
        if not wanted and name in want: note(f"{name}: uebersprungen (bereits fertig)")
        return wanted

    def buckets(name: str, step: int = 25):
        """Fortschrittsmeldungen im Terminal in groben Schritten (z. B. 25 %, 50 %, ...)."""
        last = {"p": 0}
        def callback(fraction: float) -> None:
            percent = int(fraction * 100) // step * step
            if percent > last["p"]: last["p"] = percent; note(f"{name} {percent} %")
        return callback

    def run(name: str, fn, swallow: bool = False) -> bool:
        note(f"{name} ...")
        started = time.time()
        try: detail = fn()
        except Cancelled: raise
        except Exception as error:  # noqa: BLE001
            alog.error("[%s] Stufe '%s' fehlgeschlagen fuer %s: %s", tag, name, path.name, error, exc_info=True)
            ctx.log(f"  FEHLER in {name}: {error}", logging.ERROR)
            if swallow: db.set_stage(vid, name, "error", str(error)[:300]); return False
            raise
        note(f"{name} fertig ({time.time() - started:.1f} s)" + (f" - {detail}" if detail else ""))
        return True

    if stage("probe"):
        def probe() -> str:
            info = media.probe(config, path)
            db.execute("UPDATE videos SET duration=?,width=?,height=?,fps=?,channels=?,sample_rate=?,error=NULL WHERE id=?",
                       (info["duration"], info["width"], info["height"], info["fps"], info["channels"], info["sample_rate"], vid))
            db.set_stage(vid, "probe", "done" if info["has_audio"] else "error", "" if info["has_audio"] else "keine Audiospur")
            return f"{info['duration'] / 60:.1f} min, {info['width']}x{info['height']} {info['fps']:.0f} fps" + ("" if info["has_audio"] else ", KEINE AUDIOSPUR")
        run("probe", probe)
    video = db.one("SELECT * FROM videos WHERE id=?", (vid,))
    if not video.get("duration"): raise RuntimeError("Dauer unbekannt (probe fehlgeschlagen)")

    if stage("features"):
        def features() -> str:
            feat = analysis.extract_features(config, path, duration=video["duration"], on_progress=buckets("Audio-Merkmale"))
            np.save(work / "features.npy", feat); db.set_stage(vid, "features", "done", f"{len(feat)} Bloecke")
            return f"{len(feat)} Bloecke"
        run("features", features)
    ctx.check()

    if stage("visual"):
        def visual_stage() -> str:
            cuts = media.scene_cuts(config, path)
            thumbs = work / "thumbs"; brightness = []
            for k in range(6):
                out = thumbs / f"{k}.jpg"
                if media.extract_frame(config, path, video["duration"] * (k + 0.5) / 6, out):
                    try:
                        import cv2
                        brightness.append(round(float(cv2.imread(str(out)).mean()), 1))
                    except Exception:  # noqa: BLE001
                        pass
            visual = {"cuts": cuts, "cut_rate_per_min": round(len(cuts) / max(video["duration"] / 60, 0.1), 2),
                      "brightness": round(float(np.mean(brightness)), 1) if brightness else None, "thumbs": 6}
            (work / "visual.json").write_text(dumps(visual), encoding="utf-8"); db.set_stage(vid, "visual", "done", f"{len(cuts)} Schnitte")
            return f"{len(cuts)} Schnitte"
        run("visual", visual_stage)
    ctx.check()

    if options.get("vision") and (force or not _done(db, vid, "vision")):
        from . import vision
        model = options.get("vision_model") or config.get("vision", "model")
        if not model:
            db.set_stage(vid, "vision", "error", "kein Vision-Modell konfiguriert (config.yaml vision.model)")
            note("vision: kein Vision-Modell konfiguriert (config.yaml vision.model) - uebersprungen", logging.WARNING)
        else:
            def vision_stage() -> str:
                result = vision.analyze_video(config, path, video["duration"], work, model, ctx=ctx)
                (work / "vision.json").write_text(dumps(result), encoding="utf-8"); db.set_stage(vid, "vision", "done", f"{len(result['frames'])} Frames")
                return f"{len(result['frames'])} Frames ({model})"
            run("vision", vision_stage, swallow=True)
    ctx.check()

    transcribed_now = False
    if options.get("transcribe", True) and stage("transcribe"):
        from . import transcribe
        def transcribe_stage() -> str:
            result = transcribe.transcribe(config, path, options.get("whisper_model"), options.get("language"), duration=video["duration"],
                                           on_progress=buckets(f"Transkription ({options.get('whisper_model') or config.get('whisper', 'model')})", 10))
            words = sum(len(s["text"].split()) for s in result["segments"])
            db.execute("INSERT INTO transcripts(video_id,language,words,data) VALUES(?,?,?,?) "
                       "ON CONFLICT(video_id) DO UPDATE SET language=excluded.language, words=excluded.words, data=excluded.data",
                       (vid, result["language"], words, dumps(result)))
            db.set_stage(vid, "transcribe", "done", f"{words} Woerter ({result['device']})")
            return f"{words} Woerter, {result['language']}, {result['model']} auf {result['device']}"
        transcribed_now = run("transcribe", transcribe_stage, swallow=True)      # Analyse ohne Transkript bleibt nutzbar
    ctx.check()

    segmented_now = False
    resegment = force or transcribed_now or not _done(db, vid, "segment")
    if "segment" in want and not resegment: note("segment: uebersprungen (bereits fertig)")
    if "segment" in want and resegment and (work / "features.npy").exists():
        def segment_stage() -> str:
            feat = np.load(work / "features.npy"); block_s = float(config.get("analysis", "block_seconds", 0.5)); mask = None
            row = db.one("SELECT data FROM transcripts WHERE video_id=?", (vid,))
            if row: mask = analysis.speech_mask_from_transcript(json.loads(row["data"])["segments"], len(feat), block_s)
            segments = analysis.build_segments(analysis.smooth_labels(analysis.classify_blocks(feat, mask)), feat, block_s)
            db.execute("DELETE FROM segments WHERE video_id=?", (vid,))
            db.many("INSERT INTO segments(video_id,start,end,label,features) VALUES(?,?,?,?,?)",
                    [(vid, x["start"], x["end"], x["label"], dumps(x["features"])) for x in segments])
            db.set_stage(vid, "segment", "done", f"{len(segments)} Segmente")
            return f"{len(segments)} Segmente" + ("" if mask is not None else " (ohne Transkript)")
        segmented_now = run("segment", segment_stage)

    if "triggers" in want and _done(db, vid, "segment"):          # einzelne Trigger vermessen (Resonanz, Abklingzeit, Rate, ...) -> Trigger-Bibliothek
        if force or segmented_now or not _done(db, vid, "triggers"):
            def triggers_stage() -> str:
                from . import triggers
                count = triggers.extract_video(config, db, video, ctx)
                db.set_stage(vid, "triggers", "done", f"{count} Segmente"); return f"{count} Segmente vermessen"
            run("triggers", triggers_stage, swallow=True)
        else: note("triggers: uebersprungen (bereits fertig)")


def process_creator(config: Config, db: Database, ctx: JobContext, params: dict) -> None:
    from .profile import build_profile
    creator = db.one("SELECT * FROM creators WHERE id=?", (params["creator_id"],))
    if not creator: raise ValueError("Creator nicht gefunden")
    added, total = scan_creator(db, creator, float(config.get("inbox", "settle_seconds", 45)))
    ctx.log(f"{creator['name']}: {total} Videos ({added} neu)")
    ids = params.get("video_ids")
    videos = db.query("SELECT * FROM videos WHERE creator_id=? ORDER BY id", (creator["id"],))
    if ids: videos = [v for v in videos if v["id"] in set(ids)]
    started, failed = time.time(), 0
    alog.info("--- Analyse %s: %d Videos, Optionen %s ---", creator["name"], len(videos), {k: v for k, v in params.items() if k != "video_ids"})
    for index, video in enumerate(videos):
        ctx.check(); ctx.yield_to_higher_priority()          # neue Videos (hoehere Prioritaet) duerfen vorgezogen werden
        ctx.progress(index / max(len(videos), 1), f"[{index + 1}/{len(videos)}] {video['title'][:70]}")
        alog.info("[Video %s] %s", video["id"], video["path"])
        try:
            process_video(config, db, video, ctx, params)
            db.execute("UPDATE videos SET error=NULL WHERE id=? AND error IS NOT NULL", (video["id"],))
        except Cancelled:
            raise
        except Exception as error:  # noqa: BLE001
            failed += 1
            db.execute("UPDATE videos SET error=? WHERE id=?", (str(error)[:300], video["id"]))
            ctx.log(f"Video {video['id']} uebersprungen wegen Fehler: {error}", logging.ERROR)
    ctx.log("Stil-Profil wird berechnet ...")
    build_profile(db, creator["id"])
    summary = f"Fertig in {(time.time() - started) / 60:.1f} min" + (f" - {failed} Video(s) mit Fehler (siehe logs/errors.log)" if failed else "")
    ctx.log(summary, logging.WARNING if failed else logging.INFO); alog.info("--- %s: %s ---", creator["name"], summary)
