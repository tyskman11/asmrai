"""Freigabe-gebundene Assets: Referenzstimme, Gesichts-Referenzen und Trainings-Datensaetze aus einem Creator-Archiv.

Alles hier ist **nur** erlaubt, wenn fuer den Creator eine dokumentierte Freigabe (Stimme bzw. Gesicht) im Register steht.
Die Funktionen schreiben ausschliesslich in ``data/`` (Stimmen, Figurenbilder, Datensaetze) - Quellvideos bleiben unveraendert.
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import Config
from .db import Database
from .jobs import JobContext
from .media import NO_WINDOW, run

KINDS = {"voice": "consent_voice", "face": "consent_face"}


def require_consent(db: Database, creator_id: int, kind: str) -> dict:
    creator = db.one("SELECT * FROM creators WHERE id=?", (creator_id,))
    if not creator: raise ValueError("Creator nicht gefunden.")
    if not creator.get(KINDS[kind]) or not (creator.get("consent_note") or "").strip():
        raise ValueError(f"Keine dokumentierte Freigabe fuer {'Stimme' if kind == 'voice' else 'Gesicht'} von {creator['name']} - zuerst im Freigaben-Register eintragen.")
    return creator


def set_consent(db: Database, creator_id: int, voice: bool, face: bool, note: str) -> dict:
    note = (note or "").strip()
    if (voice or face) and len(note) < 10:
        raise ValueError("Bitte Nachweis notieren (wer hat wann wie zugestimmt, z. B. 'E-Mail von X vom 12.05.2026').")
    db.execute("UPDATE creators SET consent_voice=?, consent_face=?, consent_note=?, consent_date=? WHERE id=?",
               (int(voice), int(face), note, time.time() if (voice or face) else None, creator_id))
    return db.one("SELECT id,name,consent_voice,consent_face,consent_note,consent_date FROM creators WHERE id=?", (creator_id,))


# -- Stimme ---------------------------------------------------------------------------------------

_encoder = None


def _encoder_model(config: Config):
    global _encoder
    if _encoder is None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        source = config.path("paths", "seriesai", "../SeriesAI") / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"
        device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2 * 1024 ** 3 else "cpu"
        _encoder = EncoderClassifier.from_hparams(source=str(source), savedir=str(source), run_opts={"device": device},
                                                  overrides={"pretrained_path": str(source)}, local_strategy=LocalStrategy.COPY)
    return _encoder


def _decode(config: Config, path: str, start: float, duration: float, rate: int, normalize: bool = False) -> np.ndarray:
    af = ["-af", "highpass=f=70,loudnorm=I=-23:TP=-2:LRA=7"] if normalize else []
    cmd = [config.ffmpeg, "-v", "error", "-ss", f"{start:.2f}", "-t", f"{duration:.2f}", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), *af, "-f", "f32le", "-"]
    return np.frombuffer(run(cmd).stdout, dtype=np.float32)


def voice_candidates(config: Config, db: Database, creator_id: int, min_s: float = 3.0, max_s: float = 15.0) -> list[dict]:
    """Saubere Sprechstuecke: Whisper-Segment liegt ganz in einem 'speech'-Segment, kaum Tapping/Rauschen, hoher Abstand zum Rauschboden."""
    from .profile import get_profile
    floor = (get_profile(db, creator_id) or {}).get("noise_floor_db", -55)
    out = []
    for video in db.query("SELECT id,path,title FROM videos WHERE creator_id=? AND error IS NULL", (creator_id,)):
        row = db.one("SELECT data FROM transcripts WHERE video_id=?", (video["id"],)); feat_file = config.work / str(video["id"]) / "features.npy"
        if not row or not feat_file.exists(): continue
        feat = np.load(feat_file); block = float(config.get("analysis", "block_seconds", 0.5))
        speech = [(s["start"], s["end"]) for s in db.query("SELECT start,end FROM segments WHERE video_id=? AND label='speech'", (video["id"],))]
        for seg in json.loads(row["data"])["segments"]:
            length = seg["end"] - seg["start"]
            if not (min_s <= length <= max_s) or seg["no_speech_prob"] > 0.4 or seg["avg_logprob"] < -0.6 or len(seg["text"].split()) < 4: continue
            covered = sum(max(0.0, min(e, seg["end"]) - max(b, seg["start"])) for b, e in speech)
            if covered < 0.9 * length: continue
            a, b = int(seg["start"] / block), max(int(seg["end"] / block), int(seg["start"] / block) + 1); part = feat[a:b]
            if not len(part): continue
            level, onset, crest = float(part[:, 0].mean()), float(part[:, 5].mean()), float(part[:, 2].mean())
            if level - floor < 10 or onset > 3.0: continue
            out.append({"video_id": video["id"], "path": video["path"], "start": seg["start"], "end": seg["end"], "text": seg["text"],
                        "score": (level - floor) - 2 * onset + 3 * seg["avg_logprob"] - abs(length - 9) * 0.3})
    return sorted(out, key=lambda c: -c["score"])


def _embed(config: Config, clips: list[dict], ctx: JobContext | None) -> np.ndarray:
    import torch
    encoder = _encoder_model(config); vectors = []
    for index, clip in enumerate(clips):
        if ctx: ctx.check(); ctx.progress(0.1 + 0.5 * index / max(len(clips), 1))
        wave = _decode(config, clip["path"], clip["start"], clip["end"] - clip["start"], 16000)
        with torch.no_grad(): vec = encoder.encode_batch(torch.from_numpy(wave).unsqueeze(0)).squeeze().cpu().numpy()
        vectors.append(vec / max(np.linalg.norm(vec), 1e-9))
    return np.asarray(vectors)


def same_speaker(config: Config, clips: list[dict], ctx: JobContext | None, threshold: float = 0.55) -> list[dict]:
    """Behaelt nur Stuecke der Hauptstimme (Median-Embedding); Gaeste/Fremdstimmen fallen heraus."""
    if len(clips) < 3: return clips
    vectors = _embed(config, clips, ctx); center = np.median(vectors, axis=0); center /= max(np.linalg.norm(center), 1e-9)
    sims = vectors @ center
    return [{**c, "similarity": round(float(s), 3)} for c, s in zip(clips, sims) if s >= threshold]


def build_voice_references(config: Config, db: Database, creator_id: int, ctx: JobContext, count: int = 8) -> dict:
    creator = require_consent(db, creator_id, "voice")
    ctx.log("Suche saubere Sprechstuecke ...")
    candidates = voice_candidates(config, db, creator_id)[:160]
    if len(candidates) < 3: raise ValueError("Zu wenige saubere Sprechstuecke - zuerst Analyse mit Transkription ausfuehren.")
    ctx.log(f"{len(candidates)} Kandidaten, pruefe Sprecher-Konsistenz ...")
    clips = same_speaker(config, candidates, ctx)[:count]
    if not clips: raise ValueError("Keine konsistente Hauptstimme gefunden.")
    slug = f"{creator['slug']}-ref"; folder = config.voices / slug
    if folder.exists():
        for old in folder.glob("ref_*.wav"): old.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    import soundfile
    for index, clip in enumerate(clips):
        wave = _decode(config, clip["path"], clip["start"], clip["end"] - clip["start"], 24000, normalize=True)
        soundfile.write(str(folder / f"ref_{index:02d}.wav"), wave, 24000)
    (folder / "meta.json").write_text(json.dumps({"source_creator": creator["name"], "consent_note": creator["consent_note"], "created": time.time(),
                                                  "clips": [{k: c[k] for k in ("video_id", "start", "end", "text", "similarity") if k in c} for c in clips]}, ensure_ascii=False, indent=1), encoding="utf-8")
    db.execute("INSERT INTO voices(slug,name,kind,owner,consent,created) VALUES(?,?,?,?,?,?) ON CONFLICT(slug) DO UPDATE SET owner=excluded.owner, consent=1",
               (slug, f"{creator['name']} (freigegeben)", "own", creator["name"], 1, time.time()))
    ctx.log(f"Referenzstimme '{slug}' mit {len(clips)} Clips angelegt.")
    return {"voice": f"own:{slug}", "clips": len(clips)}


def export_voice_dataset(config: Config, db: Database, creator_id: int, ctx: JobContext, limit: int = 4000) -> dict:
    """Coqui/LJSpeech-Format (wavs/, metadata_train.csv, metadata_eval.csv) fuer ein Stimm-Fine-Tuning."""
    creator = require_consent(db, creator_id, "voice")
    ctx.log("Kandidaten sammeln ...")
    candidates = voice_candidates(config, db, creator_id, 2.5, 14.0)[:limit]
    clips = same_speaker(config, candidates, ctx, 0.5)
    if not clips: raise ValueError("Keine passenden Sprechstuecke gefunden.")
    folder = config.data / "datasets" / creator["slug"] / "voice"; (folder / "wavs").mkdir(parents=True, exist_ok=True)
    import soundfile
    rows, seconds = [], 0.0
    for index, clip in enumerate(clips):
        ctx.check(); ctx.progress(0.6 + 0.4 * index / len(clips))
        name = f"{creator['slug']}_{index:05d}"
        wave = _decode(config, clip["path"], clip["start"], clip["end"] - clip["start"], 24000, normalize=True)
        soundfile.write(str(folder / "wavs" / f"{name}.wav"), wave, 24000); seconds += len(wave) / 24000
        text = clip["text"].replace("|", " ").strip(); rows.append((f"wavs/{name}.wav", text, text))
    random.Random(1).shuffle(rows); split = max(1, int(len(rows) * 0.05))
    for fname, subset in (("metadata_eval.csv", rows[:split]), ("metadata_train.csv", rows[split:])):
        with open(folder / fname, "w", encoding="utf-8", newline="") as handle: csv.writer(handle, delimiter="|").writerows([("audio_file", "text", "speaker_name")] + [(a, b, creator["slug"]) for a, b, _ in subset])
    (folder / "README.txt").write_text(f"{len(rows)} Clips, {seconds / 3600:.2f} h, 24 kHz mono. Quelle: {creator['name']}. Freigabe: {creator['consent_note']}\n"
                                       "Format: Coqui-XTTS-Fine-Tuning (formatter ljspeech-aehnlich: audio_file|text|speaker_name).\n", encoding="utf-8")
    ctx.log(f"{len(rows)} Clips ({seconds / 3600:.2f} h) -> {folder}")
    return {"clips": len(rows), "hours": round(seconds / 3600, 2), "folder": str(folder)}


# -- Gesicht --------------------------------------------------------------------------------------

def dominant_identity(embeddings: list, threshold: float = 0.363) -> list[bool]:
    """Behaelt nur Gesichter der haeufigsten Person (Median-Embedding, Cosinus-Aehnlichkeit; 0.363 = SFace-Schwelle).
    Gaeste in den Videos fallen so heraus - die Freigabe gilt nur fuer den Creator selbst."""
    vectors = np.asarray(embeddings, dtype=float); vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    center = np.median(vectors, axis=0); center /= np.linalg.norm(center) + 1e-9
    return [bool(x) for x in (vectors @ center) >= threshold]


def _face_model(config: Config) -> Path:
    path = config.path("paths", "seriesai", "../SeriesAI") / "models" / "opencv" / "face_detection_yunet_2023mar.onnx"
    if not path.exists(): raise FileNotFoundError(f"YuNet-Modell fehlt: {path}")
    return path


def extract_faces(config: Config, db: Database, creator_id: int, ctx: JobContext, per_video: int = 20, keep: int = 80, refs: int = 8) -> dict:
    """Schaerfste, frontale Einzelgesichter als 512-px-Crops. Die besten ``refs`` werden Figuren-Referenzbilder (img2video)."""
    import cv2
    creator = require_consent(db, creator_id, "face")
    detector = cv2.FaceDetectorYN.create(str(_face_model(config)), "", (320, 320), 0.9, 0.3, 5000)
    sface = config.path("paths", "seriesai", "../SeriesAI") / "models" / "opencv" / "face_recognition_sface_2021dec.onnx"
    if not sface.exists(): raise FileNotFoundError(f"SFace-Modell fehlt ({sface}) - ohne Personenabgleich wuerden auch Gaeste extrahiert.")
    recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
    videos = db.query("SELECT id,path,duration FROM videos WHERE creator_id=? AND duration IS NOT NULL AND error IS NULL", (creator_id,))
    if not videos: raise ValueError("Zuerst Analyse ausfuehren (Videodauer fehlt).")
    found = []; tmp = config.work / "_faces"; tmp.mkdir(parents=True, exist_ok=True)
    for index, video in enumerate(videos):
        ctx.check(); ctx.progress(index / len(videos))
        for k in range(per_video):
            t = video["duration"] * (k + 0.5) / per_video; frame = tmp / "f.jpg"
            run([config.ffmpeg, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", video["path"], "-frames:v", "1", "-q:v", "2", str(frame)], 60)
            image = cv2.imread(str(frame))
            if image is None: continue
            h, w = image.shape[:2]; detector.setInputSize((w, h)); _, faces = detector.detect(image)
            if faces is None or len(faces) != 1: continue                                  # genau ein Gesicht, sonst Verwechslungsgefahr
            x, y, fw, fh = faces[0][:4]; score = float(faces[0][14])
            re, le, nose = faces[0][4:6], faces[0][6:8], faces[0][8:10]
            eye_dist = max(abs(re[0] - le[0]), 1.0)
            if fw < 110 or abs(nose[0] - (re[0] + le[0]) / 2) / eye_dist > 0.3: continue   # zu klein bzw. nicht frontal
            cx, cy, half = x + fw / 2, y + fh / 2, max(fw, fh) * 0.85
            x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0)); x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
            crop = image[y0:y1, x0:x1]
            if min(crop.shape[:2]) < 200: continue
            crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
            sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            found.append({"crop": crop, "score": sharp * score, "thumb": cv2.resize(crop, (16, 16)).astype(np.float32), "video_id": video["id"], "t": round(t, 1),
                          "emb": recognizer.feature(recognizer.alignCrop(image, faces[0])).flatten()})
    if not found: raise ValueError("Keine brauchbaren Gesichter gefunden.")
    keep_mask = dominant_identity([f["emb"] for f in found]); excluded = len(found) - sum(keep_mask)
    found = [f for f, ok in zip(found, keep_mask) if ok]
    ctx.log(f"{len(found)} Gesichter der Hauptperson, {excluded} Gesichter anderer Personen (Gaeste) ausgeschlossen")
    found.sort(key=lambda f: -f["score"]); chosen: list[dict] = []
    for item in found:                                                                      # Duplikate (fast gleiche Bilder) auslassen
        if all(np.abs(item["thumb"] - c["thumb"]).mean() > 18 for c in chosen): chosen.append(item)
        if len(chosen) >= keep: break
    dataset = config.data / "datasets" / creator["slug"] / "face"; dataset.mkdir(parents=True, exist_ok=True)
    token = creator["slug"].replace("-", "")
    for index, item in enumerate(chosen):
        cv2.imwrite(str(dataset / f"{index:03d}.jpg"), item["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        (dataset / f"{index:03d}.txt").write_text(f"a photo of {token}, face, soft lighting", encoding="utf-8")
    for index, item in enumerate(chosen[:refs]):
        cv2.imwrite(str(config.characters / f"{creator['slug']}_face_{index:02d}.jpg"), item["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    (dataset / "README.txt").write_text(f"{len(chosen)} Gesichts-Crops (512 px) von {creator['name']}. Freigabe: {creator['consent_note']}\n", encoding="utf-8")
    ctx.log(f"{len(chosen)} Gesichter -> {dataset}; {min(refs, len(chosen))} Referenzbilder in data/characters")
    return {"faces": len(chosen), "references": min(refs, len(chosen)), "folder": str(dataset)}


def run_action(config: Config, db: Database, ctx: JobContext, params: dict) -> None:
    action, cid = params["action"], int(params["creator_id"])
    handlers = {"voice_refs": build_voice_references, "voice_dataset": export_voice_dataset, "face_refs": extract_faces}
    if action not in handlers: raise ValueError(f"Unbekannte Aktion: {action}")
    result = handlers[action](config, db, cid, ctx)
    ctx.log(json.dumps(result, ensure_ascii=False))
