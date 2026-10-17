"""Trigger-Erkennung: aus den groben Klassen (Tapping, Brushing, Atem, Textur) werden einzelne, nachbaubare Trigger.

1. Stufe ``triggers`` der Video-Analyse: fuer jedes Nicht-Sprach-Segment werden Klang-Merkmale gemessen (Resonanzfrequenz, Abklingzeit,
   Schlagrate und -regelmaessigkeit, Helligkeit, Rauheit, MFCC, Pegel, Stereo-Bewegung).
2. Beim Profil-Bau werden die Segmente je Klasse geclustert (KMeans, Anzahl per Silhouette) und regelbasiert benannt
   (z. B. tapping_wood, tapping_nails_fast, brushing_soft, scratching, crinkle, breath).
3. Jeder Trigger der Bibliothek hat ein *Rezept* (Art, Material, Helligkeit, Rate, Regelmaessigkeit, Abklingzeit, Pegel), mit dem die
   Klangsynthese ihn nachbaut; ``demo_wav`` erzeugt eine Hoerprobe.
"""
from __future__ import annotations

import io
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from . import media
from .config import Config
from .db import Database, dumps

log = logging.getLogger("asmrai.analysis")
ANALYZED_LABELS = ("tapping", "brushing", "breath", "texture")
MAX_SEGMENTS, MAX_SECONDS, MIN_SECONDS = 400, 6.0, 1.0
N_MFCC = 12
STAT_KEYS = ("centroid", "rolloff", "resonance_hz", "flatness", "zcr", "onset_rate", "onset_cv", "decay_ms", "level_db", "width", "pan_motion")


# ------------------------------------------------------------------------------------------------ Merkmale
def descriptors(y: np.ndarray, sr: int) -> dict | None:
    """Klang-Merkmale eines Mono-Ausschnitts (>= 0,4 s)."""
    import librosa
    from scipy.ndimage import uniform_filter1d
    if len(y) < sr * 0.4 or not np.isfinite(y).all() or float(np.max(np.abs(y))) < 1e-5: return None
    hop, n_fft = 256, 1024
    spec = np.abs(librosa.stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop)); freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    db_spec = librosa.amplitude_to_db(spec); frames = spec.shape[1]
    mean = spec.mean(axis=1) + 1e-12
    centroid = float((freqs * mean).sum() / mean.sum()); rolloff = float(freqs[min(np.searchsorted(np.cumsum(mean) / mean.sum(), 0.85), len(freqs) - 1)])
    flatness = float(librosa.feature.spectral_flatness(S=spec).mean()); zcr = float(librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop).mean())
    mfcc = librosa.feature.mfcc(y=y.astype(np.float32), sr=sr, n_mfcc=N_MFCC + 1, n_fft=n_fft, hop_length=hop).mean(axis=1)[1:]
    # Schlaege: normierte Onset-Huellkurve, strenge Spitzenwahl mit Mindestabstand 60 ms (Rauschen/Brummen liefert sonst Scheinschlaege)
    env = librosa.onset.onset_strength(S=db_spec, sr=sr, hop_length=hop); env = env / (np.percentile(env, 95) + 1e-9)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, delta=0.35, pre_max=3, post_max=3, pre_avg=12, post_avg=12, wait=max(1, int(0.06 * sr / hop)))
    seconds = len(y) / sr; times = onsets * hop / sr; intervals = np.diff(times)
    cv = float(np.std(intervals) / np.mean(intervals)) if len(intervals) >= 3 and np.mean(intervals) > 0 else 1.5      # wenige/keine Schlaege: unregelmaessig
    # Resonanz des angeschlagenen Objekts: Spektrum an den Schlaegen minus Hintergrund (Brummen/Raumklang), erst ab 300 Hz
    background = np.median(db_spec, axis=1)
    strike = (db_spec[:, np.clip(onsets + 2, 0, frames - 1)].mean(axis=1) if len(onsets) >= 2 else db_spec.mean(axis=1)) - background
    strike = uniform_filter1d(strike, 5); high = freqs > 300; resonance = float(freqs[high][int(np.argmax(strike[high]))])
    # Abklingzeit: Zeit, bis die Huellkurve nach dem Schlag um 10 dB (mindestens die Haelfte zum Rauschboden) gefallen ist
    rms = librosa.feature.rms(S=spec, frame_length=n_fft, hop_length=hop)[0]; db = 20 * np.log10(rms + 1e-9); decays = []; span = int(0.4 * sr / hop)
    for frame in onsets[:80]:
        window = db[frame: frame + span]
        if len(window) < 4: continue
        peak_at = int(np.argmax(window[:6])); window = window[peak_at:]; floor = float(np.min(window)); peak = float(window[0])
        if peak - floor < 6: continue
        target = max(peak - 10, (peak + floor) / 2); below = np.where(window <= target)[0]; decays.append((below[0] if len(below) else len(window)) * hop / sr * 1000)
    return {"centroid": round(centroid, 1), "rolloff": round(rolloff, 1), "resonance_hz": round(resonance, 1), "flatness": round(flatness, 4), "zcr": round(zcr, 4),
            "onset_rate": round(len(onsets) / seconds, 3), "onset_cv": round(min(cv, 2.5), 3), "decay_ms": round(float(np.median(decays)), 1) if decays else 200.0,
            "mfcc": [round(float(m), 2) for m in mfcc], "duration": round(seconds, 2)}


def extract_video(config: Config, db: Database, video: dict, ctx=None) -> int:
    """Stufe 'triggers': Merkmale aller Nicht-Sprach-Segmente eines Videos speichern."""
    segments = db.query("SELECT * FROM segments WHERE video_id=? AND label IN (%s) AND end-start>=? ORDER BY start" % ",".join("?" * len(ANALYZED_LABELS)),
                        (video["id"], *ANALYZED_LABELS, MIN_SECONDS))
    if len(segments) > MAX_SEGMENTS: segments = [segments[int(i * len(segments) / MAX_SEGMENTS)] for i in range(MAX_SEGMENTS)]
    db.execute("DELETE FROM trigger_events WHERE video_id=?", (video["id"],))
    if not segments: return 0
    sr = int(config.get("analysis", "analysis_sample_rate", 22050))
    mid = np.concatenate([(chunk[:, 0] + chunk[:, 1]) / 2 for chunk in media.stream_audio(config, Path(video["path"]), sr, 120.0)])
    feat_file = config.work / str(video["id"]) / "features.npy"; feat = np.load(feat_file) if feat_file.exists() else None
    block = float(config.get("analysis", "block_seconds", 0.5)); rows = []
    for index, seg in enumerate(segments):
        if ctx and index % 25 == 0: ctx.check()
        a, b = int(seg["start"] * sr), int(min(seg["end"], seg["start"] + MAX_SECONDS) * sr)
        desc = descriptors(mid[a:b], sr)
        if desc is None: continue
        if feat is not None:      # Pegel, Stereobreite und Stereo-Bewegung aus den Block-Merkmalen
            part = feat[int(seg["start"] / block): max(int(seg["end"] / block), int(seg["start"] / block) + 1)]
            if len(part): desc.update(level_db=round(float(part[:, 0].mean()), 2), width=round(float(part[:, 10].mean()), 3), pan_motion=round(float(part[:, 9].std()), 3))
        rows.append((video["id"], seg["start"], seg["end"], seg["label"], dumps(desc)))
    db.many("INSERT INTO trigger_events(video_id,start,end,label,desc) VALUES(?,?,?,?,?)", rows)
    return len(rows)


# ------------------------------------------------------------------------------------------------ Benennung
MATERIAL_REF = {"wood": (420, 45), "plastic": (1100, 20), "glass": (2600, 90), "nails": (3400, 12), "metal": (1800, 140), "skin": (220, 30)}   # (Hauptfrequenz Hz, Zeitkonstante ms) wie in der Synthese


def material_for(resonance: float, decay_ms: float, centroid: float = 0.0) -> str:
    """Naechstes Synthese-Material nach Hauptfrequenz (log) und 10-dB-Abklingzeit - so passen Analyse und Nachbau zusammen."""
    def distance(item):
        freq, tau = item[1]
        return abs(math.log(max(resonance, 50) / freq)) + 0.6 * abs(math.log(max(decay_ms, 5) / (tau * 1.15)))
    return min(MATERIAL_REF.items(), key=distance)[0]


def classify_cluster(label: str, s: dict) -> tuple[str, str, str | None]:
    """-> (Art, Name, Material). Regelbasiert und nachvollziehbar - kein trainiertes Modell."""
    rate, centroid, rolloff, cv = s["onset_rate"], s["centroid"], s["rolloff"], s["onset_cv"]
    if label == "tapping":
        material = material_for(s["resonance_hz"], s["decay_ms"], centroid)
        return "tapping", f"tapping_{material}" + ("_fast" if rate > 3.2 else "_slow" if rate < 1.4 else ""), material
    if label == "brushing":
        if rolloff > 7500 and rate > 2.5: return "scratching", "scratching", None
        return "brushing", ("brushing_bright" if centroid > 3500 else "brushing_soft" if centroid < 1800 else "brushing"), None
    if label == "breath": return "breath", ("blowing" if centroid > 1500 else "breath"), None
    if cv > 0.7 and rolloff > 5500: return "crinkle", "crinkle", None                                  # Textur: unregelmaessig und hoehenreich
    if rate > 2.5 and cv < 0.6:
        material = material_for(s["resonance_hz"], s["decay_ms"], centroid); return "tapping", f"tapping_{material}_soft", material
    return "crinkle", "rustle", None


def _vector(desc: dict) -> list[float]:
    return [math.log1p(desc["centroid"]), math.log1p(desc["rolloff"]), math.log1p(desc["resonance_hz"]), desc["flatness"], desc["zcr"] * 10, math.log1p(desc["onset_rate"]),
            desc["onset_cv"], math.log1p(desc["decay_ms"]), *desc["mfcc"]]


def build_library(db: Database, creator_id: int, seed: int = 1) -> list[dict]:
    """Trigger-Bibliothek eines Creators: geclusterte, benannte Trigger mit Rezept, Anteil und Beispielen."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    rows = db.query("SELECT t.video_id, t.start, t.end, t.label, t.desc FROM trigger_events t JOIN videos v ON v.id=t.video_id WHERE v.creator_id=?", (creator_id,))
    events = [{"video_id": r["video_id"], "start": r["start"], "end": r["end"], "label": r["label"], "desc": json.loads(r["desc"])} for r in rows]
    if not events: return []
    total_time = sum(e["end"] - e["start"] for e in events) or 1.0; library, pending = [], []
    for label in ANALYZED_LABELS:
        items = [e for e in events if e["label"] == label]
        if not items: continue
        matrix = StandardScaler().fit_transform(np.asarray([_vector(e["desc"]) for e in items]))
        k, labels = 1, np.zeros(len(items), dtype=int)
        if len(items) >= 12:
            best = -1.0
            for candidate in range(2, min(4, len(items) // 6) + 1):
                model = KMeans(candidate, n_init=5, random_state=seed).fit(matrix)
                score = silhouette_score(matrix, model.labels_, sample_size=min(len(items), 2000), random_state=seed)
                if score > best: best, k, labels = score, candidate, model.labels_
            if best < 0.12: k, labels = 1, np.zeros(len(items), dtype=int)               # kaum getrennt -> ein Trigger statt kuenstlicher Aufteilung
        for cluster in range(k):
            members = [e for e, c in zip(items, labels) if c == cluster]
            if not members: continue
            stats = {key: float(np.mean([m["desc"][key] for m in members if key in m["desc"]])) if any(key in m["desc"] for m in members) else 0.0 for key in STAT_KEYS}
            kind, name, material = classify_cluster(label, stats)
            center = np.mean([matrix[i] for i, c in enumerate(labels) if c == cluster], axis=0)
            nearest = sorted((i for i, c in enumerate(labels) if c == cluster), key=lambda i: float(np.linalg.norm(matrix[i] - center)))[:3]
            durations = [m["end"] - m["start"] for m in members]
            pending.append({"name": name, "kind": kind, "label": label, "share": round(sum(durations) / total_time, 4), "segments": len(members),
                            "duration": {"mean": round(float(np.mean(durations)), 2), "median": round(float(np.median(durations)), 2)},
                            "descriptors": {key: round(value, 3) for key, value in stats.items()},
                            "recipe": {"kind": kind, "material": material, "centroid": round(stats["centroid"], 1), "onset_rate": round(stats["onset_rate"], 2),
                                       "rate_cv": round(min(stats["onset_cv"], 2.0), 2), "decay_ms": round(stats["decay_ms"], 1), "resonance_hz": round(stats["resonance_hz"], 1),
                                       "level_db": round(stats["level_db"], 1), "width": round(stats["width"], 2)},
                            "examples": [{"video_id": items[i]["video_id"], "start": items[i]["start"], "end": min(items[i]["end"], items[i]["start"] + 8)} for i in nearest]})
    groups: dict[str, list[dict]] = {}
    for t in pending: groups.setdefault(t["name"], []).append(t)
    for name, group in groups.items():          # gleichnamige Trigger unterscheiden: hellerer = _bright, dumpferer = _dull, weitere nummeriert
        if len(group) == 1: library.append(group[0]); continue
        for rank, t in enumerate(sorted(group, key=lambda t: -t["recipe"]["centroid"])):
            t["name"] = f"{name}_bright" if rank == 0 else f"{name}_dull" if rank == len(group) - 1 else f"{name}_{rank + 1}"; library.append(t)
    total = sum(t["share"] for t in library) or 1.0
    for t in library: t["share"] = round(t["share"] / total, 4)
    return sorted(library, key=lambda t: -t["share"])


def merge_libraries(items: list[tuple[list[dict], float]], top: int = 16) -> list[dict]:
    """Fusion mehrerer Creator-Bibliotheken: gleichnamige Trigger werden gewichtet gemittelt, der Rest bleibt eigenstaendig."""
    merged: dict[str, dict] = {}
    for library, weight in items:
        for t in library:
            part = weight * t["share"]; cur = merged.get(t["name"])
            if cur is None: merged[t["name"]] = {**t, "share": part, "_w": part, "_recipe": {k: (v * part if isinstance(v, (int, float)) else v) for k, v in t["recipe"].items()}}
            else:
                cur["share"] += part; cur["_w"] += part; cur["segments"] += t["segments"]; cur["examples"] = (cur["examples"] + t["examples"])[:4]
                for k, v in t["recipe"].items():
                    if isinstance(v, (int, float)): cur["_recipe"][k] = cur["_recipe"].get(k, 0) + v * part
    out = []
    for t in merged.values():
        recipe = {k: (round(v / t["_w"], 2) if isinstance(v, (int, float)) and t["_w"] else v) for k, v in t.pop("_recipe").items()}; t.pop("_w")
        t["recipe"] = {**t["recipe"], **recipe}; out.append(t)
    out = sorted(out, key=lambda t: -t["share"])[:top]; total = sum(t["share"] for t in out) or 1.0
    for t in out: t["share"] = round(t["share"] / total, 4)
    return out


# ------------------------------------------------------------------------------------------------ Nachbau
def demo_wav(recipe: dict, seconds: float = 6.0, seed: int = 1) -> bytes:
    """Hoerprobe: der Trigger aus seinem Rezept synthetisiert und binaural im Raum bewegt."""
    import soundfile
    from .synth import dsp
    mono = dsp.make_trigger(recipe.get("kind", "tapping"), seconds, recipe, seed)
    mono = dsp.normalize_rms(mono, -22.0)
    stereo = dsp.spatialize(mono, dsp.azimuth_path(seconds, "sweep", np.random.default_rng(seed)), 0.25).astype(np.float32)
    buffer = io.BytesIO(); soundfile.write(buffer, stereo, dsp.SR, format="WAV", subtype="PCM_16"); return buffer.getvalue()


def sample_wav(config: Config, video_path: str, start: float, end: float) -> bytes | None:
    """Original-Ausschnitt (max. 12 s) zum Vergleich - nur lokal, direkt aus dem Quellvideo."""
    result = media.run([config.ffmpeg, "-v", "error", "-ss", f"{start:.2f}", "-t", f"{min(end - start, 12):.2f}", "-i", video_path, "-vn", "-ac", "2", "-ar", "44100", "-f", "wav", "-"], 60)
    return result.stdout if result.returncode == 0 and result.stdout else None
