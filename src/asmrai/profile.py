"""Stil-Profil eines Creators: rein statistische Parameter (keine Stimme, kein Gesicht, keine Audiodaten)."""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, defaultdict

import numpy as np

from .analysis import LABELS
from .db import Database, dumps

STOPWORDS = set("""the a an and or but if so to of in on at for with is are was were be been it this that you your i me my we our they them he she
of as by from not no yes just like can will do does did have has had what when where how why who which there here then than too very really
der die das und oder aber ein eine einen zu von in auf mit ist sind war nicht ja nein du dich dir ich mir wir sie es den dem des im am um""".split())


def _stats(values: list[float]) -> dict:
    if not values: return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {"n": len(arr), "mean": round(float(arr.mean()), 2), "median": round(float(np.median(arr)), 2),
            "p10": round(float(np.percentile(arr, 10)), 2), "p90": round(float(np.percentile(arr, 90)), 2)}


def build_profile(db: Database, creator_id: int) -> dict:
    videos = db.query("SELECT * FROM videos WHERE creator_id=?", (creator_id,))
    ids = [v["id"] for v in videos]
    profile: dict = {"built": time.time(), "videos": len(videos), "analyzed": 0, "hours": 0.0, "labels": {}, "transitions": {}}
    if not ids: db.execute("INSERT OR REPLACE INTO profiles(creator_id,data,updated) VALUES(?,?,?)", (creator_id, dumps(profile), time.time())); return profile
    marks = ",".join("?" * len(ids))
    segments = db.query(f"SELECT video_id,start,end,label,features FROM segments WHERE video_id IN ({marks}) ORDER BY video_id,start", ids)
    if not segments:   # noch nichts analysiert (oder alle Videos fehlgeschlagen)
        db.execute("INSERT OR REPLACE INTO profiles(creator_id,data,updated) VALUES(?,?,?)", (creator_id, dumps(profile), time.time())); return profile
    analyzed = {s["video_id"] for s in segments}
    profile["analyzed"] = len(analyzed)
    profile["hours"] = round(sum(v["duration"] or 0 for v in videos if v["id"] in analyzed) / 3600, 2)
    per_label: dict[str, list[dict]] = defaultdict(list)
    transitions: dict[str, Counter] = defaultdict(Counter)
    previous: dict[int, str] = {}
    for seg in segments:
        seg["features"] = json.loads(seg["features"] or "{}")
        seg["duration"] = seg["end"] - seg["start"]
        per_label[seg["label"]].append(seg)
        before = previous.get(seg["video_id"])
        if before and before != seg["label"]: transitions[before][seg["label"]] += 1
        previous[seg["video_id"]] = seg["label"]
    total_time = sum(s["duration"] for s in segments) or 1.0
    for label in LABELS:
        items = per_label.get(label, [])
        if not items: continue
        weights = np.asarray([s["duration"] for s in items])
        def wmean(key: str) -> float: return round(float(np.average([s["features"].get(key, 0) for s in items], weights=weights)), 3)
        profile["labels"][label] = {
            "share": round(sum(weights) / total_time, 4), "segments": len(items), "duration": _stats([s["duration"] for s in items]),
            "rms_db": wmean("rms_db"), "peak_db": wmean("peak_db"), "crest_db": wmean("crest_db"), "centroid": wmean("centroid"),
            "flatness": wmean("flatness"), "onset_rate": wmean("onset_rate"), "low": wmean("low"), "high": wmean("high"),
            "pan_abs": round(float(np.average([abs(s["features"].get("pan", 0)) for s in items], weights=weights)), 3),
            "width": wmean("width"), "corr": wmean("corr")}
    profile["transitions"] = {a: {b: round(n / sum(c.values()), 4) for b, n in c.items()} for a, c in transitions.items()}
    active = [s for s in segments if s["label"] != "silence"]
    profile["loudness_db"] = round(float(np.average([s["features"].get("rms_db", -50) for s in active], weights=[s["duration"] for s in active])), 2) if active else -45.0
    profile["noise_floor_db"] = round(float(np.percentile([s["features"].get("rms_db", -60) for s in segments], 5)), 2)
    # Sprache
    rows = db.query(f"SELECT video_id,language,words,data FROM transcripts WHERE video_id IN ({marks})", ids)
    words, speech_time, languages, counter, bigrams = 0, 0.0, Counter(), Counter(), Counter()
    for row in rows:
        data = json.loads(row["data"]); languages[row["language"]] += 1
        for seg in data["segments"]:
            if seg.get("no_speech_prob", 0) > 0.7: continue
            speech_time += seg["end"] - seg["start"]; tokens = re.findall(r"[a-zäöüß']{3,}", seg["text"].lower())
            words += len(tokens); counter.update(t for t in tokens if t not in STOPWORDS)
            bigrams.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if not (a in STOPWORDS and b in STOPWORDS))
    profile["speech"] = {"transcribed": len(rows), "languages": dict(languages), "words_per_min": round(words / (speech_time / 60), 1) if speech_time > 5 else None,
                         "top_words": [w for w, _ in counter.most_common(40)], "phrases": [p for p, n in bigrams.most_common(15) if n >= 3]}
    # Bild
    cut_rates, brightness = [], []
    from .config import Config
    config = Config.load()
    for v in videos:
        file = config.work / str(v["id"]) / "visual.json"
        if file.exists():
            visual = json.loads(file.read_text(encoding="utf-8")); cut_rates.append(visual["cut_rate_per_min"])
            if visual.get("brightness") is not None: brightness.append(visual["brightness"])
    profile["visual"] = {"cut_rate_per_min": round(float(np.mean(cut_rates)), 2) if cut_rates else None,
                         "brightness": round(float(np.mean(brightness)), 1) if brightness else None,
                         "resolution": Counter(f"{v['width']}x{v['height']}" for v in videos if v["width"]).most_common(1)[0][0] if any(v["width"] for v in videos) else None,
                         "fps": Counter(round(v["fps"]) for v in videos if v["fps"]).most_common(1)[0][0] if any(v["fps"] for v in videos) else None}
    try:
        from . import triggers
        profile["trigger_library"] = triggers.build_library(db, creator_id)               # einzelne, nachbaubare Trigger des Creators
    except Exception as error:  # noqa: BLE001 - Profil bleibt auch ohne Bibliothek nutzbar
        logging.getLogger("asmrai.analysis").error("Trigger-Bibliothek fuer Creator %s fehlgeschlagen: %s", creator_id, error, exc_info=True); profile["trigger_library"] = []
    from . import vision
    catalogs = []
    for v in videos:
        file = config.work / str(v["id"]) / "vision.json"
        if file.exists(): catalogs.append(json.loads(file.read_text(encoding="utf-8")))
    profile["visual_triggers"] = vision.aggregate(catalogs)
    db.execute("INSERT OR REPLACE INTO profiles(creator_id,data,updated) VALUES(?,?,?)", (creator_id, dumps(profile), time.time()))
    return profile


def get_profile(db: Database, creator_id: int) -> dict | None:
    row = db.one("SELECT data FROM profiles WHERE creator_id=?", (creator_id,))
    return json.loads(row["data"]) if row else None


# ----------------------------------------------------------------------------------------------------------------------------
# Fusion: die Stil-Profile mehrerer Creator zu EINEM Stil (dem des imaginaeren Creators) verschmelzen

LABEL_KEYS = ("rms_db", "peak_db", "crest_db", "centroid", "flatness", "onset_rate", "low", "high", "pan_abs", "width", "corr")


def _wavg(pairs: list[tuple[float, float]], default: float = 0.0) -> float:
    total = sum(w for w, _ in pairs)
    return sum(w * v for w, v in pairs) / total if total > 0 else default


def fuse_profiles(items: list[tuple[str, dict, float]]) -> dict:
    """items = [(Creator-Name, Profil, Gewicht)] -> ein Profil im selben Format. Anteile, Dauern, Pegel, Uebergaenge, Sprechtempo und der
    visuelle Katalog werden gewichtet gemittelt (Klassen-Kennwerte zusaetzlich nach ihrem Zeitanteil im jeweiligen Creator)."""
    items = [(n, p, float(w)) for n, p, w in items if p and p.get("labels") and w > 0]
    if not items: raise ValueError("Keine analysierten Creator-Profile vorhanden - zuerst die Analyse laufen lassen.")
    total = sum(w for _, _, w in items); norm = [(n, p, w / total) for n, p, w in items]
    labels = sorted({label for _, p, _ in norm for label in p["labels"]})
    share = {l: sum(w * p["labels"].get(l, {}).get("share", 0.0) for _, p, w in norm) for l in labels}
    scale = sum(share.values()) or 1.0
    fused_labels = {}
    for label in labels:
        parts = [(w * p["labels"][label].get("share", 0.0), p["labels"][label]) for _, p, w in norm if label in p["labels"] and p["labels"][label].get("share", 0) > 0]
        if not parts: continue
        duration = {k: round(_wavg([(x, d["duration"].get(k, 0)) for x, d in parts]), 2) for k in ("mean", "median", "p10", "p90")}
        duration["n"] = sum(d["duration"].get("n", 0) for _, d in parts)
        fused_labels[label] = {"share": round(share[label] / scale, 4), "segments": sum(d.get("segments", 0) for _, d in parts), "duration": duration,
                               **{k: round(_wavg([(x, d.get(k, 0)) for x, d in parts]), 3) for k in LABEL_KEYS}}
    transitions: dict[str, dict[str, float]] = {}
    for a in labels:
        row: dict[str, float] = {}
        for _, p, w in norm:
            weight = w * p["labels"].get(a, {}).get("share", 0.0)
            for b, prob in p.get("transitions", {}).get(a, {}).items(): row[b] = row.get(b, 0.0) + weight * prob
        s = sum(row.values())
        if s > 0: transitions[a] = {b: round(v / s, 4) for b, v in row.items()}
    wpm = [(w, p["speech"]["words_per_min"]) for _, p, w in norm if (p.get("speech") or {}).get("words_per_min")]
    languages: dict[str, int] = {}
    ranking: dict[str, float] = {}; phrase_rank: dict[str, float] = {}
    for _, p, w in norm:
        for lang, n in ((p.get("speech") or {}).get("languages") or {}).items(): languages[lang] = languages.get(lang, 0) + n
        for rank, word in enumerate((p.get("speech") or {}).get("top_words", [])): ranking[word] = ranking.get(word, 0.0) + w / (rank + 1)
        for rank, phrase in enumerate((p.get("speech") or {}).get("phrases", [])): phrase_rank[phrase] = phrase_rank.get(phrase, 0.0) + w / (rank + 1)
    cut = [(w, p["visual"]["cut_rate_per_min"]) for _, p, w in norm if (p.get("visual") or {}).get("cut_rate_per_min") is not None]
    bright = [(w, p["visual"]["brightness"]) for _, p, w in norm if (p.get("visual") or {}).get("brightness") is not None]
    heaviest = max(norm, key=lambda x: x[2])[1].get("visual") or {}
    catalogs = [(w * (p["visual_triggers"].get("frames", 0) or 1), p["visual_triggers"]) for _, p, w in norm if p.get("visual_triggers")]
    visual_triggers = {}
    if catalogs:
        for key in ("triggers", "props", "framing", "lighting", "settings"):
            merged: dict[str, float] = {}
            for weight, cat in catalogs:
                for name, value in cat.get(key, {}).items(): merged[name] = merged.get(name, 0.0) + weight * value
            top = sorted(merged.items(), key=lambda kv: -kv[1])[:12]; t = sum(v for _, v in top) or 1.0
            visual_triggers[key] = {name: round(v / t, 3) for name, v in top}
        visual_triggers["frames"] = sum(c.get("frames", 0) for _, c in catalogs); visual_triggers["videos"] = sum(c.get("videos", 0) for _, c in catalogs)
    return {"built": time.time(), "fused": True, "videos": sum(p.get("videos", 0) for _, p, _ in norm), "analyzed": sum(p.get("analyzed", 0) for _, p, _ in norm),
            "hours": round(sum(p.get("hours", 0.0) for _, p, _ in norm), 2), "labels": fused_labels, "transitions": transitions,
            "loudness_db": round(_wavg([(w, p.get("loudness_db", -40.0)) for _, p, w in norm]), 2), "noise_floor_db": round(_wavg([(w, p.get("noise_floor_db", -55.0)) for _, p, w in norm]), 2),
            "speech": {"transcribed": sum((p.get("speech") or {}).get("transcribed", 0) for _, p, _ in norm), "languages": languages,
                       "words_per_min": round(_wavg(wpm), 1) if wpm else None, "top_words": [w for w, _ in sorted(ranking.items(), key=lambda kv: -kv[1])[:40]],
                       "phrases": [p for p, _ in sorted(phrase_rank.items(), key=lambda kv: -kv[1])[:15]]},
            "visual": {"cut_rate_per_min": round(_wavg(cut), 2) if cut else None, "brightness": round(_wavg(bright), 1) if bright else None,
                       "resolution": heaviest.get("resolution"), "fps": heaviest.get("fps")},
            "visual_triggers": visual_triggers,
            "trigger_library": __import__("asmrai.triggers", fromlist=["merge_libraries"]).merge_libraries([(p.get("trigger_library") or [], w) for _, p, w in norm]),
            "fused_from": [{"creator": n, "weight": round(w, 4), "hours": p.get("hours", 0.0), "analyzed": p.get("analyzed", 0)} for n, p, w in norm]}


def save_fusion(db: Database, slug: str, weights: dict, profile: dict) -> None:
    db.execute("INSERT INTO styles(persona_slug,weights,profile,updated) VALUES(?,?,?,?) ON CONFLICT(persona_slug) DO UPDATE SET weights=excluded.weights, profile=excluded.profile, updated=excluded.updated",
               (slug, dumps(weights), dumps(profile), time.time()))


def get_fusion(db: Database, slug: str) -> dict | None:
    row = db.one("SELECT profile FROM styles WHERE persona_slug=?", (slug,))
    return json.loads(row["profile"]) if row else None


def style_profile(db: Database, session: dict) -> dict | None:
    """Das Stil-Profil einer Session: verschmolzener Stil des AI-Modells (params.fusion), sonst Profil des gewaehlten Creators, sonst None (Standard)."""
    params = json.loads(session["params"]) if isinstance(session.get("params"), str) else (session.get("params") or {})
    if params.get("fusion"): return get_fusion(db, params["fusion"])
    return get_profile(db, session["creator_id"]) if session.get("creator_id") else None
