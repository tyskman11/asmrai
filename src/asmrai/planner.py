"""Session-Planer: zieht aus dem Stil-Profil eines Creators (Segment-Uebergaenge, Dauern, Anteile) eine neue Abfolge."""
from __future__ import annotations

import random

DEFAULT_PROFILE = {
    "labels": {
        "tapping": {"share": 0.35, "duration": {"median": 6, "p90": 14}, "centroid": 2200, "onset_rate": 2.0, "rms_db": -36, "pan_abs": 0.2, "width": 1.0},
        "brushing": {"share": 0.2, "duration": {"median": 7, "p90": 16}, "centroid": 3000, "onset_rate": 0.5, "rms_db": -38, "pan_abs": 0.2, "width": 1.0},
        "breath": {"share": 0.1, "duration": {"median": 5, "p90": 9}, "centroid": 1200, "onset_rate": 0.3, "rms_db": -42, "pan_abs": 0.15, "width": 0.9},
        "texture": {"share": 0.15, "duration": {"median": 5, "p90": 12}, "centroid": 2500, "onset_rate": 1.0, "rms_db": -38, "pan_abs": 0.2, "width": 1.0},
        "speech": {"share": 0.2, "duration": {"median": 10, "p90": 25}, "centroid": 2000, "onset_rate": 2.0, "rms_db": -32, "pan_abs": 0.1, "width": 0.5}},
    "transitions": {}, "loudness_db": -38.0, "speech": {"words_per_min": 95, "top_words": []}}

MATERIALS = ["wood", "plastic", "glass", "nails", "metal", "skin"]
LIBRARY_KINDS = {"tapping": {"tapping"}, "brushing": {"brushing", "scratching"}, "breath": {"breath"}, "texture": {"crinkle", "tapping"}}


def library_by_name(library: list[dict], name: str) -> dict:
    return next(t["recipe"] for t in library if t["name"] == name)
KIND_FOR_LABEL = {"tapping": "tapping", "brushing": "brushing", "breath": "breath", "texture": "crinkle", "speech": "speech", "silence": "pause"}


def _duration(rng: random.Random, stats: dict, pace: float) -> float:
    median = max(float(stats.get("median", 5)), 2.0); p90 = max(float(stats.get("p90", median * 2)), median * 1.2)
    value = rng.lognormvariate(0, 0.45) * median * (p90 / median) ** 0.25
    return round(max(3.0, min(value * pace, 45.0)), 1)


def plan_session(profile: dict | None, duration_s: float, seed: int, options: dict) -> list[dict]:
    profile = profile if profile and profile.get("labels") else DEFAULT_PROFILE
    rng = random.Random(seed)
    labels = dict(profile["labels"]); labels.pop("silence", None)
    speech_on = bool(options.get("speech", True))
    if not speech_on: labels.pop("speech", None)
    elif "speech" not in labels: labels["speech"] = DEFAULT_PROFILE["labels"]["speech"]
    if options.get("speech_share") is not None and "speech" in labels:
        labels["speech"] = {**labels["speech"], "share": max(0.0, min(0.8, float(options["speech_share"])))}
    allowed = set(options.get("triggers") or KIND_FOR_LABEL.values())
    labels = {k: v for k, v in labels.items() if KIND_FOR_LABEL.get(k) in allowed}
    if not labels: raise ValueError("Keine Trigger-Art ausgewaehlt.")
    shares = {k: max(v.get("share", 0.05), 0.02) for k, v in labels.items()}
    transitions = profile.get("transitions", {}); pace = float(options.get("pace", 1.0))
    materials = options.get("materials") or ["wood", "plastic", "glass"]
    speech_time = 0.0
    plan, time_s, label = [], 0.0, rng.choices(list(shares), weights=list(shares.values()))[0]
    wpm = 100.0   # tatsaechliches Tempo der synthetischen Stimme (XTTS, speed 0.92); das Creator-Tempo (inkl. Pausen) ist nur Information
    while time_s < duration_s:
        stats = labels[label]
        length = _duration(rng, stats.get("duration", {}), pace)
        kind = KIND_FOR_LABEL[label]
        if kind == "brushing" and stats.get("centroid", 0) > 3300 and rng.random() < 0.5: kind = "scratching"
        trigger_name = None
        library = [t for t in (profile.get("trigger_library") or []) if kind not in ("speech", "pause") and t["recipe"]["kind"] in LIBRARY_KINDS.get(label, {kind}) and t["recipe"]["kind"] in allowed]
        if library and options.get("materials"): library = [t for t in library if not t["recipe"].get("material") or t["recipe"]["material"] in options["materials"]] or library
        if library:                      # einen konkreten, erkannten Trigger des Creators waehlen (nach seinem Anteil) und mit seinem Rezept nachbauen
            chosen = rng.choices(library, weights=[max(t["share"], 0.01) for t in library])[0]; trigger_name = chosen["name"]; kind = chosen["recipe"]["kind"]
            stats = {**stats, "centroid": chosen["recipe"]["centroid"], "onset_rate": chosen["recipe"]["onset_rate"], "rms_db": chosen["recipe"].get("level_db") or stats.get("rms_db", -38)}
        pan_abs = float(stats.get("pan_abs", 0.2))
        mode = "center" if kind == "speech" else rng.choices(["left", "right", "sweep", "orbit", "wander"], weights=[1 + pan_abs * 4, 1 + pan_abs * 4, 2, 1.5, 2])[0]
        item = {"kind": kind, "label": label, "start": round(time_s, 2), "duration": length, "spatial": mode,
                "distance": 0.12 if kind == "speech" else rng.choice([0.15, 0.25, 0.4]),
                "level_db": round(float(stats.get("rms_db", -38)) - float(profile.get("loudness_db", -38)), 1),
                "params": {"centroid": stats.get("centroid", 2200), "onset_rate": stats.get("onset_rate", 2.0)}}
        if trigger_name:
            recipe = library_by_name(library, trigger_name)
            item["trigger"] = trigger_name; item["params"].update({k: recipe[k] for k in ("rate_cv", "decay_ms") if recipe.get(k) is not None})
            if kind == "tapping": item["params"]["material"] = recipe.get("material") or rng.choice(materials)
        elif kind == "tapping": item["params"]["material"] = rng.choice(materials)
        if kind == "speech": item["words"] = max(6, int(length / 60 * wpm))
        plan.append(item); time_s += length
        if kind == "speech": speech_time += length
        row = {k: v for k, v in transitions.get(label, {}).items() if k in labels and k != label}
        label = rng.choices(list(row), weights=list(row.values()))[0] if row else rng.choices(list(shares), weights=list(shares.values()))[0]
        target = shares.get("speech", 0) / sum(shares.values())
        if "speech" in labels and kind != "speech" and speech_time < target * (time_s + 5) and rng.random() < 0.6: label = "speech"
        if rng.random() < 0.15 and time_s < duration_s:
            gap = round(rng.uniform(1.0, 3.0), 1)
            plan.append({"kind": "pause", "label": "silence", "start": round(time_s, 2), "duration": gap, "spatial": "center", "distance": 0.3, "level_db": -99, "params": {}}); time_s += gap
    return plan
