"""Anwendungskern: Konfiguration, Datenbank, Job-Handler und Fachoperationen (von WebUI und CLI gemeinsam genutzt)."""
from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from . import assets, creator_gen, generate, looks, pipeline, portrait, video
from .synth import blend
from .config import ROOT, Config
from .db import Database, dumps
from .jobs import JobManager
from .profile import build_profile, fuse_profiles, get_fusion, get_profile, save_fusion
from .synth import voice as voices


class App:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.load()
        self.db = Database(self.config.db_path)
        rebased = self.db.rebase_paths(ROOT)
        if rebased: logging.getLogger("asmrai").info("%d gespeicherte Pfade auf den aktuellen Projektordner umgestellt (%s)", rebased, ROOT)
        # Bis 0.1.3 wurde ein nicht konfiguriertes, optionales Vision-Modell
        # fälschlich als Video-Fehler gespeichert. Alte Einträge migrieren,
        # damit vollständig analysierte Creator nicht als fehlerhaft erscheinen.
        if not self.config.get("vision", "model"):
            self.db.execute("UPDATE stages SET state='skipped', detail='kein Vision-Modell konfiguriert (optional)' "
                            "WHERE stage='vision' AND state='error' AND detail LIKE 'kein Vision-Modell konfiguriert%'")
        self.jobs = JobManager(self.db)
        self.watcher = None   # wird von web.serve gesetzt
        self.jobs.register("process", lambda ctx, p: pipeline.process_creator(self.config, self.db, ctx, p))
        self.jobs.register("audio", lambda ctx, p: generate.render_audio(self.config, self.db, p["session_id"], ctx))
        self.jobs.register("assets", lambda ctx, p: assets.run_action(self.config, self.db, ctx, p))
        self.jobs.register("looks", self._import_looks_job)
        self.jobs.register("voice_blend", self._voice_blend_job)
        self.jobs.register("fusion", self._fusion_job)
        self.jobs.register("portrait", lambda ctx, p: portrait.render_portrait(self, ctx, p))
        self.jobs.register("video", lambda ctx, p: video.render_video(self.config, self.db, p["session_id"], ctx, p.get("mode", "ambient"), p))
        try:                                                                         # alte Ablageorte -> ai_models/<modell>/
            self.storage_moved = {"models": looks.migrate_to_models(self.config, self.db), **looks.migrate_storage(self.config, self.db)}
        except Exception as error: self.storage_moved = {}; logging.getLogger("asmrai").error("Migration der Personen-Dateien fehlgeschlagen: %s", error, exc_info=True)
        self.resumed_jobs = self.jobs.resume()

    # -- Creators -----------------------------------------------------------------------------
    def add_creator(self, name: str, source_dir: str, notes: str = "") -> dict:
        name = name.strip()
        if not name: raise ValueError("Name fehlt.")
        folder = Path(source_dir.strip().strip('"')).resolve()
        if not folder.is_dir(): raise ValueError(f"Ordner nicht gefunden: {folder}")
        nas_root = self.config.nas_root
        if nas_root:
            try: folder.relative_to(nas_root.resolve())
            except ValueError: raise ValueError(f"Creator-Ordner muss auf dem lokalen NAS liegen ({nas_root}).") from None
        slug = pipeline.slugify(name)
        if self.db.one("SELECT 1 AS x FROM creators WHERE slug=?", (slug,)): raise ValueError("Creator existiert bereits.")
        cid = self.db.execute("INSERT INTO creators(slug,name,source_dir,notes,created) VALUES(?,?,?,?,?)", (slug, name, str(folder), notes, time.time()))
        creator = self.db.one("SELECT * FROM creators WHERE id=?", (cid,))
        pipeline.scan_creator(self.db, creator)
        return self.creator_summary(creator)

    def creator_summary(self, creator: dict) -> dict:
        videos = self.db.query("SELECT v.id, v.duration, (SELECT COUNT(*) FROM stages s WHERE s.video_id=v.id AND s.stage='segment' AND s.state='done') AS seg, "
                               "(SELECT COUNT(*) FROM stages s WHERE s.video_id=v.id AND s.stage='transcribe' AND s.state='done') AS tr FROM videos v WHERE v.creator_id=?", (creator["id"],))
        profile = get_profile(self.db, creator["id"])
        return {**creator, "videos": len(videos), "analyzed": sum(v["seg"] for v in videos), "transcribed": sum(v["tr"] for v in videos),
                "hours": round(sum(v["duration"] or 0 for v in videos) / 3600, 2), "has_profile": bool(profile and profile.get("analyzed")),
                "failed": len(self.failed_video_ids(creator["id"])),
                "sessions": self.db.one("SELECT COUNT(*) AS n FROM sessions WHERE creator_id=?", (creator["id"],))["n"]}

    def failed_video_ids(self, creator_id: int) -> list[int]:
        """Videos mit Fehler (Video-Fehler oder eine Stufe im Zustand 'error') - nur diese werden beim Wiederholen angefasst."""
        rows = self.db.query("SELECT v.id FROM videos v WHERE v.creator_id=? AND (v.error IS NOT NULL OR EXISTS "
                             "(SELECT 1 FROM stages s WHERE s.video_id=v.id AND s.state='error')) ORDER BY v.id", (creator_id,))
        return [r["id"] for r in rows]

    def creators(self) -> list[dict]:
        return [self.creator_summary(c) for c in self.db.query("SELECT * FROM creators ORDER BY name")]

    def delete_creator(self, creator_id: int) -> None:
        """Entfernt nur Analysedaten. Die Quellvideos werden niemals angefasst."""
        for row in self.db.query("SELECT id FROM videos WHERE creator_id=?", (creator_id,)):
            shutil.rmtree(self.config.work / str(row["id"]), ignore_errors=True)
        self.db.execute("DELETE FROM creators WHERE id=?", (creator_id,))

    def start_process(self, creator_id: int, options: dict) -> int:
        creator = self.db.one("SELECT * FROM creators WHERE id=?", (creator_id,))
        if not creator: raise ValueError("Creator nicht gefunden.")
        params = {k: options.get(k) for k in ("transcribe", "whisper_model", "language", "force", "video_ids", "stages", "vision", "vision_model") if options.get(k) is not None}
        title = f"Analyse: {creator['name']}"
        if options.get("only_failed"):
            failed = self.failed_video_ids(creator_id)
            if not failed: raise ValueError("Keine fehlgeschlagenen Videos.")
            params["video_ids"] = failed; title = f"Analyse (nur fehlgeschlagene, {len(failed)}): {creator['name']}"
        priority = int(options.get("priority") or 0)          # keine automatische Bevorzugung: Reihenfolge = Eingang; Aendern per "Vorziehen" im Dashboard
        return self.jobs.submit("process", title, {"creator_id": creator_id, **params}, priority)

    def videos(self, creator_id: int) -> list[dict]:
        rows = self.db.query("SELECT * FROM videos WHERE creator_id=? ORDER BY title", (creator_id,))
        for row in rows:
            row["stages"] = {k: v["state"] for k, v in self.db.stages(row["id"]).items()}
        return rows

    def video_detail(self, video_id: int) -> dict:
        row = self.db.one("SELECT * FROM videos WHERE id=?", (video_id,))
        if not row: raise KeyError("Video nicht gefunden")
        row["stages"] = self.db.stages(video_id)
        row["segments"] = [{**s, "features": json.loads(s["features"] or "{}")} for s in self.db.query("SELECT * FROM segments WHERE video_id=? ORDER BY start", (video_id,))]
        transcript = self.db.one("SELECT language,words,data FROM transcripts WHERE video_id=?", (video_id,))
        row["transcript"] = {"language": transcript["language"], "words": transcript["words"], "segments": json.loads(transcript["data"])["segments"]} if transcript else None
        visual = self.config.work / str(video_id) / "visual.json"
        row["visual"] = json.loads(visual.read_text(encoding="utf-8")) if visual.exists() else None
        return row

    def video_series(self, video_id: int, points: int = 600) -> dict:
        import numpy as np
        file = self.config.work / str(video_id) / "features.npy"
        if not file.exists(): return {"block": 0.5, "rms_db": [], "pan": [], "onset": []}
        from .analysis import FEATURE_NAMES
        feat = np.load(file); step = max(1, len(feat) // points); n = len(feat) // step * step
        def reduce(name: str, fn=np.mean):
            return [round(float(v), 2) for v in fn(feat[:n, FEATURE_NAMES.index(name)].reshape(-1, step), axis=1)]
        return {"block": float(self.config.get("analysis", "block_seconds", 0.5)) * step, "rms_db": reduce("rms_db"), "pan": reduce("pan"), "onset": reduce("onset_rate", np.max)}

    # -- Sessions -----------------------------------------------------------------------------
    def create_session(self, data: dict) -> dict:
        persona = self.db.one("SELECT * FROM personas WHERE slug=?", (data["persona"],)) if data.get("persona") else None
        if data.get("persona") and not persona: raise ValueError("Person nicht gefunden.")
        style_source, fusion = str(data.get("style_source") or ""), None
        if style_source.isdigit(): data = {**data, "creator_id": int(style_source)}
        if persona:   # Person liefert Stimme, Aussehen, Bild, Trigger-Repertoire und (falls gesetzt) den Stil
            data = {**data, "voice": data.get("voice") or persona["voice"],         # ausdruecklich gewaehlte Stimme hat Vorrang vor der des Modells
                     "creator_id": persona["style_creator_id"] or data.get("creator_id"),
                    "triggers": json.loads(persona["triggers"] or "[]") or data.get("triggers")}
        creator_id = data.get("creator_id")
        if style_source == "default": creator_id = None
        elif persona and not style_source.isdigit() and not data.get("creator_id") or style_source == "fusion":
            if persona and get_fusion(self.db, persona["slug"]): fusion, creator_id = persona["slug"], None            # Fusion aller Creator hat Vorrang vor dem Einzelstil
            elif style_source == "fusion": raise ValueError("Fuer dieses AI-Modell wurde noch kein Stil verschmolzen (Tab Personen -> Imaginaeren Creator bauen).")
        if creator_id and not self.db.one("SELECT 1 AS x FROM creators WHERE id=?", (creator_id,)): raise ValueError("Creator nicht gefunden.")
        params = {"duration_min": max(0.5, min(float(data.get("duration_min", 10)), 120)), "seed": int(data.get("seed") or int(time.time()) % 100000),
                  "language": data.get("language", "en"), "theme": str(data.get("theme", ""))[:300], "speech": bool(data.get("speech", True)),
                  "speech_share": data.get("speech_share"), "whisper": float(data.get("whisper", 0.85)), "voice": data.get("voice"),
                  "triggers": data.get("triggers"), "materials": data.get("materials"), "pace": float(data.get("pace", 1.0)), "target_db": float(data.get("target_db", -26)),
                  "relax": float(data["relax"]) if data.get("relax") not in (None, "") else None, "style": data.get("style") or None, "hair": data.get("hair") or None,
                  "face_ref": data.get("face_ref") if data.get("face_ref") in self.characters() else None,
                  "fusion": fusion, "persona": persona["slug"] if persona else None, "appearance": persona["appearance"] if persona else "", "character": persona["image"] if persona else None,
                  "personality": persona["personality"] if persona else "", "catchphrases": json.loads(persona["catchphrases"] or "[]") if persona else [],
                  "setting": persona["setting"] if persona else ""}
        name = (data.get("name") or "").strip() or f"Session {time.strftime('%d.%m. %H:%M')}"
        sid = self.db.execute("INSERT INTO sessions(creator_id,name,params,status,created,updated) VALUES(?,?,?,?,?,?)",
                              (creator_id, name, dumps(params), "planned", time.time(), time.time()))
        session = self.db.one("SELECT * FROM sessions WHERE id=?", (sid,))
        self.db.execute("UPDATE sessions SET plan=? WHERE id=?", (dumps(generate.make_plan(self.config, self.db, session)), sid))
        return self.session(sid)

    def replan(self, session_id: int, seed: int | None = None) -> dict:
        session = self.db.one("SELECT * FROM sessions WHERE id=?", (session_id,))
        params = json.loads(session["params"])
        if seed is not None: params["seed"] = int(seed)
        self.db.execute("UPDATE sessions SET params=?, plan=NULL, status='planned' WHERE id=?", (dumps(params), session_id))
        session = self.db.one("SELECT * FROM sessions WHERE id=?", (session_id,))
        self.db.execute("UPDATE sessions SET plan=? WHERE id=?", (dumps(generate.make_plan(self.config, self.db, session)), session_id))
        return self.session(session_id)

    def session(self, session_id: int) -> dict:
        row = self.db.one("SELECT * FROM sessions WHERE id=?", (session_id,))
        if not row: raise KeyError("Session nicht gefunden")
        row["params"] = json.loads(row["params"]); row["plan"] = json.loads(row["plan"]) if row["plan"] else []
        row["has_audio"] = bool(row["audio_path"] and Path(row["audio_path"]).exists()); row["has_video"] = bool(row["video_path"] and Path(row["video_path"]).exists())
        row.pop("audio_path"), row.pop("video_path")
        return row

    def sessions(self) -> list[dict]:
        rows = self.db.query("SELECT s.id,s.name,s.status,s.created,s.params,s.audio_path,s.video_path,c.name AS creator FROM sessions s LEFT JOIN creators c ON c.id=s.creator_id ORDER BY s.id DESC")
        for row in rows:
            row["params"] = json.loads(row["params"]); row["has_audio"] = bool(row["audio_path"] and Path(row["audio_path"]).exists())
            row["has_video"] = bool(row["video_path"] and Path(row["video_path"]).exists()); row.pop("audio_path"); row.pop("video_path")
        return rows

    def delete_session(self, session_id: int) -> None:
        shutil.rmtree(self.config.sessions / str(session_id), ignore_errors=True)
        self.db.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    # -- Characters (eigene Referenzbilder fuer ComfyUI) ------------------------------------------
    def person_image(self, slug: str) -> Path | None:
        row = self.db.one("SELECT image FROM personas WHERE slug=?", (slug,))
        return looks.persona_image_path(self.config, slug, row["image"] if row else None)

    def characters(self) -> list[str]:
        return sorted(p.name for p in self.config.characters.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def add_character(self, name: str, data_b64: str, consent: bool) -> str:
        if not consent: raise ValueError("Bitte bestaetigen: eigene/KI-generierte Figur oder Einwilligung der abgebildeten Person liegt vor.")
        stem = pipeline.slugify(Path(name).stem); suffix = Path(name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}: raise ValueError("Nur JPG/PNG.")
        (self.config.characters / f"{stem}{suffix}").write_bytes(base64.b64decode(data_b64)); return f"{stem}{suffix}"

    # -- Freigaben & Assets ------------------------------------------------------------------------
    def set_consent(self, creator_id: int, data: dict) -> dict:
        return assets.set_consent(self.db, creator_id, bool(data.get("voice")), bool(data.get("face")), data.get("note", ""))

    def start_assets(self, creator_id: int, action: str) -> int:
        kind = "face" if action.startswith("face") else "voice"
        creator = assets.require_consent(self.db, creator_id, kind)       # sofort ablehnen statt erst im Job
        labels = {"voice_refs": "Referenzstimme", "voice_dataset": "Stimm-Datensatz", "face_refs": "Gesichts-Referenzen"}
        if action not in labels: raise ValueError("Unbekannte Aktion.")
        return self.jobs.submit("assets", f"{labels[action]}: {creator['name']}", {"creator_id": creator_id, "action": action})

    def _import_looks_job(self, ctx, params) -> None:
        result = looks.import_looks(self.config, self.db, ctx, params.get("slug"))
        for slug in result["models"]:
            self.sync_persona_image(slug); looks.export_person(self.config, self.db, slug)

    # -- Imaginaerer Creator: Stile aller Creator verschmelzen + Stimmen mischen ------------------------
    LABEL_TRIGGER = {"tapping": ["tapping"], "brushing": ["brushing", "scratching"], "breath": ["breath"], "texture": ["crinkle"], "speech": ["speech"]}

    def fusion_candidates(self) -> list[dict]:
        out = []
        for c in self.db.query("SELECT * FROM creators ORDER BY name"):
            p = get_profile(self.db, c["id"]) or {}
            out.append({"id": c["id"], "name": c["name"], "analyzed": p.get("analyzed", 0), "videos": p.get("videos", 0), "hours": p.get("hours", 0.0),
                        "usable": bool(p.get("labels")), "consent_voice": bool(c.get("consent_voice")), "consent_face": bool(c.get("consent_face"))})
        return out

    def start_fusion(self, slug: str, data: dict) -> int:
        persona = self.db.one("SELECT name FROM personas WHERE slug=?", (slug,))
        if not persona: raise ValueError("AI-Modell nicht gefunden.")
        usable = [c for c in self.fusion_candidates() if c["usable"]]
        if not usable: raise ValueError("Noch kein Creator analysiert - zuerst die Analyse (mit Transkription) laufen lassen.")
        given = {str(k): float(v) for k, v in (data.get("weights") or {}).items() if float(v) > 0}
        weights = {str(c["id"]): given.get(str(c["id"]), c["hours"] or 1.0) for c in usable if not given or str(c["id"]) in given}
        if not weights: raise ValueError("Kein Creator mit Gewicht gewaehlt.")
        return self.jobs.submit("fusion", f"Imaginaeren Creator bauen: {persona['name']}",
                                {"slug": slug, "weights": weights, "blend_voice": bool(data.get("blend_voice", True)), "adapt_triggers": bool(data.get("adapt_triggers", True)),
                                 "relax": float(data.get("relax", 0.6))})

    def _fusion_job(self, ctx, params) -> None:
        slug = params["slug"]; items = []
        ctx.log("Schritt 1/3: Stile der Creator werden verschmolzen ...")
        for creator in self.db.query("SELECT * FROM creators"):
            weight = params["weights"].get(str(creator["id"]))
            if weight: items.append((creator["name"], get_profile(self.db, creator["id"]), float(weight)))
        fused = fuse_profiles(items); save_fusion(self.db, slug, params["weights"], fused)
        ctx.log("  Anteile: " + ", ".join(f"{k} {v['share'] * 100:.0f} %" for k, v in sorted(fused["labels"].items(), key=lambda kv: -kv[1]["share"])))
        ctx.log("  Quellen: " + ", ".join(f"{f['creator']} {f['weight'] * 100:.0f} %" for f in fused["fused_from"]))
        ctx.progress(0.34)
        if params.get("adapt_triggers"):
            triggers = sorted({t for label, info in fused["labels"].items() if info["share"] >= 0.03 for t in self.LABEL_TRIGGER.get(label, [])})
            if len(triggers) >= 2:
                self.db.execute("UPDATE personas SET triggers=?, style_creator_id=NULL WHERE slug=?", (dumps(triggers), slug)); ctx.log(f"  Trigger-Repertoire des Modells: {', '.join(triggers)}")
        ctx.progress(0.4); ctx.check()
        if params.get("blend_voice"):
            ctx.log("Schritt 2/3: Stimmen werden zu einer entspannten Stimme gemischt ...")
            from .synth import voice as voices
            own = [v["id"] for v in voices.list_voices(self.config, self.db) if v["kind"] == "own" and v["consent"]]
            if own:
                sources = [{"voice": v, "weight": 1.0} for v in own] + [{"voice": f"builtin:{self.config.get('xtts', 'default_speaker', 'Claribel Dervla')}", "weight": 0.25}]
                blend.create_blend(self.config, self.db, slug, "Fusion-Stimme", sources, params.get("relax", 0.6), True, ctx)
            else: ctx.log("  Keine freigegebene eigene Stimme vorhanden - uebersprungen (erst 'Referenzstimme erzeugen' bei den Creatorn).", logging.WARNING)
        ctx.progress(0.95); looks.export_person(self.config, self.db, slug)
        ctx.log("Schritt 3/3: Fertig - im Generator ist der Stil jetzt automatisch 'Fusion' und die Stimme die des Modells.")

    # -- Stimmen mischen -------------------------------------------------------------------------
    def _voice_blend_job(self, ctx, params) -> None:
        result = blend.create_blend(self.config, self.db, params["slug"], params["name"], params["sources"], params.get("relax", 0.5), params.get("set_default", True), ctx, params.get("language", "en"))
        ctx.log(f"Fertig: {result['voice']} aus {result['sources']} Stimme(n)")

    def start_voice_blend(self, slug: str, data: dict) -> int:
        """Mischt Stimmen zu einer neuen Stimme des AI-Modells. use_all: alle freigegebenen eigenen Stimmen + weiche eingebaute Grundstimme."""
        from .synth import voice as voices
        if not self.db.one("SELECT 1 AS x FROM personas WHERE slug=?", (slug,)): raise ValueError("AI-Modell nicht gefunden.")
        sources = list(data.get("sources") or [])
        if data.get("use_all"):
            sources = [{"voice": v["id"], "weight": 1.0} for v in voices.list_voices(self.config, self.db) if v["kind"] == "own" and v["consent"]]
            sources.append({"voice": f"builtin:{self.config.get('xtts', 'default_speaker', 'Claribel Dervla')}", "weight": float(data.get("soft_weight", 0.25))})
        blend.validate_sources(self.db, sources)               # sofort ablehnen (z. B. fehlende Einwilligung) statt erst im Job
        name = (data.get("name") or "").strip() or "Entspannter Mix"
        return self.jobs.submit("voice_blend", f"Stimme mischen: {name} ({slug})", {"slug": slug, "name": name, "sources": sources, "relax": float(data.get("relax", 0.5)),
                                                                                  "set_default": bool(data.get("set_default", True)), "language": data.get("language", "en")})

    # -- AI-Modelle (ai_models/<modell>/: Bilder + _data) -------------------------------------------
    def models_overview(self) -> dict:
        root = looks.models_root(self.config)
        if root.is_dir(): looks.ensure_personas(self.config, self.db)        # jeder Modell-Ordner hat genau eine Person
        models = [looks.model_summary(self.config, self.db, folder) for folder in looks.model_folders(self.config)]
        return {"root": str(root), "root_exists": root.is_dir(), "pending": sum(m["pending"] for m in models), "models": models,
                "looks": self.db.query("SELECT * FROM looks ORDER BY persona_slug, group_name, id")}

    def create_model(self, name: str) -> dict:
        """Legt einen neuen Modell-Ordner (und seine Person) an; die Bilder legst du danach in den Ordner."""
        name = (name or "").strip()
        if not name: raise ValueError("Name fehlt.")
        slug = looks.slugify(name)
        if self.db.one("SELECT 1 AS x FROM personas WHERE slug=?", (slug,)) or any(looks.slugify(f.name) == slug for f in looks.model_folders(self.config)):
            raise ValueError("Ein Modell mit diesem Namen existiert bereits.")
        (looks.models_root(self.config) / slug).mkdir(parents=True, exist_ok=True)
        looks.ensure_persona(self.config, self.db, slug, name)
        return looks.model_summary(self.config, self.db, looks.model_dir(self.config, slug))

    def sync_persona_image(self, slug: str) -> None:
        """Hat die Person noch kein Bild, wird der beste frontale Look ihr Referenzbild."""
        row = self.db.one("SELECT image FROM personas WHERE slug=?", (slug,))
        best = looks.best_frontal(self.config, self.db, slug)
        if row and not row["image"] and best:
            target = looks.person_dir(self.config, slug, create=True) / "main.jpg"; shutil.copy2(best, target)
            self.db.execute("UPDATE personas SET image=? WHERE slug=?", (target.name, slug))

    def update_look(self, look_id: int, data: dict) -> dict:
        row = self.db.one("SELECT * FROM looks WHERE id=?", (look_id,))
        if not row: raise KeyError("Look nicht gefunden")
        angle = data.get("angle", row["angle"])
        if angle not in ("front", "three_quarter", "profile", "back", "unknown"): raise ValueError("Unbekannter Winkel.")
        self.db.execute("UPDATE looks SET angle=?, hair=? WHERE id=?", (angle, (data.get("hair", row["hair"]) or "").strip()[:40] or None, look_id))
        if row["persona_slug"]: looks.export_person(self.config, self.db, row["persona_slug"])
        return self.db.one("SELECT * FROM looks WHERE id=?", (look_id,))

    # -- Personen (virtuelle Figuren) --------------------------------------------------------------
    def personas(self) -> list[dict]:
        rows = self.db.query("SELECT p.*, c.name AS style_creator FROM personas p LEFT JOIN creators c ON c.id=p.style_creator_id ORDER BY p.name")
        for row in rows:
            row["triggers"] = json.loads(row["triggers"] or "[]"); row["catchphrases"] = json.loads(row["catchphrases"] or "[]")
        return rows

    def save_persona(self, data: dict) -> dict:
        name = (data.get("name") or "").strip()
        if not name: raise ValueError("Name fehlt.")
        image, slug_now = data.get("image") or None, pipeline.slugify(name)
        existing = self.db.one("SELECT image FROM personas WHERE slug=?", (slug_now,))
        if not image and existing: image = existing["image"]                    # Hauptbild aus dem Modell-Ordner beim Speichern nicht verlieren
        if image and image not in self.characters() and not looks.persona_image_path(self.config, slug_now, image):
            raise ValueError("Referenzbild unbekannt - zuerst unter 'Stimmen & Figuren' hochladen.")
        voice = data.get("voice") or None
        if voice and voice.startswith("own:"):
            row = self.db.one("SELECT consent FROM voices WHERE slug=?", (voice[4:],))
            if not row or not row["consent"]: raise ValueError("Diese Stimme hat keine dokumentierte Einwilligung.")
        slug = pipeline.slugify(name)
        triggers = [t for t in (data.get("triggers") or []) if t in {"tapping", "brushing", "scratching", "breath", "crinkle", "speech"}]
        self.db.execute("INSERT INTO personas(slug,name,appearance,image,voice,style_creator_id,triggers,notes,created) VALUES(?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(slug) DO UPDATE SET name=excluded.name, appearance=excluded.appearance, image=excluded.image, voice=excluded.voice, "
                        "style_creator_id=excluded.style_creator_id, triggers=excluded.triggers, notes=excluded.notes",
                        (slug, name, str(data.get("appearance", ""))[:500], image, voice, data.get("style_creator_id") or None, dumps(triggers), str(data.get("notes", ""))[:500], time.time()))
        looks.person_dir(self.config, slug, create=True)          # jede Person = ein Modell-Ordner
        looks.export_person(self.config, self.db, slug)
        return next(p for p in self.personas() if p["slug"] == slug)

    def generate_persona(self, data: dict) -> dict:
        """Erfindet eine neue, eigene virtuelle Person und legt sie an (Porträt wird separat als Job erzeugt)."""
        seed = int(data.get("seed") or int(time.time()) % 100000)
        taken = {p["name"] for p in self.personas()}
        info = creator_gen.generate(self.config, seed, str(data.get("hints", ""))[:300], data.get("language", "en"), taken)
        slug = pipeline.slugify(info["name"])
        default_voice = f"builtin:{self.config.get('xtts', 'default_speaker', 'Claribel Dervla')}"
        self.db.execute(
            "INSERT INTO personas(slug,name,appearance,voice,triggers,notes,created,personality,setting,catchphrases,backstory,voice_note,age,origin) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (slug, info["name"], info["appearance"], data.get("voice") or default_voice, dumps(info["triggers"]), f"generiert ({info['source']}, seed {seed})", time.time(),
             info["personality"], info["setting"], dumps(info["catchphrases"]), info["backstory"], info["voice"], info["age"], "generated"))
        for part in ("voice", "look"): (looks.person_dir(self.config, slug, create=True) / "models" / part).mkdir(parents=True, exist_ok=True)   # spaeter trainierte Modelle
        looks.export_person(self.config, self.db, slug)
        return next(p for p in self.personas() if p["slug"] == slug)

    def start_portrait(self, slug: str, data: dict | None = None) -> int:
        if not self.db.one("SELECT 1 AS x FROM personas WHERE slug=?", (slug,)): raise ValueError("Person nicht gefunden.")
        return self.jobs.submit("portrait", f"Porträt: {slug}", {"slug": slug, **(data or {})})

    def rename_persona(self, slug: str, new_name: str) -> dict:
        """Aendert nur den Anzeigenamen; die interne Kennung (slug) bleibt, damit Bilder, Looks und Sessions verbunden bleiben."""
        new_name = (new_name or "").strip()
        if not new_name: raise ValueError("Name fehlt.")
        if not self.db.one("SELECT 1 AS x FROM personas WHERE slug=?", (slug,)): raise ValueError("Person nicht gefunden.")
        if self.db.one("SELECT 1 AS x FROM personas WHERE LOWER(name)=LOWER(?) AND slug<>?", (new_name, slug)): raise ValueError("Dieser Name ist schon vergeben.")
        self.db.execute("UPDATE personas SET name=? WHERE slug=?", (new_name[:60], slug))
        looks.export_person(self.config, self.db, slug)
        return next(p for p in self.personas() if p["slug"] == slug)

    def delete_persona(self, slug: str) -> None:
        self.db.execute("DELETE FROM personas WHERE slug=?", (slug,))

    # -- Diagnose -------------------------------------------------------------------------------
    def doctor(self) -> dict:
        cfg = self.config
        def tool(cmd: list[str]) -> str | None:
            try: return subprocess.run(cmd, capture_output=True, text=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.splitlines()[0]
            except Exception: return None
        checks = {"ffmpeg": tool([cfg.ffmpeg, "-version"]), "ffprobe": tool([cfg.ffprobe, "-version"])}
        try:
            import torch
            checks["gpu"] = f"{torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info()[0] / 1024 ** 3:.1f} GB frei)" if torch.cuda.is_available() else "keine CUDA-GPU (CPU)"
        except Exception as error: checks["gpu"] = f"torch fehlt: {error}"
        for key, module in (("faster_whisper", "faster_whisper"), ("xtts (coqui-tts)", "TTS"), ("librosa", "librosa"), ("opencv", "cv2")):
            try: __import__(module); checks[key] = "ok"
            except Exception: checks[key] = None
        checks["xtts_model"] = "ok" if (cfg.path("xtts", "model_dir") / "model.pth").exists() else None
        checks["whisper_cache"] = str(cfg.path("whisper", "cache")) if cfg.path("whisper", "cache").exists() else None
        import urllib.request
        from .comfy import state
        comfy_state = state(cfg.get("comfyui", "api_url", "http://127.0.0.1:8188"))
        checks["comfyui"] = None if comfy_state == "nicht erreichbar" else ("erreichbar" if comfy_state == "laeuft" else comfy_state)
        try: urllib.request.urlopen(cfg.get("llm", "base_url", "") + "/health", timeout=2); checks["llm"] = "erreichbar"
        except Exception: checks["llm"] = None
        return checks

    def overview(self) -> dict:
        return {"creators": self.creators(), "sessions": len(self.db.query("SELECT id FROM sessions")), "jobs": self.jobs.active(),
                "voices": len(voices.list_voices(self.config, self.db))}
