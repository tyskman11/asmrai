"""AI-Modelle und ihre Looks.

Layout (ein Ordner = ein AI-Modell = eine virtuelle Person):

    ai_models\\
      <modell>\\                         z. B. elara-vane
          <style>\\*.png|jpg             deine Originalbilder: Winkel/Frisuren, Unterordner = Style (werden nie veraendert)
          _data\\                        alles, was die Software zu diesem Modell erzeugt
              person.json                Steckbrief (Spiegel der Datenbank)
              main.jpg, portrait.jpg     Hauptbild / ggf. generiertes Porträt
              looks\\<style>\\NNN_full.jpg, NNN_frame.jpg   Look-Ausschnitte (Videoformat) fuer img2video
              models\\voice\\, models\\look\\                trainierte Stimm- und Look-Modelle (spaeter)

Pro Bild wird der Winkel (front / three_quarter / profile / back) aus den Gesichts-Landmarken geschaetzt, die Frisur aus Datei-/Ordnername gelesen
(editierbar). Ordner, die mit "_" oder "." beginnen, gelten nie als Modell oder Bildquelle.
"""
from __future__ import annotations

import json
import random
import re
import shutil
import time
from pathlib import Path

from .config import Config
from .db import Database
from .jobs import JobContext

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
DATA_DIR = "_data"
HAIR_WORDS = {"bob": "bob", "long": "long", "short": "short", "ponytail": "ponytail", "pony": "ponytail", "braid": "braids", "braids": "braids", "bun": "bun",
              "curly": "curly", "wavy": "wavy", "straight": "straight", "pixie": "pixie", "bangs": "bangs", "updo": "updo", "twintails": "twintails", "down": "down", "up": "up"}
ANGLE_WORDS = {"front": "front", "frontal": "front", "side": "profile", "profile": "profile", "back": "back", "rear": "back",
               "left": "three_quarter", "right": "three_quarter", "34": "three_quarter", "3q": "three_quarter", "threequarter": "three_quarter"}
PREFERRED_ANGLES = ("front", "three_quarter")        # img2video haelt die Identitaet bei frontalen Startbildern am besten
ALL_TRIGGERS = ["tapping", "brushing", "scratching", "breath", "crinkle", "speech"]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "modell"


# ------------------------------------------------------------------------------------ Ordner
def models_root(config: Config) -> Path:
    return config.path("paths", "ai_models") or config.path("paths", "ai_model") or config.data / "ai_models"


def model_folders(config: Config) -> list[Path]:
    root = models_root(config)
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))) if root.is_dir() else []


def model_dir(config: Config, slug: str) -> Path:
    """Ordner eines Modells; passt ein vorhandener Ordner zur Kennung (z. B. 'Elara Vane' -> elara-vane), wird er verwendet."""
    for folder in model_folders(config):
        if slugify(folder.name) == slug: return folder
    return models_root(config) / slug


def person_dir(config: Config, slug: str | None, create: bool = False) -> Path:
    path = (model_dir(config, slug) if slug else models_root(config) / "_unassigned") / DATA_DIR
    if create: path.mkdir(parents=True, exist_ok=True)
    return path


def looks_dir(config: Config, slug: str | None = None, create: bool = False) -> Path:
    path = person_dir(config, slug) / "looks"
    if create: path.mkdir(parents=True, exist_ok=True)
    return path


def scan_images(folder: Path) -> list[Path]:
    """Quellbilder eines Modellordners - ohne _data und andere Ordner, die mit '_' beginnen."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT and p.is_file() and not p.relative_to(folder).parts[0].startswith(("_", ".")))


def persona_image_path(config: Config, slug: str | None, image: str | None) -> Path | None:
    """Hauptbild einer Person: zuerst im _data-Ordner ihres Modells, sonst (manuell hochgeladene Figur) in data/characters."""
    if not image: return None
    for candidate in ((person_dir(config, slug) / image) if slug else None, config.characters / image):
        if candidate and candidate.is_file(): return candidate
    return None


def ensure_persona(config: Config, db: Database, slug: str, name: str | None = None) -> dict:
    """Jeder Modell-Ordner hat genau eine Person; fehlt sie, wird sie mit Standardwerten angelegt (spaeter in der WebUI anpassbar)."""
    row = db.one("SELECT * FROM personas WHERE slug=?", (slug,))
    if row: return row
    display = name or slug.replace("-", " ").title()
    db.execute("INSERT INTO personas(slug,name,voice,triggers,notes,created,origin) VALUES(?,?,?,?,?,?,?)",
               (slug, display, f"builtin:{config.get('xtts', 'default_speaker', 'Claribel Dervla')}", json.dumps(ALL_TRIGGERS), "aus Modell-Ordner", time.time(), "model_folder"))
    for part in ("voice", "look"): (person_dir(config, slug, True) / "models" / part).mkdir(parents=True, exist_ok=True)
    export_person(config, db, slug)
    return db.one("SELECT * FROM personas WHERE slug=?", (slug,))


def ensure_personas(config: Config, db: Database) -> list[str]:
    created = []
    for folder in model_folders(config):
        slug = slugify(folder.name)
        if not db.one("SELECT 1 AS x FROM personas WHERE slug=?", (slug,)): ensure_persona(config, db, slug, folder.name); created.append(slug)
    return created


# ------------------------------------------------------------------------------------ Erkennung
def tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def guess_hair(name: str) -> str | None:
    found = [HAIR_WORDS[t] for t in tokens(name) if t in HAIR_WORDS]
    return " ".join(dict.fromkeys(found)) or None


def guess_angle_from_name(name: str) -> str | None:
    for t in tokens(name):
        if t in ANGLE_WORDS: return ANGLE_WORDS[t]
    return None


def angle_from_landmarks(face) -> tuple[str, float]:
    """YuNet-Landmarks: Nasenversatz zur Augenmitte, normiert auf den Augenabstand."""
    right_eye, left_eye, nose = face[4:6], face[6:8], face[8:10]
    yaw = float((nose[0] - (right_eye[0] + left_eye[0]) / 2) / max(abs(right_eye[0] - left_eye[0]), 1.0))
    a = abs(yaw)
    return ("front" if a < 0.14 else "three_quarter" if a < 0.38 else "profile"), round(yaw, 3)


def video_frame(image, face, width: int, height: int):
    """Kopf-Schulter-Ausschnitt im Videoformat; ohne Gesicht: Mittelausschnitt."""
    import cv2
    ih, iw = image.shape[:2]; ratio = width / height
    if face is not None:
        x, y, fw, fh = face[:4]; ch = min(max(fh * 3.4, fh * 2), ih); cw = ch * ratio
        if cw > iw: cw = iw; ch = cw / ratio
        cx = x + fw / 2
        # Kopf hat Vorrang: Stirn/Haaransatz (~25 % ueber der Gesichtsbox) landet knapp unter der Oberkante des Ausschnitts
        x0 = int(min(max(cx - cw / 2, 0), iw - cw)); y0 = int(min(max(y - 0.25 * fh - 0.1 * ch, 0), ih - ch))
    else:
        ch = min(ih, iw / ratio); cw = ch * ratio; x0, y0 = int((iw - cw) / 2), int((ih - ch) / 2)
    crop = image[y0:y0 + int(ch), x0:x0 + int(cw)]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------ Import
def import_looks(config: Config, db: Database, ctx: JobContext | None = None, slug: str | None = None) -> dict:
    """Importiert neue Bilder aller Modell-Ordner (oder nur des Modells ``slug``). Bereits importierte Bilder werden uebersprungen."""
    import cv2
    root = models_root(config)
    if not root.is_dir(): raise FileNotFoundError(f"Ordner nicht gefunden: {root}")
    ensure_personas(config, db)
    model = config.path("paths", "seriesai", "../SeriesAI") / "models" / "opencv" / "face_detection_yunet_2023mar.onnx"
    detector = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.7, 0.3, 5000) if model.exists() else None
    width, height = int(config.get("comfyui", "width", 768)), int(config.get("comfyui", "height", 448))
    known = {r["source"]: r["mtime"] for r in db.query("SELECT source,mtime FROM looks")}
    jobs = [(slugify(f.name), f, p) for f in model_folders(config) if slug in (None, slugify(f.name)) for p in scan_images(f)]
    added = skipped = 0; touched: set[str] = set()
    for index, (persona, folder_path, path) in enumerate(jobs):
        if ctx: ctx.check(); ctx.progress(index / max(len(jobs), 1))
        if known.get(str(path)) == path.stat().st_mtime: skipped += 1; continue
        image = cv2.imread(str(path))
        if image is None: continue
        ih, iw = image.shape[:2]; scale = 1024 / max(ih, iw)
        if scale < 1: image = cv2.resize(image, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)
        face, yaw = None, None
        if detector is not None:
            detector.setInputSize((image.shape[1], image.shape[0])); _, faces = detector.detect(image)
            if faces is not None and len(faces): face = max(faces, key=lambda f: f[2] * f[3])
        angle = guess_angle_from_name(path.stem)
        if face is not None:
            measured, yaw = angle_from_landmarks(face); angle = angle if angle in ("back",) else measured
        rel = path.relative_to(folder_path); style = rel.parts[0] if len(rel.parts) > 1 else None; group = style or "Stile"
        folder = looks_dir(config, persona, create=True) / re.sub(r"[^a-z0-9]+", "-", group.lower()).strip("-"); folder.mkdir(parents=True, exist_ok=True)
        stem = f"{len(list(folder.glob('*_full.jpg'))) + 1:03d}"
        cv2.imwrite(str(folder / f"{stem}_full.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(folder / f"{stem}_frame.jpg"), video_frame(image, face, width, height), [cv2.IMWRITE_JPEG_QUALITY, 95])
        db.execute("INSERT INTO looks(group_name,persona_slug,file,angle,hair,yaw,has_face,source,mtime,created,style) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (group, persona, f"{folder.name}/{stem}", angle or "unknown", guess_hair(path.stem) or guess_hair(style or ""), yaw, int(face is not None), str(path),
                    path.stat().st_mtime, time.time(), style))
        added += 1; touched.add(persona)
    for persona in touched: export_person(config, db, persona)
    if ctx: ctx.log(f"{added} neue Looks, {skipped} unveraendert, {len(jobs)} Bilder in {len({j[0] for j in jobs})} Modell(en)")
    return {"added": added, "skipped": skipped, "total": len(jobs), "models": sorted(touched)}


def count_pending(config: Config, db: Database, slug: str | None = None) -> int:
    """Bilder in den Modell-Ordnern, die noch nicht (oder in geaenderter Fassung) importiert wurden."""
    known = {r["source"]: r["mtime"] for r in db.query("SELECT source,mtime FROM looks")}
    return sum(1 for f in model_folders(config) if slug in (None, slugify(f.name)) for p in scan_images(f) if known.get(str(p)) != p.stat().st_mtime)


# ------------------------------------------------------------------------------------ Auswahl & Pfade
def pick_look(db: Database, persona_slug: str, rng: random.Random, hair: str | None = None, style: str | None = None) -> dict | None:
    """Waehlt ein Referenzbild fuer einen Clip: bevorzugt frontale Winkel, optional bestimmte Frisur oder bestimmten Style."""
    rows = db.query("SELECT * FROM looks WHERE persona_slug=? AND has_face=1", (persona_slug,))
    if style: rows = [r for r in rows if r["style"] and style.lower() == r["style"].lower()] or rows
    if hair: rows = [r for r in rows if r["hair"] and hair.lower() in r["hair"]] or rows
    good = [r for r in rows if r["angle"] in PREFERRED_ANGLES] or rows
    return rng.choice(good) if good else None


def frame_path(config: Config, look: dict) -> Path:
    return looks_dir(config, look.get("persona_slug")) / f"{look['file']}_frame.jpg"


def full_path(config: Config, look: dict) -> Path:
    return looks_dir(config, look.get("persona_slug")) / f"{look['file']}_full.jpg"


def best_frontal(config: Config, db: Database, persona_slug: str) -> Path | None:
    rows = db.query("SELECT * FROM looks WHERE persona_slug=? AND has_face=1 ORDER BY (angle='front') DESC, ABS(COALESCE(yaw,0)) ASC, id ASC LIMIT 1", (persona_slug,))
    return frame_path(config, rows[0]) if rows and frame_path(config, rows[0]).exists() else None


def delete_look(config: Config, db: Database, look_id: int) -> None:
    row = db.one("SELECT * FROM looks WHERE id=?", (look_id,))
    if row:
        for path in (full_path(config, row), frame_path(config, row)): path.unlink(missing_ok=True)
        db.execute("DELETE FROM looks WHERE id=?", (look_id,))
        if row["persona_slug"]: export_person(config, db, row["persona_slug"])


def model_summary(config: Config, db: Database, folder: Path) -> dict:
    slug = slugify(folder.name); persona = db.one("SELECT * FROM personas WHERE slug=?", (slug,)) or {}
    rows = db.query("SELECT style,angle,has_face FROM looks WHERE persona_slug=?", (slug,))
    base = person_dir(config, slug)
    fusion = db.one("SELECT profile, updated FROM styles WHERE persona_slug=?", (slug,)); fused = json.loads(fusion["profile"]) if fusion else None
    def has_files(sub: str) -> bool: return (base / "models" / sub).is_dir() and any((base / "models" / sub).iterdir())
    return {"slug": slug, "folder": folder.name, "name": persona.get("name", folder.name), "images": len(scan_images(folder)), "pending": count_pending(config, db, slug),
            "looks": len(rows), "styles": sorted({r["style"] or "Stile" for r in rows}), "usable_looks": sum(1 for r in rows if r["has_face"]),
            "main_image": bool(persona.get("image") and persona_image_path(config, slug, persona.get("image"))),
            "voice_model": has_files("voice") or bool(persona.get("voice_model")), "look_model": has_files("look") or bool(persona.get("look_model")),
            "voice": persona.get("voice"), "path": str(folder),
            "fusion": {"updated": fusion["updated"], "creators": fused.get("fused_from", []), "labels": {k: v["share"] for k, v in fused["labels"].items()}, "hours": fused.get("hours")} if fused else None,
            "blends": sorted(d.name for d in (base / "models" / "voice").iterdir() if (d / "blend.json").exists()) if (base / "models" / "voice").is_dir() else []}


# ------------------------------------------------------------------------------------ Steckbrief
def export_person(config: Config, db: Database, slug: str) -> Path | None:
    """Schreibt <modell>/_data/person.json - ein lesbarer Steckbrief (Spiegel der Datenbank)."""
    row = db.one("SELECT * FROM personas WHERE slug=?", (slug,))
    if not row: return None
    def load(value, default):
        try: return json.loads(value) if value else default
        except ValueError: return default
    data = {"name": row["name"], "slug": slug, "age": row.get("age"), "origin": row.get("origin"), "appearance": row["appearance"], "personality": row.get("personality"),
            "setting": row.get("setting"), "backstory": row.get("backstory"), "catchphrases": load(row.get("catchphrases"), []), "triggers": load(row.get("triggers"), []),
            "voice": row.get("voice"), "voice_note": row.get("voice_note"), "voice_model": row.get("voice_model"), "look_model": row.get("look_model"), "main_image": row.get("image"),
            "looks": [{"id": r["id"], "style": r["style"], "angle": r["angle"], "hair": r["hair"], "has_face": bool(r["has_face"]), "file": r["file"],
                       "original": Path(r["source"]).name if r["source"] else None} for r in db.query("SELECT * FROM looks WHERE persona_slug=? ORDER BY style, id", (slug,))],
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    target = person_dir(config, slug, create=True) / "person.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"); return target


# ------------------------------------------------------------------------------------ Migrationen
def migrate_to_models(config: Config, db: Database) -> dict:
    """Alter Ordner ``ai_model`` (Style-Ordner direkt darin, ``_data/<person>``) -> ``ai_models/<erstes Modell>/``.
    Das erste Modell ist die Person, der die bisherigen Bilder gehoerten. Idempotent: ohne alten Ordner passiert nichts."""
    # Nur wenn der alte Ordner ausdruecklich konfiguriert ist (paths.legacy_ai_model) - niemals einen Standardpfad raten:
    # sonst koennte eine andere Konfiguration (z. B. ein Test) echte Bilder wegschieben.
    legacy = config.path("paths", "legacy_ai_model"); root = models_root(config)
    if legacy is None or not legacy.is_dir() or legacy.resolve() == root.resolve(): return {}
    owner = db.setting("looks_owner")
    if not owner:
        row = db.one("SELECT persona_slug FROM looks WHERE persona_slug IS NOT NULL LIMIT 1") or db.one("SELECT slug AS persona_slug FROM personas LIMIT 1")
        owner = row["persona_slug"] if row else "ai-model"
    target = root / owner; target.mkdir(parents=True, exist_ok=True)
    moved = {"model": owner, "style_folders": 0, "data": 0}
    for child in sorted(legacy.iterdir()):
        if child.name == DATA_DIR: continue
        destination = target / child.name
        if destination.exists(): continue
        shutil.move(str(child), str(destination)); moved["style_folders"] += 1
        db.execute("UPDATE looks SET source=REPLACE(source, ?, ?) WHERE source LIKE ?", (str(child), str(destination), str(child) + "%"))
    old_data = legacy / DATA_DIR
    if old_data.is_dir():
        for person in sorted(old_data.iterdir()):
            if not person.is_dir(): continue
            new_data = person_dir(config, person.name, create=True)
            for item in sorted(person.iterdir()):
                if not (new_data / item.name).exists(): shutil.move(str(item), str(new_data / item.name)); moved["data"] += 1
            if not any(person.iterdir()): person.rmdir()
        if not any(old_data.iterdir()): old_data.rmdir()
    if not any(legacy.iterdir()): legacy.rmdir()
    db.set_setting("looks_owner", None)
    for persona in db.query("SELECT slug FROM personas"): export_person(config, db, persona["slug"])
    return moved


def migrate_storage(config: Config, db: Database) -> dict:
    """Aeltere Ablageorte (data/characters/looks, data/characters/<person>.jpg, data/models/<person>) -> <modell>/_data/."""
    moved = {"looks": 0, "images": 0, "models": 0}
    old_looks = config.characters / "looks"
    for row in db.query("SELECT * FROM looks"):
        for suffix in ("_full.jpg", "_frame.jpg"):
            old = old_looks / f"{row['file']}{suffix}"; new = looks_dir(config, row["persona_slug"]) / f"{row['file']}{suffix}"
            if old.exists() and not new.exists():
                new.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(old), str(new)); moved["looks"] += 1
    for persona in db.query("SELECT slug,image FROM personas"):
        slug, base = persona["slug"], person_dir(config, persona["slug"])
        if persona["image"] == f"{slug}.jpg" and (config.characters / persona["image"]).exists():
            base.mkdir(parents=True, exist_ok=True); shutil.move(str(config.characters / persona["image"]), str(base / "main.jpg"))
            db.execute("UPDATE personas SET image='main.jpg' WHERE slug=?", (slug,)); moved["images"] += 1
        old_models = config.data / "models" / slug
        if old_models.is_dir():
            new_models = base / "models"; new_models.parent.mkdir(parents=True, exist_ok=True)
            if not new_models.exists(): shutil.move(str(old_models), str(new_models)); moved["models"] += 1
            elif not any(old_models.iterdir()): old_models.rmdir()
        if any(moved.values()): export_person(config, db, slug)
    for folder in sorted((p for p in old_looks.rglob("*") if p.is_dir()), reverse=True) if old_looks.is_dir() else []:
        if not any(folder.iterdir()): folder.rmdir()
    if old_looks.is_dir() and not any(old_looks.iterdir()): old_looks.rmdir()
    old_models_root = config.data / "models"
    if old_models_root.is_dir() and not any(old_models_root.iterdir()): old_models_root.rmdir()
    return moved
