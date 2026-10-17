import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmrai import looks  # noqa: E402
from asmrai.app import App  # noqa: E402
from asmrai.config import Config  # noqa: E402


def portrait(path, shade):
    import cv2
    import numpy as np
    image = np.full((800, 600, 3), 40, np.uint8); cv2.circle(image, (300, 300), 140, (shade, 160, 200), -1); path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(path), image)


def make(tmp_path, **paths):
    cfg = Config({"paths": {"data": str(tmp_path / "data"), "ai_models": str(tmp_path / "ai_models"), "seriesai": str(tmp_path / "none"), **paths}})
    return cfg, App(cfg)


def files_under(path):
    return sorted(str(p.relative_to(path)).replace("\\", "/") for p in path.rglob("*") if p.is_file()) if path.exists() else []


def test_one_folder_is_one_model_with_its_own_person_and_data(tmp_path, monkeypatch):
    cfg, app = make(tmp_path)
    monkeypatch.setattr(looks, "best_frontal", lambda c, d, s: (lambda rows: looks.frame_path(c, rows[0]) if rows else None)(d.query("SELECT * FROM looks WHERE persona_slug=?", (s,))))
    root = tmp_path / "ai_models"
    portrait(root / "Elara Vane" / "bun" / "front_a.png", 200); portrait(root / "Elara Vane" / "curly hair" / "side_b.png", 120)
    portrait(root / "mia" / "long_hair" / "a_front.png", 90)
    overview = app.models_overview()
    assert [m["slug"] for m in overview["models"]] == ["elara-vane", "mia"] and overview["pending"] == 3          # Personen entstehen automatisch je Ordner
    assert {m["slug"]: m["pending"] for m in overview["models"]} == {"elara-vane": 2, "mia": 1}
    result = looks.import_looks(cfg, app.db, slug="elara-vane")                                                  # nur ein Modell importieren
    assert result["added"] == 2 and result["models"] == ["elara-vane"] and app.models_overview()["pending"] == 1
    looks.import_looks(cfg, app.db); app.sync_persona_image("elara-vane"); looks.export_person(cfg, app.db, "elara-vane")
    rows = app.db.query("SELECT persona_slug, style, hair FROM looks"); assert {(r["persona_slug"], r["style"]) for r in rows} == {("elara-vane", "bun"), ("elara-vane", "curly hair"), ("mia", "long_hair")}
    elara = files_under(root / "Elara Vane" / "_data")                                                          # vorhandener Ordnername wird verwendet
    assert "main.jpg" in elara and "person.json" in elara and "looks/bun/001_frame.jpg" in elara and "looks/curly-hair/001_full.jpg" in elara
    assert files_under(root / "mia" / "_data") != [] and "looks/long-hair/001_frame.jpg" in files_under(root / "mia" / "_data")     # Daten bleiben je Modell getrennt
    assert files_under(cfg.characters) == [] and not (cfg.data / "models").exists()
    card = json.loads((root / "Elara Vane" / "_data" / "person.json").read_text(encoding="utf-8"))
    assert card["name"] == "Elara Vane" and {x["style"] for x in card["looks"]} == {"bun", "curly hair"} and card["main_image"] == "main.jpg"
    assert looks.persona_image_path(cfg, "elara-vane", "main.jpg") == root / "Elara Vane" / "_data" / "main.jpg"


def test_import_ignores_data_folders_keeps_originals_and_is_incremental(tmp_path):
    cfg, app = make(tmp_path); root = tmp_path / "ai_models"
    original = root / "nova" / "bun" / "a_front.png"; portrait(original, 200); before = original.read_bytes()
    assert looks.import_looks(cfg, app.db)["added"] == 1
    assert app.models_overview()["pending"] == 0 and app.models_overview()["models"][0]["images"] == 1                  # _data zaehlt nicht als Quelle
    portrait(root / "nova" / "bun" / "c_front.png", 150); portrait(root / "_scratch" / "x.png", 10)               # neues Bild; '_'-Ordner wird ignoriert
    assert app.models_overview()["pending"] == 1 and [m["slug"] for m in app.models_overview()["models"]] == ["nova"]
    assert looks.import_looks(cfg, app.db)["added"] == 1 and len(app.db.query("SELECT 1 FROM looks")) == 2
    assert original.read_bytes() == before


def test_create_model_makes_folder_person_and_model_dirs(tmp_path):
    cfg, app = make(tmp_path)
    model = app.create_model("Nova Lind"); base = tmp_path / "ai_models" / "nova-lind"
    assert model["slug"] == "nova-lind" and base.is_dir() and (base / "_data" / "models" / "voice").is_dir() and (base / "_data" / "person.json").exists()
    assert app.db.one("SELECT name FROM personas WHERE slug='nova-lind'")["name"] == "Nova Lind"
    with pytest.raises(ValueError, match="existiert"): app.create_model("nova lind")
    with pytest.raises(ValueError): app.create_model("  ")


def test_session_uses_the_chosen_model(tmp_path):
    cfg, app = make(tmp_path)
    app.create_model("Alt"); app.create_model("Neu")
    app.db.execute("UPDATE personas SET appearance='alte frisur' WHERE slug='alt'"); app.db.execute("UPDATE personas SET appearance='neue frisur' WHERE slug='neu'")
    first = app.create_session({"persona": "alt", "duration_min": 1, "seed": 1, "speech": False}); second = app.create_session({"persona": "neu", "duration_min": 1, "seed": 1, "speech": False})
    assert first["params"]["persona"] == "alt" and first["params"]["appearance"] == "alte frisur"
    assert second["params"]["persona"] == "neu" and second["params"]["appearance"] == "neue frisur"


def test_saving_a_persona_keeps_main_image_and_creates_folder(tmp_path):
    cfg, app = make(tmp_path)
    app.create_model("Mira"); (looks.person_dir(cfg, "mira", True) / "main.jpg").write_bytes(b"x")
    app.db.execute("UPDATE personas SET image='main.jpg' WHERE slug='mira'")
    saved = app.save_persona({"name": "Mira", "appearance": "neu beschrieben", "image": ""})                       # Formular sendet leeres Bild
    assert saved["image"] == "main.jpg" and saved["appearance"] == "neu beschrieben"
    other = app.save_persona({"name": "Frei Erfunden"}); assert (tmp_path / "ai_models" / "frei-erfunden" / "_data").is_dir()


def test_legacy_ai_model_folder_moves_into_first_model(tmp_path):
    legacy = tmp_path / "ai_model"
    cfg = Config({"paths": {"data": str(tmp_path / "data"), "ai_models": str(tmp_path / "ai_models"), "legacy_ai_model": str(legacy), "seriesai": str(tmp_path / "none")}})
    app = App(cfg)
    for style in ("bun", "curly hair"): portrait(legacy / style / "a.png", 100)
    (legacy / "_data" / "elara-vane" / "looks" / "bun").mkdir(parents=True); (legacy / "_data" / "elara-vane" / "looks" / "bun" / "001_frame.jpg").write_bytes(b"R")
    (legacy / "_data" / "elara-vane" / "main.jpg").write_bytes(b"M")
    app.db.execute("INSERT INTO personas(slug,name,image,created) VALUES('elara-vane','Elara Vane','main.jpg',?)", (time.time(),))
    old_source = str(legacy / "bun" / "a.png")
    app.db.execute("INSERT INTO looks(group_name,persona_slug,file,angle,has_face,source,created) VALUES('bun','elara-vane','bun/001','front',1,?,?)", (old_source, time.time()))
    moved = looks.migrate_to_models(cfg, app.db)
    model = tmp_path / "ai_models" / "elara-vane"
    assert moved["model"] == "elara-vane" and moved["style_folders"] == 2 and moved["data"] == 2
    assert (model / "bun" / "a.png").exists() and (model / "curly hair" / "a.png").exists()                       # Originale: ein Ordner = ein Modell
    assert (model / "_data" / "main.jpg").read_bytes() == b"M" and (model / "_data" / "looks" / "bun" / "001_frame.jpg").read_bytes() == b"R"
    assert not legacy.exists()                                                                                      # alter Ordner leer und entfernt
    assert app.db.one("SELECT source FROM looks")["source"] == str(model / "bun" / "a.png")                          # gespeicherter Quellpfad folgt
    assert looks.migrate_to_models(cfg, app.db) == {}                                                               # idempotent


def test_migration_moves_old_characters_and_models_into_model_folder(tmp_path):
    cfg, app = make(tmp_path)
    app.db.execute("INSERT INTO personas(slug,name,image,created) VALUES('elara-vane','Elara Vane','elara-vane.jpg',?)", (time.time(),))
    app.db.execute("INSERT INTO looks(group_name,persona_slug,file,angle,has_face,source,created) VALUES('bun','elara-vane','bun/001','front',1,'s',?)", (time.time(),))
    old = cfg.characters / "looks" / "bun"; old.mkdir(parents=True); (old / "001_full.jpg").write_bytes(b"F"); (old / "001_frame.jpg").write_bytes(b"R")
    (cfg.characters / "elara-vane.jpg").write_bytes(b"M"); (cfg.data / "models" / "elara-vane").mkdir(parents=True)
    moved = looks.migrate_storage(cfg, app.db); base = tmp_path / "ai_models" / "elara-vane" / "_data"
    assert moved == {"looks": 2, "images": 1, "models": 1}
    assert (base / "looks" / "bun" / "001_frame.jpg").read_bytes() == b"R" and (base / "main.jpg").read_bytes() == b"M" and (base / "models").is_dir()
    assert files_under(cfg.characters) == [] and not (cfg.characters / "looks").exists() and not (cfg.data / "models").exists()
    assert looks.migrate_storage(cfg, app.db) == {"looks": 0, "images": 0, "models": 0}


def test_generated_person_becomes_a_model_folder(tmp_path):
    cfg, app = make(tmp_path); cfg.raw.pop("llm", None)
    person = app.generate_persona({"seed": 3}); base = tmp_path / "ai_models" / person["slug"] / "_data"
    assert (base / "models" / "voice").is_dir() and (base / "models" / "look").is_dir() and (base / "person.json").exists()
    assert person["slug"] in [m["slug"] for m in app.models_overview()["models"]]


def test_pick_look_prefers_front_and_hair_per_model(tmp_path):
    import random
    cfg, app = make(tmp_path)
    for i, (slug, angle, hair, face) in enumerate([("a", "profile", "bob", 1), ("a", "front", "bob", 1), ("a", "front", "long", 1), ("a", "front", "bob", 0), ("b", "front", "bob", 1)]):
        app.db.execute("INSERT INTO looks(group_name,persona_slug,file,angle,hair,has_face,source,created) VALUES('g',?,?,?,?,?,?,?)", (slug, f"g/{i:03d}", angle, hair, face, f"s{i}", time.time()))
    assert {looks.pick_look(app.db, "a", random.Random(s), "bob")["file"] for s in range(30)} == {"g/001"}     # frontal + Frisur + Gesicht, nur dieses Modell
    assert looks.pick_look(app.db, "nobody", random.Random(1)) is None


def test_migration_never_touches_a_default_ai_model_folder(tmp_path, monkeypatch):
    """Regression: Ohne ausdruecklich gesetztes paths.legacy_ai_model darf die Migration keinen Ordner anfassen (kein geratener Standardpfad)."""
    from asmrai.config import ROOT
    guard = ROOT / "ai_model"
    cfg, app = make(tmp_path)
    seen = []
    monkeypatch.setattr(looks.shutil, "move", lambda *a, **k: seen.append(a))
    assert looks.migrate_to_models(cfg, app.db) == {} and not seen
    assert cfg.path("paths", "legacy_ai_model") is None and (not guard.exists() or guard.is_dir())


def test_rebase_paths_moves_old_project_location(tmp_path):
    from asmrai.db import Database
    db = Database(tmp_path / "t.db")
    db.execute("INSERT INTO creators(slug,name,source_dir,created) VALUES('a','A',?,0)", (r"C:\AI\ASMRAI\input\A",))
    root = tmp_path / "ASMRAI"
    assert db.rebase_paths(root) == 1
    assert db.one("SELECT source_dir FROM creators")["source_dir"] == str(root / "input" / "A")
    assert db.rebase_paths(root) == 0
