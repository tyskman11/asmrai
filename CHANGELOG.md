# Änderungsverlauf

Dieses Projekt verwendet [Semantic Versioning](https://semver.org/lang/de/).

## 0.1.2 – 18.10.2026

### Behoben

- Ein zweiter Start erkennt die bereits aktive lokale WebUI und beendet sich sofort, statt eine konkurrierende SQLite-WAL-Verbindung auf dem NAS zu öffnen.
- `doctor` fragt bei einer laufenden Instanz deren System-Check ab und benötigt keine zweite Datenbankverbindung.
- Creator-Videoquellen sind auf den konfigurierten NAS-Stamm begrenzt.

### Geändert

- Videoordner können im Dashboard nur noch über den NAS-Ordnerwähler ausgewählt werden; die Texteingabe ist gesperrt.

## 0.1.1 – 17.10.2026

### Behoben

- HTTP-Anfragen mit ungültiger oder zu großer `Content-Length` werden sauber abgewiesen.
- Ungültige Medien-Byte-Ranges liefern `416 Range Not Satisfiable`, statt einen Handler-Fehler auszulösen.

### Verbessert

- Beim Programmstart wird eine datensparsame Laufzeitdiagnose (ASMRAI-, Python- und Betriebssystem-Version sowie Log-Ordner) geschrieben.
- Jeder CLI-Aufruf wird mit seinem Befehl im zentralen Log protokolliert.

## 0.1.0 – 01.10.2026

### Hinzugefügt

- Lokale WebUI, Creator-Analyse, Stilprofile und prozedurale binaurale ASMR-Session-Erzeugung.
- Fortschrittsfähige Hintergrund-Jobs, SQLite-Datenhaltung sowie Fehler-, Analyse- und Gesamtlogs.
- Optionale lokale Whisper-, XTTS-, LLM- und ComfyUI/LTX-Integration.
