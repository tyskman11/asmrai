# ASMRAI

Lokales System, das aus den Videos eines oder mehrerer ASMR-Creators ein **Stil-Profil** lernt und daraus
**neue, eigene ASMR-Sessions** synthetisiert (Audio binaural, optional mit Video). Alles läuft offline auf dem eigenen PC.

> **Grundsatz:** Von einem Creator wird nur *Stil* gelernt (Tempo, Trigger-Mix, Pegel, Stereo-Bewegung, Sprechtempo,
> Bildschnitt). **Stimme und Gesicht werden nicht geklont.** Gesprochen wird mit eingebauten synthetischen Stimmen oder mit
> Stimmen, die du mit Einwilligung der sprechenden Person registrierst. Nutze nur Material, an dem du die Rechte hast oder
> für das eine Erlaubnis vorliegt.

## Start

```text
python start.py            # WebUI auf http://127.0.0.1:8780 (zeigt die Adresse im Terminal an; es wird nie selbst ein Tab geöffnet)
python start.py doctor     # Umgebung prüfen
python start.py setup      # Umgebung anlegen/aktualisieren (--core: ohne KI-Pakete, --yes: ohne Rückfrage)
```

`start.py` startet bei `serve` auch **ComfyUI** (`comfyui.autostart`), falls es nicht schon läuft; ein bereits laufendes bleibt unberührt, ein selbst gestartetes wird beim Beenden mitgestoppt. `python start.py comfyui` startet nur ComfyUI.

`start.py` läuft mit jedem Python 3.10+ (nur Standardbibliothek) und sucht einen Interpreter mit allen Paketen:
zuerst `.venv`, dann `..\SeriesAI\.venv` (nie ein globaler Python). Findet es keinen, **legt es ohne Rückfrage `.venv` an und
installiert** `requirements.txt` (Basis, ca. 300 MB) und `requirements-full.txt` (PyTorch/CUDA, Whisper, XTTS; ca. 5–6 GB). ffmpeg wird (Windows) nach `runtime/ffmpeg` geladen, alle Caches/Temp-Dateien liegen in `runtime\cache`.
Für die KI-Pakete wird Python 3.10–3.12 empfohlen (3.13 hat dafür noch nicht überall Wheels).
Modelle (Whisper, XTTS-v2), ComfyUI und die LTX-Workflows werden aus dem Nachbarprojekt `..\SeriesAI` mitgenutzt
(siehe `config.yaml`). ComfyUI und das lokale LLM (llama.cpp, GGUF) sind **optional**.

## Ablauf in der WebUI

1. **Creators** → Name + Ordner mit Videos → *Hinzufügen & scannen*. Jeder weitere Creator bekommt sein eigenes Profil.
2. **Analyse starten** (Job, fortsetzbar): pro Video `probe → features → visual → transcribe → segment`, danach Profil.
   Bereits erledigte Stufen werden übersprungen; *Alles neu berechnen* erzwingt eine Wiederholung.
3. **Stil-Profil**: Zeitanteile je Klasse, Dauer/Pegel/Helligkeit/Onset-Rate je Klasse, Übergangswahrscheinlichkeiten,
   Sprechtempo, Schnittrate.
4. **Generator**: Stil wählen, Dauer/Thema/Sprache/Trigger/Materialien/Stimme einstellen → *Plan erstellen* →
   *Audio rendern* → optional *Video rendern* (Ambient per ffmpeg, oder ComfyUI/LTX mit eigener Figur).
5. **Stimmen & Figuren**: eigene Stimmen (mit Einwilligung) und Referenzbilder eigener/KI-Figuren.
6. **Jobs** zeigt Fortschritt/Log, Jobs lassen sich abbrechen. Verschiedene Job-Arten laufen parallel, gleiche Arten nacheinander.

Alles geht auch per CLI: `python start.py analyze <creator> [--model medium] [--limit 3]`,
`python start.py session --creator <slug> --minutes 10 --video`.

## Wie es funktioniert

| Stufe | Technik |
|---|---|
| Audio-Merkmale | ffmpeg-Stream → 0,5-s-Blöcke: Pegel, Crest, Spektralschwerpunkt, Flatness, Onset-Rate, Bandanteile, Pan, Stereobreite |
| Klassen | regelbasiert relativ zum Rauschboden der Datei: `silence, speech, tapping, brushing, breath, texture` (Heuristik, kein trainiertes Modell) |
| Sprache | faster-whisper **ohne** Silero-VAD (die überhört Flüstern vollständig), Pegelanhebung vor der Erkennung, Halluzinationsfilter |
| Bild | ffmpeg-Szenenerkennung, 6 Vorschaubilder, Helligkeit |
| Plan | Markov-Übergänge + Dauerverteilungen aus dem Profil, steuerbarer Sprachanteil |
| Klänge | prozedurale Synthese (Tapping je Material, Brushing, Scratching, Atem, Knistern) – parametrisiert durch das Profil |
| Räumlichkeit | Kugelkopf-Modell: interaurale Laufzeit (Woodworth) + Kopfschatten-Filter, Bewegungsbahnen links/rechts/Sweep/Orbit/Wander |
| Stimme | XTTS-v2 (eingebaute Sprecher oder eigene freigegebene Stimme) → Rausch-Vocoder erzeugt echtes Flüstern |
| Video | `ambient`: ffmpeg-Verlauf + Vektorskop/Spektrum; `comfyui`: LTX-Clips (optional img2video mit eigener Figur), als Ping-Pong-Loop |

## Grenzen (ehrlich)

* Die Klassen sind Schätzungen aus dem Signal; Mundgeräusche/Ear-Play werden nicht separat erkannt (landen in `texture`).
* Synthetische Trigger klingen sauber, aber nicht wie echte Mikrofonaufnahmen mit Objekten. Das Stil-Profil steuert
  Tempo, Helligkeit und Pegel – nicht die Klangqualität einzelner Aufnahmen.
* Das Quellaudio ist 128-kbit-AAC; für das Lernen von Statistik reicht das, für Klang-Fine-Tuning nicht.
* Mit 6 GB VRAM ist ein Video-LoRA-Training nicht realistisch; die Video-Generierung nutzt vorhandene LTX-Gewichte.
* ComfyUI-Clips sind lang (Minuten je Clip) und teilen sich die GPU mit der Whisper-Transkription.

## Daten & Sicherheit

* Quellvideos werden **nie** verändert; *Analysedaten löschen* entfernt nur DB-Einträge und `data/work/<id>`.
* Der Server bindet an `127.0.0.1`, prüft `Host` (DNS-Rebinding) und `Origin` (CSRF).
* Stimmen erfordern Besitzer + Einwilligung; Aufnahmen aus den Creator-Archiven sind als Stimmvorlage nicht auswählbar.

## AI-Modelle (`ai_models`): ein Ordner = ein Modell = eine virtuelle Person

```text
ai_models\
  elara-vane\                                     das erste Modell (jeder weitere Ordner ist ein weiteres Modell)
      bun\, curly hair\, normal straight hair\    deine Originalbilder (Unterordner = Style) - werden nie verändert
      _data\                                      alles, was die Software zu diesem Modell erzeugt
          person.json                             Steckbrief (Name, Aussehen, Wesen, Stimme, Looks) - Spiegel der Datenbank
          main.jpg, portrait.jpg                  Hauptbild / ggf. generiertes Porträt
          looks\<style>\NNN_full.jpg, NNN_frame.jpg   Look-Ausschnitte (Videoformat) für img2video
          models\voice\, models\look\             trainierte Stimm- und Look-Modelle (später)
```

* **Neues Modell:** Tab *Personen* → „+ Neues AI-Modell" (legt Ordner + Person an) oder einfach einen Ordner in `ai_models` anlegen; die Person wird automatisch erzeugt. Bilder in den Ordner legen, dann „Bilder importieren".
* **Im Generator** wählst du das **AI-Modell** aus; Stimme, Aussehen, Trigger-Repertoire, Looks und (später) trainierte Modelle kommen aus diesem Ordner.
* **Stimme mischen:** In der Modell-Karte → „Stimme mischen". Die Sprecher-Merkmale (XTTS-Latents) mehrerer Stimmen werden gewichtet gemittelt (alle freigegebenen Stimmen + weiche Grundstimme, Regler „Entspannung"). Ergebnis: `<modell>/_data/models/voice/<mix>/` mit Hörprobe; es wird die Standard-Stimme des Modells. Nur freigegebene Quellen. Reicht der GPU-Speicher nicht, wechselt XTTS automatisch in den RAM (langsamer). **Kein Gewichts-Fine-Tuning** - dafür gibt es den Stimm-Datensatz-Export.
* **Generator-Auswahl:** AI-Modell, Stimme (Standard = Stimme des Modells, auch Mixe), Entspannung, Look-Style (Gesicht/Frisur) und alternativ eine Gesichts-Referenz aus einem Creator-Archiv.
* Ordner, die mit `_` oder `.` beginnen, gelten nie als Modell oder Bildquelle. Alte Ablageorte (`data/characters/looks`, `data/models`) werden beim Start automatisch in den Modell-Ordner verschoben.

## Struktur

```text
config.yaml            Pfade, Modelle, Ports
start.py               Starter
src/asmrai/            app, pipeline, analysis, transcribe, profile, planner, generate, video, web, cli
src/asmrai/synth/      dsp (Trigger, Binaural, Flüstern), voice (XTTS, Stimmen)
src/asmrai/static/     WebUI (Single-Page, kein Build-Schritt)
data/                  asmrai.db, work/<video>, sessions/<id>, voices/, characters/   (nicht versioniert)
tests/                 pytest (python -m pytest)
```
