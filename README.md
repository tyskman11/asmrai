# ASMRAI

ASMRAI is a local system that learns a **style profile** from videos by one or more ASMR creators and uses it to synthesize **new, original ASMR sessions** (binaural audio and optional video). Everything runs offline on your own computer.

> **Principle:** The project learns a creator's *style*—pace, trigger mix, levels, stereo movement, speaking pace, and editing—not their identity. **It does not clone voices or faces.** Use only source material you own or are authorized to use. Speech uses built-in synthetic voices or voices registered with the speaker's consent.

## Start

```text
python start.py            # starts the Web UI on http://127.0.0.1:8780
python start.py doctor     # checks the environment
python start.py setup      # creates or updates the environment
```

`start.py` can start ComfyUI when `comfyui.autostart` is enabled. A ComfyUI instance that was already running is left untouched; an instance started by ASMRAI is stopped when ASMRAI exits. Run `python start.py comfyui` to start only ComfyUI.

The launcher supports Python 3.10+. It looks for `.venv` first and then for the sibling `../SeriesAI/.venv`; it never uses a global Python installation. When neither environment exists, it creates `.venv` and installs the base and optional AI dependencies. Python 3.10–3.12 is recommended for the optional AI packages. FFmpeg and all caches are kept under `runtime/`.

## Web UI workflow

1. Add a creator and a folder containing videos, then scan it. Each creator has a separate profile.
2. Start the resumable analysis job: `probe → features → visual → transcribe → segment → profile`. Completed stages are skipped unless a full recalculation is requested.
3. Review the style profile: class shares, timing, loudness, brightness, onset rate, transitions, speech rate, and edit rate.
4. Configure a session in the generator, create a plan, render audio, and optionally render video.
5. Register consented voices and references for original or AI-created characters.
6. Use Jobs to inspect progress and logs. Different job types can run in parallel; jobs of the same type run sequentially.

The CLI is available as well:

```text
python start.py analyze <creator> [--model medium] [--limit 3]
python start.py session --creator <slug> --minutes 10 --video
```

## How it works

| Stage | Implementation |
|---|---|
| Audio features | FFmpeg streaming in 0.5-second windows: level, crest factor, spectral centroid and flatness, onset rate, band shares, pan, and stereo width |
| Classes | Relative, rule-based classification: `silence`, `speech`, `tapping`, `brushing`, `breath`, and `texture` |
| Speech | Faster-Whisper without Silero VAD, with level adjustment and hallucination filtering |
| Video | FFmpeg scene detection, preview frames, and brightness extraction |
| Planning | Markov transitions and duration distributions derived from the profile |
| Sound | Procedural tapping, brushing, scratching, breath, and texture synthesis |
| Spatial audio | Head-related timing and shadow filtering with controllable movement paths |
| Voice | XTTS-v2 using built-in or explicitly consented voices |
| Video output | FFmpeg ambient visuals or optional ComfyUI/LTX clips using an original character |

## Limits

- The audio classes are signal estimates rather than a trained classifier.
- Synthetic triggers are clean and controllable, but do not replace real object recordings.
- The source audio is suitable for learning statistics, not fine-tuning recording quality.
- Video LoRA training is not realistic on a 6 GB GPU; existing local video weights are used instead.
- ComfyUI rendering, Whisper transcription, and other AI workloads share the same local GPU.

## Data and security

- Source videos are never modified. Deleting analysis data removes only database records and generated work files.
- The server binds to `127.0.0.1` and validates `Host` and `Origin` headers.
- Voices require an owner and consent. Creator-archive recordings cannot be selected as voice references.

## Layout

```text
config.yaml            paths, models, and ports
start.py               launcher
src/asmrai/            application, pipeline, analysis, synthesis, video, Web UI, and CLI
src/asmrai/static/     single-page Web UI without a build step
data/                  local database, work files, sessions, voices, and characters (not versioned)
tests/                 pytest suite
```
