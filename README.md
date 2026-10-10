# Ai Partner 2.0

An evolving terminal-first personal AI assistant inspired by JARVIS —
voice interaction, wake-word activation, multimodal perception, memory,
tools, and automation, in one terminal process.

## Status

Under active development. Honest snapshot of what exists today:

- Core architecture: implemented (`jarvis/`)
- Terminal interface: active — `python -m jarvis`
- Wake-word mode: implemented (openWakeWord, `hey_jarvis`), enabled by config
- Voice pipeline: implemented — shared-mic audio/VAD, STT, Kokoro TTS
- ASR backend: set `speech.asr_engine: "google"` to explicitly permit remote
  recognition; `local` remains unavailable until a local recognizer is configured
- AI routing: implemented — Gemini + OpenRouter + OpenCode Zen, ranked fallback
- Vision/perception: implemented — attached images always; webcam opt-in
- Memory: implemented — session context + SQLite long-term facts
- Tools/IoT foundation: implemented (system, web, media, email, weather, …)
- Test suite: see the current baseline and verification reports
- Dedicated graphical interface: intentionally deferred (terminal is the UI)

Live microphone, speaker, camera, and live-model paths are implemented in code
but not all exercised in every environment — see `scripts/*_live_check.py`.

The previous localhost React/FastAPI interface has been removed. A new
interface will be designed later as a separate presentation layer.

## Architecture

```text
Wake Word
   ↓
Audio / VAD  (one shared microphone stream)
   ↓
Speech Recognition
   ↓
AI / Reasoning  (router → Gemini / OpenRouter / Zen → tool loop)
   ↓
Memory / Vision / Tools
   ↓
Text-to-Speech  (Kokoro)
   ↓
Speaker
```

The terminal is currently the interface. No browser, no localhost server,
no second process: `python -m jarvis` runs the whole assistant.

## Features

### Voice
- Wake-word activation (`hey_jarvis`, local/offline via openWakeWord)
- Single-microphone pipeline: wake word, VAD, and STT share one stream
- Barge-in/interruption handling with echo suppression
- Kokoro local neural TTS (configured default; requires the optional Kokoro
  package and local model assets)
- `say` and Chatterbox remain selectable through `speech.tts_engine`

### Intelligence
- Provider-agnostic model layer (`jarvis/providers/`, `jarvis/model_layer.py`)
- Deterministic ranked routing with capability filtering and health cooldowns
- Bounded multi-step tool calling; honest errors instead of invented replies

### Vision
- Attached-image understanding (screenshots, photos, comparisons)
- Opt-in webcam perception: sampled frames, local change detection, cooldown-capped
- Ephemeral visual context — the camera never writes to memory

### Memory
- Short-term conversation history (capped) + persistent SQLite facts
- Explicit "remember this" writes; proactive/memory boundaries enforced

### Automation
- Tool registry: system status, files, web search, Wikipedia, screenshots,
  email, weather, NASA, jokes, dictionary, and more (`jarvis/tools/`)

## Quick Start

```bash
git clone <repository-url>
cd <repository>

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Then configure at least one enabled provider in `.env` (or export its
credential in the environment). Gemini, OpenRouter, and OpenCode Zen keys are
checked against enabled models; an unused provider key is not required. See
`.env.example`.

Run:

```bash
python -m jarvis            # wake-word mode (default)
python -m jarvis --text     # text mode, no microphone
python -m jarvis --no-wake  # voice mode without wake word
python -m jarvis --status   # system status without starting
python -m jarvis --test     # integration checks
```

Voice and webcam are opt-in via `config.yaml`
(`wake_word.enabled`, `vision.webcam.enabled`). With voice disabled, the
default startup enters typed mode and does not open a microphone. `--text`
always stays typed; `--no-wake` explicitly selects voice input without a wake
phrase.

## Configuration

All behaviour lives in `config.yaml`; secrets live only in `.env`
(never committed). Key knobs: model registry and routing weights, TTS voice
and expressiveness, VAD thresholds, conversation follow-up window,
conversation-state decay, proactive gating, and vision sampling/cooldowns.
Directory listing is restricted to the resolved `file_access.allowed_roots`
(defaults: the project and its `data` directory); symlink targets outside
those roots are hidden.

## Tests

```bash
python -m pytest tests/
```

Offline and deterministic (mock providers); live hardware/model checks live
separately in `scripts/phase*_live_check.py` and `scripts/tts_smoke_test.py`.

## Layout

```text
jarvis/    core package (brain, voice, vision, memory, tools, providers)
tests/     offline test suite
scripts/   live hardware/model diagnostics
docs/      project history (PHASE0_DIAGNOSIS.md)
config.yaml            behaviour (no secrets)
requirements.txt       Python dependencies (3.12)
start.py               startup checks + mode dispatch (via python -m jarvis)
data/                  runtime state only (git-ignored): memory DB, screenshots
```
