# JARVIS 2.0 — Local AI Assistant

Welcome to **JARVIS 2.0**, an AI companion upgraded from a monolithic keyword-matching script into an intelligent, modular voice assistant.

## Architecture Roadmap
- [x] **Phase 1: Foundation & Security Fixes**
  - Modular package structure (`jarvis/`, `jarvis/tools/`)
  - Zero hardcoded secrets (`.env` and `python-dotenv`)
  - Cross-platform file paths and execution via `pathlib` and `subprocess`
  - Cross-platform speech synthesis & recognition (Chatterbox TTS / pyttsx3)
- [x] **Phase 2: The Brain Swap** (Google Gemini `gemini-3.5-flash` + autonomous tool calling via Chat API)
- [x] **Phase 3: Memory System** (Session memory via Chat API + SQLite durable long-term storage)
- [x] **Phase 4: Character & Personality** (Kyuoko Hori persona from Horimiya — system prompt engineering)
- [x] **Phase 5: Voice Upgrade** (Chatterbox TTS local neural voice, cloned from `Model/chatterbox_emotion_test.wav`)
- [x] **Phase 6: Wake Word** (openWakeWord local offline detection — "hey jarvis")
- [x] **Phase 7: Resource Efficiency** (History capped at 20, idle wake-word listener, no local LLM)
- [x] **Phase 8: Production Polish** (Rotating logging, robust error handling, centralized config, CLI status indicators)
- [x] **Phase 9: Final Pre-Demo Checklist** (Security audit, integration tests, memory persistence, unified startup)

- [x] **Phase 0: Functionality & Reliability Audit** (see `PHASE0_DIAGNOSIS.md`)
- [x] **Phase 0.5: Provider-Agnostic Model Layer** (see below)
- [x] **Phase 3: Vision, Webcam & Perception** (image vision + opt-in webcam perception)
- [x] **Phase 4: Wake Word, Conversation State & Interruption** (see below)

## Model Layer (Phase 0.5)

AI Partner is no longer tied to one model or provider. Conversation history is
owned by the Brain in a neutral format; the model is chosen per request.

```text
user message ──> classify  (jarvis/classify.py)
            ──> route     (jarvis/router.py)   ── ranked fallback chain
            ──> provider session (jarvis/providers/) ──> Gemini | OpenAI-compatible
```

- **Providers** are configured, not coded. `config.yaml` declares them and names
  the environment variable holding each key. Adding a provider is a config edit.
- **Models** live in the `models:` block with a capability profile, priority,
  free/paid flag, and a `verified` flag recording whether its id, context length
  and tool support were actually checked against the provider.
- **Routing** is deterministic weighted scoring over capability fit, task fit,
  measured reliability, measured latency and configured priority. Models that
  cannot satisfy a request are *excluded*, not merely scored lower — in
  particular a tool-required request never reaches a model that cannot call
  tools.
- **Health** is tracked in memory: success/failure counts, latency, and
  exponential-backoff cooldown after retryable failures. Auth and bad-request
  errors deliberately do **not** trigger cooldown, so misconfiguration stays
  visible instead of being retried forever.
- Models are tried **sequentially**, never raced.

Inspect the live pool with:

```bash
curl http://localhost:8000/api/models | python3 -m json.tool
```

Supported out of the box: Google Gemini, OpenRouter (free Nemotron / Ling
models), and OpenCode Zen (community free models such as Space Bunny and Big
Pickle). See `.env.example` for the optional keys.

## Vision & Perception (Phase 3)

Two visual inputs, one perception layer:

- **Images you attach** — screenshots, photos, multi-image comparisons. Always
  available, nothing to enable.
- **The webcam** — quiet, opt-in background perception. **Off by default.**

### The webcam is off until you turn it on

`config.yaml`:

```yaml
vision:
  enabled: true
  webcam:
    enabled: false        # <- the camera never opens until this is true
```

With it off, nothing about the camera runs: no device is opened, no thread is
started, no log line is emitted. `python3 start.py --status` always prints the
camera state, so what it is doing is never a guess.

### How it stays cheap and quiet

```text
camera  ->  sample every Ns  ->  local change detection  ->  meaningful?
             (configurable)        (Pillow, no model)          no -> discard
                                                                   |
                                                                   v
                                              vision model, at most once per cooldown
                                                                   |
                                                          observation / inference
                                                                   |
                                                     current visual context (ephemeral)
```

- **Frames are sampled, not streamed.** `interval_seconds` (10s) is how often a
  frame is even looked at.
- **Change detection is local.** `jarvis/vision_change.py` reduces a frame to a
  32x32 mean-centred greyscale signature and scores the *fraction of cells that
  materially changed*. No model call, no network. Measured: a person arriving
  scores 0.054, sensor noise and lighting drift both score 0.000.
- **A cooldown caps the cost.** `analysis_cooldown_seconds` (120s) is a hard
  ceiling on vision calls no matter how much the scene changes.
- **Perception is silent.** A camera event updates internal context and says
  nothing. Ask "what am I doing right now?" and the answer uses it.
- **Perception cannot write memory.** Webcam analysis runs as an *ephemeral* turn:
  no tools, no conversation history, and nothing recorded. Only an explicit
  "remember this" writes to long-term memory, and that uses the existing memory
  system — there is no second store.

### What it will not do

No facial recognition or identity inference, no emotion or health claims, no
surveillance, no recording, no image archive. A perception turn carries no tools
at all, so the camera cannot send mail, open a browser, or store anything.

### Verify it

```bash
python3 scripts/phase3_live_check.py   # real camera if present, real model
```

## Voice & Conversation (Phase 4)

Wake -> listen -> think -> speak -> follow-up -> interrupt, over **one**
microphone.

```text
IDLE --wake--> LISTENING --speech end--> TRANSCRIBING --transcript--> THINKING
      --> SPEAKING --done--> FOLLOW_UP --speech--> LISTENING | --timeout--> IDLE
                                          --speech while busy--> INTERRUPTED
```

### Off by default, like the webcam

```yaml
wake_word:
  enabled: false
```

While `enabled` is false the microphone is never opened, the wake-word model is
never loaded, and text conversation is unaffected. `python3 start.py --status`
always prints the voice state, so it is never a guess.

### One microphone

`jarvis/audio.py` owns the input device. The wake word, the voice-activity
detector and speech recognition all consume the *same* frames as sinks on one
stream. Previously the wake-word listener held a PyAudio stream while
`speech.listen()` opened a second one through `speech_recognition.Microphone`.

### State is the backend's

`GET /api/state` is authoritative; the HUD renders it and never infers state
from whether a reply arrived. `POST /api/voice/interrupt` stops a reply mid
sentence. The frontend no longer keeps a local timer that guessed "speaking".

### Interruption

Every turn has an id (`turn_001`). Audio is tagged with the turn that produced
it, and a barge-in marks the old turn stale *before* the new one starts, so
queued audio and a slow model reply are both discarded rather than played late.
Cancellation is cooperative: nothing is killed inside a native call, so
shutdown stays race-free.

### Making barge-in possible without waking yourself

Three things, each added because measurement showed it was missing:

- **Adaptive noise floor.** Speech is judged against the room's own measured
  level, not an absolute threshold. Ambient RMS measured 0.046-0.14 on the
  development machine — above any threshold tuned in a quiet room, so the
  assistant heard permanent speech and interrupted itself.
- **Echo cancellation.** Everything played is fed to a canceller that subtracts
  the far-end reference from the microphone signal, searching a small lag
  window for the output latency.
- **Echo-gated wake word.** The wake word is suppressed while the assistant's
  audio is in the air. The VAD keeps running underneath, so barge-in stays
  possible.

### Verify it

```bash
python3 scripts/phase4_live_check.py   # real mic, real speakers, real model
```

Checks needing a human to speak are reported SKIPPED rather than assumed.

## Getting Started

### 1. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Secrets
Copy the environment template and populate your keys:
```bash
cp .env.example .env
```
Open `.env` and fill in:
- `NASA_API_KEY`: Get a free key at [api.nasa.gov](https://api.nasa.gov/) or use `DEMO_KEY`
- `GMAIL_ADDRESS` & `GMAIL_APP_PASSWORD`: [Google App Password](https://support.google.com/accounts/answer/185833) if using email sending

### 3. Run JARVIS

**Voice Mode** (default):
```bash
python3 -m jarvis.main
```

**Text Mode** (for testing tool calling without voice):
```bash
python3 -m jarvis.main --text
```

**Verification Test** (automated tool-calling checks):
```bash
python3 -m jarvis.main --test
```
