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
