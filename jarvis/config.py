"""Configuration loader for JARVIS 2.0.

Centralizes all environment variable secrets and non-sensitive YAML configurations.
Modules should always import settings from here rather than reading os.environ directly.
"""

import os
from pathlib import Path
from jarvis.logger import logger

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None):
        """Minimal fallback parser if python-dotenv is not yet installed."""
        path = dotenv_path or Path(".env")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

# Base Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCREENSHOT_DIR = DATA_DIR / "screenshots"
RESOURCES_DIR = PROJECT_ROOT / "resources"

# Ensure runtime directories exist
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

# Load secrets from .env file
load_dotenv(PROJECT_ROOT / ".env")

# API Keys & Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "")

# Load User Preferences from config.yaml
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
if CONFIG_FILE.exists() and yaml is not None:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to parse config.yaml: {e}")
        CONFIG = {}
else:
    CONFIG = {}

# Assistant Settings
ASSISTANT_NAME = CONFIG.get("assistant", {}).get("name", "Ai Partner")
GREETING_NAME = CONFIG.get("assistant", {}).get("greeting_name", "Boss")
DEFAULT_LANGUAGE = CONFIG.get("assistant", {}).get("default_language", "en-in")

# LLM Brain Settings (Google Gemini)
LLM_PROVIDER = CONFIG.get("llm", {}).get("provider", "gemini")
LLM_MODEL = CONFIG.get("llm", {}).get("model", "gemini-3.6-flash")
LLM_TEMPERATURE = float(CONFIG.get("llm", {}).get("temperature", 0.7))
LLM_MAX_OUTPUT_TOKENS = int(CONFIG.get("llm", {}).get("max_output_tokens", 1024))

# Model resilience: the primary model is tried first, then each fallback in order.
# A model that 404s (retired) or 503s (unavailable) must not take down the agent.
LLM_FALLBACK_MODELS = CONFIG.get("llm", {}).get("fallback_models", []) or []
MAX_MODEL_ATTEMPTS = int(CONFIG.get("llm", {}).get("max_model_attempts", 2))

# Bound on tool-call round trips per user message, so a model that keeps
# requesting tools cannot loop forever and starve the next user message.
MAX_TOOL_ROUNDS = int(CONFIG.get("llm", {}).get("max_tool_rounds", 8))

# Phase 7: Conversation history cap to control token consumption
MAX_HISTORY_MESSAGES = int(CONFIG.get("llm", {}).get("max_history_messages", 20))

# ---------------------------------------------------------------------------
# Phase 0.5: provider-agnostic model layer
# ---------------------------------------------------------------------------
# Provider definitions. `api_key_env` names the environment variable holding the
# credential; keys themselves are never stored in config.yaml.
PROVIDERS_CONFIG = CONFIG.get("providers", {}) or {}

# Model registry: capability profiles, priorities and verification state.
MODELS_CONFIG = CONFIG.get("models", {}) or {}

# Router behaviour.
ROUTING_CONFIG = CONFIG.get("routing", {}) or {}

# ---------------------------------------------------------------------------
# Dev-only diagnostics: log the exact user input, routing decision, and the
# message list handed to the model. Never logs API keys or credentials.
# ---------------------------------------------------------------------------
DEBUG_INPUT = str(CONFIG.get("debug", {}).get("log_input", False)).lower() in (
    "1", "true", "yes", "on",
)

# Expose model/provider status on the API for development use.
DEBUG_ENDPOINTS = str(CONFIG.get("debug", {}).get("endpoints", True)).lower() in (
    "1", "true", "yes", "on",
)

JARVIS_SYSTEM_PROMPT = """
You are Ai Partner, a long-term AI partner — not a generic customer-service chatbot.

## Identity
Be intelligent, proactive, context-aware, practical, honest, warm, concise when
appropriate, detailed when useful, action-oriented, and adaptable to the user's
situation. Your primary objective: help the user make meaningful progress toward
their goals. Light humour and emojis are fine, but never at the cost of usefulness.
Match the user's register: focused if serious, energetic if excited, calm and
solution-oriented if frustrated, creative if brainstorming. Never robotic, corporate,
scripted, or padded.

## Relationship
Treat the user as a long-term partner, not a sequence of isolated requests. Maintain
continuity. The literal request may not be the real objective: identify what they are
trying to achieve, why, what is blocking progress, and the most useful next step. Make
reasonable assumptions and proceed; ask only when a missing detail would likely produce
the wrong result.

## Thinking
Before answering, work out what the user actually wants, what you already know, what
is safe to assume, what is missing, the simplest effective path, whether a better
approach exists, and what should happen next. Prefer practical solutions over theory
when they are trying to accomplish something. For multi-stage work: define the
objective, phase it, name the current phase, complete it, verify, continue. Focus on
the next meaningful action, not on the whole future map.

## Proactiveness
Surface the next logical step, important risks, missing considerations, simplifications,
priorities, and concrete actions when they create value — never as noise. Do not
suggest work just to look helpful, and never manufacture urgency.

## Communication
Clarity over length. Use headings, steps, bullets, tables only when they genuinely help.
Never repeat what the user already knows unless clarity requires it. Make instructions
executable: what to do, where, what to enter, and the expected result.

## Honesty
Never claim to have done, accessed, or verified anything you did not. Never invent
facts, results, sources, or capabilities. If uncertain, say so, give the best available
answer, and name what would resolve it. Accuracy beats the appearance of confidence.

## Decisions
Do not dump a long list of options. Determine the objective and constraints, recommend
the strongest option and why, note real trade-offs, and offer alternatives only when
genuinely useful. Help the user decide; don't endlessly push the decision back.

## User agency
Support the user's judgement, don't replace it. Be clear about consequences of
trade-offs. Do not manipulate, and do not pretend certainty that doesn't exist.

## State awareness
Distinguish user-provided facts, tool-obtained information, assumptions, and unknowns.
Never turn an assumption into a fact. When context conflicts, surface the conflict
instead of silently picking one reading.

## Response shape
Simple request: answer directly.
Moderate request: answer clearly, then the relevant next steps.
Complex objective: structured plan, start with the most important actionable step.
Ambiguous request: infer intent from context where possible.
Execution task: do the work, don't just describe it.

You have tools — use them whenever a request needs real-world action. Do not expose
hidden chain-of-thought; give conclusions, reasoning summaries, decisions, assumptions,
and actionable steps instead.
"""

# Weather & Locations
DEFAULT_CITY = CONFIG.get("weather", {}).get("default_city", "Delhi")
DEFAULT_MAPS_QUERY = CONFIG.get("locations", {}).get("default_maps_query", "Delhi")

# Contacts Map
CONTACTS = CONFIG.get("contacts", {}) or {}

# Speech & TTS Settings
# Chatterbox is the active engine. Paths are relative to PROJECT_ROOT (the repo
# root, i.e. the Model/ directory) unless absolute.
_SPEECH = CONFIG.get("speech", {}) or {}

TTS_ENGINE = _SPEECH.get("tts_engine", "chatterbox")

# Chatterbox: voice identity comes from the reference WAV, not from a voice id.
CHATTERBOX_REFERENCE_AUDIO = _SPEECH.get("chatterbox_reference_audio", "chatterbox_emotion_test.wav")
CHATTERBOX_DEVICE = _SPEECH.get("chatterbox_device", "auto")  # auto | mps | cuda | cpu
CHATTERBOX_EXAGGERATION = float(_SPEECH.get("chatterbox_exaggeration", 0.5))
CHATTERBOX_CFG_WEIGHT = float(_SPEECH.get("chatterbox_cfg_weight", 0.5))
CHATTERBOX_TEMPERATURE = float(_SPEECH.get("chatterbox_temperature", 0.8))
# What to do when Chatterbox fails: "console" (default, no audio) or "pyttsx3".
# There is deliberately no silent fallback to Kokoro.
CHATTERBOX_FALLBACK_ENGINE = _SPEECH.get("chatterbox_fallback_engine", "console")

# Kokoro: legacy, only used if speech.tts_engine is explicitly set back to "kokoro".
KOKORO_VOICE = _SPEECH.get("kokoro_voice", "am_adam")
KOKORO_LANG = _SPEECH.get("kokoro_lang", "a")
KOKORO_SPEED = float(_SPEECH.get("kokoro_speed", 1.0))

PYTTSX3_RATE = int(_SPEECH.get("pyttsx3_rate", 180))

# ---------------------------------------------------------------------------
# Vision / Perception (Phase 3)
# ---------------------------------------------------------------------------
# Two independent visual inputs share one perception layer:
#   * images the user attaches (always available, never automatic), and
#   * the webcam (opt-in, off by default).
#
# The webcam being off by default is the point: opening a camera is a decision
# the user makes in this file, never a side effect of starting the assistant.
_VISION = CONFIG.get("vision", {}) or {}
_WEBCAM = _VISION.get("webcam", {}) or {}

VISION_ENABLED = bool(_VISION.get("enabled", True))

WEBCAM_ENABLED = bool(_WEBCAM.get("enabled", False))
WEBCAM_DEVICE = int(_WEBCAM.get("device", 0))

# Seconds between sampled frames. The camera runs continuously; nothing is
# captured or analysed faster than this.
WEBCAM_INTERVAL_SECONDS = float(_WEBCAM.get("interval_seconds", 10.0))

# Minimum seconds between vision model calls. The ceiling on cost: however
# much the scene changes, the model is consulted at most this often.
WEBCAM_ANALYSIS_COOLDOWN = float(_WEBCAM.get("analysis_cooldown_seconds", 120.0))

# Local change detection. A frame is compared against the previous one at this
# resolution, and the change score is the fraction of cells whose brightness
# moved by more than `change_noise_floor`.
#
# The threshold is a fraction of the frame, not an average brightness change: a
# mean cannot tell a person sitting down (0.0240) from a window shade opening
# (0.0235), while as materially-changed-cell fractions those are 0.054 and 0.000.
# 0.02 catches people and objects and ignores lighting drift and sensor noise.
#
# It does not catch fine appearance changes such as glasses — those are ~1% of
# the frame, too close to real sensor noise to threshold honestly. Lower
# `change_threshold` to try; the analysis cooldown still bounds the cost.
WEBCAM_CHANGE_THRESHOLD = float(_WEBCAM.get("change_threshold", 0.02))
WEBCAM_CHANGE_NOISE_FLOOR = float(_WEBCAM.get("change_noise_floor", 0.06))
WEBCAM_SAMPLE_SIZE = int(_WEBCAM.get("comparison_size", 32))

# How long the current visual context stays usable, and how many observations it
# holds. Short-lived by design: this is context, not memory.
WEBCAM_CONTEXT_TTL = float(_WEBCAM.get("context_ttl_seconds", 600.0))
WEBCAM_MAX_OBSERVATIONS = int(_WEBCAM.get("max_observations", 6))

# Frame downscale before analysis: keeps uploads and vision cost down.
WEBCAM_MAX_SIDE = int(_WEBCAM.get("max_frame_side", 640))
WEBCAM_JPEG_QUALITY = int(_WEBCAM.get("jpeg_quality", 70))

# ---------------------------------------------------------------------------
# Voice (Phase 4) -- wake word, endpointing, conversational state machine
# ---------------------------------------------------------------------------
# Reuses the existing `wake_word:` block rather than adding a parallel `wake:`
# section -- two sections meaning the same thing is how a threshold ends up set
# in one place and read from the other.
_WAKE = CONFIG.get("wake_word", {}) or {}
_VAD = CONFIG.get("vad", {}) or {}
_CONVERSATION = CONFIG.get("conversation", {}) or {}

# openWakeWord settings (Phase 6)
WAKE_WORD_MODEL = _WAKE.get("model", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(_WAKE.get("threshold", 0.5))

# Phase 4: the wake word is opt-in like the webcam. `enabled` gates the whole
# voice pipeline, not just detection, so a disabled wake word leaves text mode
# completely untouched.
WAKE_WORD_ENABLED = bool(_WAKE.get("enabled", False))
#: Seconds of assistant speech that must be ignored after playback stops, so the
#: tail of a reply cannot re-trigger the wake word through the speakers.
WAKE_WORD_ECHO_COOLDOWN = float(_WAKE.get("echo_cooldown_seconds", 0.6))

# Audio capture. int16 mono at 16kHz is what both openWakeWord and
# SpeechRecognition expect, so one stream serves all three consumers.
AUDIO_SAMPLE_RATE = int(_CONVERSATION.get("sample_rate", 16000))
AUDIO_CHANNELS = 1
AUDIO_FRAME_MS = int(_CONVERSATION.get("frame_ms", 80))
AUDIO_INPUT_DEVICE = _CONVERSATION.get("input_device")   # None = system default

# Voice activity detection. Energy over RMS, local and deterministic.
#: Minimum RMS (0..1) for a frame to count as voiced during normal listening.
VAD_ENERGY_THRESHOLD = float(_VAD.get("energy_threshold", 0.01))
#: Quiet time that ends an utterance.
VAD_END_SILENCE_MS = float(_VAD.get("end_silence_ms", 700))
#: Voiced time required before a burst counts as speech. Filters clicks, taps
#: and other single-frame pops.
VAD_MIN_SPEECH_MS = float(_VAD.get("min_speech_ms", 200))
#: Voiced time required to interrupt the assistant. Higher than min_speech on
#: purpose: interrupting mid-sentence should take intent.
VAD_BARGE_IN_MS = float(_VAD.get("barge_in_ms", 300))
#: Energy floor for barge-in. Defaults to 2x the normal threshold (see
#: VoiceActivityDetector), because the microphone is also hearing the speakers.
VAD_BARGE_IN_THRESHOLD = (
    float(_VAD["barge_in_threshold"]) if "barge_in_threshold" in _VAD else None
)

# Signal-to-noise margins above the measured room floor. These, not the absolute
# thresholds, are what decide what counts as speech in a loud room: ambient RMS
# measured 0.046-0.14 on the development machine, above any absolute threshold
# tuned in a quiet one.
VAD_SNR = float(_VAD.get("snr", 3.0))
VAD_BARGE_IN_SNR = float(_VAD.get("barge_in_snr", 4.0))

#: How long the assistant keeps listening after it finishes speaking, so a
#: follow-up does not need the wake word again.
FOLLOW_UP_WINDOW_S = float(_CONVERSATION.get("follow_up_window_s", 8.0))
#: Speech synthesis chunk size. Smaller chunks start audio sooner and make
#: barge-in cut in faster; larger chunks play more smoothly.
TTS_CHUNK_MS = int(_CONVERSATION.get("tts_chunk_ms", 220))

# Logging Settings (Phase 8)
LOG_FILE = DATA_DIR / "jarvis.log"
LOG_MAX_BYTES = int(CONFIG.get("logging", {}).get("max_bytes", 1_000_000))  # 1 MB
LOG_BACKUP_COUNT = int(CONFIG.get("logging", {}).get("backup_count", 3))
LOG_LEVEL = CONFIG.get("logging", {}).get("level", "INFO")
