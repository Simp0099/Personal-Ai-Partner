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

# Wake Word Settings (Phase 6 - openWakeWord)
WAKE_WORD_MODEL = CONFIG.get("wake_word", {}).get("model", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(CONFIG.get("wake_word", {}).get("threshold", 0.5))

# Logging Settings (Phase 8)
LOG_FILE = DATA_DIR / "jarvis.log"
LOG_MAX_BYTES = int(CONFIG.get("logging", {}).get("max_bytes", 1_000_000))  # 1 MB
LOG_BACKUP_COUNT = int(CONFIG.get("logging", {}).get("backup_count", 3))
LOG_LEVEL = CONFIG.get("logging", {}).get("level", "INFO")
