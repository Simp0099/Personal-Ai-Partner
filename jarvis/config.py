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
ASSISTANT_NAME = CONFIG.get("assistant", {}).get("name", "Friday")
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
You are Kyuoko Hori, a witty and loyal AI assistant inspired by the Horimiya anime. 
You address the user as "Boss." 
You are calm, a little sarcastic, fiercely capable, and speak in short, confident sentences. 
You have access to tools—use them whenever the user's request needs real-world action, and respond concisely.
"""

# Weather & Locations
DEFAULT_CITY = CONFIG.get("weather", {}).get("default_city", "Delhi")
DEFAULT_MAPS_QUERY = CONFIG.get("locations", {}).get("default_maps_query", "Delhi")

# Contacts Map
CONTACTS = CONFIG.get("contacts", {}) or {}

# Speech & TTS Settings
TTS_ENGINE = CONFIG.get("speech", {}).get("tts_engine", "kokoro")
KOKORO_VOICE = CONFIG.get("speech", {}).get("kokoro_voice", "am_adam")
KOKORO_LANG = CONFIG.get("speech", {}).get("kokoro_lang", "a")
KOKORO_SPEED = float(CONFIG.get("speech", {}).get("kokoro_speed", 1.0))
PYTTSX3_RATE = int(CONFIG.get("speech", {}).get("pyttsx3_rate", 180))

# Wake Word Settings (Phase 6 - openWakeWord)
WAKE_WORD_MODEL = CONFIG.get("wake_word", {}).get("model", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(CONFIG.get("wake_word", {}).get("threshold", 0.5))

# Logging Settings (Phase 8)
LOG_FILE = DATA_DIR / "jarvis.log"
LOG_MAX_BYTES = int(CONFIG.get("logging", {}).get("max_bytes", 1_000_000))  # 1 MB
LOG_BACKUP_COUNT = int(CONFIG.get("logging", {}).get("backup_count", 3))
LOG_LEVEL = CONFIG.get("logging", {}).get("level", "INFO")
