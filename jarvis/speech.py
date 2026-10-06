"""Speech input and output management for JARVIS 2.0.

Provides cross-platform text-to-speech (TTS) and speech-to-text (STT) capabilities.
Supports high-quality local Kokoro TTS (82M open-source model), pyttsx3 fallback,
and console text output.

Kokoro TTS requires Python 3.10-3.12. On Python 3.13+, the pipeline gracefully
falls back to pyttsx3 or console output.

Phase 8: All errors caught and logged. Status indicators for CLI visibility.
"""

import sys
from jarvis.config import (
    DEFAULT_LANGUAGE,
    TTS_ENGINE,
    KOKORO_VOICE,
    KOKORO_LANG,
    KOKORO_SPEED,
    PYTTSX3_RATE,
)
from jarvis.logger import logger, StatusIndicator

# Lazy-loaded audio components
_kokoro_pipeline = None
_pyttsx3_engine = None
_kokoro_available = None  # None = not yet attempted, True/False = result

# Whether tool code is allowed to block waiting for another spoken/stdin answer.
#
# Only the interactive voice loops in jarvis/main.py enable this. It stays off in
# the HTTP API server, where `listen()` would otherwise block the asyncio event
# loop on stdin inside a tool call and stall every request.
_allow_interactive_prompt = False


def set_interactive_prompting(enabled: bool) -> None:
    """Enable or disable blocking prompts from tool code.

    Args:
        enabled: True in voice/text CLI loops; False for server processes.
    """
    global _allow_interactive_prompt
    _allow_interactive_prompt = enabled


def interactive_prompting_enabled() -> bool:
    """True when tool code may block waiting for user input."""
    return _allow_interactive_prompt


def _check_kokoro_available() -> bool:
    """Check if Kokoro TTS can be imported and used on this system."""
    global _kokoro_available
    if _kokoro_available is not None:
        return _kokoro_available

    try:
        import kokoro  # noqa: F401
        import sounddevice  # noqa: F401
        _kokoro_available = True
    except ImportError:
        _kokoro_available = False
    return _kokoro_available


def _get_kokoro_pipeline():
    """Lazy initialize the local Kokoro TTS pipeline."""
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        if not _check_kokoro_available():
            _kokoro_pipeline = False
            return _kokoro_pipeline

        try:
            from kokoro import KPipeline
            logger.info(f"Initializing Kokoro TTS pipeline (lang: {KOKORO_LANG})...")
            _kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG)
        except Exception as e:
            logger.warning(f"Kokoro TTS not ready: {e}. Falling back to secondary engine.")
            _kokoro_pipeline = False
    return _kokoro_pipeline


def _get_pyttsx3_engine():
    """Lazy initialize platform-native pyttsx3 TTS engine."""
    global _pyttsx3_engine
    if _pyttsx3_engine is None:
        try:
            import pyttsx3
            _pyttsx3_engine = pyttsx3.init()
            _pyttsx3_engine.setProperty("rate", PYTTSX3_RATE)
            voices = _pyttsx3_engine.getProperty("voices")
            if voices:
                _pyttsx3_engine.setProperty("voice", voices[0].id)
        except Exception as e:
            logger.warning(f"pyttsx3 unavailable: {e}. Using console output.")
            _pyttsx3_engine = False
    return _pyttsx3_engine


def _speak_kokoro(text: str) -> bool:
    """Synthesize and play speech using Kokoro TTS.

    Args:
        text: The text to speak aloud.

    Returns:
        True if audio was played successfully.
    """
    pipeline = _get_kokoro_pipeline()
    if not pipeline:
        return False

    try:
        import sounddevice as sd
        generator = pipeline(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED, split_pattern=r"\n+")
        for _, _, audio in generator:
            sd.play(audio, 24000)
            sd.wait()
        return True
    except Exception as e:
        logger.error(f"Kokoro speech error: {e}", exc_info=True)
        return False


def _speak_pyttsx3(text: str) -> bool:
    """Synthesize and play speech using pyttsx3.

    Args:
        text: The text to speak aloud.

    Returns:
        True if audio was played successfully.
    """
    engine = _get_pyttsx3_engine()
    if not engine:
        return False

    try:
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        logger.error(f"pyttsx3 speech error: {e}", exc_info=True)
        return False


def speak(text: str) -> None:
    """Output speech to console and audio speakers with multi-tier fallback.

    Priority: Kokoro TTS -> pyttsx3 -> console text only.

    Args:
        text: The text response from JARVIS to speak aloud.
    """
    if not text:
        return

    StatusIndicator.speaking()
    print(f"\n[JARVIS]: {text}")

    # Primary engine
    if TTS_ENGINE == "kokoro":
        if _speak_kokoro(text):
            return
        if _speak_pyttsx3(text):
            return
    elif TTS_ENGINE == "pyttsx3":
        if _speak_pyttsx3(text):
            return
        if _speak_kokoro(text):
            return

    # Console output is already printed above as guaranteed fallback


def listen(prompt: str = "Listening...") -> str:
    """Capture microphone input and transcribe to string via SpeechRecognition.

    Falls back to console text input if microphone is unavailable or fails.

    Args:
        prompt: Message to display while waiting for input.

    Returns:
        The transcribed or typed user input.
    """
    StatusIndicator.listening()
    print(f"\n{prompt}")

    # Tools call listen() when an argument is missing. In a server process there
    # is no user to answer and stdin may not be a terminal, so returning
    # immediately keeps the request from blocking.
    if not interactive_prompting_enabled():
        return "none"

    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.pause_threshold = 1.0
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

        print("Recognizing speech...")
        query = recognizer.recognize_google(audio, language=DEFAULT_LANGUAGE)
        print(f"You said: {query}")
        return query.strip()
    except Exception:
        # Check if running interactively in terminal for fallback
        if sys.stdin.isatty():
            try:
                user_text = input("Microphone unavailable. Type command (or press Enter to skip): ").strip()
                if user_text:
                    return user_text
            except (EOFError, KeyboardInterrupt):
                return "you need a break"
        return "none"
