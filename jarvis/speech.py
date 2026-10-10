"""Speech input and output management for JARVIS 2.0.

Provides text-to-speech (TTS) and speech-to-text (STT) capabilities. Kokoro is
the configured local Kokoro engine.
Console output remains the guaranteed fallback.

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
from jarvis.trace import span_of

# Lazy-loaded audio components
_pyttsx3_engine = None


class TTSEngineError(RuntimeError):
    """Raised when a TTS engine is misconfigured, missing, or cannot synthesize.

    Carries an actionable message so the failure is never a silent voice swap.
    """

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
    """Speak through the local Kokoro engine."""
    try:
        from jarvis import tts
        from jarvis.speech_pipeline import _sounddevice_play_array
        _sounddevice_play_array(tts.synthesize(text), tts.KOKORO_RATE)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("Kokoro speech error: %s", e, exc_info=True)
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


def synthesize_for_engine(text: str, exaggeration=None):
    """Compatibility entry point for the configured speech engine."""
    if TTS_ENGINE == "kokoro":
        from jarvis import tts
        audio = tts.synthesize(text, exaggeration=exaggeration)
    elif TTS_ENGINE == "pyttsx3":
        raise TTSEngineError("pyttsx3 does not provide cancellable waveform synthesis.")
    else:
        raise TTSEngineError(f"Unsupported TTS engine: {TTS_ENGINE!r}.")
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1.0:
        raise TTSEngineError("Kokoro returned invalid or empty audio.")
    return audio


def speak(text: str) -> None:
    """Output speech to console and audio speakers.

    Priority: the configured engine -> explicitly supported alternative, if
    any -> console text only.

    Args:
        text: The text response from JARVIS to speak aloud.
    """
    if not text:
        return

    logger.info("[TTS] legacy speech request engine=%s", TTS_ENGINE)
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
