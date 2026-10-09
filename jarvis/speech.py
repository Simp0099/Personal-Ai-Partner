"""Speech input and output management for JARVIS 2.0.

Provides cross-platform text-to-speech (TTS) and speech-to-text (STT) capabilities.
The active TTS engine is Chatterbox: the model loads once, the reference voice
(`chatterbox_emotion_test.wav`, resolved relative to the project root) is
prepared once into reusable conditionals, and every synthesis call reuses them.
pyttsx3 and console output remain available as explicit, configured fallbacks.

Kokoro is retained as a legacy engine only (speech.tts_engine: "kokoro").

Phase 8: All errors caught and logged. Status indicators for CLI visibility.
"""

import sys
import threading
from pathlib import Path

from jarvis.config import (
    DEFAULT_LANGUAGE,
    TTS_ENGINE,
    PROJECT_ROOT,
    CHATTERBOX_REFERENCE_AUDIO,
    CHATTERBOX_DEVICE,
    CHATTERBOX_EXAGGERATION,
    CHATTERBOX_CFG_WEIGHT,
    CHATTERBOX_TEMPERATURE,
    CHATTERBOX_FALLBACK_ENGINE,
    KOKORO_VOICE,
    KOKORO_LANG,
    KOKORO_SPEED,
    PYTTSX3_RATE,
    SAY_VOICE,
)
from jarvis.logger import logger, StatusIndicator
from jarvis.trace import span_of

# Lazy-loaded audio components
_chatterbox_model = None  # loaded once, reused for every request
_chatterbox_lock = threading.RLock()  # serializes init + inference
_chatterbox_conds_key = None  # str(ref path) whose conditionals are prepared
_chatterbox_warmed = False  # one controlled warm-up per process
_CHATTERBOX_WARMUP_TEXT = "Yes, Boss?"
_pyttsx3_engine = None
_kokoro_pipeline = None
_kokoro_available = None  # None = not yet attempted, True/False = result


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


# ---------------------------------------------------------------------------
# Chatterbox TTS (active engine)
# ---------------------------------------------------------------------------


def _resolve_chatterbox_reference() -> Path:
    """Resolve the Chatterbox reference WAV against the project root.

    Raises:
        TTSEngineError: if the configured reference audio does not exist.
    """
    path = Path(CHATTERBOX_REFERENCE_AUDIO).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise TTSEngineError(
            f"Chatterbox reference audio not found: {CHATTERBOX_REFERENCE_AUDIO} "
            f"(resolved to {path}). Add the file or set "
            "speech.chatterbox_reference_audio in config.yaml."
        )
    return path


def _chatterbox_device() -> str:
    """Resolve the torch device string for Chatterbox."""
    if CHATTERBOX_DEVICE != "auto":
        return CHATTERBOX_DEVICE
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_chatterbox_model():
    """Load the Chatterbox model once and reuse it for every request.

    The model itself (weights + tokenizer) is the expensive part, so it is cached
    at module scope under a lock (double-checked). Reference-voice conditionals
    are prepared separately by `prepare_chatterbox_reference`, not here.

    Returns:
        A ready-to-use `ChatterboxTTS` instance.

    Raises:
        TTSEngineError: if Chatterbox is not installed, the model fails to
            initialize, or the reference audio is missing.
    """
    global _chatterbox_model
    with _chatterbox_lock:
        if _chatterbox_model is not None:
            return _chatterbox_model

        # Resolve (and validate) the reference voice before spending time on weights.
        reference = _resolve_chatterbox_reference()

        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError as e:
            raise TTSEngineError(
                "Chatterbox TTS is not installed. Install it with "
                "`pip install chatterbox-tts` in the project environment."
            ) from e

        device = _chatterbox_device()
        logger.info(
            f"Initializing Chatterbox TTS (device: {device}, reference: {reference.name})..."
        )
        try:
            model = ChatterboxTTS.from_pretrained(device=device)
        except Exception as e:
            raise TTSEngineError(f"Chatterbox model failed to initialize: {e}") from e

        _chatterbox_model = model
        return model


def prepare_chatterbox_reference(model=None, exaggeration=None) -> float:
    """Prepare voice conditionals once per reference file; return elapsed ms.

    The installed API (`prepare_conditionals`) does the expensive deterministic
    work -- wav load, resample, embeddings. After this, `generate()` runs
    without `audio_prompt_path` and reuses `model.conds`. Per-call
    exaggeration differences are handled by the library's cheap emotion-only
    update inside `generate()`. Returns 0.0 when the cache is reused.
    Falls back to per-call preparation when the model predates the API.
    """
    reference = _resolve_chatterbox_reference()
    model = model if model is not None else _get_chatterbox_model()
    key = str(reference)
    with _chatterbox_lock:
        global _chatterbox_conds_key
        if _chatterbox_conds_key == key:
            return 0.0
        prepare = getattr(model, "prepare_conditionals", None)
        if prepare is None:
            logger.debug("Chatterbox model lacks prepare_conditionals; "
                         "reference will be passed per request.")
            return 0.0
        import time
        start = time.perf_counter_ns()
        prepare(str(reference),
                exaggeration=(CHATTERBOX_EXAGGERATION if exaggeration is None
                              else float(exaggeration)))
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        _chatterbox_conds_key = key
        logger.info(f"Chatterbox reference prepared ({reference.name}) "
                    f"in {elapsed_ms:.0f} ms.")
        return elapsed_ms


def warm_up_chatterbox(model=None) -> float:
    """One controlled warm-up through the real inference path; ms, 0.0 if done.

    Lazy like the rest of TTS: the cost lands on first use, where Phase 5
    measured no latency benefit -- this exercises the path and surfaces
    failures early, it does not claim a speedup. Best-effort: a failed
    warm-up is logged and retried next call, never blocking real synthesis.
    """
    with _chatterbox_lock:
        global _chatterbox_warmed
        if _chatterbox_warmed:
            return 0.0
        model = model if model is not None else _get_chatterbox_model()
        import time
        start = time.perf_counter_ns()
        try:
            with _inference_context():
                model.generate(
                    _CHATTERBOX_WARMUP_TEXT,
                    exaggeration=CHATTERBOX_EXAGGERATION,
                    cfg_weight=CHATTERBOX_CFG_WEIGHT,
                    temperature=CHATTERBOX_TEMPERATURE,
                )
        except Exception as e:  # noqa: BLE001 -- synthesis still attempted below
            logger.warning(f"Chatterbox warm-up failed, will retry: {e}")
            return 0.0
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        _chatterbox_warmed = True
        logger.info(f"Chatterbox warm-up completed in {elapsed_ms:.0f} ms.")
        return elapsed_ms


def ensure_chatterbox_ready() -> object:
    """Load model, prepare reference, warm up -- each exactly once."""
    model = _get_chatterbox_model()
    prepare_chatterbox_reference(model)
    warm_up_chatterbox(model)
    return model


def _inference_context():
    """Inference-mode context where torch exists, else a no-op.

    The installed Chatterbox already runs its inference under
    `torch.inference_mode()`; this guards our boundary too without
    making torch an import-time dependency of the speech module.
    """
    try:
        import torch
    except ImportError:
        import contextlib
        return contextlib.nullcontext()
    return torch.inference_mode()


def _synthesize_chatterbox(text: str, exaggeration=None):
    """Generate speech with Chatterbox, cloning the configured reference voice.

    `exaggeration` overrides `speech.chatterbox_exaggeration` for this one call.
    It is the only expressiveness control the engine exposes, so Phase 5 uses it
    as a hint and nothing more; omitting it keeps the configured default.

    The resolved reference path is prepared once per process (see
    `prepare_chatterbox_reference`); per-request `generate()` calls reuse the
    cached conditionals and only carry the lightweight synthesis parameters.

    Args:
        text: The text to synthesize.

    Returns:
        A 1-D numpy float32 array of mono samples.

    Raises:
        TTSEngineError: on any engine failure.
    """
    reference = _resolve_chatterbox_reference()
    with span_of("tts_ready"):
        model = ensure_chatterbox_ready()
    try:
        with _chatterbox_lock, _inference_context():
            with span_of("tts_generate"):
                wav = model.generate(
                    text,
                    exaggeration=(CHATTERBOX_EXAGGERATION if exaggeration is None
                                  else float(exaggeration)),
                    cfg_weight=CHATTERBOX_CFG_WEIGHT,
                    temperature=CHATTERBOX_TEMPERATURE,
                )
    except Exception as e:
        raise TTSEngineError(
            f"Chatterbox synthesis failed with reference {reference.name}: {e}"
        ) from e

    import numpy as np

    # Chatterbox returns a torch tensor shaped (1, samples).
    array = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _speak_chatterbox(text: str) -> bool:
    """Synthesize and play speech using Chatterbox.

    Returns:
        True if audio was played successfully.
    """
    try:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise TTSEngineError(
                "sounddevice is not installed. Install it with `pip install sounddevice`."
            ) from e

        audio = _synthesize_chatterbox(text)
        sample_rate = getattr(_get_chatterbox_model(), "sr", 24000)
        sd.play(audio, sample_rate)
        sd.wait()
        return True
    except TTSEngineError as e:
        # Explicit failure: never silently substitute another voice.
        logger.error(f"Chatterbox speech error: {e}")
        return False


def _check_kokoro_available() -> bool:
    """Check if Kokoro TTS can be imported and used on this system (legacy)."""
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
    """Synthesize and play speech using Kokoro TTS (legacy, not the default).

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


#: Sample rate the voice pipeline consumes (SpeechPlayer chunks at this rate).
SAY_TARGET_RATE = 16000


def _synthesize_say(text: str) -> "np.ndarray":
    """Render speech with macOS `say` into a 16 kHz mono float32 array.

    Fast local engine for interactive responses (~1 s per short reply vs
    ~1 min for Chatterbox). No model to load, no downloads, no network.
    The voice is the configured system voice, NOT the Chatterbox clone.
    Raises TTSEngineError when `say` is missing or rendering fails.
    """
    import os
    import subprocess
    import tempfile

    import numpy as np

    if not (text or "").strip():
        raise TTSEngineError("Nothing to speak.")
    fd, path = tempfile.mkstemp(prefix="jarvis-say-", suffix=".aiff")
    os.close(fd)
    try:
        # No persistent model: always ready (no tts_ready span by design).
        with span_of("tts_generate"):
            try:
                subprocess.run(
                    ["say", "-v", SAY_VOICE, "-o", path, text],
                    check=True, capture_output=True, timeout=120,
                )
            except FileNotFoundError as e:
                raise TTSEngineError(f"macOS `say` not found: {e}") from e
            except subprocess.CalledProcessError as e:
                raise TTSEngineError(f"`say` failed (rc={e.returncode})") from e
            try:
                import soundfile as sf
                data, rate = sf.read(path, dtype="float32", always_2d=True)
            except Exception as e:
                raise TTSEngineError(f"Could not read `say` output: {e}") from e
        mono = np.asarray(data, dtype=np.float32).mean(axis=1)
        if rate != SAY_TARGET_RATE:  # linear resample; measurement-grade
            idx = np.linspace(0, len(mono) - 1, int(len(mono) * SAY_TARGET_RATE / rate))
            mono = np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)
        return np.asarray(mono, dtype=np.float32).reshape(-1)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def synthesize_for_engine(text: str, exaggeration=None):
    """Synthesize with the configured engine for array-returning pipelines.

    `say` is rendered locally; anything else keeps the existing Chatterbox
    path (today's effective voice-mode behavior for legacy engines).
    """
    if TTS_ENGINE == "say":
        return _synthesize_say(text)
    return _synthesize_chatterbox(text, exaggeration=exaggeration)


def _speak_say(text: str) -> bool:
    """Speak with macOS `say` through the speaker. Returns True on success."""
    try:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise TTSEngineError(
                "sounddevice is not installed. Install it with `pip install sounddevice`."
            ) from e

        audio = _synthesize_say(text)
        sd.play(audio, SAY_TARGET_RATE)
        sd.wait()
        return True
    except TTSEngineError as e:
        logger.error(f"say speech error: {e}")
        return False


def speak(text: str) -> None:
    """Output speech to console and audio speakers.

    Priority: the configured engine ("say" by default for interactive speed,
    "chatterbox" for the cloned voice when explicitly selected) -> the
    explicitly configured fallback -> console text only.
    Chatterbox never silently reverts to Kokoro.

    Args:
        text: The text response from JARVIS to speak aloud.
    """
    if not text:
        return

    StatusIndicator.speaking()
    print(f"\n[JARVIS]: {text}")

    # Primary engine
    if TTS_ENGINE == "say":
        if _speak_say(text):
            return
    elif TTS_ENGINE == "chatterbox":
        if _speak_chatterbox(text):
            return
        if CHATTERBOX_FALLBACK_ENGINE == "pyttsx3" and _speak_pyttsx3(text):
            return
    elif TTS_ENGINE == "kokoro":  # legacy path, opt-in only
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
