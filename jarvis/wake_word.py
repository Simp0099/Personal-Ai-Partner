"""JARVIS 2.0 Wake Word Detection using openWakeWord.

Provides continuous background audio stream listening for the "hey jarvis"
wake word. When detected with sufficient confidence, triggers the main
assistant loop.

Phase 7 optimizations:
  - No background LLM calls — only lightweight ONNX inference on audio chunks
  - Audio stream uses small chunks (1280 samples = 80ms) for low latency
  - No polling loops — blocking stream.read() keeps CPU usage near-zero
  - Daemon thread automatically terminates when main process exits

Phase 8 polish:
  - All errors caught and logged gracefully
  - Status indicators for CLI visibility

Phase 4:
  - `WakeWordEngine` is the reusable half: model loading, inference and teardown.
    `VoiceLoop` drives it over the *shared* microphone stream, and
    `WakeWordListener` keeps its own stream for the legacy single-purpose mode.
    One engine implementation, so the provider abstraction is not duplicated.
  - The listener now releases its audio stream on every exit path. It previously
    returned from the loop on wake without `stop_stream()/close()/terminate()`,
    leaving the device held for the rest of the process.

Requires: openwakeword, numpy

Capture arrives through the shared `jarvis.audio` microphone owner. This module
scores frames; it never opens a device of its own.
"""

from jarvis.config import (
    AUDIO_FRAME_MS, AUDIO_SAMPLE_RATE, WAKE_WORD_THRESHOLD, WAKE_WORD_MODEL,
)
from jarvis.logger import logger, StatusIndicator


def known_wake_models() -> set:
    """Built-in model names in the installed openWakeWord, or empty if absent."""
    try:
        import openwakeword
        return set(openwakeword.MODELS.keys())
    except Exception:  # noqa: BLE001 - validation is best-effort
        return set()


class WakeWordEngine:
    """The openWakeWord model: load once, score chunks, release cleanly.

    Deliberately free of any audio device. Callers feed it PCM from the shared stream.

    Phase 8: one engine serves every active phrase through a single
    `Model(wakeword_models=[...])` inference call -- no extra microphone
    stream per phrase. `phrases` is a list of {phrase, model, threshold}
    entries (see `WAKE_WORD_PHRASES`); omitted means the legacy single model.
    `score()`/`threshold`/`model` keep describing the primary phrase so
    existing single-phrase callers are untouched.
    """

    def __init__(self, model: str = None, threshold: float = None, phrases: list = None):
        if phrases is None:
            phrases = [{"phrase": "hey jarvis",
                        "model": model or WAKE_WORD_MODEL,
                        "threshold": (WAKE_WORD_THRESHOLD if threshold is None
                                      else float(threshold))}]
        active = [dict(p) for p in phrases if p.get("model")]
        if not active:
            raise ValueError("WakeWordEngine needs at least one phrase with a model.")
        self._phrases = active
        primary = active[0]
        self.model = primary["model"]
        self.threshold = float(primary["threshold"])
        self.phrase = primary.get("phrase", self.model)
        #: Per-phrase thresholds, config order = fire priority on ties.
        self.thresholds = {p.get("phrase", p["model"]): float(p["threshold"])
                           for p in active}
        #: Phrase -> model, for diagnostics.
        self.models = {p.get("phrase", p["model"]): p["model"] for p in active}
        self._inference = None

    def load(self) -> bool:
        """Load all active models in one inference session.

        Returns False rather than raising when unavailable; names the failing
        models so a bad entry is diagnosable instead of silent.
        """
        if self._inference is not None:
            return True

        try:
            import openwakeword
            from openwakeword.model import Model

            # ONNX Runtime starts a native background thread to upload telemetry.
            # It outlives the interpreter in some versions and aborts the process
            # at exit when it locks a mutex whose owner has already been torn
            # down ("libc++abi ... recursive_mutex lock failed"). Turning the
            # telemetry off removes the thread and the crash with it.
            try:
                import onnxruntime as ort
                ort.disable_telemetry_events()
            except Exception as e:  # noqa: BLE001 - older builds may lack it
                logger.debug(f"Could not disable ONNX telemetry: {e}")

            names = [p["model"] for p in self._phrases]
            known = set(openwakeword.MODELS.keys())
            unknown = [m for m in names if m not in known]
            if unknown:
                logger.error(f"Wake word model(s) not installed: {unknown}. "
                             f"Known: {sorted(known)}.")
                return False

            logger.info(f"Loading wake word model(s) {names}...")

            # Use built-in model names — openWakeWord resolves paths automatically
            self._inference = Model(
                wakeword_models=names,
                inference_framework="onnx",
            )
            logger.info(f"Wake word engine ready. Listening for '{self.model}'...")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize wake word engine: {e}", exc_info=True)
            self._inference = None
            return False

    @property
    def loaded(self) -> bool:
        return self._inference is not None

    def scores(self, pcm) -> dict:
        """Confidence per active phrase for one chunk of int16 PCM, in 0..1."""
        if self._inference is None:
            return {}
        import numpy as np

        expected = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS / 1000)
        if len(pcm) != expected * 2:
            raise ValueError(
                f"wake-word frame has {len(pcm)} bytes; expected {expected * 2}"
            )
        samples = np.frombuffer(pcm, dtype="<i2")
        prediction = self._inference.predict(samples)
        by_model = {p["model"]: p.get("phrase", p["model"]) for p in self._phrases}
        return {by_model.get(name, name): float(score)
                for name, score in prediction.items() if name in by_model}

    def score(self, pcm) -> float:
        """Confidence of the primary phrase. Kept for single-phrase callers."""
        if self._inference is None:
            return 0.0
        return self.scores(pcm).get(self.phrase, 0.0)

    def close(self) -> None:
        """Release the inference engine.

        Without this the ONNX session lived until the process exited, so there
        was no way to shut the wake word engine down cleanly.
        """
        self._inference = None


