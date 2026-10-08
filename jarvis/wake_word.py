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

Requires: openwakeword, pyaudio, numpy
"""

import sys
import time
import threading
from typing import Callable, Optional

from jarvis.config import WAKE_WORD_THRESHOLD, WAKE_WORD_MODEL
from jarvis.logger import logger, StatusIndicator


class WakeWordEngine:
    """The openWakeWord model: load once, score chunks, release cleanly.

    Deliberately free of any audio device. Callers feed it PCM, which is what
    lets the same engine serve both the standalone listener and the Phase 4
    shared stream.
    """

    def __init__(self, model: str = None, threshold: float = None):
        self.model = model or WAKE_WORD_MODEL
        self.threshold = float(threshold if threshold is not None else WAKE_WORD_THRESHOLD)
        self._inference = None

    def load(self) -> bool:
        """Load the model. Returns False rather than raising when unavailable."""
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

            logger.info(f"Loading wake word model '{self.model}' (threshold: {self.threshold})...")

            # Use built-in model name — openWakeWord resolves the path automatically
            self._inference = Model(
                wakeword_models=[self.model],
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

    def score(self, pcm) -> float:
        """Confidence for one chunk of int16 PCM, in 0..1."""
        if self._inference is None:
            return 0.0
        import numpy as np

        prediction = self._inference.predict(np.frombuffer(pcm, dtype=np.int16))
        return float(prediction.get(self.model, 0.0))

    def close(self) -> None:
        """Release the inference engine.

        Without this the ONNX session lived until the process exited, so there
        was no way to shut the wake word engine down cleanly.
        """
        self._inference = None


class WakeWordListener:
    """Standalone wake-word detection over its own audio stream.

    Retained for the single-purpose wake-word mode. Phase 4's conversational
    loop uses `WakeWordEngine` over the shared stream instead, so a process
    never has two microphones open.
    """

    def __init__(
        self,
        on_wake: Callable[[], None],
        threshold: float = None,
        model: str = None,
    ):
        """Initialize the wake word listener.

        Args:
            on_wake: Callback function to invoke when wake word is detected.
            threshold: Confidence threshold (0.0-1.0). Defaults to config value.
            model: Wake word model name. Defaults to config value.
        """
        self.on_wake = on_wake
        self.engine = WakeWordEngine(model=model, threshold=threshold)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # Kept for backwards compatibility with existing callers and tests.
    @property
    def threshold(self) -> float:
        return self.engine.threshold

    @property
    def model(self) -> str:
        return self.engine.model

    @property
    def _inference(self):
        return self.engine._inference

    @_inference.setter
    def _inference(self, value):
        self.engine._inference = value

    def _init_engine(self):
        return self.engine.load()

    def close(self) -> None:
        self.engine.close()

    def _listen_loop(self):
        """Background thread: continuous audio stream and wake word detection.

        Phase 7: Uses blocking stream.read() — no busy-wait polling.
        CPU usage is near-zero while waiting for audio data.

        Phase 8/4: All errors caught and logged gracefully, and the audio
        device is released on *every* exit path, including wake.
        """
        import numpy as np
        import pyaudio

        if not self._init_engine():
            return

        CHUNK = 1280  # 80ms at 16kHz — small chunks for low latency
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        audio = None
        stream = None
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            logger.info(f"Listening for '{self.model}'...")

            while self._running:
                # Blocking read — no busy-wait, CPU stays idle
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                pcm = np.frombuffer(audio_data, dtype=np.int16)

                # Run lightweight ONNX inference (no LLM, no cloud calls)
                score = self.engine.score(audio_data)

                if score > self.engine.threshold:
                    StatusIndicator.wake_detected(score)
                    self._running = False
                    self.on_wake()
                    return

        except Exception as e:
            if self._running:
                logger.error(f"Wake word audio setup failed: {e}", exc_info=True)
        finally:
            # Phase 4: this is the fix. The old code only tore down on the normal
            # exit path, so a detected wake word returned with the device still
            # open for the rest of the process.
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"wake word stream close reported: {e}")
            if audio is not None:
                try:
                    audio.terminate()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"PyAudio terminate reported: {e}")

    def start(self):
        """Start listening for the wake word in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the wake word listener."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        self.close()

    def is_listening(self) -> bool:
        """Check if the listener is currently active."""
        return self._running


def wait_for_wake_word(
    on_wake: Callable[[], None],
    threshold: float = None,
    model: str = None,
) -> WakeWordListener:
    """Convenience function to create and start a WakeWordListener.

    Args:
        on_wake: Callback to invoke when wake word is detected.
        threshold: Confidence threshold (0.0-1.0).
        model: Wake word model name.

    Returns:
        The started WakeWordListener instance.
    """
    listener = WakeWordListener(on_wake=on_wake, threshold=threshold, model=model)
    listener.start()
    return listener