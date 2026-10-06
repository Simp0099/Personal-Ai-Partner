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

Requires: openwakeword, pyaudio, numpy
"""

import sys
import time
import threading
from typing import Callable, Optional

from jarvis.config import WAKE_WORD_THRESHOLD, WAKE_WORD_MODEL
from jarvis.logger import logger, StatusIndicator


class WakeWordListener:
    """Continuous wake word detection using openWakeWord.

    Runs a background audio stream that listens for the configured wake word.
    When detected, calls the provided callback to activate the assistant.

    Phase 7: This listener is completely idle — no LLM calls, no polling,
    just lightweight ONNX inference on small audio chunks.

    Phase 8: All errors caught and logged. Status indicators for CLI.
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
        self.threshold = threshold or WAKE_WORD_THRESHOLD
        self.model = model or WAKE_WORD_MODEL
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._inference = None

    def _init_engine(self):
        """Lazy-initialize the openWakeWord inference engine."""
        if self._inference is not None:
            return True

        try:
            import openwakeword
            from openwakeword.model import Model

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

    def _listen_loop(self):
        """Background thread: continuous audio stream and wake word detection.

        Phase 7: Uses blocking stream.read() — no busy-wait polling.
        CPU usage is near-zero while waiting for audio data.

        Phase 8: All errors caught and logged gracefully.
        """
        import numpy as np
        import pyaudio

        if not self._init_engine():
            return

        CHUNK = 1280  # 80ms at 16kHz — small chunks for low latency
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

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
                try:
                    # Blocking read — no busy-wait, CPU stays idle
                    audio_data = stream.read(CHUNK, exception_on_overflow=False)
                    pcm = np.frombuffer(audio_data, dtype=np.int16)

                    # Run lightweight ONNX inference (no LLM, no cloud calls)
                    prediction = self._inference.predict(pcm)

                    # Check confidence score for our target model
                    score = prediction.get(self.model, 0.0)

                    if score > self.threshold:
                        StatusIndicator.wake_detected(score)
                        self._running = False
                        self.on_wake()
                        return

                except Exception as e:
                    if self._running:
                        logger.error(f"Wake word stream error: {e}", exc_info=True)
                    break

            stream.stop_stream()
            stream.close()
            audio.terminate()

        except Exception as e:
            logger.error(f"Wake word audio setup failed: {e}", exc_info=True)

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

    def is_listening(self) -> bool:
        """Check if the listener is currently active."""
        return self._running


def wait_for_wake_word(
    on_wake: Callable[[], None],
    threshold: float = None,
    model: str = None,
) -> WakeWordListener:
    """Convenience function to create and start a wake word listener.

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
