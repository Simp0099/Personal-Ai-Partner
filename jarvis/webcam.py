"""Phase 3 — webcam perception lifecycle.

Opt-in, quiet, and disposable:

    START -> camera opens -> sample -> change-detect -> vision (only if changed)
          -> ... -> shutdown -> camera released

Nothing here starts on import. :func:`start_webcam_perception` is called by the
entrypoints and is a no-op unless ``vision.webcam.enabled`` is true in
config.yaml, so the camera is never opened without the user asking for it.

The loop is a single daemon thread holding the only camera handle. It is joined
on :meth:`WebcamPerception.stop`, which also releases the device, so shutdown
cannot race a half-read frame against interpreter teardown.

``cv2`` is imported lazily inside the camera wrapper. A machine without OpenCV
gets a reported, honest "webcam unavailable" and keeps working text-only.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from jarvis.config import (
    VISION_ENABLED,
    WEBCAM_ANALYSIS_COOLDOWN,
    WEBCAM_DEVICE,
    WEBCAM_ENABLED,
    WEBCAM_INTERVAL_SECONDS,
)
from jarvis.logger import logger, StatusIndicator
from jarvis.vision import (
    Observation,
    VisualContext,
    frame_to_image_part,
    get_visual_context,
    parse_observations,
    scene_instruction,
)
from jarvis.vision_change import ChangeDetector


class WebcamUnavailable(RuntimeError):
    """The camera could not be opened. Never raised past :meth:`start`."""


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class OpenCVCamera:
    """Thin wrapper over ``cv2.VideoCapture``.

    Owns the native handle, so ``release()`` is the single place the device is
    given back. Frames are BGR (OpenCV's native order) and are converted to RGB
    once here, at the edge, rather than every consumer guessing.
    """

    def __init__(self, device: int = WEBCAM_DEVICE, width: int = 640, height: int = 480):
        self.device = device
        self.width = width
        self.height = height
        self._capture = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as e:
            raise WebcamUnavailable(
                "OpenCV is not installed; webcam perception needs `opencv-python`."
            ) from e

        capture = cv2.VideoCapture(self.device)
        if not capture.isOpened():
            capture.release()
            raise WebcamUnavailable(
                f"No camera available on device {self.device} (missing, in use, "
                "or permission denied)."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture
        logger.info(f"[WEBCAM] camera opened (device {self.device})")

    def read(self):
        """One frame as an RGB array, or ``None`` when the read fails."""
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as e:  # noqa: BLE001 - never fail shutdown over this
                logger.warning(f"[WEBCAM] camera release reported: {e}")
            self._capture = None
            logger.info("[WEBCAM] camera released")

    @property
    def is_open(self) -> bool:
        return self._capture is not None


# ---------------------------------------------------------------------------
# Perception loop
# ---------------------------------------------------------------------------

class WebcamPerception:
    """Samples the camera and updates visual context when the scene changes.

    The loop itself does the minimum and nothing more:

    1. wait for the sampling interval (no busy polling),
    2. read one frame,
    3. score it against the previous frame locally,
    4. only if it changed, and not inside the cooldown, ask a vision model what
       it means and record the result.

    Frames are never written anywhere. After analysis the array is dropped.
    """

    def __init__(
        self,
        analyze: Callable[[str, Any], str],
        *,
        interval: float = WEBCAM_INTERVAL_SECONDS,
        cooldown: float = WEBCAM_ANALYSIS_COOLDOWN,
        camera: Any = None,
        detector: Optional[ChangeDetector] = None,
        context: Optional[VisualContext] = None,
        clock: Callable[[], float] = time.time,
    ):
        """
        Args:
            analyze: Called with the scene prompt once a frame is worth looking
                at. Returns the model's description; any exception is treated as
                a failed analysis, never as a fake observation.
            interval: Seconds between samples.
            cooldown: Minimum seconds between vision calls.
            camera: Camera object exposing ``open``/``read``/``release``.
            detector: Local change detector.
            context: Where observations are recorded.
            clock: Time source, injectable for tests.
        """
        self.analyze = analyze
        self.interval = float(interval)
        self.cooldown = float(cooldown)
        self.camera = camera
        self.detector = detector or ChangeDetector()
        self.context = context or get_visual_context()
        self.clock = clock

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._state_lock = threading.RLock()
        self.frames_sampled = 0
        self.frames_discarded = 0
        self.analyses = 0
        self.last_error: Optional[str] = None
        self.available = False
        self.started_at: Optional[float] = None
        #: Clock reading of the last vision call, for the cooldown.
        self._last_analysis_at = 0.0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> bool:
        """Open the camera and run the perception loop in a daemon thread.

        Returns:
            True when perception is running. A False return is an honest failure
            -- the reason is logged and in :attr:`last_error` -- never a silent
            half-start.
        """
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            if self.camera is None:
                self.camera = OpenCVCamera()
            try:
                self.camera.open()
            except Exception as e:  # noqa: BLE001 - unavailability is not a crash
                self.available = False
                self.last_error = str(e)
                logger.warning(f"[WEBCAM] unavailable: {e}. Continuing without perception.")
                return False

            self._stop.clear()
            self.available = True
            self.started_at = self.clock()
            self._thread = threading.Thread(target=self._run, name="webcam-perception", daemon=True)
            self._thread.start()

        logger.info(
            f"[WEBCAM] perception active (sampling every {self.interval:.0f}s, "
            f"vision at most every {self.cooldown:.0f}s). Nothing is recorded or saved."
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the loop, join the worker, and release the camera.

        Idempotent, and safe to call when perception never started. The join is
        what keeps the native camera handle from being released underneath a
        frame read on another thread.
        """
        self._stop.set()
        with self._state_lock:
            thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("[WEBCAM] perception worker did not stop in time")
        if self.camera is not None:
            self.camera.release()
        self.available = False
        self.started_at = None

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    # -- loop -------------------------------------------------------------

    def _run(self) -> None:
        """Worker body. One frame per interval, forever, until stopped."""
        while not self._stop.is_set():
            if self._stop.wait(self.interval):
                break
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - a frame must never kill the loop
                self.last_error = str(e)
                logger.error(f"[WEBCAM] frame cycle failed: {e}", exc_info=True)

    def tick(self) -> Optional[Observation]:
        """One sample/analyse cycle. Exposed so tests need no sleeping.

        Returns:
            The recorded observation when a frame was analysed, else ``None``.
        """
        frame = self.camera.read() if self.camera is not None else None
        if frame is None:
            self.last_error = "camera read failed"
            logger.warning("[WEBCAM] frame read failed; skipping this sample")
            return None

        self.frames_sampled += 1
        score = self.detector.score(frame)
        if score < self.detector.threshold:
            self.frames_discarded += 1
            return None

        if self.cooldown and self.analyses and (self.clock() - self._last_analysis_at) < self.cooldown:
            # Scene changed again inside the cooldown. Recorded as discarded so
            # the next interval re-evaluates rather than waiting a whole cycle.
            self.frames_discarded += 1
            logger.debug(f"[WEBCAM] change {score:.3f} inside cooldown; deferred")
            return None

        observations = self._analyze_frame(frame)
        self._last_analysis_at = self.clock()
        if not observations:
            self.last_error = "vision analysis produced no usable observation"
            logger.warning(f"[WEBCAM] {self.last_error}")
            return None

        self.analyses += 1
        self.last_error = None
        if self.context.update(observations):
            StatusIndicator.info(
                "[WEBCAM] " + "; ".join(f"{o.kind}: {o.text}" for o in observations)
            )
        else:
            logger.info("[WEBCAM] observation unchanged; context refreshed only")
        return observations[0]

    def _analyze_frame(self, frame) -> List[Observation]:
        """Send one frame to the vision model and parse what comes back.

        Raises whatever the analyzer raises: a failed vision call must surface as
        a failure, never as an empty observation set that reads as "nothing
        changed".
        """
        attachment = frame_to_image_part(frame)
        prompt = scene_instruction(self.context.observations())
        return parse_observations(self.analyze(prompt, attachment))

    # -- diagnostics ------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Runtime state for the status endpoint. Contains no image data."""
        return {
            "enabled": bool(WEBCAM_ENABLED and VISION_ENABLED),
            "running": self.is_running(),
            "camera_available": self.available,
            "frames_sampled": self.frames_sampled,
            "frames_discarded": self.frames_discarded,
            "vision_analyses": self.analyses,
            "sample_interval_seconds": self.interval,
            "analysis_cooldown_seconds": self.cooldown,
            "change_threshold": self.detector.threshold,
            "uptime_seconds": round(self.clock() - self.started_at, 1) if self.started_at else None,
            "last_error": self.last_error,
            "frames_persisted": 0,
            "context": self.context.status(),
        }


#: Process-wide instance, so :func:`start_webcam_perception` and the shutdown
#: path act on the same camera even though they are called from different places.
_instance: Optional[WebcamPerception] = None
_instance_lock = threading.Lock()


def start_webcam_perception(brain, *, force: bool = False) -> Optional[WebcamPerception]:
    """Start perception if it is configured on. Returns the instance, or ``None``.

    ``brain`` is used for its vision routing only; the analysis runs as an
    ephemeral turn that is never recorded in conversation history, so a camera
    event cannot appear as something the user said.
    """
    global _instance

    if not (force or (VISION_ENABLED and WEBCAM_ENABLED)):
        logger.info("[WEBCAM] perception disabled by configuration; camera not opened")
        return None

    with _instance_lock:
        if _instance is not None and _instance.is_running():
            return _instance

        def analyze(prompt: str, attachment) -> str:
            return brain.ask(prompt, images=[attachment], ephemeral=True)

        perception = WebcamPerception(analyze=analyze)
        if not perception.start():
            _instance = perception
            return None
        _instance = perception
        return perception


def stop_webcam_perception() -> None:
    """Stop and release the process-wide instance, if there is one."""
    global _instance
    with _instance_lock:
        perception, _instance = _instance, None
    if perception is not None:
        perception.stop()


def get_webcam_perception() -> Optional[WebcamPerception]:
    return _instance


__all__ = [
    "OpenCVCamera",
    "WebcamPerception",
    "WebcamUnavailable",
    "get_webcam_perception",
    "start_webcam_perception",
    "stop_webcam_perception",
]