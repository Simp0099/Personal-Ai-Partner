"""Phase 4 — one microphone, shared by wake word, VAD and ASR.

The rule this module exists to enforce: **exactly one process opens the
microphone.** Previously the wake-word listener and
`speech.listen()` could open two capture streams through
``speech_recognition.Microphone`` — two owners of one device, which is how you
get "device busy" failures on some machines and a silent mic on others.

```text
Microphone  ->  MicrophoneStream  ->  [sinks: wake word, VAD, ASR]
(one owner)        (one thread)
```

Everything downstream is a sink that receives the same PCM frames, so they can
never disagree about what was heard or about ordering.

Everything here is injectable: :class:`MicrophoneStream` takes a ``source``
object with ``open``/``read``/``close``, so the whole pipeline is testable with a
fake microphone and no audio hardware.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from jarvis.config import (
    AUDIO_CHANNELS,
    AUDIO_FRAME_MS,
    AUDIO_INPUT_DEVICE,
    AUDIO_SAMPLE_RATE,
    VAD_BARGE_IN_SNR,
    VAD_BARGE_IN_MS,
    VAD_END_SILENCE_MS,
    VAD_ENERGY_THRESHOLD,
    VAD_MIN_SPEECH_MS,
    VAD_SNR,
)
from jarvis.logger import logger

SAMPLE_RATE = AUDIO_SAMPLE_RATE


def frame_bytes(duration_ms: float = AUDIO_FRAME_MS) -> int:
    """PCM bytes for one frame of `duration_ms` at the configured rate."""
    return int(SAMPLE_RATE * duration_ms / 1000) * 2  # int16 mono


# ---------------------------------------------------------------------------
# Frame maths
# ---------------------------------------------------------------------------

def rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of an int16 mono frame, normalised to 0..1.

    Cheaper than a full FFT and, for this purpose, better: voice energy is what
    matters, not spectral detail, and RMS cannot be fooled into reporting speech
    by a narrow-band tone the way a band-limited detector can.
    """
    import array

    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    total = 0.0
    for value in samples:
        total += float(value) * float(value)
    return (total / len(samples)) ** 0.5 / 32768.0


@dataclass
class SpeechEvent:
    """VAD found the start or the end of an utterance."""

    kind: str                      # "start" | "end"
    at: float
    #: Utterance audio captured between the two events, if it ended.
    audio: bytes = b""
    peak: float = 0.0


class VoiceActivityDetector:
    """Energy-based endpointing over the shared frame stream.

    Deliberately simple and fully local: RMS above an adaptive threshold for at
    least `min_speech_ms` is speech; `end_silence_ms` of quiet ends it.

    **The threshold is relative to the room, not absolute.** Measured on the
    development machine, ambient RMS sits at 0.046-0.14 (median 0.068) -- above
    any absolute threshold tuned in a quiet room, so every single frame looked
    like speech and the assistant interrupted itself out of its own reply. A
    fixed threshold cannot be right for both a quiet office and a room with a
    fan, so the detector tracks a noise floor and demands speech to clear it by a
    signal-to-noise margin. The configured `energy_threshold` remains the floor
    of that decision, so a genuinely quiet room behaves exactly as configured.

    Two further knobs:

    * ``min_speech_ms`` kills clicks, keyboard taps and single-frame pops.
    * ``barge_in_ms`` is a *separate, higher* bar used while Ai Partner is busy.
      Interrupting someone mid-sentence should take intent, not a door slam.
    """

    #: How fast the floor follows the room. Quiet arrives quickly (the detector
    #: notices a fan being switched off), loud more slowly, so a hum is not
    #: mistaken for speech -- and then not ignored forever either.
    _FALL = 0.5
    _RISE = 0.01
    #: Frames averaged into the initial estimate. ~1.3s at 80ms frames: long
    #: enough to characterise the room, short enough to feel instant.
    _CALIBRATION_FRAMES = 16

    def __init__(
        self,
        *,
        energy_threshold: float = VAD_ENERGY_THRESHOLD,
        end_silence_ms: float = VAD_END_SILENCE_MS,
        min_speech_ms: float = VAD_MIN_SPEECH_MS,
        barge_in_ms: float = VAD_BARGE_IN_MS,
        barge_in_threshold: Optional[float] = None,
        snr: float = VAD_SNR,
        barge_in_snr: float = VAD_BARGE_IN_SNR,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.energy_threshold = float(energy_threshold)
        self.end_silence_ms = float(end_silence_ms)
        self.min_speech_ms = float(min_speech_ms)
        self.barge_in_ms = float(barge_in_ms)
        #: Barge-in defaults to a stricter level than normal speech detection.
        self.barge_in_threshold = float(
            barge_in_threshold if barge_in_threshold is not None
            else max(self.energy_threshold * 2.0, 0.02)
        )
        self.snr = float(snr)
        self.barge_in_snr = float(barge_in_snr)
        self.clock = clock

        self._speaking = False
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._buffer = bytearray()
        self._peak = 0.0
        #: Running ambient level. None until the first frame arrives.
        self._noise_floor: Optional[float] = None
        self._calibration: List[float] = []

    @property
    def noise_floor(self) -> float:
        """Current ambient estimate; 0 before anything has been heard."""
        return self._noise_floor or 0.0

    def _track_noise(self, level: float) -> None:
        """Estimate the room, but never from frames that look like speech.

        The opening window seeds the floor from a low percentile rather than the
        mean, so a wake word landing inside it cannot set the floor to speech
        level. Tracking is not gated on `_speaking` during that window -- doing
        so deadlocked in exactly the room this exists for -- but no event can be
        emitted before the floor exists, so `_speaking` cannot rise there either.
        """
        if self._noise_floor is None:
            # Seed from a short window rather than from a single frame, so one
            # door slam does not become the room's noise floor.
            self._calibration.append(level)
            if len(self._calibration) >= self._CALIBRATION_FRAMES:
                self._noise_floor = float(np.percentile(self._calibration, 20))
                self._calibration.clear()
            return
        if self._speaking:
            return
        rate = self._FALL if level < self._noise_floor else self._RISE
        self._noise_floor += rate * (level - self._noise_floor)

    def _level_needed(self, barge_in: bool) -> float:
        """RMS a frame must reach to count as speech right now.

        Never below the configured absolute threshold, so a quiet room behaves
        exactly as configured rather than becoming hypersensitive.
        """
        floor = self.noise_floor
        if barge_in:
            return max(self.barge_in_threshold, floor * self.barge_in_snr)
        return max(self.energy_threshold, floor * self.snr)

    def reset(self) -> None:
        """Forget any partial utterance. Called on every interruption."""
        self._speaking = False
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._buffer.clear()
        self._peak = 0.0

    @property
    def in_speech(self) -> bool:
        return self._speaking

    def push(self, pcm: bytes, *, barge_in: bool = False) -> Optional[SpeechEvent]:
        """Feed one frame. Returns an event when an utterance starts or ends.

        Args:
            pcm: One int16 mono frame.
            barge_in: Apply the stricter interruption bar instead of the normal
                one. Set while the assistant is busy.
        """
        now = self.clock()
        level = rms(pcm)
        self._peak = max(self._peak, level)
        self._track_noise(level)
        if self._noise_floor is None:
            # Still characterising the room (~1.3s from loop start, while the
            # assistant is idle). Emitting an event now would mean deciding
            # speech against a floor that does not exist yet, which in a noisy
            # room means hearing silence.
            return None
        frame_ms = len(pcm) / 2 / SAMPLE_RATE * 1000
        threshold = self._level_needed(barge_in)
        voiced = level >= threshold

        if not self._speaking:
            if voiced:
                self._voiced_ms += frame_ms
                self._buffer.extend(pcm)
            else:
                # A short burst is not an utterance; discard it rather than
                # making the user repeat themselves.
                self._voiced_ms = 0.0
                self._buffer.clear()
            if self._voiced_ms >= self.min_speech_ms:
                self._speaking = True
                self._silence_ms = 0.0
                return SpeechEvent("start", now, bytes(self._buffer), self._peak)
            return None

        self._buffer.extend(pcm)
        if voiced:
            self._silence_ms = 0.0
        else:
            self._silence_ms += frame_ms
            if self._silence_ms >= self.end_silence_ms:
                audio = bytes(self._buffer)
                self.reset()
                return SpeechEvent("end", now, audio, self._peak)
        return None


class EchoGuard:
    """Knows when the assistant's own audio is in the air.

    The microphone hears the speakers. Without this, Ai Partner's own reply
    trips the VAD, trips the wake word, and starts a conversation with itself.

    Suppression is time-based rather than "turn the mic off": a hard mute during
    playback would make barge-in impossible, which is the opposite of what the
    user wants. Instead playback is *known*, so assistant audio is ignored while
    genuine user speech still interrupts, and a short cooldown covers the tail of
    the output after playback ends.
    """

    def __init__(self, *, cooldown_s: float = 0.0, clock: Callable[[], float] = time.monotonic):
        self.cooldown_s = float(cooldown_s)
        self.clock = clock
        self._playback_until = 0.0

    def begin_playback(self) -> None:
        self._playback_until = float("inf")

    def end_playback(self) -> None:
        self._playback_until = self.clock() + self.cooldown_s

    @property
    def playing(self) -> bool:
        return self.clock() < self._playback_until

    def should_suppress(self) -> bool:
        """True while the microphone is hearing assistant audio.

        During playback this gates the *wake word* only -- VAD keeps running, at
        its stricter barge-in level, so the user can still interrupt. The
        cooldown after playback gates both.
        """
        return self.playing

    def barge_in(self) -> bool:
        """True when VAD should use the stricter interruption bar."""
        return self.playing


class EchoCanceller:
    """Subtracts the assistant's own playback from the captured microphone.

    Necessary because a louder VAD threshold cannot answer the question barge-in
    actually asks. The assistant's voice through the speakers is exactly as loud
    as the user's and lasts exactly as long, so *any* threshold that lets a person
    interrupt will also let the assistant interrupt itself. Measured live: with
    thresholding alone the assistant cut itself off mid-sentence on its first
    reply.

    So the far-end reference is used directly. Everything Ai Partner plays is
    handed to :meth:`play_reference` as it is played, keeping a short history
    time-stamped. When a captured frame arrives, the reference covering the same
    wall-clock instant is resampled onto it, an echo gain is estimated by the
    usual projection, and that much of the reference is subtracted. What is left
    is the near end -- the user.

    Deliberately a single-tap affine canceller, not a filter bank. It is a few
    dozen lines, has no dependencies, and does the one job that matters: remove
    a direct speaker-to-mic leak while leaving a person alone. It cannot remove
    room reverberation, which is the honest ceiling of this approach.
    """

    #: How much played audio to keep for alignment. Long enough to cover a lag
    #: search, short enough that the history is a constant few hundred KB.
    HISTORY_S = 2.0

    def __init__(self, *, rate: int = SAMPLE_RATE, history_s: float = HISTORY_S,
                 clock: Callable[[], float] = time.monotonic):
        self.rate = int(rate)
        self.history_s = float(history_s)
        self.clock = clock
        self._history: List[Any] = []          # [(start_time, float32 array)]
        #: Last estimated echo gain, and how much energy was removed. Diagnostics
        #: only, but they are what tells you the canceller is working.
        self.last_gain = 0.0
        self.last_reduction = 0.0
        self.last_lag_ms = 0
        self._lag_ms = 0
        self.frames_cancelled = 0

    # -- far end ---------------------------------------------------------

    def play_reference(self, pcm: bytes) -> None:
        """Record audio about to be played, stamped with when it starts."""
        if not pcm:
            return
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self._history.append((self.clock(), samples))
        self._trim()

    def _trim(self) -> None:
        cutoff = self.clock() - self.history_s
        while len(self._history) > 1 and self._history[1][0] <= cutoff:
            self._history.pop(0)

    #: Output latency between "we called play" and "the speakers moved" is a few
    #: tens of milliseconds and varies by device and buffer. Rather than assume
    #: it, the gain is estimated at several candidate lags and the best one is
    #: used. Measured as necessary: assuming zero lag left enough echo to
    #: interrupt the assistant out of its own reply.
    _LAGS_MS = (-80, -60, -40, -20, 0, 20, 40, 60, 80)

    def _best(self, frame, reference, energy: float) -> float:
        """Echo gain at the lag that best explains `frame`.

        The lag is chosen on the exact least-squares residual,
        ``||f||² - (f.c)²/(c.c)``, computed with the *unclamped* gain, and the
        clamp is applied only to the value that is actually subtracted.
        Selecting on the clamped gain instead distorted the choice and left echo
        behind -- measured, not assumed.
        """
        frame_energy = float((frame * frame).sum())
        best, best_gain, best_lag = float("inf"), 0.0, 0
        for lag_ms in self._LAGS_MS:
            shift = int(self.rate * lag_ms / 1000)
            if shift == 0:
                candidate = reference
            elif shift < 0:
                if len(reference) - (-shift) < len(frame):
                    continue
                candidate = reference[-shift:len(frame)]
            else:
                candidate = np.concatenate(
                    (np.zeros(shift, dtype=np.float32), reference[:-shift])
                )
            energy_c = float((candidate * candidate).sum())
            if energy_c < 1e-9:
                continue
            explained = (float((frame * candidate).sum()) ** 2) / energy_c
            residual = frame_energy - explained        # lower is a better fit
            if residual < best:
                best = residual
                best_gain = float((frame * candidate).sum()) / energy_c
                best_lag = lag_ms
        self._lag_ms = best_lag
        return max(0.0, min(best_gain, 4.0))

    def _reference(self, start: float, end: float):
        """Reference samples for [start, end), zero where nothing was played."""
        count = int((end - start) * self.rate)
        out = np.zeros(max(count, 0), dtype=np.float32)
        if count <= 0:
            return out
        times = np.linspace(start, end, count, endpoint=False, dtype=np.float64)
        for played_at, samples in self._history:
            n = len(samples)
            if played_at + n / self.rate < start or played_at > end:
                continue
            # Map absolute time onto this chunk's own sample index.
            positions = (times - played_at) * self.rate
            valid = (positions >= 0) & (positions < n)
            if not valid.any():
                continue
            indices = positions[valid].astype(np.int64)
            out[valid] += samples[indices]
        return out

    # -- near end --------------------------------------------------------

    def _align(self, frame, reference):
        """The reference shifted to the lag `_best` chose."""
        shift = int(self.rate * self._lag_ms / 1000)
        if shift == 0:
            return reference
        if shift < 0:
            return reference[-shift:][:len(frame)]
        return np.concatenate((np.zeros(shift, dtype=np.float32), reference[:-shift]))

    def cancel(self, pcm: bytes) -> bytes:
        """Return `pcm` with the assistant's own playback removed."""
        if not pcm:
            return pcm
        frame = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if not self._history:
            return pcm

        end = self.clock()
        start = end - len(frame) / self.rate
        reference = self._reference(start, end)
        if reference.shape != frame.shape:
            return pcm

        energy = float((reference * reference).sum())
        if energy < 1e-9:
            return pcm                     # nothing of ours in this frame

        gain = self._best(frame, reference, energy)
        reference = self._align(frame, reference)

        residual = frame - gain * reference
        self.last_gain = gain
        self.last_lag_ms = self._lag_ms
        self.last_reduction = 1.0 - (
            float((residual * residual).sum()) / max(float((frame * frame).sum()), 1e-12)
        )
        self.frames_cancelled += 1
        return np.clip(residual * 32767.0, -32768, 32767).astype("<i2").tobytes()

    def close(self) -> None:
        self._history.clear()

    def status(self) -> Dict[str, Any]:
        return {
            "frames_cancelled": self.frames_cancelled,
            "last_gain": round(self.last_gain, 3),
            "last_reduction": round(self.last_reduction, 3),
            "last_lag_ms": self.last_lag_ms,
            "history_s": self.history_s,
        }


@dataclass
class AudioChunk:
    """One queued piece of synthesised speech, tagged with its turn.

    The tag is the whole point: :meth:`AudioQueue.pop` refuses chunks whose turn
    is stale, so audio generated for an interrupted turn can never resume once a
    new turn is underway.
    """

    turn_id: str
    index: int
    data: bytes
    sample_rate: int = SAMPLE_RATE


class AudioQueue:
    """Ordered, turn-tagged, cancellable audio.

    ``discarded_stale`` counts what was thrown away, which is how a test proves
    old audio did not leak into the new turn rather than merely asserting the
    absence of a symptom.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: List[AudioChunk] = []
        self._next_index = 0
        self.discarded_stale = 0
        self.cleared = 0

    def push(self, turn_id: str, data: bytes, sample_rate: int = SAMPLE_RATE) -> AudioChunk:
        with self._lock:
            chunk = AudioChunk(turn_id, self._next_index, data, sample_rate)
            self._next_index += 1
            self._chunks.append(chunk)
            return chunk

    def pop(self, is_stale: Callable[[Optional[str]], bool]) -> Optional[AudioChunk]:
        """Next chunk that is still worth playing, or None.

        Chunks belonging to a stale turn are dropped rather than returned, so the
        caller cannot accidentally play them.
        """
        while True:
            with self._lock:
                if not self._chunks:
                    return None
                chunk = self._chunks.pop(0)
                if not is_stale(chunk.turn_id):
                    return chunk
                # Taken off the queue and thrown away. Anything still in flight
                # from a cancelled turn is discarded here rather than played.
                self.discarded_stale += 1

    def peek_len(self) -> int:
        with self._lock:
            return len(self._chunks)

    def clear(self) -> int:
        """Drop everything queued. Returns how many chunks were discarded."""
        with self._lock:
            count = len(self._chunks)
            self._chunks.clear()
            self.cleared += count
            return count


# ---------------------------------------------------------------------------
# The microphone owner
# ---------------------------------------------------------------------------

class MicrophoneUnavailable(RuntimeError):
    """The input device could not be opened. Never raised past ``start``."""


class SoundDeviceSource:
    """Microphone capture through sounddevice (PortAudio).

    The device callback pushes frames into a bounded queue. read() blocks until a
    frame arrives or the timeout expires. A timeout returns None, which is normal.
    Any other failure raises, so the caller can see it.
    """

    def __init__(self, device=None, rate=SAMPLE_RATE, frame_ms=AUDIO_FRAME_MS, max_queued=200):
        self.device = device
        self.rate = int(rate)
        self.block = int(self.rate * frame_ms / 1000)   # samples per frame
        self.frame_size = self.block * 2                 # bytes per frame, int16 mono
        self._queue = queue.Queue(maxsize=max_queued)
        self._stream = None
        self.device_name = ""
        self.overflows = 0

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def open(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise MicrophoneUnavailable(
                "sounddevice is not installed. Run: pip install sounddevice") from e

        try:
            info = sd.query_devices(self.device, "input")
        except Exception as e:
            raise MicrophoneUnavailable(f"no input device found: {e}") from e
        self.device_name = info.get("name", "unknown")

        try:
            self._stream = sd.InputStream(
                samplerate=self.rate,
                channels=1,
                dtype="int16",
                blocksize=self.block,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise MicrophoneUnavailable(
                f"could not open '{self.device_name}': {e}. On macOS, allow your terminal "
                "under System Settings > Privacy and Security > Microphone."
            ) from e
        logger.info(f"[AUDIO] microphone: {self.device_name}")

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.overflows += 1
        chunk = indata.tobytes()
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Keep the newest audio: drop the oldest frame instead of blocking the device.
            self.overflows += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                pass

    def read(self, timeout: float = 1.0) -> Optional[bytes]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"microphone close reported: {e}")
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class MicrophoneStream:
    """The single owner of the microphone, fanning frames out to sinks.

    One reader thread. Sinks are called in registration order, on that thread, so
    a sink must be fast and must never block: a sink that blocks stalls every
    other consumer of the same audio.
    """

    def __init__(
        self,
        *,
        source: Any = None,
        clock: Callable[[], float] = time.monotonic,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.source = source or SoundDeviceSource(device=AUDIO_INPUT_DEVICE)
        self.clock = clock
        self.on_error = on_error
        self._sinks: List[Callable[[bytes], None]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self.frames_read = 0
        self.available = False
        self.last_error: Optional[str] = None

    # -- wiring ----------------------------------------------------------

    def add_sink(self, sink: Callable[[bytes], None]) -> None:
        """Register a consumer of raw frames."""
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: Callable[[bytes], None]) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    # -- lifecycle -------------------------------------------------------

    def start(self) -> bool:
        """Open the device and start pumping frames. Honest False on failure."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            try:
                self.source.open()
            except Exception as e:  # noqa: BLE001 - unavailability is not a crash
                self.available = False
                self.last_error = str(e)
                logger.warning(f"[AUDIO] microphone unavailable: {e}")
                if self.on_error:
                    self.on_error(str(e))
                return False
            self._stop.clear()
            self.available = True
            self.last_error = None
            self._thread = threading.Thread(target=self._pump, name="microphone", daemon=True)
            self._thread.start()
        logger.info("[AUDIO] microphone open (one stream, shared by wake word, VAD and ASR)")
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Stop pumping and release the device. Idempotent."""
        self._stop.set()
        with self._lock:
            thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self.source.close()
        self.available = False

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    # -- reading ---------------------------------------------------------

    def _pump(self) -> None:
        """Read loop. Failures are counted, logged, and retried. They are never hidden."""
        failures = 0
        while not self._stop.is_set():
            try:
                pcm = self.source.read()
            except Exception as e:  # noqa: BLE001
                failures += 1
                self.last_error = f"read failed: {e}"
                logger.error(f"[AUDIO] microphone read failed ({failures}): {e}")
                if failures >= 20:
                    self.available = False
                    logger.error("[AUDIO] giving up on the microphone after repeated read failures")
                    return
                time.sleep(min(0.05 * failures, 0.5))
                continue
            if pcm is None:          # timeout with no audio is normal: keep waiting
                continue
            failures = 0
            self.frames_read += 1
            with self._lock:
                sinks = list(self._sinks)
            for sink in sinks:
                try:
                    sink(pcm)
                except Exception as e:  # noqa: BLE001 - one bad sink, not the mic
                    logger.warning(f"[AUDIO] sink failed: {e}")

    # -- diagnostics -----------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.is_running(),
            "available": self.available,
            "frames_read": self.frames_read,
            "sinks": len(self._sinks),
            "sample_rate": SAMPLE_RATE,
            "frame_ms": AUDIO_FRAME_MS,
            "last_error": self.last_error,
        }


__all__ = [
    "AudioChunk",
    "AudioQueue",
    "EchoCanceller",
    "EchoGuard",
    "MicrophoneStream",
    "MicrophoneUnavailable",
    "SoundDeviceSource",
    "SpeechEvent",
    "VoiceActivityDetector",
    "frame_bytes",
    "rms",
]