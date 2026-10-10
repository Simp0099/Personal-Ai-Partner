"""Phase 4 — cancellable speech output, and transcription.

Two adapters over the existing engines, each adding only the one thing Phase 4
needs and nothing else:

* :class:`SpeechPlayer` — turns a reply into turn-tagged audio chunks and plays
  them through a queue that can be emptied mid-sentence. Playback owns its
  ``sounddevice.OutputStream`` on one worker, so cancellation never closes a
  native stream concurrently.
* :class:`Transcriber` — wraps the existing ``speech_recognition`` engine. It is
  not a new ASR architecture: same recogniser, same Google endpoint, fed PCM
  from the shared microphone instead of opening a second capture device.

Both are injectable. The whole voice pipeline runs with no audio hardware and no
network in the test suite.
"""

from __future__ import annotations

import threading
import os
from typing import Any, Callable, List, Optional

from jarvis.audio import SAMPLE_RATE, AudioQueue
from jarvis.config import (
    ASR_ENGINE, ASR_TIMEOUT_S, AUDIO_OUTPUT_DEVICE, DEFAULT_LANGUAGE, TTS_CHUNK_MS,
    TTS_ENGINE,
)
from jarvis.logger import logger
from jarvis.tts import KOKORO_RATE


class ASRUnavailable(RuntimeError):
    """Speech recognition is not installed or not usable.

    Raised, not swallowed: the caller reports it as a state the HUD can show and
    returns to IDLE. Fabricating a transcript is the one unacceptable outcome.
    """


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _engine_synthesize(text: str, exaggeration=None) -> Any:
    """Synthesize with the configured engine, resolved at call time."""
    from jarvis import tts

    return tts.synthesize(text, exaggeration=exaggeration)


_output_owner_lock = threading.Lock()
_output_state_lock = threading.Lock()
_active_output_cancel: Optional[threading.Event] = None


def _sounddevice_play_array(data: Any, sample_rate: int, cancel_event=None) -> str:
    """Play mono float32 audio; the worker that opens the stream also closes it."""
    with _output_owner_lock:
        return _sounddevice_play_array_owned(data, sample_rate, cancel_event)


def _sounddevice_play_array_owned(data: Any, sample_rate: int, cancel_event=None) -> str:
    global _active_output_cancel
    import sounddevice as sd
    import numpy as np

    samples = np.asarray(data, dtype=np.float32).reshape(-1)
    if not samples.size or (cancel_event is not None and cancel_event.is_set()):
        return "cancelled"

    device = AUDIO_OUTPUT_DEVICE if AUDIO_OUTPUT_DEVICE is not None else sd.default.device[1]
    if device is None or int(device) < 0:
        raise RuntimeError(
            "No default output device is selected. Run `python scripts/audio_devices.py` "
            "and set conversation.output_device in config.yaml."
        )
    device = int(device)
    channels = 1
    try:
        info = sd.query_devices(device, "output")
        sd.check_output_settings(
            device=device, channels=channels, dtype="float32", samplerate=sample_rate,
        )
    except Exception:
        logger.exception(
            "[AUDIO] output configuration rejected device=%s rate=%d channels=%d "
            "format=float32",
            device, sample_rate, channels,
        )
        raise
    local_cancel = cancel_event or threading.Event()
    with _output_state_lock:
        _active_output_cancel = local_cancel
    logger.info(
        "[AUDIO] playback opening pid=%d engine=%s device=%s name=%s rate=%d channels=%d "
        "dtype=float32 samples=%d duration_s=%.3f",
        os.getpid(), TTS_ENGINE, device, info.get("name"), sample_rate, channels, len(samples),
        len(samples) / sample_rate,
    )
    finished = threading.Event()
    offset = 0

    def callback(outdata, frames, _time_info, _status):
        nonlocal offset
        if _status:
            logger.warning("[AUDIO] output callback reported PortAudio status: %s", _status)
        if local_cancel.is_set():
            outdata.fill(0)
            raise sd.CallbackStop
        count = min(frames, len(samples) - offset)
        if count:
            outdata[:count, 0] = samples[offset:offset + count]
            offset += count
        if count < frames:
            outdata[count:, 0] = 0
        if offset >= len(samples):
            raise sd.CallbackStop

    try:
        with sd.OutputStream(
            device=device, samplerate=sample_rate, channels=channels,
            dtype="float32", callback=callback,
            finished_callback=finished.set,
        ):
            if not finished.wait(timeout=max(5.0, len(samples) / sample_rate * 2 + 5.0)):
                raise TimeoutError("output stream did not finish before its playback deadline")
    except Exception:
        logger.exception(
            "[AUDIO] output stream failed device=%s name=%s rate=%d channels=%d "
            "format=float32 frames=%d",
            device, info.get("name"), sample_rate, channels, len(samples),
        )
        raise
    finally:
        with _output_state_lock:
            if _active_output_cancel is local_cancel:
                _active_output_cancel = None
    result = "cancelled" if local_cancel.is_set() else "completed"
    logger.info(
        "[AUDIO] playback %s pid=%d engine=%s device=%s rate=%d channels=%d "
        "frames=%d/%d duration_s=%.3f",
        result, os.getpid(), TTS_ENGINE, device, sample_rate, channels, offset, len(samples),
        offset / sample_rate,
    )
    return result


def _sounddevice_play(data: bytes, sample_rate: int = SAMPLE_RATE, cancel_event=None) -> str:
    """Play one int16 PCM chunk through sounddevice's float32 interface."""
    import numpy as np

    if len(data) % 2:
        raise ValueError("playback PCM must contain complete int16 samples")
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return _sounddevice_play_array(samples, sample_rate, cancel_event)


def _sounddevice_stop() -> None:
    """Ask the active output owner to stop at its next callback and close its stream."""
    with _output_state_lock:
        event = _active_output_cancel
    if event is not None:
        event.set()


class SpeechPlayer:
    """Plays a reply as turn-tagged chunks, cancellable between chunks.

    The chunk queue is shared with the HUD path, so a browser consuming the same
    queue gets the same turn ids and the same stale-audio rejection. Two
    consumers, one queue, no way for them to disagree.

    Cancellation is cooperative: the play function returns between chunks, and
    that is the only place it is checked. Nothing is killed mid-call, which is
    what keeps native audio teardown race-free.
    """

    def __init__(
        self,
        queue: Optional[AudioQueue] = None,
        *,
        synthesize: Optional[Callable[[str], Any]] = None,
        play: Optional[Callable[[bytes, int], None]] = None,
        stop_playback: Optional[Callable[[], None]] = None,
        chunk_ms: int = TTS_CHUNK_MS,
        sample_rate: int = KOKORO_RATE,
        on_play: Optional[Callable[[bytes], None]] = None,
        exaggeration: Optional[float] = None,
    ):
        self.queue = queue if queue is not None else AudioQueue()
        self.synthesize = synthesize or _engine_synthesize
        self.play = play or _sounddevice_play
        self.stop_playback = stop_playback or _sounddevice_stop
        self.chunk_ms = int(chunk_ms)
        self.sample_rate = int(sample_rate)
        #: Called with each chunk immediately before it is played, so the echo
        #: canceller knows exactly what the speakers are about to emit.
        self.on_play = on_play
        #: Optional per-reply expressiveness hint, bounded by the engine's own
        #: configured range. None means "use the configured default".
        self.exaggeration = exaggeration

        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self.chunks_played = 0
        self.chunks_cancelled = 0
        self.playing = False
        self.last_error: Optional[str] = None

    # -- synthesis -------------------------------------------------------

    def synthesize_to_queue(self, text: str, turn_id: str, is_stale) -> int:
        """Synthesize `text` and enqueue it as chunks tagged with `turn_id`.

        Stops early if the turn goes stale mid-synthesis, so an interrupted
        reply is not even fully generated. Returns the number of chunks queued.
        """
        if not (text or "").strip():
            return 0
        engine = TTS_ENGINE if self.synthesize is _engine_synthesize else "custom"
        try:
            audio = _synthesize_with(self.synthesize, text, self.exaggeration)
        except Exception:
            logger.error("[TTS] synthesis failed engine=%s turn=%s", engine, turn_id, exc_info=True)
            raise
        waveform = audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio
        import numpy as np
        waveform = np.asarray(waveform)
        sample_count = waveform.size
        logger.info(
            "[TTS] synthesis complete engine=%s turn=%s shape=%s dtype=%s "
            "sample_rate=%d channels=1 samples=%d duration_s=%.3f",
            engine, turn_id, waveform.shape, waveform.dtype, self.sample_rate,
            sample_count, sample_count / self.sample_rate,
        )
        data = _to_pcm16(audio, self.sample_rate)
        queued = 0
        for chunk in _split(data, self.sample_rate, self.chunk_ms):
            if is_stale(turn_id):
                break
            self.queue.push(turn_id, chunk, self.sample_rate)
            queued += 1
        return queued

    # -- playback --------------------------------------------------------

    def start(self, turn_id: str, is_stale) -> None:
        """Play the current turn's queued audio on a worker thread.

        A cancelled drain thread may still be finishing. Wait for it, so the new
        reply is never dropped because the old thread still looks alive (V8).
        The wait is bounded, and it happens outside the lock, so cancelling from
        another thread can still reach `_cancel` while we join.
        """
        with self._lock:
            previous = self._thread
        if previous is not None and previous.is_alive():
            previous.join(timeout=2.0)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                # Still playing after the wait. Leave the reply queued rather than
                # running a second drain thread over the same queue.
                return
            self._cancel.clear()
            self.last_error = None
            self._thread = threading.Thread(
                target=self._drain, args=(turn_id, is_stale), name="speech-player", daemon=True
            )
            self._thread.start()

    def _drain(self, turn_id: str, is_stale) -> None:
        self.playing = True
        try:
            if self.play is _sounddevice_play:
                chunks = []
                while not self._cancel.is_set():
                    chunk = self.queue.pop(is_stale)
                    if chunk is None:
                        break
                    if is_stale(chunk.turn_id):
                        self.chunks_cancelled += 1
                    else:
                        chunks.append(chunk)
                if chunks and not self._cancel.is_set():
                    data = b"".join(chunk.data for chunk in chunks)
                    if self.on_play is not None:
                        self.on_play(data)
                    result = self.play(data, chunks[0].sample_rate, self._cancel)
                    if result == "cancelled":
                        self.chunks_cancelled += len(chunks)
                    else:
                        self.chunks_played += len(chunks)
                return
            while not self._cancel.is_set():
                chunk = self.queue.pop(is_stale)
                if chunk is None:
                    break
                # Cooperative cancellation point. Between chunks, never inside
                # a native call.
                if self._cancel.is_set() or is_stale(chunk.turn_id):
                    self.chunks_cancelled += 1
                    continue
                if self.on_play is not None:
                    self.on_play(chunk.data)
                self.play(chunk.data, chunk.sample_rate)
                self.chunks_played += 1
        except Exception as e:  # noqa: BLE001 - playback failure must not hang
            self.last_error = str(e)
            logger.error(f"speech playback failed: {e}", exc_info=True)
        finally:
            self.playing = False

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until playback finishes. False if it is still going."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def cancel(self) -> int:
        """Stop playback and empty the queue. Returns chunks discarded.

        Safe from another thread, safe mid-utterance, and idempotent. This is the
        barge-in primitive.
        """
        with self._lock:
            self._cancel.set()
            discarded = self.queue.clear()
        try:
            self.stop_playback()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"stop playback reported: {e}")
        self.chunks_cancelled += discarded
        logger.info("[AUDIO] playback cancellation requested queued_chunks=%d", discarded)
        return discarded

    def is_playing(self) -> bool:
        return self.playing

    def close(self, timeout: float = 5.0) -> None:
        self.cancel()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise TimeoutError("speech playback worker did not stop before timeout")
        with self._lock:
            if self._thread is thread:
                self._thread = None


def _synthesize_with(synthesize, text: str, exaggeration) -> Any:
    """Call a synthesizer that may or may not accept an expressiveness hint.

    Injected synthesizers in tests take only the text, so the hint is passed
    only when the callable can actually receive it.
    """
    if exaggeration is None:
        return synthesize(text)
    try:
        return synthesize(text, exaggeration=exaggeration)
    except TypeError:
        return synthesize(text)


def _to_pcm16(audio: Any, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Normalise any engine output to interleaved int16 little-endian bytes."""
    import numpy as np

    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _split(data: bytes, sample_rate: int, chunk_ms: int) -> List[bytes]:
    """Split PCM into fixed-duration chunks."""
    size = max(1, int(sample_rate * chunk_ms / 1000)) * 2
    return [data[i:i + size] for i in range(0, len(data), size)] or [b""]


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class Transcriber:
    """Local faster-whisper transcription with visible Google fallback."""
    def __init__(self, language: str = "en", model_size: str = "base.en",
                 recognizer: Any = None, google_language: str = "en-IN"):
        self.language = language
        self.model_size = model_size
        self.google_language = google_language
        self._recognizer = recognizer
        self._whisper = None
        self.engine: Optional[str] = None
        self.last_error: Optional[str] = None

    def _load_whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._whisper

    def transcribe(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> str:
        if not pcm:
            return ""
        if len(pcm) % 2 or sample_rate <= 0:
            raise ValueError("captured PCM must contain complete int16 samples and a positive sample rate")
        try:
            model = self._load_whisper()
        except ImportError:
            return self._google(pcm, sample_rate)
        self.engine = "faster-whisper"
        import numpy as np
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = model.transcribe(samples, language=self.language, beam_size=1, vad_filter=False)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        self.last_error = None if text else "no speech recognised"
        return text

    def _google(self, pcm: bytes, sample_rate: int) -> str:
        try:
            import speech_recognition as sr
        except ImportError as e:
            raise ASRUnavailable("No speech recognition is installed. Run: pip install faster-whisper") from e
        recognizer = self._recognizer or sr.Recognizer()
        self._recognizer = recognizer
        self.engine = "google"
        try:
            result = recognizer.recognize_google(sr.AudioData(pcm, sample_rate, 2), language=self.google_language)
            self.last_error = None
            return result
        except sr.UnknownValueError:
            self.last_error = "speech was not understood"
            return ""


__all__ = [
    "ASRUnavailable",
    "SpeechPlayer",
    "Transcriber",
    "_split",
    "_to_pcm16",
]
