"""Phase 4 — cancellable speech output, and transcription.

Two adapters over the existing engines, each adding only the one thing Phase 4
needs and nothing else:

* :class:`SpeechPlayer` — turns a reply into turn-tagged audio chunks and plays
  them through a queue that can be emptied mid-sentence. ``speech.speak()``
  blocks on ``sd.wait()`` with no way out, which is the single biggest reason
  barge-in cannot work; this keeps the same ``sounddevice`` output and adds a
  cancel point between chunks.
* :class:`Transcriber` — wraps the existing ``speech_recognition`` engine. It is
  not a new ASR architecture: same recogniser, same Google endpoint, fed PCM
  from the shared microphone instead of opening a second capture device.

Both are injectable. The whole voice pipeline runs with no audio hardware and no
network in the test suite.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional

from jarvis.audio import SAMPLE_RATE, AudioQueue
from jarvis.config import TTS_CHUNK_MS
from jarvis.logger import logger


class ASRUnavailable(RuntimeError):
    """Speech recognition is not installed or not usable.

    Raised, not swallowed: the caller reports it as a state the HUD can show and
    returns to IDLE. Fabricating a transcript is the one unacceptable outcome.
    """


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _chatterbox_synthesize(text: str, exaggeration=None) -> Any:
    """Reuse the existing Chatterbox engine. Returns a mono float32 array.

    `exaggeration` is passed straight through to the engine when supplied and
    left alone when not, so the default voice is byte-for-byte what it was
    before Phase 5.
    """
    from jarvis import speech

    return speech._synthesize_chatterbox(text, exaggeration=exaggeration)


def _sounddevice_play(data: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Play one chunk, blocking until it finishes."""
    import sounddevice as sd
    import numpy as np

    samples = np.frombuffer(data, dtype=np.float32)
    sd.play(samples, sample_rate)
    sd.wait()


def _sounddevice_stop() -> None:
    """Abort whatever is currently playing."""
    try:
        import sounddevice as sd
        sd.stop()
    except Exception as e:  # noqa: BLE001 - nothing to stop is not an error
        logger.debug(f"audio stop reported: {e}")


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
        sample_rate: int = SAMPLE_RATE,
        on_play: Optional[Callable[[bytes], None]] = None,
        exaggeration: Optional[float] = None,
    ):
        self.queue = queue if queue is not None else AudioQueue()
        self.synthesize = synthesize or _chatterbox_synthesize
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

    # -- synthesis -------------------------------------------------------

    def synthesize_to_queue(self, text: str, turn_id: str, is_stale) -> int:
        """Synthesize `text` and enqueue it as chunks tagged with `turn_id`.

        Stops early if the turn goes stale mid-synthesis, so an interrupted
        reply is not even fully generated. Returns the number of chunks queued.
        """
        if not (text or "").strip():
            return 0
        audio = _synthesize_with(self.synthesize, text, self.exaggeration)
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
        """Play the current turn's queued audio on a worker thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._cancel.clear()
            self._thread = threading.Thread(
                target=self._drain, args=(turn_id, is_stale), name="speech-player", daemon=True
            )
            self._thread.start()

    def _drain(self, turn_id: str, is_stale) -> None:
        self.playing = True
        try:
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
        self._cancel.set()
        discarded = self.queue.clear()
        try:
            self.stop_playback()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"stop playback reported: {e}")
        self.chunks_cancelled += discarded
        return discarded

    def is_playing(self) -> bool:
        return self.playing

    def close(self) -> None:
        self.cancel()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
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
    """Speech-to-text over PCM already captured from the shared microphone.

    Uses the project's existing ``speech_recognition`` engine and Google
    endpoint. The only thing that changed is where the audio comes from: a PCM
    buffer handed in, instead of ``sr.Microphone()`` opening a second device.
    """

    def __init__(self, language: str = "en-in", recognizer: Any = None):
        self.language = language
        self._recognizer = recognizer
        self.last_error: Optional[str] = None

    def _get(self):
        if self._recognizer is not None:
            return self._recognizer
        try:
            import speech_recognition as sr
        except ImportError as e:
            raise ASRUnavailable(
                "SpeechRecognition is not installed; voice input is unavailable. "
                "Text conversation still works."
            ) from e
        self._recognizer = sr.Recognizer()
        return self._recognizer

    def transcribe(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe captured audio. Raises on failure rather than guessing."""
        if not pcm:
            return ""
        recognizer = self._get()
        try:
            import speech_recognition as sr
        except ImportError as e:  # pragma: no cover - _get already raised
            raise ASRUnavailable("SpeechRecognition is not installed.") from e

        try:
            # AudioData takes the rate it was captured at directly. There is no
            # SAMPLE_RATE constant to borrow here.
            audio = sr.AudioData(_to_float_samples(pcm), sample_rate, 2)
            return recognizer.recognize_google(audio, language=self.language)
        except sr.UnknownValueError:
            # Speech was detected but not understood. That is an honest "I did
            # not catch that", not a fabricated transcript.
            self.last_error = "speech was not understood"
            return ""
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)
            raise


def _to_float_samples(pcm: bytes):
    import numpy as np

    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


__all__ = [
    "ASRUnavailable",
    "SpeechPlayer",
    "Transcriber",
    "_split",
    "_to_pcm16",
]