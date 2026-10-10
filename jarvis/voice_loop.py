"""Phase 4 — the voice conversation loop.

Drives one turn:

```text
IDLE --wake--> LISTENING --speech end--> TRANSCRIBING --transcript--> THINKING
      --> SPEAKING --done--> FOLLOW_UP --speech--> LISTENING | --timeout--> IDLE
                                          --speech while busy--> INTERRUPTED
```

Everything it needs is injected, so the whole loop runs in tests with no
microphone, no model and no sound card. That is not a testing convenience — it
is what makes barge-in and cancellation testable at all, because those are race
conditions and races need determinism.

Three rules the loop enforces:

* **Vision never enters here.** A camera observation updates visual context and
  says nothing; it cannot move the state machine. Only a wake word or speech can.
* **A barge-in cancels before it starts.** The old turn is marked stale and its
  audio dropped *before* the new turn begins, so nothing from the old response
  can surface in the new one.
* **Cancellation is cooperative.** The Brain's network call cannot be recalled,
  but its result is discarded if the turn went stale, and synthesis stops between
  chunks. Nothing is killed mid-native-call, so shutdown stays race-free.
"""

from __future__ import annotations

import threading
import time
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from jarvis.audio import (
    EchoCanceller, EchoGuard, MicrophoneStream, VoiceActivityDetector, frame_bytes,
)
from jarvis.config import (
    FOLLOW_UP_WINDOW_S,
    WAKE_WORD_ENABLED,
    WAKE_WORD_PHRASES,
)
from jarvis.conversation import (
    AssistantStateMachine,
    ConversationMachine,
    State,
)
from jarvis.trace import new_trace, span_of, use_trace
from jarvis.tone import get_tone
from jarvis.logger import logger, StatusIndicator
from jarvis.wake_word import WakeWordEngine
from jarvis.speech_pipeline import ASRUnavailable, SpeechPlayer, Transcriber


#: States in which new user speech means "stop, I'm talking", not "start".
INTERRUPTABLE = (State.TRANSCRIBING, State.THINKING, State.SPEAKING)


class _VoiceProcessLock:
    """Keep two Jarvis processes from listening and echo-cancelling independently."""

    def __init__(self):
        self.path = Path(tempfile.gettempdir()) / f"jarvis-voice-{os.getuid()}.lock"
        self._file = None

    def acquire(self) -> Optional[int]:
        import fcntl

        if self._file is not None:
            return None
        handle = self.path.open("a+", encoding="ascii")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            owner = handle.read().strip()
            handle.close()
            return int(owner) if owner.isdigit() else 0
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._file = handle
        return None

    def release(self) -> None:
        if self._file is None:
            return
        import fcntl

        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None


class VoiceLoop:
    """One microphone, one conversation at a time, interruptible at any point."""

    def __init__(
        self,
        brain: Any = None,
        *,
        machine: Optional[ConversationMachine] = None,
        assistant: Optional[AssistantStateMachine] = None,
        microphone: Optional[MicrophoneStream] = None,
        vad: Optional[VoiceActivityDetector] = None,
        echo: Optional[EchoGuard] = None,
        canceller: Optional[EchoCanceller] = None,
        player: Optional[SpeechPlayer] = None,
        transcriber: Optional[Transcriber] = None,
        wake_engine: Optional[WakeWordEngine] = None,
        respond: Optional[Callable[[str], str]] = None,
        follow_up_window: float = FOLLOW_UP_WINDOW_S,
        wake_enabled: bool = WAKE_WORD_ENABLED,
        clock: Callable[[], float] = time.monotonic,
        events: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
    ):
        """
        Args:
            brain: Used for its default `ask` when `respond` is not supplied.
            machine: Authoritative state. One is created if omitted.
            assistant: Optional 4-state lifecycle mirror. When given, every
                detailed transition is projected onto it (never raises).
            microphone: The single audio owner. One is created if omitted.
            player: Cancellable speech output.
            respond: Given a transcript, returns the reply text. Injectable so
                tests need no model.
            follow_up_window: Seconds of listening after a reply finishes.
            wake_engine: Scores frames for the wake phrase. The shared
                microphone feeds it; it never opens a device.
            wake_enabled: Opt-in. When false the loop rests in LISTENING and
                speech alone starts a turn. When true it rests in IDLE, which
                only a wake word can leave.
            events: Callables notified of state events, for the HUD.
        """
        self.machine = machine or ConversationMachine(clock=clock)
        self.microphone = microphone or MicrophoneStream(
            clock=clock, on_error=self._on_microphone_error,
        )
        self.vad = vad or VoiceActivityDetector(clock=clock)
        self.echo = echo or EchoGuard(clock=clock)
        #: Removes the assistant's own playback from the microphone signal. The
        #: player feeds it through `on_play`, wired up in __init__ below. The
        #: playback rate is the engine's own (Kokoro, 24 kHz), which is not the
        #: capture rate, so the canceller is told both.
        from jarvis.tts import KOKORO_RATE

        self.canceller = canceller or EchoCanceller(clock=clock, playback_rate=KOKORO_RATE)
        self.player = player or SpeechPlayer()
        self.transcriber = transcriber
        #: Scores frames for the wake phrase. Never opens a device: the shared
        #: microphone is the only owner of the input stream.
        self.wake_engine = wake_engine or WakeWordEngine(phrases=WAKE_WORD_PHRASES)
        self.brain = brain
        self.respond = respond or self._ask_brain
        self.follow_up_window = float(follow_up_window)
        self.wake_enabled = bool(wake_enabled)
        self.clock = clock
        self._events: List[Callable[[Dict[str, Any]], None]] = list(events or [])

        self._lock = threading.RLock()
        self._process_lock = _VoiceProcessLock()
        self._thread: Optional[threading.Thread] = None
        #: The turn currently being served. Joined on shutdown.
        self._turn_thread: Optional[threading.Thread] = None
        self._stopping = False
        self._stop = threading.Event()
        #: Set while a wake word is being handled, so one "hey jarvis" produces
        #: exactly one transition even if several overlapping chunks score high.
        self._wake_latched = False
        self._wake_rejections = 0
        self.wake_events = 0
        #: Last firing-frame wake inference time in ms, attached to the turn trace.
        self._last_wake_ms = 0.0
        self._utterance = b""
        self.interrupts = 0
        self.running = False
        self.last_error: Optional[str] = None
        self._last_audio_diag = 0.0
        self._last_inference_error = 0.0
        self._invalid_frame_reported = False

        if getattr(self.player, "on_play", None) is None:
            self.player.on_play = self.canceller.play_reference

        # Subscribe once so every transition reaches the HUD event list.
        self.machine.subscribe(self._forward_event)
        self._assistant = assistant
        if assistant is not None:
            self.machine.subscribe(self._mirror_lifecycle)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _forward_event(self, event) -> None:
        payload = event.to_dict()
        payload["timestamp"] = round(self.clock(), 3)
        for listener in list(self._events):
            try:
                listener(payload)
            except Exception:  # noqa: BLE001 - a bad observer cannot break the loop
                pass

    def _on_microphone_error(self, message: str) -> None:
        """Stop the conversation loop if capture fails after startup."""
        self.last_error = message
        if self.running:
            logger.error("[VOICE] microphone capture stopped: %s", message)
            self.running = False
            self._stop.set()
            self.machine.fail(message)
            self.machine.transition(State.IDLE, reason="microphone failure", force=True)

    def _mirror_lifecycle(self, event) -> None:
        """Project detailed voice states onto the 4-state lifecycle view."""
        try:
            if self._assistant is not None:
                self._assistant.observe(event.state, reason="voice loop")
        except Exception:  # noqa: BLE001 - mirror never breaks the driver
            pass

    def add_event_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._events.append(listener)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Open the microphone and begin the conversation loop.

        Returns False — honestly, with the reason in `last_error` — when the
        microphone or wake-word model is unavailable. Text conversation is
        unaffected either way.
        """
        with self._lock:
            if self._stopping:
                raise RuntimeError("voice loop shutdown is already in progress")
            if self.running:
                return True

            owner_pid = self._process_lock.acquire()
            if owner_pid is not None:
                self.last_error = (
                    "another Jarvis process owns the microphone and speaker"
                    + (f" (pid {owner_pid})" if owner_pid else "")
                )
                logger.error("[VOICE] %s; stop that instance before starting another", self.last_error)
                return False

            # The wake model must be ready BEFORE audio starts arriving. While the
            # ONNX session is still being built, WakeWordEngine.scores() returns an
            # empty dict for every frame, so any speech spoken during that window is
            # discarded silently -- the loop rests in IDLE and the first request is
            # simply never heard. Loading first costs a few hundred milliseconds
            # before the device opens and costs nothing afterwards.
            if self.wake_enabled and not self.wake_engine.load():
                self.last_error = "wake word model unavailable"
                logger.warning(
                    "[VOICE] wake word model unavailable; voice input is disabled. "
                    "Text conversation still works."
                )
                self.wake_enabled = False

            self.microphone.add_sink(self._on_frame)
            if not self.microphone.start():
                self.microphone.remove_sink(self._on_frame)
                self.last_error = self.microphone.last_error or "microphone unavailable"
                self.machine.fail(self.last_error)
                self.machine.transition(State.IDLE, reason="voice unavailable", force=True)
                self._process_lock.release()
                return False

            self._stop.clear()
            self.running = True
            self.machine.transition(self._rest_state(), reason="voice loop started", force=True)

        logger.info(
            "[VOICE] conversation loop active pid=%d (wake %s, follow-up %.0fs)",
            os.getpid(), "on" if self.wake_enabled else "off", self.follow_up_window,
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Shut down in the documented order, releasing everything.

        Order matters: stop accepting turns, cancel the active one, stop taking
        audio, stop producing audio, then release devices and join workers. Each
        step removes a source of work before the resource it uses is freed.
        """
        self._stop.set()

        # Claim the threads under the lock, then release it before joining.
        # `_run_turn` takes this same lock in its `finally`, so joining while
        # holding it deadlocks until the join times out -- which is exactly what
        # happened: every shutdown left a turn thread finishing after stop()
        # had already returned.
        with self._lock:
            if self._stopping:
                raise RuntimeError("voice loop shutdown is already in progress")
            self._stopping = True
            self._stop.set()
            self.running = False
            worker = self._thread
            turn_thread = self._turn_thread

        failures = []
        try:
            # Stop accepting turns before releasing resources, even after capture failed.
            self.machine.cancel_turn("shutdown")
            self.machine.transition(State.IDLE, reason="shutdown", force=True)
            for cleanup in (
                lambda: self.microphone.remove_sink(self._on_frame),
                self.microphone.stop,
                self.player.close,
                self.canceller.close,
            ):
                try:
                    cleanup()
                except Exception as e:
                    failures.append(e)
                    logger.error("[VOICE] shutdown cleanup failed: %s", e, exc_info=True)
            for name, thread in (("voice-loop", worker), ("voice-turn", turn_thread)):
                if thread is not None and thread.is_alive():
                    thread.join(timeout=timeout)
                if thread is not None and thread.is_alive():
                    failures.append(TimeoutError(f"{name} worker did not stop before timeout"))
            self.machine.reset()
        finally:
            with self._lock:
                if self._thread is worker and (worker is None or not worker.is_alive()):
                    self._thread = None
                if self._turn_thread is turn_thread and (turn_thread is None or not turn_thread.is_alive()):
                    self._turn_thread = None
                self._stopping = False
        if failures:
            raise RuntimeError("voice loop shutdown was incomplete: " + "; ".join(map(str, failures)))
        self._process_lock.release()
        logger.info("[VOICE] conversation loop stopped")

    def is_running(self) -> bool:
        return self.running

    def wait_for_stop(self, timeout: Optional[float] = None) -> bool:
        """Wait for loop cancellation; returns True when shutdown was requested."""
        return self._stop.wait(timeout)

    # ------------------------------------------------------------------
    # Audio sink
    # ------------------------------------------------------------------

    def _on_frame(self, pcm: bytes) -> None:
        """Called on the microphone thread for every frame. Must stay fast.

        Three consumers of the same audio, exactly as the architecture requires:
        the wake-word engine, the VAD, and (via the loop thread) ASR.
        """
        if not isinstance(pcm, (bytes, bytearray, memoryview)) or len(pcm) != frame_bytes():
            if not self._invalid_frame_reported:
                logger.warning(
                    "[AUDIO] discarded malformed microphone frame (%s bytes; expected %s)",
                    len(pcm) if hasattr(pcm, "__len__") else "unknown", frame_bytes(),
                )
                self._invalid_frame_reported = True
            return

        raw_pcm = pcm
        state = self.machine.state
        now = self.clock()

        # Remove the assistant's own voice from the signal *before* anything
        # looks at it. Both the wake word and the VAD then see the near end --
        # the user -- rather than a leak of what we just played.
        pcm = self.canceller.cancel(pcm)

        # While the assistant is busy, speech must clear the stricter
        # interruption bar to cut in. Whether the user *can* interrupt is decided
        # by state, never by echo: gating it on echo meant a user speaking during
        # a model call, before any audio had played, was ignored. And using the
        # lenient bar while busy meant a cough during a slow model call could
        # cancel the turn. Barge-in is the stricter bar, always, while busy.
        barge_in = state in INTERRUPTABLE

        # The wake word alone is gated while the assistant's own audio is in the
        # air: it must not wake itself. The VAD keeps running underneath, so
        # barge-in remains possible throughout.
        if self.wake_enabled and not barge_in and not self.echo.should_suppress():
            fired = self._wake_fired(pcm)
            if fired is not None:
                phrase, score = fired
                logger.debug(f"wake phrase matched: {phrase} score={score:.2f}")
                if self._wake_latched:
                    # Overlapping chunks all score high for the same utterance.
                    self._wake_rejections += 1
                else:
                    self.on_wake()
                # Deliberately no return. The wake word is usually the first
                # syllable of what the user is saying -- "Hey Jarvis, what's the
                # time" is one breath -- and discarding these frames clipped the
                # utterance. They still go to the VAD; only the transition is
                # suppressed.

        if now - self._last_audio_diag >= 5.0:
            from jarvis.audio import rms
            import array

            raw_samples = array.array("h")
            raw_samples.frombytes(bytes(raw_pcm))
            peak = max((abs(sample) for sample in raw_samples), default=0) / 32768.0
            logger.debug(
                "[AUDIO] callback frames=%d bytes=%d format=s16le-mono rate=%dHz "
                "frame_ms=%d rms_raw=%.4f rms_post_echo=%.4f peak=%.4f "
                "preprocessing_changed=%s echo_cancelled_frames=%s "
                "state=%s",
                self.microphone.frames_read, len(pcm),
                getattr(self.microphone, "status", lambda: {})().get("sample_rate", 0),
                getattr(self.microphone, "status", lambda: {})().get("frame_ms", 0),
                rms(raw_pcm), rms(pcm), peak, raw_pcm != pcm,
                getattr(self.canceller, "status", lambda: {})().get("frames_cancelled", "unknown"),
                state.value,
            )
            self._last_audio_diag = now

        event = self.vad.push(pcm, barge_in=barge_in)
        if event is None:
            return
        if event.kind == "start":
            self._on_speech_start(event, barge_in)
        else:
            self._on_speech_end(event)

    # ------------------------------------------------------------------
    # Wake and speech
    # ------------------------------------------------------------------

    def _on_speech_start(self, event, barge_in: bool) -> None:
        """What the beginning of an utterance means, decided purely by state."""
        logger.info(
            "[VOICE] VAD speech started pid=%d state=%s barge_in=%s",
            os.getpid(), self.machine.state.value, barge_in,
        )
        self._utterance = event.audio
        state = self.machine.state
        if state in INTERRUPTABLE:
            # The user is talking over the assistant -- during transcription, a
            # model call, or its reply. They have the floor.
            self.interrupt()
        elif state is State.FOLLOW_UP:
            # Follow-up: no wake word needed.
            self.machine.transition(State.LISTENING, reason="follow-up speech")

    def _on_speech_end(self, event) -> None:
        if not event.audio:
            return
        logger.info(
            "[VOICE] VAD speech ended bytes=%d peak=%.4f; submitting to ASR",
            len(event.audio), event.peak,
        )
        with self._lock:
            # Re-checked under the lock against stop(): a turn started after
            # shutdown began would not be joined by stop(), and would keep
            # touching state it had already finalised.
            if self._stop.is_set() or not self.running:
                return
            self._utterance = event.audio
            # Tracked so shutdown can wait for it, and started under the same
            # lock: stop() cannot otherwise observe a registered-but-unstarted
            # thread, see is_alive() False, skip the join, and leave it running.
            thread = threading.Thread(target=self._run_turn, name="voice-turn", daemon=True)
            self._turn_thread = thread
            thread.start()

    # ------------------------------------------------------------------
    # Turn
    # ------------------------------------------------------------------

    def _run_turn(self) -> None:
        """LISTENING -> TRANSCRIBING -> THINKING -> SPEAKING -> FOLLOW_UP.

        Every step re-checks staleness: by the time a slow model call returns,
        the user may already have interrupted.

        The running thread claims the tracked-turn slot for itself. Registering
        only at the call site left a hole: a turn started any other way was never
        joined by `stop()`, and outlived shutdown still touching state it had
        already finalised.
        """
        with self._lock:
            tracked = self._turn_thread
            if tracked is None or not tracked.is_alive():
                self._turn_thread = threading.current_thread()

        turn = None
        trace = new_trace()
        with use_trace(trace):
            try:
                if self.machine.state is State.IDLE:
                    return  # speech arrived with no conversation to serve

                turn = self.machine.begin_turn()
                self.machine.mark("speech_end", turn.id)

                # -- TRANSCRIBING ------------------------------------------------
                self.machine.transition(State.TRANSCRIBING, reason="speech ended")
                with span_of("stt"):
                    text = self._transcribe(self._utterance, turn.id)
                turn.transcript = text
                if not text.strip():
                    # Never invent a transcript. This is the one failure the user
                    # most needs told about.
                    self.machine.fail("I didn't catch that.")
                    self._settle()
                    return

                # V11: the user must see what was heard. Without this, a wrong
                # transcript is indistinguishable from a wrong reply.
                print(f"\n[YOU]: {text}", flush=True)

                from jarvis.main import is_shutdown_request
                if is_shutdown_request(text):
                    from jarvis.config import GREETING_NAME
                    self.machine.transition(State.THINKING, reason="shutdown requested")
                    StatusIndicator.shutdown()
                    self._speak(f"Going offline. You can call me anytime, {GREETING_NAME}!", turn.id)
                    self._stop.set()
                    self.machine.cancel_turn("shutdown command")
                    self.machine.transition(State.IDLE, reason="shutdown command", force=True)
                    return

                # -- THINKING ----------------------------------------------------
                self.machine.transition(State.THINKING, reason="transcript ready")
                with span_of("think"):
                    reply = self._respond(text, turn.id)
                if self.machine.is_stale(turn.id):
                    return  # interrupted while the model was thinking
                print(f"[JARVIS]: {reply}", flush=True)
                if not (reply or "").strip():
                    self._settle()
                    return

                # -- SPEAKING ----------------------------------------------------
                with span_of("speak"):
                    self._speak(reply, turn.id)
                if self.machine.is_stale(turn.id):
                    return
                self.machine.end_turn(turn.id)

                # -- FOLLOW_UP ---------------------------------------------------
                self.machine.transition(State.FOLLOW_UP, reason="reply finished")
                self.machine.mark("playback_end")
                self._await_follow_up(turn.id)

            except Exception as e:  # noqa: BLE001 - one bad turn must not kill the loop
                # An interrupted turn is not a failure: the user took the floor and
                # the next turn is already under way. Reporting ERROR here would
                # stomp the state the interruption just established.
                if turn is not None and self.machine.is_stale(turn.id):
                    return
                self.last_error = str(e)
                logger.error(f"[VOICE] turn failed: {e}", exc_info=True)
                try:
                    self.machine.fail(str(e))
                except Exception:  # noqa: BLE001
                    self.machine.transition(State.ERROR, reason=str(e), force=True)
                self._settle()
            finally:
                trace.finish()
                with self._lock:
                    # Unlatch here, so the wake word can fire again for the next
                    # utterance. Latching for the whole turn is what stopped one
                    # utterance from waking the assistant exactly once.
                    self._wake_latched = False
                    self.vad.reset()
                self._utterance = b""

    def _transcribe(self, audio: bytes, turn_id: str) -> str:
        if self.machine.is_stale(turn_id):
            return ""
        if self.transcriber is None:
            raise ASRUnavailable(
                "No transcriber configured; voice input is unavailable."
            )
        logger.info("[VOICE] ASR processing %d bytes", len(audio))
        text = self.transcriber.transcribe(audio)
        logger.info("[VOICE] ASR returned transcript (%d characters)", len(text or ""))
        if not text:
            reason = getattr(self.transcriber, "last_error", None) or "empty transcript"
            logger.warning("[VOICE] ASR returned no transcript: %s", reason)
        if not self.machine.is_stale(turn_id):
            self.machine.mark("transcript", turn_id)
        return text

    def _respond(self, text: str, turn_id: str) -> str:
        """Ask the Brain. Runs inline; the result is discarded if interrupted."""
        self.machine.mark("first_token", turn_id)
        return self.respond(text)

    def _ask_brain(self, text: str) -> str:
        return self.brain.ask(text)

    def _wake_fired(self, pcm) -> Optional[tuple]:
        """First (phrase, score) above its threshold, config order wins ties.

        One inference call scores every active phrase on the same shared chunk;
        the wake latch then guarantees one utterance wakes the assistant once.
        Engines without `scores()` fall back to the legacy single check.
        """
        engine = self.wake_engine
        start = time.perf_counter()
        try:
            scores_fn = getattr(engine, "scores", None)
            if callable(scores_fn):
                thresholds = getattr(engine, "thresholds", None) or {}
                for phrase, s in scores_fn(pcm).items():
                    if s > thresholds.get(phrase, engine.threshold):
                        return (phrase, s)
                return None
            s = engine.score(pcm)
            if s > engine.threshold:
                return (getattr(engine, "model", "wake word"), s)
            return None
        finally:
            self._last_wake_ms = (time.perf_counter() - start) * 1000.0

    def on_wake(self) -> None:
        """IDLE -> LISTENING. Immediate, and idempotent while latched."""
        with self._lock:
            if self._wake_latched:
                self._wake_rejections += 1
                return
            self._wake_latched = True
            self.wake_events += 1
        self.machine.mark("wake_detected")
        try:
            self.machine.transition(State.LISTENING, reason="wake word")
        except Exception as e:  # noqa: BLE001 - already listening, etc.
            logger.debug(f"wake transition skipped: {e}")
            return

    def _speak(self, reply: str, turn_id: str) -> None:
        """Synthesize and play, checking staleness between chunks."""
        self.machine.transition(State.SPEAKING, reason="reply ready")
        is_stale = self.machine.is_stale

        # The conversational state only reaches the voice through one numeric
        # expressiveness hint. It is a hint, not a mode: the same engine, the
        # same voice, the same speed range. It is a number, never a bool --
        # `True` would clamp to the top of the range on every excited reply.
        state = get_tone().state
        if state.exaggerated():
            self.player.exaggeration = 0.8     # livelier delivery: faster, a little brisker
        elif state.composure():
            self.player.exaggeration = 0.3     # calmer delivery: slower, steadier
        else:
            self.player.exaggeration = None    # configured default

        queued = self.player.synthesize_to_queue(reply, turn_id, is_stale)
        if is_stale(turn_id) or queued == 0:
            return

        self.machine.mark("first_audio", turn_id)
        self.echo.begin_playback()
        self.player.start(turn_id, is_stale)
        if not self.player.wait(timeout=120.0):
            self.player.cancel()
            if not self.player.wait(timeout=5.0):
                raise TimeoutError("speech playback did not stop after cancellation")
        if getattr(self.player, "last_error", None):
            raise RuntimeError(f"speech playback failed: {self.player.last_error}")
        self.echo.end_playback()
        self.machine.mark("playback_end", turn_id)

    # ------------------------------------------------------------------
    # Interruption
    # ------------------------------------------------------------------

    def interrupt(self) -> None:
        """Cancel the current turn and go back to listening.

        Order matters and is the whole design: invalidate first, then drop audio.
        Doing it the other way round leaves a window where the old turn is still
        live and its audio can still be queued.
        """
        with self._lock:
            self.interrupts += 1
            self.machine.mark("interrupt_detected")
            turn = self.machine.cancel_turn("barge-in")

            # Drop the old turn's queued audio and stop playback mid-utterance.
            dropped = self.player.cancel()
            self.vad.reset()

            try:
                self.machine.transition(State.INTERRUPTED, reason="barge-in")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"interrupt transition skipped: {e}")
                return

            self.machine.transition(State.LISTENING, reason="after barge-in")
            self.machine.mark("interrupt_complete")

        logger.info(
            f"[VOICE] interrupted (dropped {dropped} queued audio chunks); listening"
        )

    # ------------------------------------------------------------------
    # Follow-up
    # ------------------------------------------------------------------

    def _await_follow_up(self, turn_id: str) -> None:
        """Stay open briefly, then return to IDLE. Never permanent."""
        deadline = self.clock() + self.follow_up_window
        while self.clock() < deadline:
            if self._stop.is_set():
                return
            if self.machine.state is not State.FOLLOW_UP:
                return  # the user already spoke
            time.sleep(0.05)
        if self.machine.state is State.FOLLOW_UP:
            target = self._rest_state()
            try:
                self.machine.transition(target, reason="follow-up window elapsed")
            except Exception:  # noqa: BLE001
                self.machine.transition(target, reason="follow-up", force=True)

    def _rest_state(self) -> State:
        """Where the loop waits between turns. IDLE can only be left by a wake word."""
        return State.IDLE if self.wake_enabled else State.LISTENING

    def _settle(self) -> None:
        """Return to rest after a turn that produced no reply."""
        target = self._rest_state()
        try:
            self.machine.transition(target, reason="turn settled")
        except Exception:  # noqa: BLE001
            self.machine.transition(target, reason="turn settled", force=True)

    def check_proactive(self, candidate=None, *, user_speaking=False,
                        wake_active=False) -> object:
        """Decision-only proactive hook: candidate -> gates -> YES/NO.

        The candidate arrives as data (built by the caller from the
        ObservationEngine); this method never reads the camera itself, so the
        Phase 4 invariant holds: perception cannot move the state machine, only a
        wake word or speech can. Speaks nothing and touches no audio: on YES
        the caller drives the existing Brain and the existing :meth:`_speak`,
        so barge-in, stale-turn drops and the machine stay authoritative.
        """
        from jarvis.proactive import (
            ContextSnapshot,
            get_orchestrator,
            get_proactive_engine,
        )

        if candidate is None:
            return None
        state = self.machine.state
        snap = ContextSnapshot(
            user_availability="unavailable" if state in INTERRUPTABLE
            else "probably_available",
            active_conversation=state is not State.IDLE,
            user_speaking=bool(user_speaking),
            assistant_speaking=state is State.SPEAKING,
            wake_active=bool(wake_active),
            recently_spoken=get_orchestrator().recent_reasons(),
        )
        result = get_proactive_engine().decide(candidate, snap)
        return result if result.should_speak else None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "interrupts": self.interrupts,
            "follow_up_window_s": self.follow_up_window,
            "microphone": self.microphone.status(),
            "audio_queue_depth": self.player.queue.peek_len(),
            "audio_discarded_stale": self.player.queue.discarded_stale,
            "audio_discarded_cancelled": self.player.chunks_cancelled,
            "echo_suppressing": self.echo.should_suppress(),
            "echo_canceller": self.canceller.status(),
            "wake_enabled": self.wake_enabled,
            "wake_events": self.wake_events,
            "wake_rejections": self._wake_rejections,
            "last_error": self.last_error,
            "state": self.machine.snapshot(include_history=False),
        }


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------

_instance: Optional[VoiceLoop] = None
_lock = threading.Lock()
_last_start_error: Optional[str] = None


def start_voice_loop(brain, *, force: bool = False) -> Optional[VoiceLoop]:
    """Start the voice loop if it is configured on. Returns the instance or None."""
    global _instance, _last_start_error
    _last_start_error = None

    if not force:
        logger.info("[VOICE] voice disabled by configuration; microphone not opened")
        _last_start_error = "voice disabled by configuration"
        return None

    with _lock:
        if _instance is not None and _instance.is_running():
            return _instance
        loop = VoiceLoop(brain=brain, transcriber=Transcriber())
        _instance = loop
        try:
            if loop.start():
                return loop
        except Exception:
            loop.stop()
            _instance = None
            raise
        loop.stop()
        _last_start_error = loop.last_error
        _instance = None
        return None


def stop_voice_loop() -> None:
    """Stop and release the process-wide voice loop, if there is one."""
    global _instance
    with _lock:
        loop = _instance
        try:
            if loop is not None:
                loop.stop()
        finally:
            if _instance is loop:
                _instance = None


def get_voice_loop() -> Optional[VoiceLoop]:
    return _instance


def get_voice_start_error() -> Optional[str]:
    """Reason the most recent configured voice loop did not start."""
    return _last_start_error


__all__ = [
    "VoiceLoop",
    "get_voice_loop",
    "get_voice_start_error",
    "start_voice_loop",
    "stop_voice_loop",
]
