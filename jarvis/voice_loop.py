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
from typing import Any, Callable, Dict, List, Optional

from jarvis.audio import EchoCanceller, EchoGuard, MicrophoneStream, VoiceActivityDetector
from jarvis.config import (
    FOLLOW_UP_WINDOW_S,
    WAKE_WORD_ECHO_COOLDOWN,
    WAKE_WORD_ENABLED,
    WAKE_WORD_THRESHOLD,
)
from jarvis.conversation import (
    AssistantStateMachine,
    ConversationMachine,
    State,
)
from jarvis.tone import get_tone
from jarvis.logger import logger, StatusIndicator
from jarvis.speech_pipeline import ASRUnavailable, SpeechPlayer, Transcriber
from jarvis.wake_word import WakeWordEngine


#: States in which new user speech means "stop, I'm talking", not "start".
INTERRUPTABLE = (State.TRANSCRIBING, State.THINKING, State.SPEAKING)


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
            events: Callables notified of state events, for the HUD.
        """
        self.machine = machine or ConversationMachine(clock=clock)
        self.microphone = microphone or MicrophoneStream(clock=clock)
        self.vad = vad or VoiceActivityDetector(clock=clock)
        self.echo = echo or EchoGuard(cooldown_s=WAKE_WORD_ECHO_COOLDOWN, clock=clock)
        #: Removes the assistant's own playback from the microphone signal. The
        #: player feeds it through `on_play`, wired up in __init__ below.
        self.canceller = canceller or EchoCanceller(clock=clock)
        self.player = player or SpeechPlayer()
        self.transcriber = transcriber
        self.wake_engine = wake_engine or WakeWordEngine(threshold=WAKE_WORD_THRESHOLD)
        self.brain = brain
        self.respond = respond or self._ask_brain
        self.follow_up_window = float(follow_up_window)
        self.wake_enabled = bool(wake_enabled)
        self.clock = clock
        self._events: List[Callable[[Dict[str, Any]], None]] = list(events or [])

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        #: The turn currently being served. Joined on shutdown.
        self._turn_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        #: Set while a wake word is being handled, so one "hey jarvis" produces
        #: exactly one transition even if several overlapping chunks score high.
        self._wake_latched = False
        self._utterance = b""
        self._wake_rejections = 0
        self.wake_events = 0
        self.interrupts = 0
        self.running = False
        self.last_error: Optional[str] = None

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
            if self.running:
                return True

            if not self.microphone.start():
                self.last_error = self.microphone.last_error or "microphone unavailable"
                self.machine.fail(self.last_error)
                self.machine.transition(State.IDLE, reason="voice unavailable", force=True)
                return False

            if self.wake_enabled and not self.wake_engine.load():
                self.last_error = "wake word model unavailable"
                logger.warning(
                    "[VOICE] wake word model unavailable; voice input is disabled. "
                    "Text conversation still works."
                )
                self.wake_enabled = False

            self.microphone.add_sink(self._on_frame)
            self._stop.clear()
            self.running = True
            self.machine.transition(State.IDLE, reason="voice loop started", force=True)

        logger.info(
            f"[VOICE] conversation loop active (wake "
            f"{'on' if self.wake_enabled else 'off'}, follow-up "
            f"{self.follow_up_window:.0f}s)"
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
            running = self.running
            self.running = False
            worker, self._thread = self._thread, None
            turn_thread, self._turn_thread = self._turn_thread, None

        if running:
            # 1. Stop accepting turns and invalidate whatever is in flight.
            self.machine.cancel_turn("shutdown")
            self.machine.transition(State.IDLE, reason="shutdown", force=True)

            # 2. Stop taking audio.
            self.microphone.remove_sink(self._on_frame)
            self.microphone.stop()

            # 3. Stop producing audio: cancel synthesis and empty the queue.
            self.player.close()

            # 5. Release the wake-word engine and the far-end reference history.
            self.wake_engine.close()
            self.canceller.close()

            if worker is not None and worker.is_alive():
                worker.join(timeout=timeout)

            # 4. Wait for the turn that is mid-flight. Bounded: a model call
            #    already in flight cannot be recalled, and shutdown must not
            #    hang waiting for a network call that may never return.
            if turn_thread is not None and turn_thread.is_alive():
                turn_thread.join(timeout=timeout)
                if turn_thread.is_alive():
                    logger.warning(
                        "[VOICE] turn thread did not stop in time; "
                        "abandoning it rather than hanging shutdown"
                    )

        self.machine.reset()
        logger.info("[VOICE] conversation loop stopped")

    def is_running(self) -> bool:
        return self.running

    # ------------------------------------------------------------------
    # Audio sink
    # ------------------------------------------------------------------

    def _on_frame(self, pcm: bytes) -> None:
        """Called on the microphone thread for every frame. Must stay fast.

        Three consumers of the same audio, exactly as the architecture requires:
        the wake-word engine, the VAD, and (via the loop thread) ASR.
        """
        state = self.machine.state

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
            score = self.wake_engine.score(pcm)
            if score > self.wake_engine.threshold:
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
        self.machine.mark("listening")
        StatusIndicator.wake_detected(self.wake_engine.threshold)

    def _on_speech_start(self, event, barge_in: bool) -> None:
        """What the beginning of an utterance means, decided purely by state."""
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
        try:
            if self.machine.state is State.IDLE:
                return  # speech arrived with no conversation to serve

            turn = self.machine.begin_turn()
            self.machine.mark("speech_end", turn.id)

            # -- TRANSCRIBING ------------------------------------------------
            self.machine.transition(State.TRANSCRIBING, reason="speech ended")
            text = self._transcribe(self._utterance, turn.id)
            turn.transcript = text
            if not text.strip():
                # Never invent a transcript. This is the one failure the user
                # most needs told about.
                self.machine.fail("I didn't catch that.")
                self._settle()
                return

            # -- THINKING ----------------------------------------------------
            self.machine.transition(State.THINKING, reason="transcript ready")
            reply = self._respond(text, turn.id)
            if self.machine.is_stale(turn.id):
                return  # interrupted while the model was thinking
            if not (reply or "").strip():
                self._settle()
                return

            # -- SPEAKING ----------------------------------------------------
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
            with self._lock:
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
        text = self.transcriber.transcribe(audio)
        if not self.machine.is_stale(turn_id):
            self.machine.mark("transcript", turn_id)
        return text

    def _respond(self, text: str, turn_id: str) -> str:
        """Ask the Brain. Runs inline; the result is discarded if interrupted."""
        self.machine.mark("first_token", turn_id)
        return self.respond(text)

    def _ask_brain(self, text: str) -> str:
        return self.brain.ask(text)

    def _speak(self, reply: str, turn_id: str) -> None:
        """Synthesize and play, checking staleness between chunks."""
        self.machine.transition(State.SPEAKING, reason="reply ready")
        is_stale = self.machine.is_stale

        # The conversational state only reaches the voice through the one
        # expressiveness control Chatterbox actually exposes. It is a hint, not
        # a mode: the same engine, same voice, same reference audio.
        state = get_tone().state
        self.player.exaggeration = state.exaggerated() or None

        queued = self.player.synthesize_to_queue(reply, turn_id, is_stale)
        if is_stale(turn_id) or queued == 0:
            return

        self.machine.mark("first_audio", turn_id)
        self.echo.begin_playback()
        self.player.start(turn_id, is_stale)
        self.player.wait(timeout=120.0)
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
            try:
                self.machine.transition(State.IDLE, reason="follow-up window elapsed")
            except Exception:  # noqa: BLE001
                self.machine.transition(State.IDLE, reason="follow-up", force=True)

    def _settle(self) -> None:
        """Return to IDLE after a turn that produced no reply."""
        try:
            self.machine.transition(State.IDLE, reason="turn settled")
        except Exception:  # noqa: BLE001
            self.machine.transition(State.IDLE, reason="turn settled", force=True)

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
            "wake_enabled": self.wake_enabled,
            "wake_model": self.wake_engine.model,
            "wake_threshold": self.wake_engine.threshold,
            "wake_events": self.wake_events,
            "wake_rejections": self._wake_rejections,
            "interrupts": self.interrupts,
            "follow_up_window_s": self.follow_up_window,
            "microphone": self.microphone.status(),
            "audio_queue_depth": self.player.queue.peek_len(),
            "audio_discarded_stale": self.player.queue.discarded_stale,
            "audio_discarded_cancelled": self.player.chunks_cancelled,
            "echo_suppressing": self.echo.should_suppress(),
            "echo_canceller": self.canceller.status(),
            "last_error": self.last_error,
            "state": self.machine.snapshot(include_history=False),
        }


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------

_instance: Optional[VoiceLoop] = None
_lock = threading.Lock()


def start_voice_loop(brain, *, force: bool = False) -> Optional[VoiceLoop]:
    """Start the voice loop if it is configured on. Returns the instance or None."""
    global _instance

    if not (force or WAKE_WORD_ENABLED):
        logger.info("[VOICE] voice disabled by configuration; microphone not opened")
        return None

    with _lock:
        if _instance is not None and _instance.is_running():
            return _instance
        loop = VoiceLoop(brain=brain, transcriber=Transcriber())
        _instance = loop
    return loop if loop.start() else None


def stop_voice_loop() -> None:
    """Stop and release the process-wide voice loop, if there is one."""
    global _instance
    with _lock:
        loop, _instance = _instance, None
    if loop is not None:
        loop.stop()


def get_voice_loop() -> Optional[VoiceLoop]:
    return _instance


__all__ = [
    "VoiceLoop",
    "get_voice_loop",
    "start_voice_loop",
    "stop_voice_loop",
]