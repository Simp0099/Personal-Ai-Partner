"""Phase 4 — the conversational state machine.

One explicit object owns "what is Ai Partner doing right now". Everything else
reads it: the voice loop drives it, the API publishes it, the HUD renders it.

Three properties this module exists to guarantee:

* **Invalid transitions are refused, not absorbed.** ``THINKING -> SPEAKING ->
    LISTENING`` is a bug that would leave the assistant claiming to listen while
    audio plays. Raising makes it visible instead of corrupting state.
* **Turns are identified.** Every conversational turn gets a monotonic id, and
    anything carrying audio is tagged with the turn that produced it. A response
    from turn 1 that arrives after turn 2 started is *stale* and is dropped at
    every boundary, which is what stops two answers talking over each other.
* **It holds no resources.** No threads, no audio, no model. It is pure state,
    so every rule in it is testable without a microphone.

Timing is recorded per stage rather than guessed at: the machine is where the
pipeline's latency is actually measured (Part R), not in a comment somewhere.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from jarvis.logger import logger


class State(str, Enum):
    """Every state Ai Partner can be in.

    Values are the wire format: the HUD and the API speak these strings, so the
    backend stays the single definition of them.
    """

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOLLOW_UP = "follow_up"
    INTERRUPTED = "interrupted"
    ERROR = "error"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Legal transitions. Anything absent is refused.
#:
#: Written out in full rather than derived, because the whole value of a state
#: machine is that the illegal edges are visible in one place. Two edges are
#: worth noting:
#:
#: * ``* -> ERROR`` and ``ERROR -> IDLE`` exist so a failure is reported as a
#:   state the HUD can show, not swallowed into a silent reset.
#: * ``INTERRUPTED -> LISTENING`` is the barge-in path; ``INTERRUPTED -> IDLE``
#:   covers an interruption with no user audio to listen to afterwards.
_ALLOWED: Dict[State, Set[State]] = {
    State.IDLE: {State.LISTENING, State.ERROR},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.THINKING, State.INTERRUPTED, State.ERROR, State.IDLE},
    State.THINKING: {State.SPEAKING, State.INTERRUPTED, State.ERROR,
                      State.FOLLOW_UP, State.IDLE},
    State.SPEAKING: {State.FOLLOW_UP, State.INTERRUPTED, State.ERROR,
                     State.THINKING, State.IDLE},
    State.FOLLOW_UP: {State.LISTENING, State.IDLE, State.ERROR},
    State.INTERRUPTED: {State.LISTENING, State.IDLE, State.ERROR},
    State.ERROR: {State.IDLE, State.LISTENING},
}


class InvalidTransition(RuntimeError):
    """A transition that the state machine does not allow was attempted.

    Raised rather than logged: silently entering an impossible state is how a
    conversation loop starts speaking to itself.
    """


#: Pipeline stages timed for latency reporting. The names are the wire format.
STAGES = (
    "wake_detected",
    "listening",
    "speech_end",
    "transcript",
    "first_token",
    "first_audio",
    "playback_start",
    "playback_end",
    "interrupt_detected",
    "interrupt_complete",
)


@dataclass
class Turn:
    """One conversational exchange, from the user's speech to the last audio."""

    id: str
    transcript: str = ""
    #: Set when the turn is cancelled or superseded. Stale work checks this.
    cancelled: bool = False
    started_at: float = field(default_factory=time.monotonic)
    ended_at: Optional[float] = None
    #: Stage name -> monotonic timestamp, for the latency report.
    marks: Dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str, when: Optional[float] = None) -> None:
        """Record when a pipeline stage happened. First mark per stage wins."""
        if stage in STAGES:
            self.marks.setdefault(stage, when if when is not None else time.monotonic())

    @property
    def duration_ms(self) -> float:
        return round(((self.ended_at or time.monotonic()) - self.started_at) * 1000, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "transcript": self.transcript,
            "cancelled": self.cancelled,
            "duration_ms": self.duration_ms,
            "marks": {k: round(v - self.started_at, 3) for k, v in self.marks.items()},
        }


class ConversationMachine:
    """Authoritative conversational state, plus the current turn's identity.

    Thread-safe: the voice loop, the audio thread and the API all touch it.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.RLock()
        self._state = State.IDLE
        self._turn: Optional[Turn] = None
        self._turn_counter = 0
        self._turn_serial = 0
        self._listeners: List[Callable[["StateEvent"], None]] = []
        self._last_error: Optional[str] = None
        #: Stage timings for the current pipeline, including stages that happen
        #: before a turn exists (the wake word is detected before turn 1 begins).
        self._marks: Dict[str, float] = {}
        #: Rolling window of recent transitions, newest last. Bounded so a long
        #: session cannot grow this without limit.
        self._history: List[Dict[str, Any]] = []
        self._history_limit = 50

    # ------------------------------------------------------------------
    # Reading state
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def turn(self) -> Optional[Turn]:
        with self._lock:
            return self._turn

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def is_stale(self, turn_id: Optional[str]) -> bool:
        """True when `turn_id` is not the turn currently in progress.

        The one question every audio and LLM boundary must ask before acting on
        work it started earlier. A cancelled turn is stale even if it is still
        the current one, so a cancelled response can never resume.
        """
        with self._lock:
            if turn_id is None:
                return True
            if self._turn is None or self._turn.id != turn_id:
                return True
            return self._turn.cancelled

    # ------------------------------------------------------------------
    # Writing state
    # ------------------------------------------------------------------

    def transition(
        self,
        state: State,
        *,
        reason: str = "",
        force: bool = False,
    ) -> StateEvent:
        """Move to `state`, or raise :class:`InvalidTransition`.

        Args:
            state: Target state.
            reason: Short human-readable cause, surfaced in the event stream.
            force: Skip the legality check. Reserved for shutdown, where the
                process is ending and every state is equally valid.

        Returns:
            The emitted :class:`StateEvent`.
        """
        with self._lock:
            previous = self._state
            if state is not previous and not force and state not in _ALLOWED[previous]:
                raise InvalidTransition(
                    f"cannot go {previous.value} -> {state.value}"
                    + (f" ({reason})" if reason else "")
                )
            self._state = state
            # `last_error` is deliberately NOT cleared here. A turn that fails
            # reports ERROR and then settles to IDLE, and clearing on that return
            # would erase the only explanation of why the assistant went quiet.
            # It is cleared when the next turn starts instead.
            event = StateEvent(
                state=state,
                previous=previous,
                reason=reason,
                turn_id=self._turn.id if self._turn else None,
                at=self._clock(),
            )
            self._history.append(event.to_dict())
            del self._history[:-self._history_limit]
            listeners = list(self._listeners)
        # Outside the lock: a listener that blocks must not freeze the machine.
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - a bad observer cannot break the loop
                pass
        return event

    def fail(self, error: str) -> StateEvent:
        """Enter ERROR carrying `error`, from any state."""
        with self._lock:
            self._last_error = error
        return self.transition(State.ERROR, reason=error, force=True)

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------

    def begin_turn(self, transcript: str = "") -> Turn:
        """Start a new turn, cancelling whatever came before it.

        Cancelling the previous turn here is what makes barge-in safe: the old
        turn is marked stale *before* the new one starts, so any audio or model
        output still in flight from it is rejected on arrival.
        """
        with self._lock:
            if self._turn is not None:
                self._turn.cancelled = True
                self._turn.ended_at = self._clock()
            self._turn_serial += 1
            turn = Turn(id=f"turn_{self._turn_serial:03d}", transcript=transcript)
            self._turn = turn
            # A new turn supersedes whatever the last one failed on.
            self._last_error = None
            self._turn_counter = self._turn_serial
        return turn

    def cancel_turn(self, reason: str = "") -> Optional[Turn]:
        """Cancel the current turn. Its audio and model output become stale."""
        with self._lock:
            turn = self._turn
            if turn is None:
                return None
            turn.cancelled = True
            turn.ended_at = self._clock()
        return turn

    def end_turn(self, turn_id: Optional[str] = None) -> Optional[Turn]:
        """Close a turn without cancelling it (it completed normally)."""
        with self._lock:
            turn = self._turn
            if turn is None or (turn_id is not None and turn.id != turn_id):
                return None
            turn.ended_at = self._clock()
        return turn

    def mark(self, stage: str, turn_id: Optional[str] = None) -> None:
        """Timestamp a pipeline stage. Ignored if it names a stale turn.

        Always recorded on the machine as well, because the wake word is
        detected before any turn exists and its latency is part of the number
        that matters.
        """
        with self._lock:
            now = self._clock()
            self._marks.setdefault(stage, now)
            turn = self._turn
            if turn is None or (turn_id is not None and turn.id != turn_id):
                return
            turn.mark(stage, now)

    def reset(self) -> None:
        """Return to IDLE, cancelling any turn. Used on shutdown."""
        with self._lock:
            if self._turn is not None:
                self._turn.cancelled = True
                self._turn.ended_at = self._clock()
            self._turn = None
        self.transition(State.IDLE, reason="reset", force=True)

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def subscribe(self, listener: Callable[["StateEvent"], None]) -> Callable[[], None]:
        """Register a state listener. Returns an unsubscribe callable."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def snapshot(self, *, include_history: bool = True) -> Dict[str, Any]:
        """Current state for the API and the HUD. Contains no audio."""
        with self._lock:
            turn = self._turn
            return {
                "state": self._state.value,
                "turn_id": turn.id if turn else None,
                "turn_cancelled": bool(turn and turn.cancelled),
                "transcript": turn.transcript if turn else "",
                "last_error": self._last_error,
                "turn": turn.to_dict() if turn else None,
                "turns_completed": self._turn_counter,
                # Offsets from the first stage to happen. `min` over the values,
                # not the keys: min(self._marks) would return an alphabetically
                # first stage name and produce negative offsets.
                "marks": ({k: round(v - min(self._marks.values()), 3)
                           for k, v in self._marks.items()} if self._marks else {}),
                "history": list(self._history) if include_history else [],
            }


@dataclass(frozen=True)
class StateEvent:
    """One state change, as published to observers and the HUD."""

    state: State
    previous: State
    reason: str = ""
    turn_id: Optional[str] = None
    at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "state",
            "state": self.state.value,
            "previous": self.previous.value,
            "reason": self.reason,
            "turn_id": self.turn_id,
        }


# ============================================================================
# Phase 7 — assistant lifecycle (4-state view over the detailed machine)
# ============================================================================
# The detailed ConversationMachine stays authoritative for voice internals.
# This layer coordinates the cross-mode lifecycle (voice + text loops) in the
# four states the assistant can be observed in. It never owns audio, models,
# or TTS lifetime: IDLE means "no active request", not "unload anything".


class AssistantState(str, Enum):
    """Observable assistant lifecycle. Values are the log wire format."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Allowed lifecycle edges. Barge-in is real architecture (INTERRUPTABLE
#: voice states, echo guard), so PROCESSING/SPEAKING may return to LISTENING
#: when the user takes the floor; everything else follows the normal
#: IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE chain plus
#: cancellation/error recovery to IDLE.
_LIFECYCLE_ALLOWED: Dict[AssistantState, Set[AssistantState]] = {
    AssistantState.IDLE: {AssistantState.LISTENING},
    AssistantState.LISTENING: {AssistantState.PROCESSING, AssistantState.IDLE},
    AssistantState.PROCESSING: {AssistantState.SPEAKING, AssistantState.IDLE,
                                AssistantState.LISTENING},
    AssistantState.SPEAKING: {AssistantState.IDLE, AssistantState.LISTENING},
}


def detailed_to_lifecycle(state: State) -> AssistantState:
    """Project a detailed conversation state onto the 4-state lifecycle."""
    if state is State.LISTENING:
        return AssistantState.LISTENING
    if state in (State.TRANSCRIBING, State.THINKING):
        return AssistantState.PROCESSING
    if state is State.SPEAKING:
        return AssistantState.SPEAKING
    return AssistantState.IDLE


class AssistantStateMachine:
    """Single authoritative coordinator for the observable assistant lifecycle.

    Thread-safe (RLock), atomic transitions, DEBUG-only logging. Holds no
    resources: no audio, no model, no TTS handle -- so IDLE can never unload
    anything. One instance is shared process-wide via
    :func:`get_assistant_machine`; tests construct their own.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._state = AssistantState.IDLE

    @property
    def state(self) -> AssistantState:
        with self._lock:
            return self._state

    def transition_to(self, state: AssistantState, reason: str = "") -> bool:
        """Move to `state`. Same-state is a no-op success; illegal edges raise
        :class:`InvalidTransition` and leave state untouched. Reasons must be
        fixed lifecycle phrases, never user content."""
        with self._lock:
            previous = self._state
            if state is previous:
                return True
            if state not in _LIFECYCLE_ALLOWED[previous]:
                raise InvalidTransition(
                    f"cannot go {previous.value} -> {state.value}"
                    + (f" ({reason})" if reason else "")
                )
            self._state = state
        logger.debug(f"assistant lifecycle: {previous.value} -> {state.value}"
                     + (f" ({reason})" if reason else ""))
        return True

    def observe(self, detailed: State, reason: str = "") -> bool:
        """Mirror a detailed-machine transition. Never raises: a derived view
        must not break its driver; impossible projections are logged + skipped."""
        target = detailed_to_lifecycle(detailed)
        try:
            return self.transition_to(target, reason or f"observed {detailed.value}")
        except InvalidTransition as e:
            logger.debug(f"assistant lifecycle: skipped {e}")
            return False

    def recover(self, reason: str = "recovered") -> bool:
        """Force IDLE after a failure or cancellation. Always succeeds."""
        with self._lock:
            previous = self._state
            self._state = AssistantState.IDLE
        if previous is not AssistantState.IDLE:
            logger.debug(f"assistant lifecycle: {previous.value} -> idle ({reason})")
        return True

    def reset(self) -> None:
        """Return to IDLE. Used on shutdown."""
        self.recover("reset")


_assistant_machine: Optional[AssistantStateMachine] = None
_assistant_lock = threading.Lock()


def get_assistant_machine() -> AssistantStateMachine:
    """Process-wide lifecycle coordinator. One assistant, one lifecycle."""
    global _assistant_machine
    with _assistant_lock:
        if _assistant_machine is None:
            _assistant_machine = AssistantStateMachine()
        return _assistant_machine


__all__ = [
    "AssistantState",
    "AssistantStateMachine",
    "ConversationMachine",
    "InvalidTransition",
    "STAGES",
    "State",
    "StateEvent",
    "Turn",
    "detailed_to_lifecycle",
    "get_assistant_machine",
]