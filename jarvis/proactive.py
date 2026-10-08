"""Phase 6 — proactive partner: knowing when talking is worth it.

Moves the interaction model from ``User -> AI`` toward ``User <-> AI`` by
letting the assistant occasionally initiate a useful, natural interaction.

The pipeline is deliberately one-way and deterministic:

```text
PERCEPTION (Phase 3 VisualContext, owned elsewhere)
    -> OBSERVATION (ObservationEngine: did something meaningful happen?)
    -> CONTEXT (ContextSnapshot: compact facts, no raw frames)
    -> DECISION (ProactiveDecisionEngine: deterministic gates, default NO)
    -> REACTION (caller runs the existing Brain + existing Speech Pipeline)
    -> CONTINUITY (ephemeral history: cooldowns, dedup, rate limits)
```

Rules that keep it honest:

* **Default NO.** Most observations result in silence. Every gate failure is NO.
* **Camera never triggers speech.** The camera only feeds ``VisualContext``;
  this module reads observations and still usually says NO. There is no import
  path from here to any microphone, TTS engine, or state machine.
* **Observation, not inference.** Only ``observation``-kind lines from vision
  are eligible. ``inferred:``/``unclear:`` lines can never become candidates,
  so the LLM cannot hallucinate sensory evidence into a spoken event.
* **No second anything.** No model call, no thread, no database, no emotion
  system. Tone state is *read* for style hints, never written. Memory is never
  touched. The expensive Brain runs only after YES, driven by the caller.
* **Fail closed.** Any exception in ``maybe_proactive`` returns ``None``.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from jarvis.config import (
    PROACTIVE_CONFIDENCE_THRESHOLD,
    PROACTIVE_COOLDOWN_S,
    PROACTIVE_DEDUP_S,
    PROACTIVE_ENABLED,
    PROACTIVE_MAX_PER_WINDOW,
    PROACTIVE_MIN_REPEAT,
    PROACTIVE_QUIET_END,
    PROACTIVE_QUIET_ENABLED,
    PROACTIVE_QUIET_START,
    PROACTIVE_WINDOW_S,
)
from jarvis.logger import logger
from jarvis.tone import ConversationMode, ConversationState, Mood
from jarvis.vision import OBSERVATION, Observation

# ---------------------------------------------------------------------------
# Availability: an interaction/system concept, never a psychological diagnosis
# ---------------------------------------------------------------------------

UNKNOWN = "unknown"
UNAVAILABLE = "unavailable"
PROBABLY_AVAILABLE = "probably_available"
CLEARLY_AVAILABLE = "clearly_available"
_AVAILABLE = (PROBABLY_AVAILABLE, CLEARLY_AVAILABLE)

# ---------------------------------------------------------------------------
# Candidate reasons
# ---------------------------------------------------------------------------

USER_RETURNED = "user_returned"
GLASSES_ABSENT = "glasses_absent"
PROLONGED_WORK = "prolonged_work"

_RETURNED_RE = re.compile(r"\b(returned|came back|is back|re-?entered)\b", re.I)
_ABSENT_RE = re.compile(r"\b(absent|missing|not (?:present|visible|there)|gone|left|empty)\b", re.I)
_GLASSES_RE = re.compile(r"\bglasses\b", re.I)
_WORK_RE = re.compile(r"\b(working|typing|coding|writing|editing)\b", re.I)
_PERSON_RE = re.compile(r"\b(person|user|someone|figure|individual)\b", re.I)


@dataclass(frozen=True)
class Candidate:
    """A meaningful observation asking to be considered. Not permission."""

    reason: str
    confidence: float
    text: str
    repeat_count: int = 1
    at: float = field(default_factory=time.time)


def _classify(texts: List[str]) -> Optional[Tuple[str, float, str]]:
    """Map observation texts to (reason, confidence, short text). Pure."""
    combined = " ".join(texts)
    has_person = bool(_PERSON_RE.search(combined))
    if _RETURNED_RE.search(combined) and (has_person or "desk" in combined.lower()):
        return USER_RETURNED, 0.7, "user returned"
    if _GLASSES_RE.search(combined) and _ABSENT_RE.search(combined):
        return GLASSES_ABSENT, 0.6, "glasses absent"
    if _WORK_RE.search(combined):
        return PROLONGED_WORK, 0.5, "prolonged work session"
    # Alone these are noise, never candidates: presence, light, movement.
    return None


class ObservationEngine:
    """Answers only 'did something meaningful happen?' — never 'should I speak?'.

    Persistence is counted across *updates*, not within one scene: one frame is
    never enough, no matter how many lines it produced. ``note()`` is fed each
    fresh VisualContext reading; only observation-kind lines count.
    """

    def __init__(
        self,
        *,
        min_repeat: int = PROACTIVE_MIN_REPEAT,
        clock: Callable[[], float] = time.time,
    ):
        self.min_repeat = max(1, int(min_repeat))
        self.clock = clock
        self._sightings: Dict[str, int] = {}
        self._lock = threading.RLock()

    def note(self, observations: List[Observation]) -> Optional[Candidate]:
        """Fold one context reading into repeat counts. Returns a candidate once
        its reason has been seen ``min_repeat`` consecutive times."""
        texts = [o.text for o in observations or [] if o.kind == OBSERVATION]
        found = _classify(texts) if texts else None
        with self._lock:
            if found is None:
                # Nothing meaningful: a gap breaks every streak. A return that
                # vanishes for a frame was noise, not an event.
                self._sightings.clear()
                return None
            reason, confidence, short = found
            count = self._sightings.get(reason, 0) + 1
            # A different reason arriving resets the others: only one streak runs.
            self._sightings = {reason: count}
            if count < self.min_repeat:
                logger.debug(f"PROACTIVE candidate {reason} seen {count}x, waiting")
                return None
            return Candidate(reason=reason, confidence=confidence, text=short,
                             repeat_count=count, at=self.clock())

    def reset(self) -> None:
        with self._lock:
            self._sightings.clear()


# ---------------------------------------------------------------------------
# Context snapshot: everything the decision may look at, nothing more
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextSnapshot:
    """Compact facts for one decision. No frames, no histories, no reasoning."""

    user_availability: str = UNKNOWN
    active_conversation: bool = False
    user_speaking: bool = False
    assistant_speaking: bool = False
    wake_active: bool = False
    proactive_enabled: bool = PROACTIVE_ENABLED
    suppressed: bool = False
    quiet_hours: bool = False
    cooldown_remaining: float = 0.0
    recently_spoken: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ProactiveDecisionResult:
    """The decision. ``directive`` is the only thing the Brain ever sees."""

    should_speak: bool
    reason: str
    confidence: float = 0.0
    directive: str = ""


class ProactiveDecisionEngine:
    """Deterministic gates. Any failure is NO; there is no scoring blend."""

    def __init__(
        self,
        *,
        confidence_threshold: float = PROACTIVE_CONFIDENCE_THRESHOLD,
    ):
        self.confidence_threshold = float(confidence_threshold)

    # -- gates -----------------------------------------------------------
    def decide(self, candidate: Optional[Candidate],
               snap: ContextSnapshot) -> ProactiveDecisionResult:
        """Run the gates in cheapest-first order."""
        if candidate is None:
            return ProactiveDecisionResult(False, "no_observation")
        if not snap.proactive_enabled:
            return ProactiveDecisionResult(False, "proactive_disabled",
                                           candidate.confidence)
        if snap.suppressed or snap.quiet_hours:
            return ProactiveDecisionResult(False, "suppressed",
                                           candidate.confidence)
        if candidate.confidence < self.confidence_threshold:
            return ProactiveDecisionResult(False, "low_confidence",
                                           candidate.confidence)
        if snap.user_availability not in _AVAILABLE:
            return ProactiveDecisionResult(False, "user_unavailable",
                                           candidate.confidence)
        if snap.active_conversation or snap.user_speaking:
            return ProactiveDecisionResult(False, "active_conversation",
                                           candidate.confidence)
        if snap.wake_active:
            return ProactiveDecisionResult(False, "wake_word_active",
                                           candidate.confidence)
        if snap.assistant_speaking:
            return ProactiveDecisionResult(False, "assistant_speaking",
                                           candidate.confidence)
        if snap.cooldown_remaining > 0:
            return ProactiveDecisionResult(False, "cooldown_active",
                                           candidate.confidence)
        if candidate.reason in set(snap.recently_spoken or ()):
            return ProactiveDecisionResult(False, "duplicate",
                                           candidate.confidence)
        return ProactiveDecisionResult(True, candidate.reason,
                                       candidate.confidence,
                                       _directive_for(candidate.reason))


def _directive_for(reason: str) -> str:
    """Concise structured context for the existing Brain. No internals leak:
    no scores, no gate names, no state labels."""
    behaviour = {
        USER_RETURNED: "Brief natural acknowledgement, e.g. a short greeting.",
        GLASSES_ABSENT: "Brief low-pressure question about the glasses.",
        PROLONGED_WORK: "Brief light check-in, no nagging, no durations.",
    }.get(reason, "Brief natural remark.")
    return (
        "\n\n## Proactive Context\n"
        "You are initiating this turn because a quiet background observation "
        f"suggested it ({reason}). {behaviour} Keep it to one short sentence, "
        "do not mention internal state, scores, the perception pipeline, or "
        "how you noticed. Do not ask unnecessary follow-up questions."
    )


def style_hint(state: Optional[ConversationState]) -> str:
    """One-line tone flavour for proactive wording. Reads Phase 5, writes nothing."""
    if state is None:
        return "natural and conversational"
    if state.user_state.value == "frustrated":
        return "extra restrained; do not interrupt unless it truly helps"
    if state.mood is Mood.PLAYFUL:
        return "light and natural"
    if state.mood in (Mood.SERIOUS, Mood.FOCUSED) \
            or state.mode in (ConversationMode.TECHNICAL, ConversationMode.SERIOUS,
                              ConversationMode.TASK_EXECUTION):
        return "restrained, direct, practical"
    if state.mood is Mood.CALM or state.mode is ConversationMode.FOCUSED:
        return "calm and concise"
    return "natural and conversational"


# ---------------------------------------------------------------------------
# Orchestrator: continuity (cooldowns, dedup, rate limits, history). No threads.
# ---------------------------------------------------------------------------

class ProactiveOrchestrator:
    """Owns ephemeral continuity. No I/O, no model, no clock surprises."""

    def __init__(
        self,
        engine: Optional[ProactiveDecisionEngine] = None,
        *,
        cooldown_s: float = PROACTIVE_COOLDOWN_S,
        dedup_s: float = PROACTIVE_DEDUP_S,
        max_per_window: int = PROACTIVE_MAX_PER_WINDOW,
        window_s: float = PROACTIVE_WINDOW_S,
        quiet_enabled: bool = PROACTIVE_QUIET_ENABLED,
        quiet_start: int = PROACTIVE_QUIET_START,
        quiet_end: int = PROACTIVE_QUIET_END,
        clock: Callable[[], float] = time.time,
        hour: Optional[Callable[[], int]] = None,
    ):
        self.engine = engine or ProactiveDecisionEngine()
        self.cooldown_s = float(cooldown_s)
        self.dedup_s = float(dedup_s)
        self.max_per_window = int(max_per_window)
        self.window_s = float(window_s)
        self.quiet_enabled = bool(quiet_enabled)
        self.quiet_start = int(quiet_start)
        self.quiet_end = int(quiet_end)
        self.clock = clock
        self._hour = hour
        self._lock = threading.RLock()
        self._spoken_at: Dict[str, float] = {}
        self._spoken_times: List[float] = []
        self._history: List[Dict[str, Any]] = []

    # -- derived state ---------------------------------------------------
    def quiet_active(self) -> bool:
        if not self.quiet_enabled:
            return False
        h = self._hour() if self._hour else time.localtime().tm_hour
        if self.quiet_start <= self.quiet_end:
            return self.quiet_start <= h < self.quiet_end
        return h >= self.quiet_start or h < self.quiet_end

    def cooldown_remaining(self, now: Optional[float] = None) -> float:
        now = self.clock() if now is None else now
        with self._lock:
            if not self._spoken_times:
                return 0.0
            return max(0.0, self.cooldown_s - (now - self._spoken_times[-1]))

    def recent_reasons(self, now: Optional[float] = None) -> Tuple[str, ...]:
        now = self.clock() if now is None else now
        with self._lock:
            return tuple(r for r, at in self._spoken_at.items()
                         if now - at < self.dedup_s)

    def _window_count(self, now: float) -> int:
        cutoff = now - self.window_s
        self._spoken_times = [t for t in self._spoken_times if t >= cutoff]
        return len(self._spoken_times)

    # -- main entry ------------------------------------------------------
    def maybe_proactive(
        self,
        candidate: Optional[Candidate],
        snap: ContextSnapshot,
        generate: Callable[[str], str],
    ) -> Optional[str]:
        """Decide, then generate via the caller's Brain exactly once.

        ``generate`` receives the directive and returns reply text. Any failure
        — provider, TTS, or a bug in generate — returns ``None`` silently and
        records nothing, so there are never autonomous retry loops.
        """
        now = self.clock()
        with self._lock:
            windowed = ContextSnapshot(
                user_availability=snap.user_availability,
                active_conversation=snap.active_conversation,
                user_speaking=snap.user_speaking,
                assistant_speaking=snap.assistant_speaking,
                wake_active=snap.wake_active,
                proactive_enabled=snap.proactive_enabled,
                suppressed=snap.suppressed,
                quiet_hours=snap.quiet_hours or self.quiet_active(),
                cooldown_remaining=max(snap.cooldown_remaining,
                                       self.cooldown_remaining(now)),
                recently_spoken=snap.recently_spoken or self.recent_reasons(now),
            )
            if self._window_count(now) >= self.max_per_window:
                logger.info("PROACTIVE rejected: reason=rate_limited")
                self._history.append({"reason": "rate_limited", "at": now,
                                      "spoken": False})
                return None
            result = self.engine.decide(candidate, windowed)
        if not result.should_speak:
            logger.info(f"PROACTIVE rejected: reason={result.reason}")
            with self._lock:
                self._history.append({"reason": result.reason, "at": now,
                                      "spoken": False})
            return None
        try:
            # ponytail: single attempt, no retries; failure means silence
            reply = generate(result.directive)
        except Exception as e:  # noqa: BLE001 - proactive must never break JARVIS
            logger.warning(f"PROACTIVE generation failed silently: {e}")
            with self._lock:
                self._history.append({"reason": result.reason, "at": now,
                                      "spoken": False})
            return None
        if not (reply or "").strip():
            with self._lock:
                self._history.append({"reason": result.reason, "at": now,
                                      "spoken": False})
            return None
        logger.info(f"PROACTIVE approved: reason={result.reason}")
        with self._lock:
            self._spoken_at[result.reason] = now
            self._spoken_times.append(now)
            self._history.append({"reason": result.reason, "at": now,
                                  "spoken": True})
        return reply

    def status(self) -> Dict[str, Any]:
        now = self.clock()
        with self._lock:
            return {
                "enabled": bool(PROACTIVE_ENABLED),
                "quiet_hours_active": self.quiet_active(),
                "cooldown_remaining": round(self.cooldown_remaining(now), 1),
                "recent_reasons": list(self.recent_reasons(now)),
                "interactions_in_window": self._window_count(now),
                "max_per_window": self.max_per_window,
                "history": list(self._history[-10:]),
            }

    def reset(self) -> None:
        with self._lock:
            self._spoken_at.clear()
            self._spoken_times.clear()
            self._history.clear()


# ---------------------------------------------------------------------------
# Process-wide instances (match jarvis.tone / jarvis.vision patterns)
# ---------------------------------------------------------------------------

_orchestrator: Optional[ProactiveOrchestrator] = None
_observer: Optional[ObservationEngine] = None
_lock = threading.Lock()


def get_orchestrator() -> ProactiveOrchestrator:
    global _orchestrator
    with _lock:
        if _orchestrator is None:
            _orchestrator = ProactiveOrchestrator()
        return _orchestrator


def get_observer() -> ObservationEngine:
    global _observer
    with _lock:
        if _observer is None:
            _observer = ObservationEngine()
        return _observer


def reset_proactive() -> None:
    global _orchestrator, _observer
    with _lock:
        _orchestrator = None
        _observer = None


# Backwards-compatible aliases for the first-draft integration.
def get_proactive_engine() -> ProactiveDecisionEngine:
    return get_orchestrator().engine


def get_observation_engine() -> ObservationEngine:
    return get_observer()


__all__ = [
    "CLEARLY_AVAILABLE",
    "GLASSES_ABSENT",
    "PROBABLY_AVAILABLE",
    "PROLONGED_WORK",
    "UNAVAILABLE",
    "UNKNOWN",
    "USER_RETURNED",
    "Candidate",
    "ContextSnapshot",
    "ObservationEngine",
    "ProactiveDecisionEngine",
    "ProactiveDecisionResult",
    "ProactiveOrchestrator",
    "get_observation_engine",
    "get_observer",
    "get_orchestrator",
    "get_proactive_engine",
    "reset_proactive",
    "style_hint",
]
