"""Phase 5 — conversation state: how JARVIS is talking right now.

Personality answers *who* JARVIS is and lives in the identity prompt. This
module answers *how* JARVIS is behaving in this conversation, and it is a
deliberately different thing: a small, fast-changing set of numbers that shapes
style without ever replacing personality, judgment or accuracy.

```text
PERSONALITY        = who JARVIS is          (stable, identity prompt)
CONVERSATION STATE = how JARVIS talks now   (this module, decays)
MEMORY             = what JARVIS may keep   (jarvis.memory, explicit only)
VISUAL CONTEXT     = what JARVIS can see    (jarvis.vision, ephemeral)
```

Design rules that keep it honest:

* **No second model.** Everything here is deterministic keyword/regex work on
  text the Brain already has. No emotion-detection LLM, no extra network call,
  no background thread — state updates are microseconds and cost nothing.
* **Conservative inference.** ``user_state`` is set only from what the user
  *said about themselves*. Absence of evidence is ``UNKNOWN``, not a guess.
  Nothing here ever looks at the camera, a face, a voice or a posture; visual
  perception is structurally unable to reach this module.
* **A soft signal.** The state shapes style. It cannot outrank accuracy, an
  instruction, safety, or a tool requirement — that is stated in the prompt the
  model receives, and asserted in the tests.
* **Ephemeral.** Nothing here writes to memory. A mood is not a fact about
  someone, and "I'm tired today" must not become a durable memory.
* **Not a state machine.** Phase 4's :class:`jarvis.conversation.
  ConversationMachine` decides *when* JARVIS listens and speaks. This decides
  *how* it speaks. They are independent and both may be true at once.

Naming note: Phase 4 owns "conversational state machine"; this module owns
"conversation state". They are deliberately separate names to keep them from
being confused.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from jarvis.config import (
    TONE_CURIOUSITY_DEFAULT,
    TONE_DECAY_ENABLED,
    TONE_DECAY_HALF_LIFE,
    TONE_ENABLED,
    TONE_ENERGY_DEFAULT,
    TONE_MOOD_DEFAULT,
    TONE_MODE_DEFAULT,
    TONE_WARMTH_DEFAULT,
)
from jarvis.logger import logger


class Mood(str, Enum):
    """The prevailing flavour of the reply. Style, not identity."""

    NEUTRAL = "neutral"
    PLAYFUL = "playful"
    CURIOUS = "curious"
    FOCUSED = "focused"
    SUPPORTIVE = "supportive"
    CELEBRATORY = "celebratory"
    SERIOUS = "serious"
    CALM = "calm"
    ENERGETIC = "energetic"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ConversationMode(str, Enum):
    """What kind of conversation this is. Drives register and brevity."""

    CASUAL = "casual"
    FOCUSED = "focused"
    TECHNICAL = "technical"
    SUPPORTIVE = "supportive"
    PLAYFUL = "playful"
    CELEBRATORY = "celebratory"
    SERIOUS = "serious"
    BRAINSTORMING = "brainstorming"
    TASK_EXECUTION = "task_execution"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class UserState(str, Enum):
    """What the user said about themselves.

    ``UNKNOWN`` is the default and is preferred over a guess. This is set only
    from explicit self-report or unambiguous first-person phrasing — never from
    the camera, a face, a voice, a posture, or a demographic.
    """

    UNKNOWN = "unknown"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    TIRED = "tired"
    SAD = "sad"
    STRESSED = "stressed"
    CONFUSED = "confused"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    """Coerce anything into a bounded float.

    Public configuration and inference both feed this, so a bad value degrades to
    a sane number instead of raising into the conversation.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if number != number:            # NaN
        return low
    return max(low, min(high, number))


@dataclass(frozen=True)
class ConversationState:
    """One snapshot of how JARVIS is talking right now.

    Immutable on purpose: a snapshot handed to the prompt builder cannot be
    mutated into something the tracker never decided.
    """

    mood: Mood = Mood.NEUTRAL
    energy: float = TONE_ENERGY_DEFAULT
    warmth: float = TONE_WARMTH_DEFAULT
    curiosity: float = TONE_CURIOUSITY_DEFAULT
    mode: ConversationMode = ConversationMode.CASUAL
    user_state: UserState = UserState.UNKNOWN

    def __post_init__(self):
        # Frozen dataclasses still allow object.__setattr__ in __post_init__.
        # Clamping here means an invalid value can never leave this class, so
        # every consumer is safe without re-checking.
        object.__setattr__(self, "energy", _clamp(self.energy))
        object.__setattr__(self, "warmth", _clamp(self.warmth))
        object.__setattr__(self, "curiosity", _clamp(self.curiosity))
        object.__setattr__(self, "mood", _as_enum(Mood, self.mood, Mood.NEUTRAL))
        object.__setattr__(self, "mode", _as_enum(ConversationMode, self.mode,
                                                   ConversationMode.CASUAL))
        object.__setattr__(self, "user_state", _as_enum(UserState, self.user_state,
                                                         UserState.UNKNOWN))

    def as_dict(self) -> Dict[str, Any]:
        """Plain data for the API and the HUD. No reasoning, no prompts."""
        return {
            "mood": self.mood.value,
            "energy": round(self.energy, 3),
            "warmth": round(self.warmth, 3),
            "curiosity": round(self.curiosity, 3),
            "conversation_mode": self.mode.value,
            "user_emotional_state": self.user_state.value,
        }

    def exaggerated(self) -> bool:
        """Whether this state calls for a livelier delivery."""
        return self.energy >= 0.7 or self.mood in (
            Mood.CELEBRATORY, Mood.ENERGETIC, Mood.PLAYFUL
        )

    def composure(self) -> bool:
        """Whether this state calls for a calmer, flatter delivery."""
        return self.mood in (Mood.CALM, Mood.SERIOUS, Mood.FOCUSED) or self.energy <= 0.35


def _as_enum(enum_cls, value, default):
    """Coerce a value into an enum member, tolerating strings and junk."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).strip().lower())
    except (ValueError, AttributeError):
        return default


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

#: First-person self-report. The only strong evidence for `user_state`: the user
#: said it about themselves. Everything else is at most suggestive.
_EXPLICIT_STATE: Tuple[Tuple[re.Pattern, UserState], ...] = (
    (re.compile(r"\bi(?:'m| am)\s+(?:really\s+|so\s+|very\s+)?frustrat", re.I), UserState.FRUSTRATED),
    (re.compile(r"\bfrustrat(?:ed|ing)\b", re.I), UserState.FRUSTRATED),
    (re.compile(r"\b(?:stuck|blocked)\b.*\b(?:on|with|again)\b", re.I), UserState.FRUSTRATED),
    (re.compile(r"\bi(?:'m| am)\s+(?:so\s+|really\s+|very\s+)?(?:tired|exhausted|knocked)", re.I), UserState.TIRED),
    (re.compile(r"\b(?:exhausted|slept (?:like|on) (?:four|five|six|seven)|no sleep)\b", re.I), UserState.TIRED),
    (re.compile(r"\bi(?:'m| am)\s+(?:so\s+|really\s+|very\s+)?(?:excited|pumped|thrilled)\b", re.I), UserState.EXCITED),
    (re.compile(r"\b(?:can'?t wait|so hyped|so excited)\b", re.I), UserState.EXCITED),
    (re.compile(r"\bi(?:'m| am)\s+(?:really\s+|so\s+|very\s+)?(?:stressed|overwhelmed|swamped)\b", re.I), UserState.STRESSED),
    (re.compile(r"\b(?:stressed|overwhelmed)\b", re.I), UserState.STRESSED),
    (re.compile(r"\bi(?:'m| am)\s+(?:really\s+|so\s+|very\s+)?(?:sad|disappointed|gutted)\b", re.I), UserState.SAD),
    (re.compile(r"\bi\s+(?:don'?t|do not)\s+understand\b", re.I), UserState.CONFUSED),
    (re.compile(r"\b(?:i'?m|i am)\s+(?:really\s+|so\s+)?confused\b", re.I), UserState.CONFUSED),
    (re.compile(r"\bthat(?:'s| is)\s+(?:weird|strange|confusing)\b", re.I), UserState.CONFUSED),
    (re.compile(r"\b(?:it finally works|it works now|it'?s working|we got it|that worked|tests? pass(?:ed|ing)?)\b", re.I), UserState.POSITIVE),
    (re.compile(r"\b(?:thank you|thanks|perfect|awesome|brilliant|nailed it)\b", re.I), UserState.POSITIVE),
)

#: "Actually I'm fine." — an explicit resolution, which is as much the user's
#: authority over their own state as a denial is.
_RESOLVED = re.compile(
    r"\b(?:actually|i'?m|i am|we'?re|we are)\s+(?:fine|good|okay|ok|happy|"
    r"better now|all good)\b",
    re.I,
)

#: "I'm not frustrated." — an explicit correction always wins over inference.
_NEGATED_STATE = re.compile(
    r"\b(?:not|isn'?t|don'?t|no longer|never)\b[^.!?]{0,24}\b"
    r"(frustrat\w*|tired|exhausted|stressed|sad|confused|angry|upset|anxious)\b",
    re.I,
)

#: Mode signals. These steer *style* only; they never set `user_state`, because
#: a technical request is evidence about the topic, not about the person.
_TECHNICAL = re.compile(
    r"```|\b(?:stack ?trace|traceback|exception|segfault|null pointer|deadlock|"
    r"refactor|debug|deploy|schema|migration|endpoint|api|regex|sql|docker|"
    r"kubernetes|compile|import error|type ?error|test(?:s)? failing|\berrors?\b)\b",
    re.I,
)
_CELEBRATORY = re.compile(
    r"\b(?:it(?:'s| is)? ?(?:finally )?works|finally works|we did it|"
    r"it passed|all tests pass(?:ed)?|shipped it|fixed it)\b|[!]{2,}|\b(?:yess+|yay|woohoo)\b",
    re.I,
)
_SUPPORTIVE = re.compile(
    r"\b(?:been (?:at|stuck on|debugging) this for|still not working|"
    r"doesn'?t work|keeps? failing|for the (?:third|fourth|fifth) time|"
    r"ugh|argh|so annoying)\b",
    re.I,
)
_PLAYFUL = re.compile(
    r"\b(?:haha|hehe|lol|lmao|joke|funny|hilarious|banter|tease)\b|😂|😀|😉",
    re.I,
)
_SERIOUS = re.compile(
    r"\b(?:incident|outage|production is down|urgent|critical|legal|medical|"
    r"safety|serious issue)\b",
    re.I,
)
#: An imperative request to do something. Checked before the technical pattern
#: because "fix the failing test" is a task to carry out, not an explanation
#: to give -- and because without it a task following a supportive exchange was
#: never recognised as a change of register.
#: Filler a real request is wrapped in. "okay now help me optimise it" is a
#: task; requiring the verb in the first word threw that away.
_TASK_FILLER = (
    r"(?:okay|ok|alright|right|sure|great|cool|awesome|perfect|now|so|well|"
    r"please|and|then|also|hey jarvis|jarvis|first|next|again|let'?s|lets|"
    r"can you|could you|would you|will you|help me|try to|go ahead and)"
)
_TASK = re.compile(
    r"^\s*(?:" + _TASK_FILLER + r"[,\s]*)*(?:please\s+|help me\s+|now\s+)*"
    r"(?:add|remove|update|set up|configure|implement|build|write|create|make|"
    r"fix|optimi[sz]e|refactor|migrate|deploy|run|test|check|clean up|"
    r"rename|delete|install|switch|merge|revert|ship)\b",
    re.I,
)
_BRAINSTORMING = re.compile(
    r"\b(?:what if|ideas?|brainstorm|maybe we could|how about we|"
    r"thinking of|options? for)\b",
    re.I,
)
_CURIOUS = re.compile(
    r"\b(?:why does|how does|what happens if|i wonder|curious|"
    r"can you explain|walk me through)\b",
    re.I,
)
_SHORT_QUESTION = re.compile(r"\?\s*$")


#: Target state for each mode, as (mood, energy, warmth, curiosity). Keeping
#: these together makes the transitions reviewable in one place instead of
#: scattered through the update logic.
_MODE_TARGETS: Dict[ConversationMode, Tuple[Mood, float, float, float]] = {
    ConversationMode.CASUAL:        (Mood.NEUTRAL, 0.55, 0.75, 0.60),
    ConversationMode.TECHNICAL:     (Mood.FOCUSED, 0.45, 0.60, 0.70),
    ConversationMode.FOCUSED:       (Mood.FOCUSED, 0.55, 0.70, 0.70),
    ConversationMode.SUPPORTIVE:    (Mood.CALM,     0.40, 0.90, 0.55),
    ConversationMode.PLAYFUL:       (Mood.PLAYFUL,  0.75, 0.80, 0.75),
    ConversationMode.CELEBRATORY:  (Mood.CELEBRATORY, 0.85, 0.90, 0.70),
    ConversationMode.SERIOUS:      (Mood.SERIOUS,  0.40, 0.65, 0.60),
    ConversationMode.BRAINSTORMING: (Mood.CURIOUS, 0.70, 0.75, 0.95),
    ConversationMode.TASK_EXECUTION: (Mood.FOCUSED, 0.65, 0.65, 0.55),
}

#: Blend strength per turn. Below 1.0 so one message nudges rather than
#: commandeers, and a single enthusiastic sentence cannot hijack the tone.
_BLEND = 0.55


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ConversationTone:
    """Runtime owner of the conversation state.

    Decay is computed lazily on read rather than on a timer, which is why there
    is no background thread here: an idle assistant costs nothing, and there is
    nothing to leak at shutdown.
    """

    def __init__(
        self,
        *,
        defaults: Optional[ConversationState] = None,
        decay_enabled: bool = TONE_DECAY_ENABLED,
        half_life: float = TONE_DECAY_HALF_LIFE,
        blend: float = _BLEND,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.defaults = defaults or ConversationState(
            mood=_as_enum(Mood, TONE_MOOD_DEFAULT, Mood.NEUTRAL),
            energy=TONE_ENERGY_DEFAULT,
            warmth=TONE_WARMTH_DEFAULT,
            curiosity=TONE_CURIOUSITY_DEFAULT,
            mode=_as_enum(ConversationMode, TONE_MODE_DEFAULT, ConversationMode.CASUAL),
        )
        self.decay_enabled = bool(decay_enabled)
        self.half_life = float(half_life)
        self.blend = _clamp(blend)
        self.clock = clock

        self._state = self.defaults
        self._last_update = clock()
        self._lock = threading.RLock()
        #: Why the current state looks the way it does. Diagnostics and tests;
        #: never sent to the model.
        self.signals: List[str] = []
        self.updates = 0

    # -- reading ---------------------------------------------------------

    @property
    def state(self) -> ConversationState:
        """The current state, with decay applied up to now."""
        with self._lock:
            self._apply_decay()
            return self._state

    def _factor(self) -> float:
        """How much of the current state survives to now."""
        if not self.decay_enabled or self.half_life <= 0:
            return 1.0
        elapsed = max(0.0, self.clock() - self._last_update)
        return 0.5 ** (elapsed / self.half_life)

    def _apply_decay(self) -> None:
        """Exponentially relax the numbers toward their defaults.

        Only the numbers decay. `user_state` is dropped rather than relaxed: a
        stale "frustrated" is worse than an honest "unknown", and mood/mode
        follow the decay so a celebration settles into calm on its own.
        """
        factor = self._factor()
        if factor >= 1.0:
            return
        # Consume the elapsed time. Without this the elapsed window is never
        # retired, so every read re-applied the full decay and the state kept
        # collapsing towards the defaults just from being looked at.
        self._last_update = self.clock()
        state = self._state
        default = self.defaults
        blended = ConversationState(
            mood=state.mood,
            mode=state.mode,
            user_state=state.user_state,
            **{
                name: getattr(state, name) * factor + getattr(default, name) * (1 - factor)
                for name in ("energy", "warmth", "curiosity")
            },
        )
        # Once most of the excursion has decayed the labels follow it back to
        # default, and an inferred user state is withdrawn rather than left to
        # rot. A stale "frustrated" is worse than an honest "unknown".
        if factor < 0.5:
            blended = ConversationState(
                mood=default.mood,
                mode=default.mode,
                user_state=UserState.UNKNOWN,
                **{name: getattr(blended, name)
                   for name in ("energy", "warmth", "curiosity")},
            )
        self._state = blended

    # -- writing ---------------------------------------------------------

    def observe(self, message: str, *, context: Optional[ConversationState] = None) -> ConversationState:
        """Fold one user message into the state.

        Args:
            message: What the user just said.
            context: An explicit starting state, used to override the tracked
                one. Tests and callers that drive the state directly use this;
                normal turns pass nothing.

        Returns:
            The new state.
        """
        with self._lock:
            if not TONE_ENABLED:
                self._last_update = self.clock()
                return self.defaults

            self._apply_decay()
            if context is not None:
                self._state = _as_state(context)
                self._last_update = self.clock()
                self.signals = ["explicit context"]
                return self._state

            try:
                state, signals = _derive(message, self._state)
            except Exception as e:  # noqa: BLE001 - tone must never break a turn
                logger.warning(f"Conversation state update failed, keeping last: {e}")
                self._last_update = self.clock()
                return self._state

            self._state = state
            self.signals = signals
            self.updates += 1
            self._last_update = self.clock()
            return state

    def set(self, state: Any) -> ConversationState:
        """Force the state, used by tests, the API, and explicit overrides."""
        with self._lock:
            self._state = _as_state(state)
            self._last_update = self.clock()
            return self._state

    def reset(self) -> ConversationState:
        """Return to configured defaults, forgetting any inference."""
        with self._lock:
            self._state = self.defaults
            self._last_update = self.clock()
            self.signals = ["reset"]
            return self._state

    def status(self) -> Dict[str, Any]:
        """State plus non-identifying diagnostics for the API/HUD."""
        state = self.state
        return {
            **state.as_dict(),
            "enabled": bool(TONE_ENABLED),
            "decay_enabled": self.decay_enabled,
            "half_life_seconds": self.half_life,
            "signals": list(self.signals),
            "updates": self.updates,
        }


def _as_state(value: Any) -> ConversationState:
    """Accept a ConversationState, a dict, or anything state-like."""
    if isinstance(value, ConversationState):
        return value
    if isinstance(value, dict):
        data = value
        return ConversationState(
            mood=data.get("mood", Mood.NEUTRAL),
            energy=data.get("energy", 0.55),
            warmth=data.get("warmth", 0.75),
            curiosity=data.get("curiosity", 0.60),
            mode=data.get("mode", data.get("conversation_mode", ConversationMode.CASUAL)),
            user_state=data.get("user_state", data.get("user_emotional_state", UserState.UNKNOWN)),
        )
    return ConversationState()


def _blend(a: float, b: float, amount: float) -> float:
    return _clamp(a * (1 - amount) + b * amount)


def _derive(message: str, current: ConversationState) -> Tuple[ConversationState, List[str]]:
    """Compute the next state from a message. Pure; no I/O, no model call."""
    text = (message or "").strip()
    lowered = text.lower()
    signals: List[str] = []
    if not text:
        return current, signals

    # -- mode, most specific first --------------------------------------
    mode: Optional[ConversationMode] = None
    for pattern, candidate, name in (
        (_CELEBRATORY, ConversationMode.CELEBRATORY, "celebratory"),
        (_SERIOUS, ConversationMode.SERIOUS, "serious"),
        (_SUPPORTIVE, ConversationMode.SUPPORTIVE, "frustration"),
        (_TASK, ConversationMode.TASK_EXECUTION, "task"),
        (_TECHNICAL, ConversationMode.TECHNICAL, "technical"),
        (_BRAINSTORMING, ConversationMode.BRAINSTORMING, "brainstorming"),
        (_PLAYFUL, ConversationMode.PLAYFUL, "playful"),
    ):
        if pattern.search(text):
            mode = candidate
            signals.append(name)
            break

    # No mode signal: keep whatever conversation this already is. An
    # explanatory question is not by itself a register change, but it is a clear
    # signal that the user wants to understand something, so curiosity is raised
    # even when the mode holds.
    wants_detail = bool(_CURIOUS.search(text))
    if mode is None:
        mode = ConversationMode.BRAINSTORMING if wants_detail else current.mode
    if wants_detail:
        signals.append("curiosity")

    # -- user state: explicit self-report only --------------------------
    user_state = UserState.UNKNOWN
    if _NEGATED_STATE.search(text) or _RESOLVED.search(text):
        # "I'm not frustrated." / "actually I'm fine." An explicit correction
        # outranks any inference, so the state is withdrawn rather than
        # replaced with another guess.
        user_state = UserState.UNKNOWN
        signals.append("correction")
        mode = ConversationMode.SUPPORTIVE if mode is current.mode else mode
    else:
        for pattern, candidate in _EXPLICIT_STATE:
            if pattern.search(text):
                user_state = candidate
                signals.append(f"user:{candidate.value}")
                break
        else:
            # No self-report. An inferred feeling is carried forward only while
            # the conversation is still in the supportive register; the moment
            # the user moves on -- "okay, now optimise it" -- it is dropped.
            # Carrying it indefinitely is how a bad guess becomes the
            # assistant's permanent assumption about someone.
            user_state = (
                current.user_state
                if mode is ConversationMode.SUPPORTIVE
                else UserState.UNKNOWN
            )

    # -- blend ----------------------------------------------------------
    target_mood, t_energy, t_warmth, t_curiosity = _MODE_TARGETS[mode]

    # Frustration is about the user, so it warms and calms regardless of mode.
    if user_state is UserState.FRUSTRATED and mode is not ConversationMode.SUPPORTIVE:
        target_mood = Mood.CALM
        t_energy = min(t_energy, 0.45)
        t_warmth = max(t_warmth, 0.85)
    if user_state is UserState.TIRED:
        target_mood = Mood.CALM
        t_energy = min(t_energy, 0.4)
        t_warmth = max(t_warmth, 0.8)
    if user_state in (UserState.SAD, UserState.STRESSED):
        target_mood = Mood.CALM
        t_energy = min(t_energy, 0.42)
        t_warmth = max(t_warmth, 0.88)
    if user_state is UserState.EXCITED and mode is not ConversationMode.SUPPORTIVE:
        t_energy = max(t_energy, 0.75)
    if user_state is UserState.CONFUSED and mode in (
        ConversationMode.BRAINSTORMING, ConversationMode.PLAYFUL
    ):
        # Someone who said they are confused needs clarity, not ideas.
        mode = ConversationMode.FOCUSED
        target_mood, t_energy, t_warmth, t_curiosity = _MODE_TARGETS[mode]

    amount = _BLEND
    return ConversationState(
        mood=target_mood if amount >= 0.5 else current.mood,
        mode=mode,
        user_state=user_state,
        energy=_blend(current.energy, t_energy, amount),
        warmth=_blend(current.warmth, t_warmth, amount),
        curiosity=_blend(current.curiosity, t_curiosity, amount),
    ), signals


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

#: One line per mode. Deliberately behavioural ("what to do"), never an
#: instruction to perform an emotion.
_MODE_GUIDANCE: Dict[ConversationMode, str] = {
    ConversationMode.CASUAL: "Keep it natural and conversational. Match the user's length.",
    ConversationMode.TECHNICAL: "Be precise and concrete. Name the actual cause and the fix; avoid warmth that pads the answer.",
    ConversationMode.FOCUSED: "Stay on the point. One clear direction, minimal preamble.",
    ConversationMode.SUPPORTIVE: "Be supportive without being patronising. Acknowledge the difficulty, then give a practical next step.",
    ConversationMode.PLAYFUL: "Light and witty is fine here. Still answer the actual question.",
    ConversationMode.CELEBRATORY: "Match the moment. Let the satisfaction show, briefly, then move to what is next.",
    ConversationMode.SERIOUS: "Drop the levity. Plain, direct, and careful with claims.",
    ConversationMode.BRAINSTORMING: "Offer options and trade-offs. Build on their idea rather than resetting.",
    ConversationMode.TASK_EXECUTION: "Confirm what you are doing, do it, report what happened.",
}

_ENERGY_GUIDANCE = (
    (0.75, "Raise the energy: it is a good moment for it."),
    (0.60, "Keep the energy up."),
    (0.40, "Moderate energy."),
    (0.0, "Low energy: calm and steady."),
)

_USER_STATE_GUIDANCE: Dict[UserState, str] = {
    UserState.FRUSTRATED: "The user said they are frustrated. Acknowledge it once, do not dwell, do not over-apologise.",
    UserState.STRESSED: "The user said they are stressed. Reduce load; offer to narrow the scope.",
    UserState.TIRED: "The user said they are tired. Be economical with their attention.",
    UserState.SAD: "The user said they are disappointed. Be gentle, and practical rather than effusive.",
    UserState.CONFUSED: "The user said they are confused. Re-explain more simply and check where it broke down.",
    UserState.EXCITED: "The user said they are excited. Match it without inflating it.",
    UserState.POSITIVE: "The user is in a good headspace. Keep the tone light.",
}

#: Stated every time, because a mood is a style hint and nothing more.
_SOFT_SIGNAL = (
    "This shapes style only. It never overrides accuracy, the user's explicit "
    "instructions, safety, tool requirements, or task requirements. A playful "
    "mood does not mean make every answer a joke, and high energy does not mean "
    "be enthusiastic about things that are not good news. If accuracy and tone "
    "disagree, accuracy wins."
)

_HONESTY = (
    "You do not have feelings. If asked whether you are genuinely excited, "
    "happy, upset or feeling anything, say plainly that this is a conversational "
    "tone setting used to match how the conversation is going, not a subjective "
    "experience. Put it warmly rather than defensively."
)

_STATE_META = (
    "This is an internal note about how to talk in this moment. Never mention "
    "it, never name these values, and never tell the user their mood was "
    "detected or guessed."
)


def format_prompt(state: Optional[ConversationState] = None) -> str:
    """Render the state as a compact behavioural block for the turn.

    Returns an empty string when there is nothing worth saying, so a neutral
    conversation costs no tokens.
    """
    state = state or get_tone().state
    if state.mood is Mood.NEUTRAL and state.mode is ConversationMode.CASUAL \
            and state.user_state is UserState.UNKNOWN and state.energy < 0.7:
        return ""

    lines = [
        "\n\n## How To Talk Right Now",
        _STATE_META,
        f"- Mode: {state.mode.value}",
        f"- Mood: {state.mood.value}",
        f"- Energy {_band(state.energy)}, warmth {_band(state.warmth)}, "
        f"curiosity {_band(state.curiosity)}",
    ]
    guidance = _MODE_GUIDANCE.get(state.mode)
    if guidance:
        lines.append(f"- {guidance}")
    for threshold, text in _ENERGY_GUIDANCE:
        if state.energy >= threshold:
            lines.append(f"- {text}")
            break
    if state.warmth >= 0.85 and state.mode not in (
        ConversationMode.TECHNICAL, ConversationMode.TASK_EXECUTION
    ):
        lines.append("- Warm, but do not pad.")
    if state.curiosity >= 0.8:
        lines.append("- They seem to want to understand it, not just be told; offer the why.")
    user_note = _USER_STATE_GUIDANCE.get(state.user_state)
    if user_note:
        lines.append(f"- {user_note}")
    lines.append(f"- {_SOFT_SIGNAL}")
    lines.append(f"- {_HONESTY}")
    return "\n".join(lines)


def _band(value: float) -> str:
    """A coarse word instead of a number: the model acts on words better."""
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "moderate"
    if value >= 0.25:
        return "low"
    return "minimal"


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------

#: One tracker per process, matching how `jarvis.vision` keeps its visual
#: context: the state must survive a fresh Brain, a model switch, and the gap
#: between an HTTP request and a voice turn.
_tone = ConversationTone()
_lock = threading.Lock()


def get_tone() -> ConversationTone:
    """The process-wide conversation state tracker."""
    return _tone


def observe(message: str, *, context: Optional[ConversationState] = None) -> ConversationState:
    """Fold a user message into the shared state."""
    return _tone.observe(message, context=context)


def reset_tone() -> ConversationState:
    """Forget the current state, returning to defaults."""
    return _tone.reset()


__all__ = [
    "ConversationMode",
    "ConversationState",
    "ConversationTone",
    "Mood",
    "UserState",
    "format_prompt",
    "get_tone",
    "observe",
    "reset_tone",
]