"""Phase 3 — perception: the one layer both visual inputs feed.

Two sources land here and nothing else:

* a webcam frame, after local change detection decided it was worth a look
  (:mod:`jarvis.webcam`), and
* an image the user attached to a turn (handled by :mod:`jarvis.brain`, which
  passes the same neutral ``image_part`` attachments through the same provider
  adapters).

This module owns the parts that are shared: what an *observation* is, how it is
kept apart from an *inference*, how the short-lived visual context expires, and
how a raw frame becomes an attachment a vision model can read. Provider-specific
image encoding stays in the adapters, as it already does.

Deliberately not here: camera lifecycle (that is :mod:`jarvis.webcam`), long-term
memory (that is :mod:`jarvis.memory`, and perception never writes to it), and
personality (that is :mod:`jarvis.behavior`).
"""

from __future__ import annotations

import io
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jarvis.config import (
    WEBCAM_CONTEXT_TTL,
    WEBCAM_MAX_OBSERVATIONS,
    WEBCAM_MAX_SIDE,
    WEBCAM_JPEG_QUALITY,
)
from jarvis.providers.base import image_part

# --- Observation kinds -----------------------------------------------------

#: Supported by visible evidence.
OBSERVATION = "observation"
#: Reasonable, clearly derived, not itself visible.
INFERENCE = "inference"
#: Something is there but not reliably determinable from the image.
UNCERTAINTY = "uncertainty"

_KIND_PREFIXES = {
    "observed": OBSERVATION,
    "observation": OBSERVATION,
    "inferred": INFERENCE,
    "inference": INFERENCE,
    "unclear": UNCERTAINTY,
    "uncertain": UNCERTAINTY,
    "uncertainty": UNCERTAINTY,
}

#: Hedging language. Used only as a fallback when the model did not label the
#: line itself: "the user appears to be typing" is an inference whether or not
#: the model tagged it.
_HEDGE_RE = re.compile(
    r"\b(appears?|seems?|likely|suggests?|probably|possibly|might|may be|"
    r"it looks like|looks like|presumably)\b",
    re.IGNORECASE,
)

#: Matches a kind tag in any of the shapes a model actually emits:
#: ``observed:``, ``**inferred:**``, ``- unclear -``. Both sides of the colon are
#: optional-bold because models bold inconsistently.
_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*)?(observed|inferred|unclear|observation|inference|uncertainty)"
    r"(?:\*\*)?\s*[:\-]\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Observation:
    """One thing the vision model reported about a frame."""

    text: str
    #: One of :data:`OBSERVATION`, :data:`INFERENCE`, :data:`UNCERTAINTY`.
    kind: str = OBSERVATION
    at: float = field(default_factory=time.time)

    @property
    def hedged(self) -> bool:
        return self.kind != OBSERVATION


class VisualContext:
    """Short-lived record of what the camera last showed.

    Deliberately *not* memory. It is a handful of recent observations that expire
    on their own, so a desk lamp being switched on at 09:42 does not follow the
    user into next week. Nothing here is ever written to the long-term store.
    """

    def __init__(
        self,
        ttl: float = WEBCAM_CONTEXT_TTL,
        max_observations: int = WEBCAM_MAX_OBSERVATIONS,
    ):
        self.ttl = float(ttl)
        self.max_observations = int(max_observations)
        self._observations: List[Observation] = []
        self._changed_at: float = 0.0
        self._lock = threading.RLock()

    # -- reading ----------------------------------------------------------

    def observations(self, now: Optional[float] = None) -> List[Observation]:
        """Observations still in their window, oldest first. Empty once stale."""
        with self._lock:
            if self._is_stale(now):
                return []
            return list(self._observations)

    def is_fresh(self, now: Optional[float] = None) -> bool:
        with self._lock:
            return bool(self._observations) and not self._is_stale(now)

    def _is_stale(self, now: Optional[float] = None) -> bool:
        return not self._changed_at or (now if now is not None else time.time()) - self._changed_at > self.ttl

    # -- writing ----------------------------------------------------------

    def update(self, observations: List[Observation]) -> bool:
        """Replace the current scene with a new set of observations.

        A new analysis describes the scene as it is *now*, so it replaces the
        previous set rather than accumulating: "the user left the frame" must be
        able to replace "the user is at the desk", not sit beside it.

        Returns:
            True when the context actually changed. An identical repeat only
            refreshes the clock, which is what suppresses duplicate observations
            without spending another vision call.
        """
        cleaned = [o for o in observations if (o.text or "").strip()]
        with self._lock:
            same = (
                len(cleaned) == len(self._observations)
                and all(
                    a.text == b.text and a.kind == b.kind
                    for a, b in zip(cleaned, self._observations)
                )
            )
            self._observations = cleaned[-self.max_observations:]
            self._changed_at = time.time()
            return not same

    def clear(self) -> None:
        with self._lock:
            self._observations = []
            self._changed_at = 0.0

    # -- diagnostics ------------------------------------------------------

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            age = (now if now is not None else time.time()) - self._changed_at
            return {
                "observations": [
                    {"kind": o.kind, "text": o.text} for o in self._observations
                ],
                "count": len(self._observations),
                "age_seconds": round(age, 1) if self._changed_at else None,
                "ttl_seconds": self.ttl,
                "fresh": bool(self._observations) and age <= self.ttl,
            }


#: One context per process: one camera, one current picture of the room. This is
#: what lets a freshly constructed Brain still see it, and what keeps the
#: context coherent across a model switch.
_visual_context = VisualContext()


def get_visual_context() -> VisualContext:
    """The process-wide current visual context."""
    return _visual_context


def reset_visual_context() -> None:
    """Forget the current visual context (used by tests and on shutdown)."""
    _visual_context.clear()


# ---------------------------------------------------------------------------
# Scene understanding
# ---------------------------------------------------------------------------

#: Asked of the vision model for every webcam frame worth analysing. The three
#: clauses that matter: report what matters rather than cataloguing the room,
#: keep observation and inference apart, and refuse to profile the person.
SCENE_INSTRUCTION = (
    "This is a webcam frame from the user's own desk. Report what matters in it, "
    "in at most three short lines: what is there, what has changed since the "
    "previous frame if one was given, and anything the user would plausibly want "
    "known. Skip background furniture, lighting and surfaces unless they changed.\n"
    "Rules:\n"
    "- One line per point. Start each line with 'observed:', 'inferred:' or "
    "'unclear:' so the kind of claim is explicit.\n"
    "- 'observed' is only what you can actually see. 'inferred' is a conclusion "
    "drawn from what you see. 'unclear' is something present but not readable.\n"
    "- If the frame is empty or the person has left, say so plainly.\n"
    "- Never guess a person's identity, age, gender, emotion, mood, health, "
    "thoughts or intentions. Never describe a person beyond what is physically "
    "visible. Do not report any text that looks like a password, token or key.\n"
    "- This is context, not a report: no preamble, no closing summary."
)


def scene_instruction(previous: Optional[List[Observation]] = None) -> str:
    """The scene prompt, with the previous frame's observations when known.

    Carrying the previous state forward is what lets the model answer "what
    changed" instead of re-describing the room, which is most of the duplicate
    suppression without a second model call.
    """
    if not previous:
        return SCENE_INSTRUCTION
    known = "; ".join(f"{o.kind}: {o.text}" for o in previous)
    return f"{SCENE_INSTRUCTION}\nPrevious frame reported — report only what is different now, or say nothing changed: {known}"


def parse_observations(text: str) -> List[Observation]:
    """Turn a scene description into labelled observations.

    The model is asked to label each line; when it does not, hedging language is
    used as the fallback so an inference is never filed as an observation just
    because the tag was missing.
    """
    out: List[Observation] = []
    for raw in str(text or "").splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if not line:
            continue
        kind: Optional[str] = None
        match = _PREFIX_RE.match(line)
        if match:
            kind = _KIND_PREFIXES.get(match.group(1).lower())
            line = line[match.end():].strip()
        if not line:
            continue
        if kind is None:
            kind = INFERENCE if _HEDGE_RE.search(line) else OBSERVATION
        out.append(Observation(text=line, kind=kind))
    return out


# ---------------------------------------------------------------------------
# Frame handling
# ---------------------------------------------------------------------------

def frame_to_image_part(frame, *, max_side: int = WEBCAM_MAX_SIDE,
                        quality: int = WEBCAM_JPEG_QUALITY):
    """Turn one camera frame into a provider-neutral attachment.

    Downscales and JPEG-encodes before it leaves this function: a 1080p raw
    frame is a far larger upload and a more expensive vision call for no extra
    perception, and a downscaled frame is what keeps frames under the existing
    per-image size cap without a second copy of the bytes in memory.
    """
    from PIL import Image

    pil = Image.fromarray(frame)
    if max_side and max(pil.size) > max_side:
        scale = max_side / float(max(pil.size))
        pil = pil.resize(
            (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
            Image.BILINEAR,
        )
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=int(quality))
    return image_part(buffer.getvalue(), "image/jpeg")


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def format_context_block(context: Optional[VisualContext] = None) -> str:
    """Render the current visual context as a prompt section.

    Empty when there is nothing fresh, so a quiet camera costs nothing on
    unrelated turns. Says plainly that this is the camera's current view and not
    something remembered, which is what stops the model narrating it back as a
    stored preference.
    """
    context = context or _visual_context
    observations = context.observations()
    if not observations:
        return ""

    lines = [
        "\n\n## Current Visual Context",
        "What the camera saw most recently. This is the live camera view, not a "
        "stored memory and not something the user asked you to remember. Use it "
        "only if the question is about what is visible now; otherwise ignore it. "
        "Never claim to see anything not listed here, and say the camera has no "
        "fresh view rather than inventing one.",
    ]
    for observation in observations:
        lines.append(f"- [{observation.kind}] {observation.text}")
    return "\n".join(lines)


__all__ = [
    "INFERENCE",
    "OBSERVATION",
    "Observation",
    "UNCERTAINTY",
    "VisualContext",
    "format_context_block",
    "frame_to_image_part",
    "get_visual_context",
    "parse_observations",
    "reset_visual_context",
    "scene_instruction",
]