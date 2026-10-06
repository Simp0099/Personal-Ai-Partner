"""Phase 2 — the persona as an operating layer, not only prompt wording.

Phase 1 put Ai Partner's identity in the system prompt. That defines *who* it
is. This module is the per-turn layer that decides *how* it behaves on this
particular message: what to calibrate towards, how much continuity to lean on,
and how hard to work the task.

Everything here is deterministic and free. It reads the
:class:`~jarvis.classify.Classification` the Brain already computes plus whether
the conversation already has turns. No extra model call, no second memory
system, no parallel intent framework — this is a projection of signals the
routing layer was already producing.
"""

from __future__ import annotations

from typing import List, Optional

from jarvis.classify import Classification, TaskType
from jarvis.memory import CATEGORIES, tokenize

#: How to calibrate the answer for each task type. Kept short and reviewable:
#: these are nudges, not a response-template system.
_CALIBRATION = {
    TaskType.CASUAL: (
        "This is conversational. Be warm and human, keep it brief, and do not "
        "wrap the reply in structure it does not need."
    ),
    TaskType.SIMPLE_QUESTION: (
        "This is a small factual question. Answer in a sentence or two. No "
        "headings, no preamble, no restating the question."
    ),
    TaskType.REASONING: (
        "Give the conclusion first, then the reasoning that actually changed it. "
        "If this is a diagnosis, name the cause before the fix. No internal "
        "deliberation — summarise the reasoning, do not narrate the thinking."
    ),
    TaskType.CODING: (
        "This is a technical task. Be specific and correct: state the approach, "
        "give the code or command, and say how to verify it worked."
    ),
    TaskType.STRUCTURED_OUTPUT: (
        "Return exactly the format asked for and nothing around it. Do not add "
        "explanation the format did not leave room for."
    ),
    TaskType.CREATIVE: (
        "Collaborate, not document. Build on what the user already offered "
        "instead of resetting with a formal brief."
    ),
    TaskType.RESEARCH: (
        "Separate what you verified from what you inferred. If you did not look "
        "something up, do not imply that you did."
    ),
    TaskType.LONG_CONTEXT: (
        "This conversation is long. Rely on what was already established here "
        "rather than restarting the analysis from scratch."
    ),
    TaskType.TOOL_USE: (
        "This needs a real action. Use the tools, then report what actually "
        "happened — including the parts that failed."
    ),
}

#: Applied when the request reads as a concrete task rather than a question.
_OWNERSHIP = (
    "Take ownership: do the requested work rather than describing how it could "
    "be done. Report the result and any genuine blocker. Never claim an action "
    "you did not perform."
)

#: Applied when the request is narrow: stop the agent padding the answer.
_RESTRAINT = (
    "Do the part that was asked for. No unsolicited extras, no trailing menu of "
    "other things you could also do, no offers of work nobody requested."
)

_PROACTIVE = (
    "If something in this work would bite the user later, say so in a sentence, "
    "with the reason. Guidance that is genuinely useful is welcome; a list of "
    "extra work is not."
)

_CONTINUITY = (
    "This is an ongoing conversation. Build on the earlier turns and the "
    "decisions already made in them. Do not re-introduce yourself, and do not "
    "ask for information that is already established here."
)


def turn_directive(classification: Classification, *, has_history: bool = False) -> str:
    """Build this turn's behavioral directive.

    Appended to the system prompt for the turn, so it lands immediately before
    the user's message rather than being buried in a long static preamble.

    Args:
        classification: The Brain's read of the request, already computed for
            routing. Reused as-is; no second classifier is involved.
        has_history: Whether earlier turns exist in this conversation.

    Returns:
        Directive text for this turn.
    """
    lines: list[str] = []

    if has_history:
        lines.append(_CONTINUITY)

    calibration = _CALIBRATION.get(classification.task_type)
    if calibration:
        lines.append(calibration)

    # A request that asks for something done is a task, not a consultation.
    wants_action = classification.tool_required or TaskType.CODING in classification.all_types
    if wants_action:
        lines.append(_OWNERSHIP)
    else:
        lines.append(_RESTRAINT)

    lines.append(_PROACTIVE)

    return "\n".join(f"- {line}" for line in lines)


def tool_failure_note(results: List[str]) -> str:
    """Honest reporting note for a turn whose tool calls failed.

    The directive cannot carry this: the session is already open by the time
    tools run, so it is appended to the reply instead. Returns an empty string
    when every tool succeeded, so a clean turn is never annotated.
    """
    failures = [str(r) for r in results if str(r or "").startswith("Error")]
    if not failures:
        return ""
    detail = failures[0] if len(failures) == 1 else f"{failures[0]} (+{len(failures) - 1} more)"
    return (
        f"\n\n_(This did not complete: {detail} "
        f"— treat it as not done rather than as a result.)_"
    )


# ---------------------------------------------------------------------------
# Memory posture
# ---------------------------------------------------------------------------

#: Sent verbatim when the user asks something to be remembered. The strongest
#: write signal there is: an explicit instruction from the user.
_REMEMBER_INTENT = (
    "The user has explicitly asked you to remember something. Call save_memory "
    "for it with source 'explicit', and pick the category that fits from: "
    f"{', '.join(CATEGORIES)}. Prefer a stable subject key (for example "
    "'user:name' or 'project:jarvis') so a later correction replaces this entry "
    "instead of contradicting it. Say plainly whether it was saved -- never claim "
    "you will remember something you did not store."
)

#: Sent when the user asks something to be forgotten.
_FORGET_INTENT = (
    "The user has explicitly asked you to forget something. Call forget_memory "
    "with what they described. If nothing matched, say so plainly -- do not "
    "claim a memory was removed when it is still active."
)

#: Sent when the user corrects something they previously established.
_UPDATE_INTENT = (
    "The user is correcting something previously established. Save the new "
    "version under the same subject key so it supersedes the old entry rather "
    "than sitting beside it as a contradiction."
)


def memory_posture(user_text: str, subjects: Optional[List[str]] = None) -> str:
    """Additional directive for turns that carry an explicit memory intent.

    Detection is keyword-based and advisory: it changes how the turn is
    handled, it does not write anything by itself. Writing still goes through
    the tool, so persistence failures stay visible.

    Args:
        user_text: The user's message this turn.
        subjects: Existing subject keys. Supplied because supersession only
            happens when the writer reuses a key, and a model that invents a
            fresh one silently leaves two contradicting memories active.
    """
    lowered = (user_text or "").lower()
    tokens = tokenize(lowered)

    forget = bool(tokens & {"forget", "unremember"}) or any(
        p in lowered for p in ("don't remember", "do not remember", "stop remembering")
    )
    remember = any(
        p in lowered for p in ("remember this", "remember that", "keep in mind",
                               "don't forget", "do not forget", "from now on",
                               "note that", "keep this in mind")
    )
    update = any(p in lowered for p in ("actually,", "changed my mind", "i was wrong",
                                        "no longer", "instead of", "correction",
                                        "update the", "changed to", "switch to"))

    if forget:
        return _FORGET_INTENT

    if remember or (update and tokens):
        intent = _REMEMBER_INTENT if remember else _UPDATE_INTENT
        if subjects:
            keys = ", ".join(subjects)
            intent += (
                f"\n- These subject keys already exist: {keys}. If what the user "
                "is saying corrects one of them, reuse that exact key so it is "
                "replaced rather than stored alongside a contradiction."
            )
        return intent
    return ""


def memory_context(facts: List, greeting_name: str) -> str:
    """Directive describing how to treat the retrieved memory and the user.

    Memory is context to apply, not a script to recite. The user's configured
    name is surfaced here -- through the context layer, not by hardcoding it
    into replies.
    """
    lines = []
    if facts:
        lines.append(
            "Use the long-term memory below only where it is relevant. Apply "
            "preferences silently; do not recite stored facts back at the user "
            "unless they ask or the answer depends on it. If memory conflicts "
            "with what the user says right now, the user wins."
        )
    lines.append(
        f"The user is addressed as \"{greeting_name}\". Use the name naturally "
        "and sparingly -- not as the opening word of every reply."
    )
    return "\n".join(f"- {line}" for line in lines)