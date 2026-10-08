"""Deterministic local-first intent router.

Runs before the model/provider router. A match executes an existing local
tool with zero LLM involvement; anything else returns ``matched=False`` and
the caller continues into the existing AI router unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from jarvis.intents import patterns
from jarvis.intents.local_intents import handle_intent
from jarvis.logger import logger


@dataclass
class LocalIntentResult:
    matched: bool
    handled: bool = False
    response: str = ""
    intent: str = ""
    tool: str = ""


# Intent -> registered tool name, for diagnostics only. Dispatch itself goes
# through the caller's registry callable, so this table can never drift from
# the real tool set without a loud KeyError at dispatch time.
_INTENT_TOOLS = {
    "time": "tell_time",
    "dictionary": "lookup_dictionary",
    "weather": "get_temperature",
    "screenshot": "take_screenshot",
    "launch_app": "launch_app",
    "media_play": "play_music",
}


def handle(user_text: str,
           dispatch: Callable[[str, dict], str]) -> LocalIntentResult:
    """Match `user_text` and run the local handler, or return no-match."""
    found = patterns.match(user_text)
    if found is None:
        logger.debug("Local intent: no match; falling through to AI router.")
        return LocalIntentResult(matched=False)
    try:
        response = handle_intent(found.intent, found.args, dispatch)
    except Exception as e:  # noqa: BLE001 -- fall through, LLM reports honestly
        logger.error(f"Local intent '{found.intent}' failed: {e}", exc_info=True)
        return LocalIntentResult(matched=False)
    logger.debug(f"Local intent matched '{found.intent}'; LLM call skipped.")
    return LocalIntentResult(
        matched=True,
        handled=True,
        response=response,
        intent=found.intent,
        tool=_INTENT_TOOLS.get(found.intent, ""),
    )


def match_only(user_text: str) -> Optional[str]:
    """Intent name for `user_text`, or None. Test/observability helper."""
    found = patterns.match(user_text)
    return found.intent if found else None
