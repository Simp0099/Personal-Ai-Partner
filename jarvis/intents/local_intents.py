"""Local intent handlers: argument shaping + existing-tool dispatch.

Handlers contain no tool implementations. They call the already-registered
tool functions through the `dispatch(name, args)` callable the caller
provides (in production: ``JarvisBrain._execute_tool_call``, i.e. the
existing TOOL_REGISTRY). Results are shaped into a speakable sentence only
when the tool itself returns a fragment; sentence tools pass through.
"""

from __future__ import annotations

from typing import Callable


def handle_time(dispatch: Callable[[str, dict], str]) -> str:
    result = (dispatch("tell_time", {}) or "").strip()
    if result:
        return f"The current time is {result}."
    return "I couldn't read the clock just now."


def handle_dictionary(dispatch: Callable[[str, dict], str], word: str) -> str:
    result = (dispatch("lookup_dictionary", {"word": word}) or "").strip()
    if result:
        return result
    return f"I couldn't find a definition for '{word}'."


def handle_weather(dispatch: Callable[[str, dict], str], city: str) -> str:
    target = city or "Delhi"  # same default the weather tool uses
    result = (dispatch("get_temperature", {"city": target}) or "").strip()
    if result:
        return f"The temperature in {target} is {result}."
    return f"I couldn't get the weather for {target} right now."


def handle_screenshot(dispatch: Callable[[str, dict], str]) -> str:
    # The registered wrapper already returns a full sentence.
    return dispatch("take_screenshot", {})


def handle_media_play(dispatch: Callable[[str, dict], str], song: str) -> str:
    # The registered wrapper already returns a full sentence.
    return dispatch("play_music", {"song_name": song})


def handle_launch_app(dispatch: Callable[[str, dict], str], app_name: str) -> str:
    # The registered wrapper enforces the allowlist and safe execution,
    # and already returns a full user-facing sentence for every outcome
    # (launched, not approved, not installed, invalid). All local, no LLM.
    return dispatch("launch_app", {"app_name": app_name})


_HANDLERS = {
    "time": lambda dispatch, args: handle_time(dispatch),
    "dictionary": lambda dispatch, args: handle_dictionary(dispatch, args["word"]),
    "weather": lambda dispatch, args: handle_weather(dispatch, args.get("city", "")),
    "screenshot": lambda dispatch, args: handle_screenshot(dispatch),
    "launch_app": lambda dispatch, args: handle_launch_app(dispatch, args["app_name"]),
    "media_play": lambda dispatch, args: handle_media_play(dispatch, args["song"]),
}


def handle_intent(intent: str, args: dict,
                  dispatch: Callable[[str, dict], str]) -> str:
    """Run the handler for a matched intent. Raises KeyError if unknown."""
    return _HANDLERS[intent](dispatch, args)
