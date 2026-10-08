"""Deterministic command patterns for local-first intent routing.

Pure matching only: no LLM, no network, no provider, no disk. Every matcher
is anchored to the whole utterance so a keyword inside a longer request can
never trigger a local command ("explain why the weather affects aircraft"
is not a weather lookup). Precision over recall: doubt means no match and
the request falls through to the existing AI router.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class IntentMatch:
    """A recognized local command with extracted arguments."""

    intent: str
    args: Dict[str, str] = field(default_factory=dict)


def normalize(text: str) -> str:
    """Normalize for matching without destroying handler arguments."""
    cleaned = " ".join((text or "").strip().split())
    cleaned = re.sub(r"^(hey jarvis|jarvis|ok jarvis)[,\s]+", "", cleaned,
                     flags=re.IGNORECASE)
    return cleaned


def _bare(text: str) -> str:
    """Lowercased command stripped of trailing punctuation for shape matching."""
    return re.sub(r"[?!.\u2019']+$", "", text.strip().lower()).strip()


_TIME = re.compile(
    r"^(what time is it|what'?s the time|tell me the time|current time|"
    r"the time please|time please|do you have the time)\??$"
)


def match_time(text: str) -> Optional[IntentMatch]:
    if _TIME.match(_bare(text)):
        return IntentMatch(intent="time")
    return None


_DEFINE = re.compile(
    r"^(define|definition of|meaning of)\s+(?P<word>[a-zA-Z][a-zA-Z\-]{0,39})$"
)
_MEAN = re.compile(
    r"^what do(?:es)?\s+(?P<word>[a-zA-Z][a-zA-Z\-]{0,39})\s+mean$"
)
_MEANING_OF = re.compile(
    r"^what(?:'?s| is) the meaning of\s+(?P<word>[a-zA-Z][a-zA-Z\-]{0,39})$"
)


def match_dictionary(text: str) -> Optional[IntentMatch]:
    for pattern in (_DEFINE, _MEAN, _MEANING_OF):
        found = pattern.match(_bare(text))
        if found:
            return IntentMatch(intent="dictionary",
                               args={"word": found.group("word").lower()})
    return None


_CITY = r"(?P<city>[a-zA-Z][a-zA-Z .\-]{0,47}[a-zA-Z])"
_WEATHER_TODAY = re.compile(
    r"^(what(?:'?s| is) the weather|how(?:'?s| is) the weather)"
    r"( today| right now)?$"
)
_WEATHER_IN = re.compile(
    rf"^(?:weather|what(?:'?s| is) the weather|how(?:'?s| is) the weather)"
    rf" in {_CITY}$",
    re.IGNORECASE,
)


def match_weather(text: str) -> Optional[IntentMatch]:
    cleaned = _bare(text)
    if cleaned == "weather" or _WEATHER_TODAY.match(cleaned):
        return IntentMatch(intent="weather")
    # Shape matched case-insensitively; extract the city from the
    # case-preserving normalization so "Mumbai" stays "Mumbai".
    found = _WEATHER_IN.match(normalize(text).rstrip("?!."))
    if found:
        return IntentMatch(intent="weather",
                           args={"city": " ".join(found.group("city").split())})
    return None


_SCREENSHOT = re.compile(
    r"^(take a screenshot( of (the|my) screen)?|capture( my)? screen|"
    r"screenshot( my screen)?)\??$"
)


def match_screenshot(text: str) -> Optional[IntentMatch]:
    if _SCREENSHOT.match(_bare(text)):
        return IntentMatch(intent="screenshot")
    return None


_LAUNCH = re.compile(r"^(open|launch|start)\s+(?P<app>.{1,80})$",
                     re.IGNORECASE)


def match_launch_app(text: str) -> Optional[IntentMatch]:
    """Launch command with a free-text app name. Safety lives in the
    allowlist, not here: anything not allowlisted is rejected locally."""
    found = _LAUNCH.match(text.strip())
    if not found:
        return None
    app = " ".join(found.group("app").strip().split())
    if not app or len(app) > 80:
        return None
    return IntentMatch(intent="launch_app", args={"app_name": app})


_PLAY = re.compile(r"^play\s+(?P<song>.{3,100})$")
# Bare generics carry no track: let the LLM path ask which song, as today.
_GENERIC_SONGS = {"music", "song", "songs", "something", "anything", "it"}


def match_media(text: str) -> Optional[IntentMatch]:
    found = _PLAY.match(text.strip())
    if not found:
        return None
    song = " ".join(found.group("song").strip().split())
    if len(song) < 3 or song.lower().strip("?!.'") in _GENERIC_SONGS:
        return None
    return IntentMatch(intent="media_play", args={"song": song})


# NOTE: volume is intentionally absent. Phase 0 verified the repository has
# no existing volume tool, and inventing OS-level implementations here is out
# of scope. Such utterances fall through to the existing AI router.
_MATCHERS = (match_time, match_dictionary, match_weather, match_screenshot,
             match_launch_app, match_media)


def match(text: str) -> Optional[IntentMatch]:
    """First matching local intent for `text`, or None (fall through)."""
    cleaned = normalize(text)
    if not cleaned:
        return None
    for matcher in _MATCHERS:
        found = matcher(cleaned)
        if found:
            return found
    return None
