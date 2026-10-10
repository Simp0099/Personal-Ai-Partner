"""NASA Astronomy Picture of the Day (APOD) tool for JARVIS 2.0.

APOD is one endpoint: a picture and its explanation. It is not space *news*,
so it is named and documented as what it is rather than as a news feed.

Credentials come from the environment via config; nothing is read at import
time beyond that. All errors are caught and reported honestly -- a failed
fetch returns no data instead of a fabricated summary.
"""

import datetime

import requests

from jarvis.config import NASA_API_KEY, NASA_TIMEOUT_S, USER_AGENT
from jarvis.logger import logger


def _iso_days_ago(days: int) -> str:
    """Date `days` before today, as YYYY-MM-DD."""
    return (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

_APOD_URL = "https://api.nasa.gov/planetary/apod"


def normalize_spoken_date(spoken: str) -> str:
    """Normalise a spoken or typed date into YYYY-MM-DD.

    Accepts an already-correct date and the common spoken forms; anything it
    cannot parse becomes today rather than an error, because "the picture for
    whenever" is a reasonable reading of a garbled date.
    """
    text = (spoken or "").strip().lower()
    if not text:
        return datetime.date.today().isoformat()

    # Already ISO.
    try:
        return datetime.datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError:
        pass

    # "2024 03 15" / "2024/03/15"
    for sep in (" ", "/", "."):
        candidate = text.replace(sep, "-")
        try:
            return datetime.datetime.strptime(candidate, "%Y-%m-%d").date().isoformat()
        except ValueError:
            continue

    if "today" in text:
        return datetime.date.today().isoformat()

    logger.debug(f"Unrecognised date {spoken!r}; using today")
    return datetime.date.today().isoformat()


def _extract_date(data) -> str:
    """The date APOD actually served, which may differ from the one asked for.

    NASA silently serves the latest available picture when a date has none, so
    reporting the requested date would be a lie.
    """
    return str((data or {}).get("date") or "")


def get_nasa_apod(date_str: str = None) -> dict:
    """Fetch NASA's Astronomy Picture of the Day.

    Args:
        date_str: Date in YYYY-MM-DD format, or None for today.

    Returns:
        The APOD payload, or {} when it could not be fetched.
    """
    target = date_str or datetime.date.today().isoformat()
    try:
        response = requests.get(
            _APOD_URL,
            params={"api_key": NASA_API_KEY or "DEMO_KEY", "date": target},
            timeout=NASA_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        data = response.json()

        title = data.get("title") or "NASA Astronomy Picture"
        served = _extract_date(data)
        logger.info(f"NASA APOD {served or target}: {title}")
        return data
    except Exception as e:  # noqa: BLE001 -- logged, never raised to the caller
        logger.error(f"NASA APOD request failed for {target}: {e}", exc_info=True)
        return {}


def summarize_apod(data: dict, max_sentences: int = 2) -> str:
    """Short spoken summary of an APOD payload.

    Truncates on a sentence boundary so the model is not handed a 2000-word
    essay to read aloud, and falls back to the caption or media type when
    there is no explanation at all.
    """
    explanation = (data.get("explanation") or "").strip()
    if explanation:
        sentences = [s for s in explanation.split(". ") if s.strip()]
        return ". ".join(sentences[:max_sentences]).rstrip(".") + "."

    media_type = (data.get("media_type") or "").strip()
    caption = (data.get("caption") or data.get("title") or "").strip()
    if caption and media_type:
        return f"{caption} ({media_type})."
    return caption or "No details were provided for this picture."


def format_apod(data: dict) -> str:
    """One-line answer for the tool caller: what it is, and when."""
    if not data:
        return "NASA's astronomy picture service is not responding right now."

    served = _extract_date(data)
    title = data.get("title") or "Astronomy Picture of the Day"
    line = f"NASA Astronomy Picture of the Day for {served or 'today'}: {title}."
    if data.get("media_type"):
        line += f" Media type: {data['media_type']}."
    if data.get("url"):
        line += f" Image: {data['url']}"
    return line
