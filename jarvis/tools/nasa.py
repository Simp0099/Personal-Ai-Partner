"""NASA Astronomy Picture of the Day (APOD) Tool for JARVIS 2.0.

Loads NASA API key from environment variables without side-effects on import.

Phase 8: All errors caught and logged gracefully.
"""

import datetime
import requests
from jarvis.speech import speak, listen
from jarvis.config import NASA_API_KEY
from jarvis.logger import logger


def normalize_spoken_date(spoken: str) -> str:
    """Attempt to parse or normalize spoken date into YYYY-MM-DD."""
    cleaned = spoken.lower().replace(" and ", "-").replace("and", "-").replace(" ", "")
    # Check if format is already YYYY-MM-DD
    try:
        datetime.datetime.strptime(cleaned, "%Y-%m-%d")
        return cleaned
    except ValueError:
        pass

    # Default to today if invalid or unrecognized
    return datetime.datetime.now().strftime("%Y-%m-%d")


def get_nasa_apod(date_str: str = None) -> dict:
    """Fetch Astronomy Picture of the Day for a given date."""
    target_date = date_str or datetime.datetime.now().strftime("%Y-%m-%d")
    url = "https://api.nasa.gov/planetary/apod"
    params = {
        "api_key": NASA_API_KEY or "DEMO_KEY",
        "date": target_date
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        title = data.get("title", "NASA Astronomy Feature")
        explanation = data.get("explanation", "No details available.")

        speak(f"Space feature for {target_date}: {title}")
        logger.info(f"NASA APOD: {title}")
        # Read the first couple sentences so user isn't overwhelmed
        short_summary = ". ".join(explanation.split(". ")[:2]) + "."
        speak(short_summary)
        return data
    except Exception as e:
        logger.error(f"NASA tool error: {e}", exc_info=True)
        speak("Sorry, I could not retrieve space data from NASA at this moment.")
        return {}


def handle_nasa_command() -> None:
    """Interactive handler for space queries."""
    speak("Which date's space picture would you like to see? You can speak a date like 2026-09-15, or say today.")
    spoken_date = listen()
    if spoken_date == "none" or "today" in spoken_date.lower():
        get_nasa_apod()
    else:
        normalized = normalize_spoken_date(spoken_date)
        get_nasa_apod(normalized)
