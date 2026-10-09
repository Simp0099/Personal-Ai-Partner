"""Weather tool for JARVIS 2.0.

Phase 8: All errors caught and logged gracefully.
"""

import requests
from jarvis.speech import speak
from jarvis.config import DEFAULT_CITY
from jarvis.logger import logger


def _wttr_temperature(city: str):
    """Current temperature via wttr.in (no key, plain-text/JSON API).

    Returns the temperature string (e.g. "+36°C") or None when unavailable.
    Never invents a value: empty or malformed responses mean None.
    """
    import urllib.parse
    try:
        query = urllib.parse.quote_plus(city)
        response = requests.get(f"https://wttr.in/{query}?format=%t",
                                timeout=10,
                                headers={"User-Agent": "jarvis-assistant"})
        text = (response.text or "").strip()
        if response.status_code == 200 and text and "°" in text:
            return text
        logger.debug(f"wttr.in unusable for {city!r}: "
                     f"status={response.status_code} body={text[:60]!r}")
        return None
    except Exception as e:
        logger.debug(f"wttr.in request failed for {city!r}: {e}")
        return None


def _google_temperature(city: str):
    """Legacy Google-scrape fallback. Returns temp string or None."""
    target_city = city or DEFAULT_CITY
    search_query = f"temperature in {target_city}"
    url = f"https://www.google.com/search?q={search_query}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        from bs4 import BeautifulSoup
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Check typical Google temperature elements
        temp_div = soup.find("div", class_="BNeawe iBp4i AP7Wnd") or soup.find("div", class_="BNeawe")
        if temp_div:
            return temp_div.text.split("\n")[0]

        logger.debug(f"Google scrape found no temperature for {target_city} "
                     f"(status={response.status_code}).")
        return None
    except Exception as e:
        logger.debug(f"Google weather scrape failed for {target_city}: {e}")
        return None


def get_temperature(city: str = None) -> str:
    """Fetch current temperature for a given city.

    wttr.in first (reliable, keyless); the legacy Google scrape remains as
    fallback because its markup/consumers already exist. Returns "" with an
    honest message when neither source has data -- never an invented value.
    """
    target_city = city or DEFAULT_CITY
    temp = _wttr_temperature(target_city)
    if not temp:
        temp = _google_temperature(target_city)
    if temp:
        speak(f"The temperature in {target_city} is currently {temp}.")
        return temp

    logger.warning(f"Weather unavailable for {target_city} "
                   f"(wttr.in and google both failed).")
    speak(f"Unable to parse exact temperature for {target_city}.")
    return ""
