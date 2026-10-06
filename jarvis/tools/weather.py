"""Weather tool for JARVIS 2.0.

Phase 8: All errors caught and logged gracefully.
"""

import requests
from jarvis.speech import speak
from jarvis.config import DEFAULT_CITY
from jarvis.logger import logger


def get_temperature(city: str = None) -> str:
    """Fetch current temperature for a given city."""
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
            temp = temp_div.text.split("\n")[0]
            speak(f"The temperature in {target_city} is currently {temp}.")
            return temp

        speak(f"Unable to parse exact temperature for {target_city}.")
        return ""
    except Exception as e:
        logger.error(f"Weather error: {e}", exc_info=True)
        speak("Unable to check weather information right now.")
        return ""
