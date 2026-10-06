"""Web navigation and search tools for JARVIS 2.0."""

import urllib.parse
import webbrowser
from jarvis.speech import speak
from jarvis.config import DEFAULT_MAPS_QUERY

# Predefined quick URLs
KNOWN_SERVICES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "amazon": "https://www.amazon.in",
    "photos": "https://photos.google.com",
    "intel": "https://www.intel.com",
}


def open_service(service_name: str) -> str:
    """Open a predefined common web service."""
    service_clean = service_name.lower().strip()
    for name, url in KNOWN_SERVICES.items():
        if name in service_clean:
            webbrowser.open(url)
            speak(f"Opening {name} now.")
            return f"Opened {url}"

    # Generic domain fallback
    cleaned = service_clean.replace("open", "").replace("website", "").replace(" ", "").strip()
    url = f"https://www.{cleaned}.com"
    webbrowser.open(url)
    speak(f"Launching {cleaned} website.")
    return f"Opened {url}"


def search_youtube(query: str) -> str:
    """Search YouTube for a query."""
    clean_query = query.replace("youtube search", "").replace("search youtube for", "").strip()
    if not clean_query:
        clean_query = query
    encoded = urllib.parse.quote_plus(clean_query)
    url = f"https://www.youtube.com/results?search_query={encoded}"
    webbrowser.open(url)
    speak(f"Here are the YouTube results for {clean_query}.")
    return f"Searched YouTube for {clean_query}"


def search_google(query: str) -> str:
    """Search Google for a query."""
    clean_query = query.replace("google search", "").replace("search google for", "").strip()
    if not clean_query:
        clean_query = query
    encoded = urllib.parse.quote_plus(clean_query)
    url = f"https://www.google.com/search?q={encoded}"
    webbrowser.open(url)
    speak(f"Searching Google for {clean_query}.")
    return f"Searched Google for {clean_query}"


def open_maps(location: str = None) -> str:
    """Open Google Maps for a given location or default city."""
    loc = location or DEFAULT_MAPS_QUERY
    encoded = urllib.parse.quote_plus(loc)
    url = f"https://www.google.com/maps/search/{encoded}"
    webbrowser.open(url)
    speak(f"Opening Google Maps for {loc}.")
    return f"Opened maps for {loc}"
