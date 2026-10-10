"""Web navigation and search tools for JARVIS 2.0."""

import urllib.parse
import webbrowser
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
    """Open a known service; unknown names never become guessed domains."""
    cleaned = " ".join((service_name or "").lower().split())
    for prefix in ("open ", "launch ", "visit "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    cleaned = cleaned.removesuffix(" website").strip()
    url = KNOWN_SERVICES.get(cleaned)
    if url is None:
        return f"I don't have a trusted URL for '{service_name}'."
    webbrowser.open(url)
    return f"Opened {url}"


def search_youtube(query: str) -> str:
    """Search YouTube for a query."""
    clean_query = query.replace("youtube search", "").replace("search youtube for", "").strip()
    if not clean_query:
        clean_query = query
    encoded = urllib.parse.quote_plus(clean_query)
    url = f"https://www.youtube.com/results?search_query={encoded}"
    webbrowser.open(url)
    return f"Searched YouTube for {clean_query}"


def search_google(query: str) -> str:
    """Search Google for a query."""
    clean_query = query.replace("google search", "").replace("search google for", "").strip()
    if not clean_query:
        clean_query = query
    encoded = urllib.parse.quote_plus(clean_query)
    url = f"https://www.google.com/search?q={encoded}"
    webbrowser.open(url)
    return f"Searched Google for {clean_query}"


def open_maps(location: str = None) -> str:
    """Open Google Maps for a given location or default city."""
    loc = location or DEFAULT_MAPS_QUERY
    encoded = urllib.parse.quote_plus(loc)
    url = f"https://www.google.com/maps/search/{encoded}"
    webbrowser.open(url)
    return f"Opened maps for {loc}"
