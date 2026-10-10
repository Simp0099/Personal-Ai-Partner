"""General utility tools for JARVIS 2.0 (Jokes, Time, Wikipedia, wikiHow).

Every tool here returns text and stays silent. Speaking belongs to whoever
owns the turn -- the conversation layer, which then says the answer once
instead of once per tool plus once in the reply.
"""

import datetime
from jarvis.logger import logger


def tell_time() -> str:
    """Return the current system time for the conversation layer to present."""
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    return current_time


def tell_joke() -> str:
    """Fetch a random programming/general joke.

    Returns the joke as text. The conversation layer speaks the reply, so the
    tool does not speak it: speaking here would say it twice.
    """
    try:
        import pyjokes
        return pyjokes.get_joke()
    except Exception as e:
        logger.warning(f"Joke fetch failed, using a local one: {e}")
        return "Why do programmers prefer dark mode? Because light attracts bugs."


def search_wikipedia(query: str) -> str:
    """Fetch a concise 2-sentence summary from Wikipedia."""
    clean_query = _clean(query, ("wikipedia", "search", "look up", "who is", "what is"))
    if not clean_query:
        return "I need a topic to look up on Wikipedia."

    try:
        import wikipedia
        summary = wikipedia.summary(clean_query, sentences=2)
        logger.info(f"Wikipedia summary for {clean_query!r}: {summary[:100]}")
        return summary
    except Exception as e:
        logger.warning(f"Wikipedia lookup failed for {clean_query!r}: {e}")
        return f"I could not find a Wikipedia summary for {clean_query}."


def search_wikihow(query: str) -> str:
    """Search wikiHow for instructional steps."""
    clean_query = _clean(query, ("how to", "how do i", "jarvis"))
    if not clean_query:
        return "I need to know what you want to learn."

    try:
        from pywikihow import search_wikihow as wikihow_search
        results = wikihow_search(clean_query, max_results=1)
        if results:
            return results[0].summary
        return f"I could not find wikiHow instructions for {clean_query}."
    except Exception as e:
        logger.warning(f"wikiHow lookup failed for {clean_query!r}: {e}")
        return "I could not retrieve wikiHow instructions right now."


def _clean(query: str, noise: tuple) -> str:
    """Strip filler words from a query. No voice prompt: the caller asks."""
    text = (query or "").strip()
    for word in noise:
        text = text.replace(word, " ")
    return " ".join(text.split())
