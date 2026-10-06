"""General utility tools for JARVIS 2.0 (Jokes, Time, Wikipedia, wikiHow).

Phase 8: All errors caught and logged gracefully.
"""

import datetime
from jarvis.speech import speak, listen
from jarvis.logger import logger


def tell_time() -> str:
    """Announce the current system time."""
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    speak(f"The current time is {current_time}.")
    return current_time


def tell_joke() -> str:
    """Fetch and speak a random programming/general joke."""
    try:
        import pyjokes
        joke = pyjokes.get_joke()
        speak(joke)
        return joke
    except Exception as e:
        logger.error(f"Joke error: {e}", exc_info=True)
        speak("Why do programmers prefer dark mode? Because light attracts bugs.")
        return "Fallback joke"


def search_wikipedia(query: str) -> str:
    """Fetch a concise 2-sentence summary from Wikipedia."""
    clean_query = query.replace("wikipedia", "").replace("search", "").replace("who is", "").replace("what is", "").strip()
    if not clean_query:
        speak("What would you like me to look up on Wikipedia?")
        clean_query = listen()
        if clean_query == "none":
            return ""

    try:
        import wikipedia
        speak("Searching Wikipedia...")
        summary = wikipedia.summary(clean_query, sentences=2)
        logger.info(f"Wikipedia summary for '{clean_query}': {summary[:100]}")
        speak(f"According to Wikipedia: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Wikipedia error: {e}", exc_info=True)
        speak(f"Sorry, I could not find a Wikipedia summary for {clean_query}.")
        return ""


def search_wikihow(query: str) -> str:
    """Search wikiHow for instructional steps."""
    clean_query = query.replace("how to", "").replace("jarvis", "").replace("friday", "").strip()
    if not clean_query:
        speak("What do you want to learn how to do?")
        clean_query = listen()
        if clean_query == "none":
            return ""

    try:
        from pywikihow import search_wikihow as wikihow_search
        speak(f"Searching instructions for {clean_query}...")
        results = wikihow_search(clean_query, max_results=1)
        if results:
            summary = results[0].summary
            speak(summary)
            return summary
        speak("I could not find instructions on wikiHow for that.")
        return ""
    except Exception as e:
        logger.error(f"wikiHow error: {e}", exc_info=True)
        speak("Could not retrieve wikiHow instructions at this time.")
        return ""


def repeat_words() -> None:
    """Echo back user speech."""
    speak("I am listening. Speak now.")
    content = listen()
    if content != "none":
        speak(f"You said: {content}")
