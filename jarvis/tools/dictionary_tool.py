"""Dictionary tools for JARVIS 2.0 (Meanings, Synonyms, Antonyms).

Phase 8: All errors caught and logged gracefully.
"""

from jarvis.speech import speak, listen
from jarvis.logger import logger


def get_meaning(word: str) -> str:
    """Lookup the dictionary definition of a word."""
    try:
        from PyDictionary import PyDictionary
        dictionary = PyDictionary()
        result = dictionary.meaning(word)
        if result:
            first_pos = next(iter(result))
            first_def = result[first_pos][0] if result[first_pos] else "No definition found"
            summary = f"{word} as a {first_pos}: {first_def}"
            speak(summary)
            return summary
        speak(f"Could not find definitions for {word}.")
        return "No definition found"
    except Exception as e:
        logger.error(f"Dictionary error: {e}", exc_info=True)
        speak(f"Could not retrieve definition for {word}.")
        return ""


def get_synonym(word: str) -> str:
    """Lookup synonyms of a word."""
    try:
        from PyDictionary import PyDictionary
        dictionary = PyDictionary()
        result = dictionary.synonym(word)
        if result:
            synonyms_str = ", ".join(result[:4])
            summary = f"Synonyms for {word} include: {synonyms_str}"
            speak(summary)
            return summary
        speak(f"No synonyms found for {word}.")
        return ""
    except Exception as e:
        logger.error(f"Synonym error: {e}", exc_info=True)
        speak(f"Could not retrieve synonyms for {word}.")
        return ""


def get_antonym(word: str) -> str:
    """Lookup antonyms of a word."""
    try:
        from PyDictionary import PyDictionary
        dictionary = PyDictionary()
        result = dictionary.antonym(word)
        if result:
            antonyms_str = ", ".join(result[:4])
            summary = f"Antonyms for {word} include: {antonyms_str}"
            speak(summary)
            return summary
        speak(f"No antonyms found for {word}.")
        return ""
    except Exception as e:
        logger.error(f"Antonym error: {e}", exc_info=True)
        speak(f"Could not retrieve antonyms for {word}.")
        return ""


def handle_dictionary_command(query: str) -> None:
    """Interactive command handler for dictionary features."""
    cleaned = (
        query.lower()
        .replace("what is the", "")
        .replace("meaning of", "")
        .replace("synonym of", "")
        .replace("antonym of", "")
        .replace("dictionary", "")
        .replace("jarvis", "")
        .replace("friday", "")
        .strip()
    )

    if "synonym" in query:
        get_synonym(cleaned or "happy")
    elif "antonym" in query:
        get_antonym(cleaned or "happy")
    elif "meaning" in query or cleaned:
        get_meaning(cleaned or "intelligent")
    else:
        speak("Which word would you like me to look up?")
        word = listen()
        if word != "none":
            get_meaning(word)
