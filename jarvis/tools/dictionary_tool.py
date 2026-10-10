"""Dictionary tools for JARVIS 2.0 (meanings, synonyms, antonyms).

Returns text and stays silent; the conversation layer speaks the reply.
Failures are reported as text rather than raised, so the model can tell the
user what happened instead of the tool vanishing.
"""

from jarvis.logger import logger

#: Longest definition kept. PyDictionary returns every sense in every
#: part of speech, which is unreadable aloud and wastes context.
_MAX_DEF = 220
_MAX_WORDS = 8


def get_meaning(word: str) -> str:
    """Definition of a word."""
    if not (word or "").strip():
        return "I need a word to define."
    try:
        from PyDictionary import PyDictionary
        result = PyDictionary().meaning(word)
        if not result:
            return f"I could not find a definition for {word}."

        senses = [
            f"{part}: {defs[0]}"
            for part, defs in result.items()
            if defs
        ]
        if not senses:
            return f"I could not find a definition for {word}."
        return f"{word} -- " + "; ".join(senses[:_MAX_WORDS])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Dictionary lookup failed for {word!r}: {e}")
        return f"I could not retrieve a definition for {word}."


def _join(words, limit: int = 8) -> str:
    return ", ".join(list(words)[:limit])


def get_synonym(word: str) -> str:
    """Synonyms of a word."""
    if not (word or "").strip():
        return "I need a word to find synonyms for."
    try:
        from PyDictionary import PyDictionary
        result = PyDictionary().synonym(word)
        if not result:
            return f"I found no synonyms for {word}."
        return f"Synonyms for {word}: {_join(result)}"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Synonym lookup failed for {word!r}: {e}")
        return f"I could not retrieve synonyms for {word}."


def get_antonym(word: str) -> str:
    """Antonyms of a word."""
    if not (word or "").strip():
        return "I need a word to find antonyms for."
    try:
        from PyDictionary import PyDictionary
        result = PyDictionary().antonym(word)
        if not result:
            return f"I found no antonyms for {word}."
        return f"Antonyms for {word}: {_join(result)}"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Antonym lookup failed for {word!r}: {e}")
        return f"I could not retrieve antonyms for {word}."
