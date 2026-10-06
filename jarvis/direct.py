"""Minimal direct-model path for diagnostics.

This module deliberately bypasses every layer that could alter the user's
message: no tools, no long-term memory injection, no goal/specialist routing,
no autonomous loop, no history reuse.

    user message -> model -> response

It exists to answer one question: does the base model interaction work at all?
If this path fails, the problem is the model/key/quota. If it works but
`JarvisBrain` fails, the problem is in the orchestration/prompt/state layer.

Usage:
    from jarvis.direct import direct_ask
    direct_ask("What is 2 + 2?")
"""

from typing import List, Optional

from jarvis.config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_FALLBACK_MODELS,
    MAX_MODEL_ATTEMPTS,
)
from jarvis.logger import logger

_client = None


def get_client():
    """Lazily create the GenAI client. Raises if no API key is configured."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not configured in your .env file. "
                "Please add GEMINI_API_KEY to your .env file."
            )
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def model_candidates() -> List[str]:
    """Primary configured model first, then configured fallbacks, de-duplicated."""
    ordered: List[str] = []
    for name in [LLM_MODEL, *LLM_FALLBACK_MODELS]:
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def direct_ask(user_text: str, model: Optional[str] = None) -> str:
    """Send `user_text` straight to the model and return the raw reply.

    No system prompt, no tools, no memory. The user message is transmitted
    verbatim as a single `user` turn.

    Args:
        user_text: The exact message to send.
        model: Optional explicit model name; otherwise tries candidates in order.

    Returns:
        The model's text response.

    Raises:
        ValueError: If `user_text` is empty.
        RuntimeError: If every candidate model fails.
    """
    if not user_text or not user_text.strip():
        raise ValueError("direct_ask requires a non-empty user message.")

    from google.genai import types

    client = get_client()
    candidates = [model] if model else model_candidates()
    failures: List[str] = []

    for name in candidates:
        for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=name,
                    contents=user_text,
                    config=types.GenerateContentConfig(
                        temperature=LLM_TEMPERATURE,
                        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                    ),
                )
                text = (response.text or "").strip()
                if not text:
                    failures.append(f"{name}: empty response")
                    continue
                logger.info(f"[DIRECT] model={name} attempt={attempt} OK")
                return text
            except Exception as e:  # noqa: BLE001 - diagnostic path reports all
                failures.append(f"{name} attempt {attempt}: {_describe(e)}")
                logger.warning(f"[DIRECT] model={name} attempt={attempt} failed: {_describe(e)}")

    raise RuntimeError(
        "Direct model call failed for every candidate model.\n"
        + "\n".join(failures)
    )


def _describe(exc: Exception) -> str:
    """Compact, non-secret description of an API error."""
    text = str(exc).replace("\n", " ")
    return f"{type(exc).__name__}: {text[:180]}"