"""Provider registry.

Providers are constructed from configuration rather than hard-coded, so adding
a provider never requires editing routing code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from jarvis.providers.base import (
    Capabilities,
    ChatSession,
    ErrorKind,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
    classify_error,
    estimate_tokens,
    redact,
)
from jarvis.providers.gemini import GeminiProvider
from jarvis.providers.openai_compat import OpenAICompatProvider

__all__ = [
    "Capabilities",
    "ChatSession",
    "ErrorKind",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ProviderError",
    "ToolCall",
    "classify_error",
    "estimate_tokens",
    "redact",
    "build_provider",
]


def build_provider(name: str, settings: Dict[str, Any], env: Optional[Dict[str, str]] = None) -> Optional[ModelProvider]:
    """Construct a provider from its configuration block.

    ``settings`` should contain ``type`` (gemini | openai_compatible),
    ``api_key_env``, and type-specific fields. Credentials are read from the
    environment at construction time and are never stored in config.

    Returns None if the provider type is unknown.
    """
    env = env if env is not None else {}
    ptype = str(settings.get("type", "")).lower()
    api_key = str(env.get(settings.get("api_key_env", ""), "") or "")

    if ptype == "gemini":
        return GeminiProvider(
            api_key=api_key,
            default_model=str(settings.get("default_model", "")),
            context_window=int(settings.get("context_window", 1_048_576)),
        )

    if ptype == "openai_compatible":
        caps = settings.get("capabilities")
        capabilities = Capabilities(**caps) if isinstance(caps, dict) else None
        return OpenAICompatProvider(
            name=name,
            api_key=api_key,
            base_url=str(settings.get("base_url", "")),
            context_window=int(settings.get("context_window", 128_000)),
            capabilities=capabilities,
            headers={k: str(v) for k, v in (settings.get("headers") or {}).items()},
        )

    return None