"""Gemini provider adapter.

This is the **only** module that knows the Gemini SDK exists. It translates the
neutral message format in :mod:`jarvis.providers.base` to and from the
``google-genai`` wire format, so the Brain can stay provider-agnostic.

Gemini keeps conversation history server-side in a ``Chat`` object, but history
is still owned by the Brain and passed in explicitly on session creation. That
is what allows a conversation to survive a model switch.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from jarvis.providers.base import (
    Capabilities,
    ChatSession,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
    classify_error,
    new_tool_call_id,
    redact,
)

_CLIENT_LOCK = threading.Lock()
_client_cache: Dict[str, Any] = {}


def _get_client(api_key: str) -> Any:
    """Lazily build and cache a GenAI client for an API key."""
    if api_key not in _client_cache:
        with _CLIENT_LOCK:
            if api_key not in _client_cache:
                from google import genai
                _client_cache[api_key] = genai.Client(api_key=api_key)
    return _client_cache[api_key]


def _to_gemini_content(messages: List[Dict[str, Any]], types: Any) -> List[Any]:
    """Convert neutral messages to Gemini ``Content`` objects.

    Roles are mapped: ``assistant`` -> ``model``, ``tool`` -> ``user`` carrying
    a ``function_response`` part (this is Gemini's required shape).
    """
    contents: List[Any] = []
    for msg in messages:
        role = msg.get("role")

        if role == "assistant":
            parts: List[Any] = []
            if msg.get("content"):
                parts.append(types.Part(text=msg["content"]))
            for call in msg.get("tool_calls") or []:
                args = call.get("arguments") or {}
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                parts.append(types.Part(
                    function_call=types.FunctionCall(
                        id=call.get("id") or new_tool_call_id(),
                        name=call.get("name", ""),
                        args=args,
                    )
                ))
            if parts:
                contents.append(types.Content(role="model", parts=parts))

        elif role == "tool":
            results = msg.get("results")
            if results is None:
                results = [{
                    "id": msg.get("tool_call_id"),
                    "name": msg.get("name", ""),
                    "result": msg.get("content", ""),
                }]
            parts = [
                types.Part(function_response=types.FunctionResponse(
                    id=r.get("id") or new_tool_call_id(),
                    name=r.get("name", ""),
                    response={"result": r.get("result", "")},
                ))
                for r in results
            ]
            if parts:
                contents.append(types.Content(role="user", parts=parts))

        else:  # user / system-fallback
            content = msg.get("content")
            if content:
                contents.append(types.Content(
                    role="user", parts=[types.Part(text=content)]
                ))

    return contents


class GeminiChatSession(ChatSession):
    """A Gemini ``Chat`` wrapped in the neutral session interface."""

    def __init__(self, chat: Any, model_id: str, types: Any):
        self._chat = chat
        self._model_id = model_id
        self._types = types

    def send_message(self, payload: Dict[str, Any]) -> ModelResponse:
        types = self._types
        kind = payload.get("kind")

        if kind == "user":
            content: Any = payload["text"]
        elif kind == "tool_results":
            parts = [
                types.Part(function_response=types.FunctionResponse(
                    id=r.get("id") or new_tool_call_id(),
                    name=r.get("name", ""),
                    response={"result": r.get("result", "")},
                ))
                for r in payload.get("results", [])
            ]
            if not parts:
                return ModelResponse(text="", raw=None)
            content = types.Content(role="user", parts=parts)
        else:
            raise ValueError(f"Unsupported payload kind: {kind!r}")

        try:
            response = self._chat.send_message(content)
        except Exception as exc:  # noqa: BLE001 - classified and re-raised
            raise ProviderError(
                classify_error(exc),
                self._model_id,
                detail=str(exc),
                provider="gemini",
            ) from exc

        return self._to_response(response)

    def _to_response(self, response: Any) -> ModelResponse:
        types = self._types
        texts: List[str] = []
        tool_calls: List[ToolCall] = []

        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    tool_calls.append(ToolCall(
                        id=getattr(fc, "id", None) or new_tool_call_id(),
                        name=fc.name,
                        arguments=dict(getattr(fc, "args", None) or {}),
                    ))

        if not texts and not tool_calls:
            fallback = getattr(response, "text", None)
            if fallback:
                texts.append(fallback)

        return ModelResponse(
            text="\n".join(texts).strip() or None,
            tool_calls=tool_calls,
            raw=response,
        )

    def close(self) -> None:
        self._chat = None


class GeminiProvider(ModelProvider):
    """Google Gemini via the official ``google-genai`` SDK."""

    name = "gemini"

    def __init__(self, api_key: str, default_model: str = "", context_window: int = 1_048_576):
        self.api_key = api_key or ""
        self.default_model = default_model
        self._context_window = context_window

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            reasoning=True,
            coding=True,
            conversation=True,
            tool_calling=True,
            vision=True,
            structured_output=True,
            long_context=True,
            streaming=True,
        )

    def context_window(self) -> int:
        return self._context_window

    def open_session(
        self,
        model_id: str,
        system_prompt: str,
        tools: List[Any],
        history: Optional[List[Dict[str, Any]]] = None,
        **options: Any,
    ) -> ChatSession:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=options.get("temperature", 0.7),
            max_output_tokens=options.get("max_output_tokens", 1024),
            tools=tools or None,
        )

        client = _get_client(self.api_key)
        contents = _to_gemini_content(history or [], types)

        try:
            chat = client.chats.create(
                model=model_id,
                config=config,
                history=contents or None,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                classify_error(exc),
                model_id,
                detail=str(exc),
                provider=self.name,
            ) from exc

        return GeminiChatSession(chat, model_id, types)

    def probe(self, timeout: float = 10.0) -> bool:
        """Cheapest available availability signal: can we list models?"""
        if not self.api_key:
            return False
        try:
            client = _get_client(self.api_key)
            next(iter(client.models.list()), None)
            return True
        except Exception as exc:  # noqa: BLE001
            # Authorisation problems mean configured-but-unusable, which the
            # status endpoint should surface, so this is reported not raised.
            _ = redact(str(exc))
            return False