"""OpenAI-compatible provider adapter.

One adapter serves every OpenAI-compatible endpoint, so adding a new
OpenAI-compatible provider (OpenRouter, OpenCode Zen, a local vLLM server, ...)
is configuration only -- no new code.

Conversation history is held by the Brain and resent with every request, which
is how the Chat Completions API works and what keeps a conversation intact
across a model switch.

Endpoints are public information and are baked into code (they are not secrets);
API keys are always read from the environment and never stored here.
"""

from __future__ import annotations

import base64
import inspect
import json
import ssl
import threading
import typing
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from jarvis.providers.base import (
    Capabilities,
    ChatSession,
    ModelProvider,
    ModelResponse,
    ProviderError,
    ToolCall,
    classify_error,
    new_tool_call_id,
)

DEFAULT_TIMEOUT = 60.0


def _ssl_context() -> ssl.SSLContext:
    """TLS context with a usable CA bundle.

    urllib on this platform has no trust store it can find, so every HTTPS call
    died with CERTIFICATE_VERIFY_FAILED -- the adapter could never reach a real
    OpenRouter or OpenCode Zen endpoint. `requests` is already a hard
    dependency and ships certifi, so borrow that bundle. Verification stays on;
    only the CA source changes.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - certifi absent: fall back to system store
        return ssl.create_default_context()


def _openai_user_parts(text: Optional[str], images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Render a text+image turn as OpenAI content parts.

    Images travel as inline base64 data URLs, which is what the Chat Completions
    API expects for uploaded pictures. Text comes first so the model reads the
    question before the evidence; an image-only turn sends no text part at all.
    """
    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for img in images:
        encoded = base64.b64encode(img["data"]).decode("ascii")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img['mime']};base64,{encoded}"},
        })
    return parts


class OpenAICompatSession(ChatSession):
    """Stateless-per-call chat session over the Chat Completions API."""

    def __init__(
        self,
        provider: "OpenAICompatProvider",
        model_id: str,
        system_prompt: str,
        tools: List[Any],
        history: List[Dict[str, Any]],
        options: Dict[str, Any],
    ):
        self._provider = provider
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._tools = tools
        self._messages: List[Dict[str, Any]] = []
        self._options = options

        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})
        for msg in history or []:
            self._append(msg)

    # -- neutral -> OpenAI wire format ------------------------------------

    def _append(self, msg: Dict[str, Any]) -> None:
        role = msg.get("role")
        if role == "assistant":
            entry: Dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
            calls = msg.get("tool_calls") or []
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": c.get("id") or new_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": c.get("name", ""),
                            "arguments": c.get("arguments")
                            if isinstance(c.get("arguments"), str)
                            else json.dumps(c.get("arguments") or {}),
                        },
                    }
                    for c in calls
                ]
            self._messages.append(entry)

        elif role == "tool":
            results = msg.get("results")
            if results is None:
                results = [{
                    "id": msg.get("tool_call_id"),
                    "name": msg.get("name", ""),
                    "result": msg.get("content", ""),
                }]
            # The OpenAI API expects one message per tool result.
            for r in results:
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": r.get("id") or new_tool_call_id(),
                    "content": str(r.get("result", "")),
                })

        else:
            content = msg.get("content")
            images = msg.get("images") or []
            if images:
                self._messages.append({
                    "role": "user",
                    "content": _openai_user_parts(content, images),
                })
            elif content:
                self._messages.append({"role": "user", "content": content})

    # -- transport ---------------------------------------------------------

    def send_message(self, payload: Dict[str, Any]) -> ModelResponse:
        kind = payload.get("kind")
        if kind == "user":
            # The user's text is appended verbatim; images ride alongside it.
            self._append({
                "role": "user",
                "content": payload["text"],
                "images": payload.get("images"),
            })
        elif kind == "tool_results":
            self._append({"role": "tool", "results": payload.get("results", [])})
        else:
            raise ValueError(f"Unsupported payload kind: {kind!r}")

        body: Dict[str, Any] = {
            "model": self._model_id,
            "messages": self._messages,
        }
        if self._options.get("temperature") is not None:
            body["temperature"] = self._options["temperature"]
        if self._options.get("max_output_tokens"):
            body["max_tokens"] = self._options["max_output_tokens"]

        tools_payload = self._provider.build_tools_payload(self._tools)
        if tools_payload:
            body["tools"] = tools_payload
            body["tool_choice"] = "auto"

        # Per request, from the turn budget the Brain passed when opening the
        # session -- so a tool loop cannot outlive the user.
        response = self._provider.post_chat(
            body, model_id=self._model_id, timeout=self._options.get("timeout", DEFAULT_TIMEOUT)
        )
        return self._parse(response)

    def _parse(self, data: Dict[str, Any]) -> ModelResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(
                classify_error(RuntimeError(str(data.get("error") or "empty response"))),
                self._model_id,
                detail="no choices in response",
                provider=self._provider.name,
            )

        message = choices[0].get("message") or {}
        text = message.get("content")

        tool_calls: List[ToolCall] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (TypeError, ValueError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(ToolCall(
                id=call.get("id") or new_tool_call_id(),
                name=fn.get("name", ""),
                arguments=args,
            ))

        return ModelResponse(text=(text or "").strip() or None, tool_calls=tool_calls, raw=data)


class OpenAICompatProvider(ModelProvider):
    """Any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        context_window: int = 128_000,
        capabilities: Optional[Capabilities] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self._context_window = context_window
        self._capabilities = capabilities or Capabilities(
            reasoning=True, coding=True, conversation=True,
            tool_calling=True, long_context=False,
        )
        self._extra_headers = headers or {}
        self._tool_cache: Dict[str, List[Dict[str, Any]]] = {}

    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.base_url)

    def capabilities(self) -> Capabilities:
        return self._capabilities

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
        return OpenAICompatSession(
            provider=self,
            model_id=model_id,
            system_prompt=system_prompt,
            tools=tools,
            history=list(history or []),
            options=options,
        )

    # -- tools -------------------------------------------------------------

    def build_tools_payload(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert callable Python functions to OpenAI tool JSON schemas."""
        if not tools:
            return []
        cached = self._tool_cache.get(id(tools))
        if cached is not None:
            return cached

        payload: List[Dict[str, Any]] = []
        for fn in tools:
            schema = _function_to_schema(fn)
            if schema:
                payload.append(schema)

        # Cache keyed on the tool list identity; invalidated when it changes.
        self._tool_cache = {id(tools): payload}
        return payload

    # -- transport ---------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # Without this urllib sends "Python-urllib/x.y", which Cloudflare
            # in front of OpenRouter and OpenCode Zen rejects with
            # "HTTP 403 error code: 1010" (banned browser signature).
            "User-Agent": "jarvis-2.0-model-layer",
        }
        headers.update(self._extra_headers)
        return headers

    def post_chat(
        self, body: Dict[str, Any], model_id: str, timeout: float = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        """POST to /chat/completions and return parsed JSON, classifying errors."""
        url = f"{self.base_url}/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=_ssl_context()
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
            raise ProviderError(
                classify_error(RuntimeError(f"{exc.code} {raw}")),
                model_id,
                detail=f"HTTP {exc.code}: {raw}",
                provider=self.name,
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                classify_error(exc.reason if isinstance(exc.reason, BaseException) else exc),
                model_id,
                detail=f"connection failure: {exc.reason}",
                provider=self.name,
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                classify_error(exc), model_id, detail="request timed out", provider=self.name
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(
                classify_error(exc), model_id, detail="invalid JSON response", provider=self.name
            ) from exc

    def probe(self, timeout: float = 10.0) -> bool:
        """Check the models endpoint to confirm the key/endpoint is usable."""
        if not self.is_configured():
            return False
        request = urllib.request.Request(
            f"{self.base_url}/models", headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=_ssl_context()
            ) as resp:
                json.loads(resp.read().decode("utf-8"))
            return True
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


def _function_to_schema(fn: Any) -> Optional[Dict[str, Any]]:
    """Build an OpenAI function-tool schema from a Python callable.

    Uses type hints and the docstring so a tool's existing documentation is
    reused rather than duplicated in config.
    """
    name = getattr(fn, "__name__", None)
    if not name:
        return None

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None

    properties: Dict[str, Any] = {}
    required: List[str] = []

    if signature:
        try:
            hints = _resolve_hints(fn)
        except Exception:  # noqa: BLE001
            hints = {}
        for param_name, param in signature.parameters.items():
            if param_name in ("self", "cls"):
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            json_type = _json_type(hints.get(param_name, param.annotation))
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n\n")[0].strip() or name

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description[:500],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _resolve_hints(fn: Any) -> Dict[str, Any]:
    return typing.get_type_hints(fn)


def _json_type(annotation: Any) -> str:
    """Map a Python annotation to a JSON-schema type name."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return "string"

    text = str(annotation).lower()
    if "bool" in text:
        return "boolean"
    if "int" in text:
        return "integer"
    if "float" in text:
        return "number"
    if "list" in text or "sequence" in text:
        return "array"
    if "dict" in text or "mapping" in text:
        return "object"
    return "string"