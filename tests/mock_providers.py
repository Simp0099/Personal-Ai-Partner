"""Deterministic mock providers for routing tests.

Real APIs cannot be used to test routing: they are slow, rate-limited, and
non-deterministic, which is exactly the opposite of what a routing test needs.
These fakes let every routing decision be asserted exactly.

Included fakes:

``FastHealthyModel``      quick, reliable, tool-capable
``SlowHealthyModel``      reliable but slow
``FailingModel``          always raises a retryable server error
``RateLimitedModel``      always raises a rate-limit error
``ToolCapableModel``      supports tool calling
``NonToolModel``          cannot call tools (must never receive tool requests)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from jarvis.health import HealthTracker
from jarvis.model_layer import ModelLayer
from jarvis.models import ModelRegistry
from jarvis.providers.base import (
    Capabilities,
    ChatSession,
    ErrorKind,
    ModelProvider,
    ModelResponse,
    ProviderError,
    ToolCall,
    new_tool_call_id,
)


class ScriptedSession(ChatSession):
    """A session driven by a per-model behaviour callable."""

    def __init__(self, behaviour: Callable[[List[Dict[str, Any]], Dict[str, Any]], Any],
                 history: List[Dict[str, Any]], calls: List[Dict[str, Any]]):
        self._behaviour = behaviour
        self._history = list(history or [])
        self._calls = calls

    def send_message(
        self, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> ModelResponse:
        self._calls.append(payload)
        result = self._behaviour(self._history, payload)

        if isinstance(result, ModelResponse):
            return result

        # Let a behaviour append to history the way a real model would.
        if result is None:
            if payload.get("kind") == "user":
                self._history.append({"role": "user", "content": payload.get("text", "")})
            return ModelResponse(text="")

        # Convenience: return plain text.
        self._history.append({"role": "assistant", "content": str(result)})
        return ModelResponse(text=str(result))

    def close(self) -> None:
        pass


class ScriptedProvider(ModelProvider):
    """A provider whose behaviour is supplied per model id.

    ``behaviours`` maps model id -> callable(history, payload). The callable may
    return a :class:`ModelResponse`, a string, or ``None`` to append to history.
    """

    def __init__(
        self,
        name: str = "mock",
        behaviours: Optional[Dict[str, Callable]] = None,
        capabilities: Optional[Capabilities] = None,
        context_window: int = 128_000,
        configured: bool = True,
        record_calls: bool = True,
    ):
        self.name = name
        self._behaviours = dict(behaviours or {})
        self._capabilities = capabilities or Capabilities(
            reasoning=True, coding=True, conversation=True,
            tool_calling=True, long_context=False,
        )
        self._context_window = context_window
        self._configured = configured
        #: Every payload sent to this provider, in order. Routing tests assert
        #: on this to prove a request went to exactly one model.
        self.sent: List[Dict[str, Any]] = []
        #: Sessions opened, for continuity assertions.
        self.sessions: List[SessionRecord] = []
        #: Times `probe()` was called. Endpoint tests assert on this to prove a
        #: cached read did not go back to the provider.
        self.probes = 0

    def is_configured(self) -> bool:
        return self._configured

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def context_window(self) -> int:
        return self._context_window

    def open_session(self, model_id, system_prompt, tools, history=None, **options):
        record = SessionRecord(
            model_id=model_id,
            system_prompt=system_prompt,
            history=list(history or []),
            tools=tools,
            options=options,
        )
        self.sessions.append(record)

        behaviour = self._behaviours.get(model_id)
        if behaviour is None:
            behaviour = lambda h, p: "ok"

        session = ScriptedSession(behaviour, history or [], self.sent)
        record.session = session
        return session

    def probe(self, timeout: float = 10.0) -> bool:
        self.probes += 1
        return self._configured


class SessionRecord:
    """Captures what a session was opened with."""

    def __init__(self, model_id, system_prompt, history, tools, options):
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.history = history
        self.tools = tools
        self.options = options
        self.session: Optional[ChatSession] = None

    @property
    def user_messages(self) -> List[str]:
        return [
            str(m.get("content"))
            for m in self.history
            if m.get("role") == "user" and m.get("content")
        ]


# ---------------------------------------------------------------------------
# Behaviour factories
# ---------------------------------------------------------------------------

def reply(text: str) -> Callable:
    """A model that answers `text` and remembers the turn."""
    def _behaviour(history, payload):
        if payload.get("kind") == "user":
            history.append({"role": "user", "content": payload.get("text", "")})
        history.append({"role": "assistant", "content": text})
        return ModelResponse(text=text)
    return _behaviour


def tool_then_reply(tool_name: str, result: str, text: str,
                    args: Optional[Dict[str, Any]] = None) -> Callable:
    """A model that requests one tool call, then answers."""
    state = {"done": False}

    def _behaviour(history, payload):
        if payload.get("kind") == "tool_results":
            history.append({"role": "tool", "results": payload.get("results", [])})
            history.append({"role": "assistant", "content": text})
            return ModelResponse(text=text)
        if state["done"]:
            history.append({"role": "assistant", "content": text})
            return ModelResponse(text=text)
        state["done"] = True
        call = ToolCall(id=new_tool_call_id(), name=tool_name, arguments=args or {})
        return ModelResponse(text=None, tool_calls=[call])
    return _behaviour


def error(kind: ErrorKind, detail: str = "mock failure") -> Callable:
    """A model that always fails with the given error kind."""
    def _behaviour(history, payload):
        raise ProviderError(kind, "mock", detail, "mock")
    return _behaviour


def fail_then_success(kind: ErrorKind, text: str, times: int = 1) -> Callable:
    """Fails `times`, then succeeds."""
    state = {"remaining": times}

    def _behaviour(history, payload):
        if payload.get("kind") == "user" and state["remaining"] > 0:
            state["remaining"] -= 1
            raise ProviderError(kind, "mock", "transient mock failure", "mock")
        if payload.get("kind") == "user":
            history.append({"role": "user", "content": payload.get("text", "")})
        history.append({"role": "assistant", "content": text})
        return ModelResponse(text=text)
    return _behaviour


def empty() -> Callable:
    """A model that returns nothing usable."""
    def _behaviour(history, payload):
        return ModelResponse(text=None)
    return _behaviour


# ---------------------------------------------------------------------------
# Layer builder
# ---------------------------------------------------------------------------

def build_layer(
    models: Sequence[Dict[str, Any]],
    behaviours: Optional[Dict[str, Callable]] = None,
    provider_name: str = "mock",
    provider_capabilities: Optional[Capabilities] = None,
    context_window: int = 128_000,
    routing: Optional[Dict[str, Any]] = None,
    configured: bool = True,
    unconfigured_providers: Sequence[str] = (),
) -> ModelLayer:
    """Assemble a ModelLayer wired to scripted providers.

    ``models`` is a list of dicts with at least ``key``, ``model`` and
    ``capabilities``; other ModelSpec fields are passed through.

    Providers named in ``unconfigured_providers`` are registered but report
    themselves as having no credentials, so routing must route around them --
    the situation the real system hits when an API key is absent.
    """
    from jarvis.router import RoutingConfig, RoutingWeights

    specs = []
    for raw in models:
        entry = dict(raw)
        entry.setdefault("provider", provider_name)
        entry.setdefault("enabled", True)
        entry.setdefault("priority", 50)
        entry.setdefault("free", False)
        entry.setdefault("verified", True)
        entry.setdefault("context_window", context_window)
        caps = entry.get("capabilities") or {}
        entry["capabilities"] = {
            "reasoning": bool(caps.get("reasoning", True)),
            "coding": bool(caps.get("coding", True)),
            "conversation": bool(caps.get("conversation", True)),
            "tool_calling": bool(caps.get("tool_calling", True)),
            "vision": bool(caps.get("vision", False)),
            "structured_output": bool(caps.get("structured_output", True)),
            "long_context": bool(caps.get("long_context", False)),
            "streaming": bool(caps.get("streaming", True)),
        }
        specs.append(entry)

    registry = ModelRegistry.from_config({entry["key"]: entry for entry in specs})

    # Register the scripted provider under every provider name the models
    # reference, so a test can group models under arbitrary provider names
    # (useful for asserting cross-provider routing and identity consistency).
    provider_names = {entry["provider"] for entry in specs} | {provider_name}
    unconfigured = set(unconfigured_providers)
    providers: Dict[str, ScriptedProvider] = {
        name: ScriptedProvider(
            name=name,
            behaviours=behaviours or {},
            capabilities=provider_capabilities,
            context_window=context_window,
            configured=configured and name not in unconfigured,
        )
        for name in provider_names
    }

    routing_config = RoutingConfig()
    if routing:
        routing_config = RoutingConfig(
            enabled=routing.get("enabled", True),
            weights=RoutingWeights(
                capability=routing.get("capability_weight", 0.30),
                task=routing.get("task_weight", 0.25),
                reliability=routing.get("reliability_weight", 0.25),
                latency=routing.get("latency_weight", 0.10),
                priority=routing.get("priority_weight", 0.10),
            ),
            cooldown_seconds=routing.get("cooldown_seconds", 60.0),
            max_attempts=routing.get("max_attempts", 4),
            probe_cooling_models=routing.get("probe_cooling_models", True),
            latency_reference_ms=routing.get("latency_reference_ms", 4000.0),
        )

    health = HealthTracker(base_backoff=routing_config.cooldown_seconds)

    return ModelLayer(
        registry=registry,
        providers=providers,
        health=health,
        routing_config=routing_config,
        temperature=0.7,
        max_output_tokens=1024,
    )