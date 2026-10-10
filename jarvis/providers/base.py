"""Provider-agnostic model abstractions.

Nothing in this module knows about any specific vendor SDK. The Brain talks
only to :class:`ModelProvider`, so it never needs to know whether it is talking
to Gemini, OpenRouter, or OpenCode Zen.

Design rules enforced here:

* **One message format.** Every provider receives and returns the same neutral
  message dicts, so conversation history stays portable across models.
* **Errors are classified, not lumped.** A 429 must not be retried like a 400,
  and a 401 must never be hidden behind endless retries.
* **Errors are redacted.** Raw provider payloads can embed request URLs or
  headers, so :class:`ProviderError` only ever carries a short safe detail.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class ErrorKind(str, Enum):
    """What kind of failure occurred.

    Used by the router to decide between retry, fallback, cooldown, or
    fail-fast. Treating every error the same is how a dead endpoint ends up
    being hammered and how real misconfiguration gets masked.
    """

    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED = "unsupported"
    SERVER = "server"
    TIMEOUT = "timeout"
    NETWORK = "network"
    UNKNOWN = "unknown"


#: Failures where trying a different model may still succeed.
RETRYABLE_KINDS = frozenset({
    ErrorKind.RATE_LIMIT,
    ErrorKind.SERVER,
    ErrorKind.TIMEOUT,
    ErrorKind.NETWORK,
})

#: Failures that are the operator's problem: retrying cannot help.
FATAL_KINDS = frozenset({ErrorKind.AUTH, ErrorKind.INVALID_REQUEST, ErrorKind.UNSUPPORTED})


#: Substrings that identify an error kind across provider SDKs/HTTP payloads.
_ERROR_SIGNATURES: tuple[tuple[str, ErrorKind], ...] = (
    # Checked before numeric codes so the most specific match wins.
    ("resource_exhausted", ErrorKind.RATE_LIMIT),
    ("rate_limit", ErrorKind.RATE_LIMIT),
    ("rate limit", ErrorKind.RATE_LIMIT),
    ("too many requests", ErrorKind.RATE_LIMIT),
    ("quota", ErrorKind.RATE_LIMIT),
    ("insufficient_quota", ErrorKind.RATE_LIMIT),
    ("overloaded", ErrorKind.RATE_LIMIT),
    ("api_key_invalid", ErrorKind.AUTH),
    ("api key not valid", ErrorKind.AUTH),
    ("unauthorized", ErrorKind.AUTH),
    ("invalid_api_key", ErrorKind.AUTH),
    ("authentication", ErrorKind.AUTH),
    ("permission", ErrorKind.AUTH),
    ("unsupported", ErrorKind.UNSUPPORTED),
    ("not supported", ErrorKind.UNSUPPORTED),
    ("does not support", ErrorKind.UNSUPPORTED),
    ("invalid argument", ErrorKind.INVALID_REQUEST),
    ("invalid_request", ErrorKind.INVALID_REQUEST),
    ("bad request", ErrorKind.INVALID_REQUEST),
    ("deadline exceeded", ErrorKind.TIMEOUT),
    ("timed out", ErrorKind.TIMEOUT),
    ("timeout", ErrorKind.TIMEOUT),
    ("service unavailable", ErrorKind.SERVER),
    ("unavailable", ErrorKind.SERVER),
    ("internal error", ErrorKind.SERVER),
    ("bad gateway", ErrorKind.SERVER),
    ("connection", ErrorKind.NETWORK),
    ("gaierror", ErrorKind.NETWORK),
    ("nodename nor servname", ErrorKind.NETWORK),
    ("name or service not known", ErrorKind.NETWORK),
    ("network is unreachable", ErrorKind.NETWORK),
    ("reset by peer", ErrorKind.NETWORK),
    ("temporarily unavailable", ErrorKind.SERVER),
)

_NUMERIC_SIGNATURES: tuple[tuple[str, ErrorKind], ...] = (
    ("429", ErrorKind.RATE_LIMIT),
    ("401", ErrorKind.AUTH),
    ("403", ErrorKind.AUTH),
    ("400", ErrorKind.INVALID_REQUEST),
    ("404", ErrorKind.INVALID_REQUEST),
    ("408", ErrorKind.TIMEOUT),
    ("504", ErrorKind.TIMEOUT),
    ("500", ErrorKind.SERVER),
    ("502", ErrorKind.SERVER),
    ("503", ErrorKind.SERVER),
    ("529", ErrorKind.SERVER),
)


def classify_error(exc: BaseException) -> ErrorKind:
    """Map a provider exception onto an :class:`ErrorKind`.

    Works on exception text rather than vendor types so the same heuristic
    serves every provider.
    """
    if isinstance(exc, TimeoutError):
        return ErrorKind.TIMEOUT

    text = f"{type(exc).__name__} {exc}".lower()

    for needle, kind in _ERROR_SIGNATURES:
        if needle in text:
            return kind

    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        for needle, kind in _NUMERIC_SIGNATURES:
            if str(status) == needle:
                return kind

    for needle, kind in _NUMERIC_SIGNATURES:
        if needle in text:
            return kind

    return ErrorKind.UNKNOWN


#: Credential *prefixes* that identify a token by shape alone. Redacting by
#: prefix matters because an upstream error can echo a key without any nearby
#: "Bearer"/"api_key" label to key off.
_KEY_PREFIXES = (
    "AIza",        # Google / Gemini
    "sk-",         # generic + OpenAI-style
    "sk-or-v1-",   # OpenRouter
    "ghp_",        # GitHub
    "ghu_",
    "xai-",        # xAI
    "AKIA",        # AWS access key id
)

#: Contextual markers that indicate a credential follows.
_REDACTIONS = ("Bearer", "api_key=", "apikey", "authorization", "x-api-key", "access_token")


def redact(text: str, limit: int = 200) -> str:
    """Strip anything credential-shaped from a string bound for logs or clients.

    Defensive: even though providers should not echo keys back, an unexpected
    upstream payload must never be able to leak one into a log or an HTTP
    response. Two passes are applied:

    1. contextual markers (``Bearer``, ``api_key=``) and everything after them
    2. bare key prefixes (``AIza``, ``sk-``, ``ghp_`` ...) anywhere in the text

    The prefix pass is what catches a provider that echoes a key on its own.
    """
    import re

    if not text:
        return ""

    out = " ".join(str(text).split())
    lowered = out.lower()

    for marker in _REDACTIONS:
        idx = lowered.find(marker.lower())
        while idx != -1:
            out = out[:idx] + "[REDACTED]"
            lowered = out.lower()
            idx = lowered.find(marker.lower(), idx + len("[REDACTED]"))

    for prefix in _KEY_PREFIXES:
        pattern = re.compile(re.escape(prefix) + r"[A-Za-z0-9_\-]{6,}")
        out = pattern.sub("[REDACTED]", out)

    return out[:limit]


class ProviderError(Exception):
    """A classified, redacted model/provider failure.

    Carries ``kind`` so the router can react appropriately, and ``safe_detail``
    which is guaranteed free of credentials.
    """

    def __init__(
        self,
        kind: ErrorKind,
        model_id: str,
        detail: str = "",
        provider: str = "",
        status_code: Optional[int] = None,
    ):
        self.kind = kind
        self.model_id = model_id
        self.provider = provider
        self.status_code = status_code
        self.safe_detail = redact(detail)
        super().__init__(f"{kind.value} from {provider or 'provider'}:{model_id}: {self.safe_detail}")

    @property
    def retryable(self) -> bool:
        """True when a different model could plausibly serve the request."""
        return self.kind in RETRYABLE_KINDS


# ---------------------------------------------------------------------------
# Capability profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capabilities:
    """What a model can actually do.

    Only advertise what has been verified for that model; the router uses these
    flags to exclude models that cannot satisfy a request.
    """

    reasoning: bool = False
    coding: bool = False
    conversation: bool = True
    tool_calling: bool = False
    vision: bool = False
    structured_output: bool = False
    long_context: bool = False
    streaming: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return {
            "reasoning": self.reasoning,
            "coding": self.coding,
            "conversation": self.conversation,
            "tool_calling": self.tool_calling,
            "vision": self.vision,
            "structured_output": self.structured_output,
            "long_context": self.long_context,
            "streaming": self.streaming,
        }


# ---------------------------------------------------------------------------
# Neutral wire format
# ---------------------------------------------------------------------------


def new_tool_call_id() -> str:
    """Synthetic tool-call id for providers that do not supply one."""
    return f"call_{uuid.uuid4().hex[:16]}"


#: Image types accepted for visual input. Explicit allowlist: an unknown type
#: is more likely to be a mistake or an attack than a real screenshot.
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")

#: Per-image cap. Provider limits are lower still (OpenAI ~20MB total), but
#: refusing early keeps a malformed request from becoming a large upload.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

#: Rough token cost of one image, used only for context-window routing. A
#: vision model charges far more than a few text tokens for an image, and
#: routing has to know that or it will pick a small-window model.
IMAGE_TOKEN_COST = 1_600


def image_part(data: bytes, mime: str = "image/png") -> Dict[str, Any]:
    """One image in the provider-neutral attachment format.

    Deliberately just bytes plus a MIME type: that is the intersection of what
    every vision API accepts. Provider-specific wrapping (base64 data URLs for
    OpenAI-compatible, inline blobs for Gemini) belongs in the adapters.
    """
    if not data:
        raise ValueError("image_part requires non-empty image data.")
    if mime not in IMAGE_MIME_TYPES:
        raise ValueError(f"Unsupported image type '{mime}'.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is {len(data) // 1024}KB; the limit is {MAX_IMAGE_BYTES // 1024}KB."
        )
    return {"kind": "image", "mime": mime, "data": data}


def user_message(text: str, images: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A user turn. The user's text is carried verbatim.

    Images ride alongside the text rather than inside it, so every existing
    reader of ``content`` keeps working unchanged and a text-only turn is
    byte-for-byte what it was before images existed.
    """
    msg: Dict[str, Any] = {"role": "user", "content": text}
    if images:
        msg["images"] = list(images)
    return msg


def assistant_message(
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls or [],
    }


def tool_message(tool_call_id: str, name: str, content: str) -> Dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate (~4 chars/token) used for context-window routing.

    Images are charged separately: they dominate a vision turn's context cost,
    and estimating them as a handful of characters would route a screenshot to
    a model whose window cannot hold it.
    """
    chars = 0
    image_tokens = 0
    for msg in messages:
        chars += len(str(msg.get("content") or ""))
        image_tokens += IMAGE_TOKEN_COST * len(msg.get("images") or [])
        for call in msg.get("tool_calls") or []:
            chars += len(call.get("name", "")) + len(str(call.get("arguments", "")))
    return chars // 4 + image_tokens


# ---------------------------------------------------------------------------
# Request / response
# ---------------------------------------------------------------------------


@dataclass
class ModelRequest:
    """A provider-neutral model call.

    ``messages`` is the full conversation, oldest first, already trimmed by the
    caller. The final message is what the user just said.
    """

    messages: List[Dict[str, Any]]
    system_prompt: str = ""
    tools: List[Any] = field(default_factory=list)
    temperature: float = 0.7
    max_output_tokens: int = 1024
    timeout: float = 60.0
    # Routing metadata (adapters may ignore).
    task_type: str = "conversation"
    tool_required: bool = False

    @property
    def last_user_text(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""


@dataclass
class ToolCall:
    """A model-requested tool invocation, in neutral form."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """A provider-neutral model reply."""

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class ChatSession(ABC):
    """A stateful binding to one model for one conversation.

    The Brain calls :meth:`send_message` with a neutral payload and receives a
    :class:`ModelResponse`. Implementations own all wire-format translation.
    """

    @abstractmethod
    def send_message(self, payload: Dict[str, Any]) -> ModelResponse:
        """Send one turn.

        ``payload`` is either ``{"kind": "user", "text": str}`` or
        ``{"kind": "tool_results", "results": [{"id","name","result"}]}.

        The deadline comes from the session's ``timeout`` option, which the
        Brain sets from what is left of the *turn's* budget rather than a
        fresh per-call allowance: a tool loop that keeps calling tools must
        not restart the clock each round. Adapters apply it per HTTP request,
        which is the granularity every supported transport exposes.
        """

    def close(self) -> None:
        """Release any provider-side resources. Optional."""


class ModelProvider(ABC):
    """Base class for all model providers."""

    #: Short stable identifier used in configuration (e.g. ``gemini``).
    name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        """True when required credentials/endpoint are present."""

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Capabilities of the models this provider serves."""

    @abstractmethod
    def context_window(self) -> int:
        """Default context window in tokens for this provider's models."""

    def open_session(
        self,
        model_id: str,
        system_prompt: str,
        tools: List[Any],
        history: Optional[List[Dict[str, Any]]] = None,
        **options: Any,
    ) -> ChatSession:
        """Open a conversation session, optionally seeded with prior history.

        History is passed in rather than held provider-side so that switching
        models mid-conversation preserves continuity.
        """
        raise NotImplementedError

    def probe(self, timeout: float = 10.0) -> bool:
        """Cheap availability check. Returns True when the provider responds."""
        return False

    # -- health -----------------------------------------------------------
    def record_success(self, latency: float, model_id: str = "") -> None:
        """Hook for provider-side health bookkeeping."""

    def record_failure(self, error: ProviderError, model_id: str = "") -> None:
        """Hook for provider-side health bookkeeping."""


def now() -> float:
    """Indirection so tests can reason about time."""
    return time.time()