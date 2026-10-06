"""JARVIS 2.0 — Backend API Bridge (FastAPI)

Exposes HTTP endpoints for the React frontend to communicate with
the Python backend (Gemini brain, tools, memory).

Phase 0 reliability fixes:
  - One `JarvisBrain` per `conversation_id`, so conversations cannot leak
    history into each other. Previously a single global brain was shared by
    every client and `/api/clear` wiped everyone's history.
  - The blocking brain call runs in a worker thread so the event loop is not
    stalled for the duration of a model call.
  - Failures are reported with a non-2xx status and `error: true`. The previous
    behaviour returned HTTP 200 with the raw exception text as the assistant's
    "reply", making outages indistinguishable from model output.

Run with:
    pip install fastapi uvicorn
    python api_server.py
"""

import asyncio
import threading
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import JARVIS brain
from jarvis.brain import JarvisBrain, BrainError
from jarvis.config import DEBUG_ENDPOINTS
from jarvis.logger import logger

app = FastAPI(title="JARVIS 2.0 API", version="2.0.0")

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Conversation registry
# ---------------------------------------------------------------------------
# Keyed by the client-supplied conversation id. Each entry is its own brain, so
# Gemini history cannot bleed between conversations.
_brains: Dict[str, JarvisBrain] = {}
_brains_lock = threading.Lock()

# Bound the registry so long-running servers cannot grow it without limit.
MAX_SESSIONS = 200


def get_brain(conversation_id: str) -> JarvisBrain:
    """Return the brain for `conversation_id`, creating it on first use."""
    with _brains_lock:
        brain = _brains.get(conversation_id)
        if brain is None:
            if len(_brains) >= MAX_SESSIONS:
                # Drop the oldest session rather than refusing new ones.
                oldest = next(iter(_brains))
                logger.info(f"Session limit reached; evicting oldest session {oldest}")
                _brains.pop(oldest, None)
            brain = JarvisBrain(conversation_id=conversation_id)
            _brains[conversation_id] = brain
            logger.info(f"[SESSION] created {conversation_id}")
        return brain


def drop_brain(conversation_id: str) -> bool:
    """Remove a conversation entirely. Returns True if one was removed."""
    with _brains_lock:
        return _brains.pop(conversation_id, None) is not None


def clear_all_brains() -> int:
    """Reset every conversation. Returns the number cleared."""
    with _brains_lock:
        count = len(_brains)
        _brains.clear()
        return count


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message, sent verbatim.")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Client-scoped conversation id. Omit for a single shared session.",
    )


class ChatResponse(BaseModel):
    response: str
    latency: int
    error: bool = False
    conversation_id: str = "default"


class ClearRequest(BaseModel):
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    api_key_configured: bool = True


class ModelInfo(BaseModel):
    """Per-model status. Contains no credentials."""

    key: str
    provider: str
    model_id: str
    enabled: bool
    free: bool
    verified: bool
    priority: int
    context_window: int
    max_output_tokens: int
    capabilities: Dict[str, bool]
    notes: str
    configured: bool
    provider_available: Optional[bool]
    health: str
    average_latency: float
    failure_rate: float
    cooldown_remaining: float
    last_error_kind: Optional[str]
    success_count: int
    failure_count: int


class ModelStatusResponse(BaseModel):
    routing_enabled: bool
    weights: Dict[str, float]
    max_attempts: int
    cooldown_seconds: float
    models: List[ModelInfo]


# ---------------------------------------------------------------------------
# /api/models status cache
# ---------------------------------------------------------------------------
# Building the status snapshot probes every *configured model* with a
# synchronous network request, so every call cost one round-trip per model --
# a dev tab polling this endpoint hammered the providers. It is pure
# diagnostics, so a reading up to MODEL_STATUS_TTL seconds old is fine.
#
# Only the assembled status is cached. It carries `configured` (a bool
# meaning "a key is present") and never a key, header or token.
MODEL_STATUS_TTL = 30.0

_model_status_cache: Optional[ModelStatusResponse] = None
_model_status_cached_at = 0.0
_model_status_lock = threading.Lock()


def _clear_model_status_cache() -> None:
    """Forget the cached snapshot. Exposed for tests."""
    global _model_status_cache, _model_status_cached_at
    with _model_status_lock:
        _model_status_cache = None
        _model_status_cached_at = 0.0


def _build_model_status() -> ModelStatusResponse:
    """Probe the providers and assemble the status response (uncached)."""
    from jarvis.model_layer import get_model_layer

    snapshot = get_model_layer().status()

    # Keep the response shape strict: drop any field the schema does not declare.
    models = [
        {k: v for k, v in row.items() if k in ModelInfo.model_fields}
        for row in snapshot.get("models", [])
    ]

    return ModelStatusResponse(
        routing_enabled=snapshot["routing_enabled"],
        weights=snapshot["weights"],
        max_attempts=snapshot["max_attempts"],
        cooldown_seconds=snapshot["cooldown_seconds"],
        models=models,
    )


def _get_model_status(now_ts: float) -> ModelStatusResponse:
    """Return the status snapshot, probing at most once per TTL.

    The lock is held across the probe so concurrent callers that all arrive on
    an expired cache perform exactly one probe between them rather than one
    each.
    """
    global _model_status_cache, _model_status_cached_at

    with _model_status_lock:
        cached = _model_status_cache
        if cached is not None and (now_ts - _model_status_cached_at) < MODEL_STATUS_TTL:
            return cached

        fresh = _build_model_status()
        _model_status_cache = fresh
        _model_status_cached_at = now_ts
        return fresh


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from jarvis.config import LLM_MODEL, GEMINI_API_KEY
    return HealthResponse(
        status="online",
        model=LLM_MODEL,
        version="2.0.0",
        api_key_configured=bool(GEMINI_API_KEY),
    )


@app.get("/api/models", response_model=ModelStatusResponse)
async def model_status():
    """Development endpoint: model/provider health, latency and capabilities.

    Reports which credentials are *present* (`configured`) but never returns a
    key. Each provider is probed with a cheap request, and the assembled result
    is cached for `MODEL_STATUS_TTL` seconds so repeated calls do not re-probe.
    Gated by `debug.endpoints` in config.
    """
    if not DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")

    # Probing is blocking and network-bound; keep it off the event loop, the
    # same way /api/chat keeps brain.ask off it.
    return await asyncio.to_thread(_get_model_status, time.monotonic())


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the JARVIS brain.

    The message is forwarded to the brain exactly as received. Blank input is
    rejected with 400 rather than being answered by the model.
    """
    start = time.time()

    if request.message is None or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conversation_id = request.conversation_id or "default"
    brain = get_brain(conversation_id)

    try:
        # brain.ask is blocking (network + tools); keep the event loop free.
        response = await asyncio.to_thread(brain.ask, request.message.strip())
        latency = int((time.time() - start) * 1000)
        return ChatResponse(
            response=response,
            latency=latency,
            error=False,
            conversation_id=conversation_id,
        )
    except ValueError as e:
        # Blank/invalid input — a client error, not a backend failure.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except BrainError as e:
        # Genuine backend failure. Report it as a failure; do not present the
        # error text as if the assistant had said it.
        logger.error(f"Brain error (session={conversation_id}): {e.detail or e}")
        latency = int((time.time() - start) * 1000)
        return ChatResponse(
            response=e.message,
            latency=latency,
            error=True,
            conversation_id=conversation_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Chat error (session={conversation_id}): {e}", exc_info=True)
        latency = int((time.time() - start) * 1000)
        return ChatResponse(
            response="The assistant hit an unexpected internal error.",
            latency=latency,
            error=True,
            conversation_id=conversation_id,
        )


@app.post("/api/clear")
async def clear_conversation(request: Optional[ClearRequest] = None):
    """Clear one conversation, or all of them when no id is supplied.

    The frontend previously never called this, so its "clean slate" left the
    backend conversation fully intact.
    """
    request = request or ClearRequest()
    if request.conversation_id:
        removed = drop_brain(request.conversation_id)
        return {"status": "cleared", "conversation_id": request.conversation_id, "existed": removed}

    count = clear_all_brains()
    return {"status": "cleared", "cleared": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)