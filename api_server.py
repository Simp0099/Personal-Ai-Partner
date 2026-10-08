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
import base64
import mimetypes
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import JARVIS brain
from jarvis.brain import JarvisBrain, BrainError
from jarvis.config import DEBUG_ENDPOINTS
from jarvis.logger import logger
from jarvis.providers.base import image_part

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
    images: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional images, each either a 'data:<mime>;base64,...' URL or a "
            "server-local file path. Images are forwarded to a vision-capable "
            "model for this turn and are not persisted."
        ),
    )


class ChatResponse(BaseModel):
    response: str
    latency: int
    error: bool = False
    conversation_id: str = "default"
    #: Conversational state at the time the reply was produced. Lets the client
    #: tell a live reply from one that arrived after the user interrupted.
    state: Optional[str] = None
    turn_id: Optional[str] = None


class VoiceControl(BaseModel):
    """Enable or disable voice input."""

    enabled: bool = True


class InterruptRequest(BaseModel):
    """Barge-in request. Carries the turn being abandoned, if the client knows it."""

    turn_id: Optional[str] = None


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


def _decode_images(raw: Optional[List[str]]) -> List[dict]:
    """Turn request image references into provider-neutral image parts.

    Accepts data URLs and server-local paths. Decoding happens here so the Brain
    never deals with transport formats, and the resulting bytes are held only
    for the turn -- nothing about an image is written to disk or to memory.
    """
    parts = []
    for item in raw or []:
        try:
            if item.startswith("data:"):
                header, _, payload = item.partition(",")
                mime = header[5:].split(";")[0] or "image/png"
                parts.append(image_part(base64.b64decode(payload), mime))
            else:
                path = Path(item).expanduser()
                mime = mimetypes.guess_type(str(path))[0] or "image/png"
                parts.append(image_part(path.read_bytes(), mime))
        except (ValueError, OSError) as e:
            # A bad reference is a client error, not a reason to answer as if
            # the image had been seen.
            raise ValueError(f"Could not read image '{item[:40]}': {e}") from e
    return parts


@app.get("/api/vision")
async def vision_status():
    """Vision and webcam-perception status.

    Read-only on purpose. It reports whether the camera is *enabled by config*,
    *actually open*, and what it has seen so far — so the state of the camera is
    never a guess. It contains no image data, no frames and no credentials, and
    it will never start the camera: opening a device is a user decision made in
    config.yaml.
    """
    from jarvis.config import VISION_ENABLED, WEBCAM_ENABLED
    from jarvis.webcam import get_webcam_perception
    from jarvis.vision import get_visual_context

    perception = get_webcam_perception()
    return {
        "vision_enabled": VISION_ENABLED,
        "webcam_enabled_by_config": WEBCAM_ENABLED,
        "webcam": perception.status() if perception else {
            "enabled": False,
            "running": False,
            "camera_available": False,
            "note": "Webcam perception is disabled; no camera has been opened.",
        },
        "visual_context": get_visual_context().status(),
    }


@app.get("/api/state")
async def conversation_state():
    """Authoritative conversational state.

    The HUD renders this and nothing else. It deliberately does not let the
    frontend infer state from timing or from whether a reply arrived: a client
    that guesses "speaking" when a reply is slow ends up talking over the
    assistant, which is precisely what Phase 4 exists to prevent.

    Read-only. Nothing here starts the microphone.
    """
    from jarvis.tone import get_tone
    from jarvis.vision import get_visual_context
    from jarvis.webcam import get_webcam_perception
    from jarvis.voice_loop import get_voice_loop

    loop = get_voice_loop()
    machine = loop.machine if loop is not None else _state_machine()

    # Proactive state (Phase 6): ephemeral continuity only. No reasoning,
    # no frames, no memory contents — just whether proactive speech is
    # currently allowed and why the last candidate was held back.
    try:
        from jarvis.config import PROACTIVE_ENABLED
        from jarvis.proactive import get_orchestrator
        _proactive_status = get_orchestrator().status()
        _proactive_status["enabled"] = bool(PROACTIVE_ENABLED)
    except Exception:  # noqa: BLE001 - diagnostics must never break /api/state
        _proactive_status = {"enabled": False, "error": "unavailable"}

    return {
        **machine.snapshot(),
        "voice": loop.status() if loop is not None else {
            "running": False,
            "wake_enabled": False,
            "note": "Voice input is not running; text conversation is unaffected.",
        },
        "webcam_running": bool(get_webcam_perception() and get_webcam_perception().is_running()),
        # Application state, not a claim about inner experience. No reasoning,
        # no prompts, nothing the model was told.
        "conversation_state": get_tone().status(),
        "visual_context": get_visual_context().status(),
        "proactive": _proactive_status,
    }


_shared_machine_lock = threading.Lock()
_shared_machine = None


def _state_machine():
    """The machine the API reports, whether or not voice is running.

    Text conversation still moves the state machine, so the HUD shows a coherent
    picture even with the microphone switched off.
    """
    global _shared_machine
    with _shared_machine_lock:
        if _shared_machine is None:
            from jarvis.conversation import ConversationMachine
            _shared_machine = ConversationMachine()
        return _shared_machine


@app.post("/api/voice/start")
async def voice_start(request: VoiceControl):
    """Start or stop voice input. Reports honestly when audio is unavailable."""
    from jarvis.voice_loop import get_voice_loop, start_voice_loop, stop_voice_loop

    if not request.enabled:
        stop_voice_loop()
        return {"status": "stopped", "voice": get_voice_loop()}

    brain = get_brain("default")
    loop = start_voice_loop(brain, force=True)
    if loop is None:
        # start_voice_loop only returns None on failure when forced, so this is a
        # genuine "could not start" and must say so rather than looking idle.
        previous = get_voice_loop()
        return {
            "status": "unavailable",
            "error": True,
            "detail": (previous.last_error if previous else "voice input unavailable"),
            "note": "Text conversation still works.",
        }
    return {"status": "running", "voice": loop.status()}


@app.post("/api/voice/interrupt")
async def voice_interrupt(request: Optional[InterruptRequest] = None):
    """Stop the assistant speaking now and start listening to the user."""
    from jarvis.voice_loop import get_voice_loop

    loop = get_voice_loop()
    if loop is None or not loop.is_running():
        return {"status": "idle", "interrupted": False,
                "note": "Voice input is not running."}
    loop.interrupt()
    return {
        "status": loop.machine.state.value,
        "interrupted": True,
        "turn_id": loop.machine.turn.id if loop.machine.turn else None,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message, optionally with images, through the JARVIS brain.

    The message is forwarded to the brain exactly as received. Blank input is
    rejected with 400 unless images are attached, since a picture alone is a
    valid turn.
    """
    start = time.time()

    has_text = request.message is not None and request.message.strip()
    if not has_text and not request.images:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        images = _decode_images(request.images)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    conversation_id = request.conversation_id or "default"
    brain = get_brain(conversation_id)

    # Typing during a spoken reply is also an interruption. The user does not
    # have to wait for a voice command to get the floor back.
    from jarvis.voice_loop import get_voice_loop
    loop = get_voice_loop()
    machine = loop.machine if loop is not None else _state_machine()
    if loop is not None and loop.is_running() and machine.state.value in ("speaking", "thinking"):
        loop.interrupt()

    turn = machine.begin_turn(request.message or "")
    machine.mark("first_token")

    try:
        # brain.ask is blocking (network + tools); keep the event loop free.
        response = await asyncio.to_thread(
            brain.ask, (request.message or "").strip(), images
        )
        latency = int((time.time() - start) * 1000)
        machine.end_turn(turn.id)
        return ChatResponse(
            response=response,
            latency=latency,
            error=False,
            conversation_id=conversation_id,
            state=machine.state.value,
            turn_id=turn.id,
        )
    except ValueError as e:
        # Blank/invalid input — a client error, not a backend failure.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except BrainError as e:
        # Genuine backend failure. Report it as a failure; do not present the
        # error text as if the assistant had said it.
        logger.error(f"Brain error (session={conversation_id}): {e.detail or e}")
        latency = int((time.time() - start) * 1000)
        machine.end_turn(turn.id)
        return ChatResponse(
            response=e.message,
            latency=latency,
            error=True,
            conversation_id=conversation_id,
            state=machine.state.value,
            turn_id=turn.id,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Chat error (session={conversation_id}): {e}", exc_info=True)
        latency = int((time.time() - start) * 1000)
        machine.end_turn(turn.id)
        return ChatResponse(
            response="The assistant hit an unexpected internal error.",
            latency=latency,
            error=True,
            conversation_id=conversation_id,
            state=machine.state.value,
            turn_id=turn.id,
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