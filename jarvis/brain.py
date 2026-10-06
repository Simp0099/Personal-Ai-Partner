"""JARVIS 2.0 Brain — provider-agnostic LLM reasoning engine.

The Brain owns the conversation and the tools. It does **not** own the model.

Flow for one user message:

    user text (verbatim)
      -> classify the request            (jarvis.classify)
      -> route to a ranked fallback chain (jarvis.router)
      -> open a provider-agnostic session (jarvis.providers)
      -> serve tool requests until the model returns text
      -> return the text

Conversation history lives here in a neutral format and is handed to whichever
model is selected. That is what lets a conversation switch models mid-thread
without losing context, and what keeps the AI Partner identity independent of
the model behind it.

Error handling is honest: failures raise :class:`BrainError` rather than being
returned as if the assistant had said them.
"""

import json
import threading
from typing import Any, Dict, List, Optional, Tuple

from jarvis.config import (
    MAX_TOOL_ROUNDS,
    MAX_HISTORY_MESSAGES,
    DEBUG_INPUT,
    JARVIS_SYSTEM_PROMPT,
    DEFAULT_CITY,
    LLM_MODEL,
)
from jarvis.tools import (
    web,
    email_tool,
    nasa,
    media,
    dictionary_tool,
    weather,
    general,
    system_tools,
)
from jarvis import memory
from jarvis.logger import logger, StatusIndicator
from jarvis.model_layer import (
    ModelLayer,
    ModelSpec,
    ProviderError,
    get_model_layer,
    set_model_layer,
)
from jarvis.providers.base import (
    ChatSession,
    ModelResponse,
    ToolCall,
    estimate_tokens,
)

# Lazy-loaded memory database connection
_memory_conn = None


class BrainError(RuntimeError):
    """Raised when no configured model can serve the request.

    Carries a short, user-safe ``message`` plus the underlying ``detail`` for
    logs. Raising instead of returning a string is what keeps transport errors
    from masquerading as assistant replies.
    """

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


def get_memory_conn():
    """Lazy-initialize the long-term memory database connection."""
    global _memory_conn
    if _memory_conn is None:
        _memory_conn = memory.init_memory_db()
    return _memory_conn


def _build_system_prompt() -> str:
    """Build the system prompt with recalled long-term memories injected.

    The same identity is passed to every compatible model, so switching models
    does not change who the AI Partner is.

    Returns:
        The base system prompt with any stored user facts appended.
    """
    facts = memory.recall_all(get_memory_conn())
    return JARVIS_SYSTEM_PROMPT + memory.format_facts_for_prompt(facts)


def _model_candidates() -> List[str]:
    """Configured model keys, in declared order.

    Kept for backwards compatibility with configuration and diagnostics; the
    router is what actually decides the order.
    """
    layer = get_model_layer()
    return [s.key for s in layer.registry.enabled()]


# ============================================================================
# Callable Tools
# ============================================================================

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a summary of a person, place, concept, or event.

    Args:
        query: The search term or topic to look up.
    """
    return general.search_wikipedia(query)


def open_website(service_or_url: str) -> str:
    """Open a web service (e.g. YouTube, Google, Gmail, Amazon) or custom URL in the browser.

    Args:
        service_or_url: The name of the service or direct URL to open.
    """
    return web.open_service(service_or_url)


def search_youtube(query: str) -> str:
    """Search YouTube for videos matching the query.

    Args:
        query: The topic or title to search on YouTube.
    """
    return web.search_youtube(query)


def search_google(query: str) -> str:
    """Search Google for web results.

    Args:
        query: The search keywords or question.
    """
    return web.search_google(query)


def open_maps(location: str = "Delhi") -> str:
    """Open Google Maps for a specific city, address, or coordinates.

    Args:
        location: City or address to view on the map.
    """
    return web.open_maps(location)


def get_temperature(city: str = "Delhi") -> str:
    """Check the current temperature and weather for a city.

    Args:
        city: Name of the city (e.g. Delhi, London, Tokyo).
    """
    return weather.get_temperature(city or DEFAULT_CITY)


def get_nasa_apod(date: str = None) -> str:
    """Fetch NASA Astronomy Picture of the Day and space facts for a given date.

    Args:
        date: Date in YYYY-MM-DD format, or None for today.
    """
    data = nasa.get_nasa_apod(date)
    if data:
        return f"Title: {data.get('title')}. Explanation: {data.get('explanation')}"
    return "Could not fetch space news."


def take_screenshot(filename: str = None) -> str:
    """Capture a screenshot of the user's screen.

    Args:
        filename: Optional name for the screenshot file.
    """
    path = media.take_screenshot(filename)
    return f"Screenshot captured and saved to {path}" if path else "Screenshot failed."


def play_music(song_name: str) -> str:
    """Play a song or music track on YouTube.

    Args:
        song_name: Name of the track or artist to play.
    """
    media.play_music(song_name)
    return f"Playing '{song_name}' on YouTube."


def lookup_dictionary(word: str) -> str:
    """Look up dictionary definitions, meanings, or synonyms for an English word.

    Args:
        word: The word to define.
    """
    return dictionary_tool.get_meaning(word)


def send_email(to_address: str, subject: str, message: str) -> str:
    """Send an email via SMTP.

    Args:
        to_address: Recipient email address.
        subject: Subject line of the email.
        message: Body content of the email.
    """
    success = email_tool.send_email(to_address, subject, message)
    return "Email sent successfully." if success else "Failed to send email. Check credentials."


def tell_time() -> str:
    """Get the current system time."""
    return general.tell_time()


def tell_joke() -> str:
    """Get a random funny programming or general joke."""
    return general.tell_joke()


# --- System Introspection Tools ---

def get_system_time() -> str:
    """Get the current system date and time.

    Returns:
        A human-readable string with today's date and the current time.
    """
    return system_tools.get_system_time()


def get_directory_contents(directory: str = ".") -> str:
    """List the files and subdirectories in a local directory.

    Args:
        directory: Path to the directory to inspect. Defaults to the current working directory.

    Returns:
        A formatted listing of directory contents with file sizes.
    """
    return system_tools.get_directory_contents(directory)


def get_system_status() -> str:
    """Check local system status including OS, CPU, memory, and disk usage.

    Returns:
        A formatted summary of key system metrics.
    """
    return system_tools.get_system_status()


# --- Memory Tools ---

def save_memory(fact: str, category: str = "general") -> str:
    """Save a fact, preference, or note to long-term memory for future sessions.

    Args:
        fact: The fact or information to remember about the user.
        category: Optional category like 'preference', 'fact', or 'note'. Defaults to 'general'.

    Returns:
        Confirmation message indicating whether the fact was saved.
    """
    try:
        conn = get_memory_conn()
        success = memory.remember(conn, fact, category)
        if success:
            count = memory.get_memory_count(conn)
            StatusIndicator.memory(f"Saved fact #{count}: '{fact}'")
            return f"Got it, Boss. I've saved that to long-term memory (fact #{count}). I won't forget."
        return "Sorry, I couldn't save that to memory. Try again."
    except Exception as e:
        logger.error(f"save_memory failed: {e}", exc_info=True)
        return "Sorry, I couldn't save that to memory. Try again."


def recall_memories(query: str = "") -> str:
    """Recall facts from long-term memory, optionally filtered by a search query.

    Args:
        query: Optional search term to filter memories. If empty, returns all stored facts.

    Returns:
        A formatted list of recalled facts, or a message if none match.
    """
    try:
        conn = get_memory_conn()

        if query:
            facts = memory.search_memories(conn, query)
            if facts:
                lines = [f"Here's what I remember about '{query}':"]
                for i, fact in enumerate(facts, 1):
                    lines.append(f"  {i}. {fact}")
                return "\n".join(lines)
            return f"I don't have anything stored about '{query}', Boss."

        facts = memory.recall_all(conn)
        if facts:
            lines = ["Here's everything I remember about you:"]
            for i, fact in enumerate(facts, 1):
                lines.append(f"  {i}. {fact}")
            return "\n".join(lines)
        return "My long-term memory is empty, Boss. Tell me something about yourself and I'll remember it."
    except Exception as e:
        logger.error(f"recall_memories failed: {e}", exc_info=True)
        return "Sorry, I had trouble accessing my memory. Try again."


GEMINI_TOOLS = [
    search_wikipedia,
    open_website,
    search_youtube,
    search_google,
    open_maps,
    get_temperature,
    get_nasa_apod,
    take_screenshot,
    play_music,
    lookup_dictionary,
    send_email,
    tell_time,
    tell_joke,
    get_system_time,
    get_directory_contents,
    get_system_status,
    save_memory,
    recall_memories,
]

#: Neutral name for the tool set; kept under the old name for compatibility.
TOOL_REGISTRY = {fn.__name__: fn for fn in GEMINI_TOOLS}

#: Tools with external side effects. These must not be executed twice for the
#: same user message -- not even with different arguments, and not even after a
#: model switch mid-turn. An email cannot be unsent and a song cannot be
#: un-played, so a repeated request reuses the first result instead.
#:
#: Read-only tools are deliberately absent: repeating `get_temperature` for two
#: different cities in one turn is legitimate.
NON_IDEMPOTENT_TOOLS = frozenset({
    "send_email",
    "play_music",
    "open_website",
    "open_maps",
    "take_screenshot",
    "save_memory",
})


# ============================================================================
# Core LLM Reasoning Loop
# ============================================================================

class JarvisBrain:
    """Stateful conversation engine that delegates model choice to the router.

    Each instance owns exactly one conversation. Conversations are isolated by
    construction: two ``JarvisBrain`` objects never share history, and the model
    serving a conversation can change freely without losing it.
    """

    def __init__(self, conversation_id: str = "default", model_layer: Optional[ModelLayer] = None):
        self.conversation_id = conversation_id
        self._layer = model_layer
        self._session: Optional[ChatSession] = None
        self._model_key: Optional[str] = None
        self._model_id: Optional[str] = None
        #: Neutral conversation history, oldest first. Owned here, not by a model.
        self._history: List[Dict[str, Any]] = []
        self._system_prompt = ""
        self._message_count = 0
        # One in-flight request per conversation, so overlapping requests cannot
        # interleave and reorder the user's messages.
        self._lock = threading.Lock()
        #: Which model actually served the last turn (diagnostics only).
        self.last_model_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    @property
    def layer(self) -> ModelLayer:
        """The shared model layer, resolved lazily."""
        if self._layer is None:
            self._layer = get_model_layer()
        return self._layer

    def _resolve_spec(self, model: Any) -> ModelSpec:
        """Accept a ModelSpec or a registry key and return the spec."""
        if isinstance(model, ModelSpec):
            return model
        spec = self.layer.spec_for(str(model))
        if spec is None:
            raise BrainError(f"Unknown model '{model}'.", detail=f"unknown model key {model!r}")
        return spec

    def _create_chat(self, model: Any, history: Optional[List[Dict[str, Any]]] = None) -> ChatSession:
        """Open a provider session for `model`, seeded with prior history."""
        spec = self._resolve_spec(model)
        provider = self.layer.providers.get(spec.provider)
        if provider is None:
            raise ProviderError(
                kind=_UNCONFIGURED_KIND,
                model_id=spec.model_id,
                detail=f"provider '{spec.provider}' is not configured",
                provider=spec.provider,
            )

        self._system_prompt = self._system_prompt or _build_system_prompt()

        session = provider.open_session(
            model_id=spec.model_id,
            system_prompt=self._system_prompt,
            tools=GEMINI_TOOLS,
            history=list(history or []),
            **self.layer.options_for(spec),
        )
        self._session = session
        self._model_key = spec.key
        self._model_id = spec.model_id
        return session

    def _get_chat(self) -> ChatSession:
        """Return the current session, opening one on the best model if needed."""
        if self._session is not None:
            return self._session

        classification = self.layer.classify(
            "", tools_available=bool(GEMINI_TOOLS)
        )
        ranked = self.layer.plan(classification, context_tokens=0)
        if not ranked:
            raise BrainError(_NO_MODEL_MESSAGE, detail="no enabled, configured models")

        return self._create_chat(ranked[0].spec)

    def reset_conversation(self):
        """Clear this conversation's history, keeping the system prompt."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass
        self._session = None
        self._history = []
        self._message_count = 0

    def _record(self, *messages: Dict[str, Any]) -> None:
        """Append neutral turns to the conversation history."""
        self._history.extend(messages)
        self._message_count += len(messages)
        cap = MAX_HISTORY_MESSAGES * 2
        if len(self._history) > cap:
            self._history = self._history[-cap:]

    def _trim_history(self) -> None:
        """Trim history to a recent window, keeping the conversation usable.

        Trimming drops the oldest turns and rebuilds the session. The system
        prompt and recent context survive, so the model still knows who it is
        and what was just discussed.
        """
        if self._message_count < MAX_HISTORY_MESSAGES:
            return

        keep = max(2, MAX_HISTORY_MESSAGES)
        window = self._history[-keep:]
        logger.info(
            f"[BRAIN] session={self.conversation_id} trimming history to "
            f"{len(window)} most recent turns"
        )
        self._session = None
        self._history = window
        self._message_count = len(window)
        if self._model_key:
            try:
                self._create_chat(self._model_key, history=window)
            except Exception as e:  # noqa: BLE001 - never fail a turn over trimming
                logger.warning(f"History trim could not rebuild session: {e}")
                self._session = None

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _execute_tool_call(self, name: str, args: dict) -> str:
        """Execute a registered tool by name with the given arguments."""
        func = TOOL_REGISTRY.get(name)
        if func is None:
            logger.error(f"Unknown tool requested: {name}")
            return f"Error: Unknown tool '{name}'."

        try:
            result = func(**args)
            return str(result) if result is not None else f"Tool '{name}' executed successfully."
        except TypeError as e:
            logger.error(f"Tool '{name}' got unexpected arguments: {e}")
            return f"Error: '{name}' received unexpected arguments ({e})."
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
            return f"Error executing '{name}': {e}"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def ask(self, user_text: str) -> str:
        """Process one user message.

        The user's text is forwarded verbatim as the latest user turn. Nothing in
        this method rewrites, truncates, or replaces it.

        Args:
            user_text: The user's input message.

        Returns:
            The model's final text response after any tool calls.

        Raises:
            ValueError: If `user_text` is empty.
            BrainError: If no configured model can serve the request.
        """
        if user_text is None or not user_text.strip():
            raise ValueError("ask() requires a non-empty user message.")

        message = user_text.strip()

        if DEBUG_INPUT:
            logger.info(f"[INPUT] session={self.conversation_id} message={message!r}")

        # Serialize turns so rapid messages cannot interleave or reorder.
        with self._lock:
            return self._ask_locked(message)

    def _ask_locked(self, message: str) -> str:
        StatusIndicator.thinking()

        self._trim_history()

        classification = self.layer.classify(
            message,
            conversation_tokens=estimate_tokens(self._history),
            tools_available=bool(GEMINI_TOOLS),
        )
        context_tokens = estimate_tokens(self._history) + estimate_tokens([{"content": message}])

        ranked = self.layer.plan(classification, context_tokens=context_tokens)

        if DEBUG_INPUT:
            logger.info(
                f"[ROUTING] session={self.conversation_id} task={classification.task_type.value} "
                f"tool_required={classification.tool_required} signals={classification.signals} "
                f"candidates={[c.key for c in ranked] or 'none'}"
            )

        if not ranked:
            raise BrainError(
                _NO_MODEL_MESSAGE,
                detail=(
                    "no eligible model. "
                    f"task={classification.task_type.value} "
                    f"tool_required={classification.tool_required} "
                    f"enabled={[s.key for s in self.layer.registry.enabled()]}"
                ),
            )

        attempts = ranked[: max(1, self.layer.router.config.max_attempts)]
        failures: List[str] = []
        executed_this_turn: Dict[str, str] = {}
        turn_messages: List[Dict[str, Any]] = []

        for attempt, candidate in enumerate(attempts, 1):
            spec = candidate.spec
            provider = self.layer.providers.get(spec.provider)
            if provider is None or not provider.is_configured():
                failures.append(f"{spec.key}: provider not configured")
                continue

            started = _now()

            try:
                session = self._create_chat(spec, history=self._history)
                response = session.send_message({"kind": "user", "text": message})
            except ProviderError as e:
                latency = _now() - started
                self.layer.record_failure(spec, e, latency)
                failures.append(f"{spec.key}: {e.kind.value} ({e.safe_detail})")
                if not e.retryable:
                    # Auth / bad request: retrying other models cannot help, and
                    # hiding a misconfiguration behind retries is how outages get
                    # misdiagnosed. Surface it immediately.
                    logger.error(f"[ROUTER] fatal error from {spec.key}: {e.safe_detail}")
                    raise BrainError(_friendly_error_message(e), detail="; ".join(failures)) from e
                self._reset_after_failure()
                continue

            latency = _now() - started
            self.layer.record_success(spec, latency)
            self.last_model_key = spec.key

            if DEBUG_INPUT:
                logger.info(f"[MODEL OUTPUT] session={self.conversation_id} model={spec.key} "
                            f"latency={latency:.2f}s")

            try:
                reply = self._run_tool_loop(
                    session=session,
                    spec=spec,
                    response=response,
                    turn_messages=turn_messages,
                    executed_this_turn=executed_this_turn,
                )
            except ProviderError as e:
                self.layer.record_failure(spec, e)
                failures.append(f"{spec.key}: {e.kind.value} ({e.safe_detail})")
                self._reset_after_failure()
                if not e.retryable:
                    raise BrainError(_friendly_error_message(e), detail="; ".join(failures)) from e
                continue

            # Commit the turn: user message plus everything the model produced.
            self._record({"role": "user", "content": message}, *turn_messages)
            return reply

        raise BrainError(_friendly_error_message_from_failures(failures), detail="; ".join(failures))

    def _reset_after_failure(self) -> None:
        """Drop the failed session so the next candidate starts clean."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass
        self._session = None
        self._model_key = None

    def _run_tool_loop(
        self,
        *,
        session: ChatSession,
        spec: ModelSpec,
        response: ModelResponse,
        turn_messages: List[Dict[str, Any]],
        executed_this_turn: Dict[str, str],
    ) -> str:
        """Serve tool requests until the model returns text.

        `turn_messages` accumulates the assistant/tool turns produced so far, so
        they are committed to history even if the turn ultimately fails -- the
        conversation must not lose the fact that a tool ran.
        """
        for round_index in range(MAX_TOOL_ROUNDS):
            if not response.has_tool_calls:
                break

            if round_index == MAX_TOOL_ROUNDS - 1:
                logger.warning(
                    f"[BRAIN] session={self.conversation_id} hit tool round limit "
                    f"({MAX_TOOL_ROUNDS}); answering without further tools"
                )
                break

            results: List[Dict[str, Any]] = []
            assistant_calls: List[Dict[str, Any]] = []

            for call in response.tool_calls:
                tool_name = call.name
                tool_args = dict(call.arguments or {})

                StatusIndicator.tool_call(tool_name, tool_args)

                # Duplicate-execution guard. After a model switch mid-turn, or a
                # retry, the model may re-request a tool it already got a
                # result for -- possibly with different or empty arguments.
                # For side-effecting tools that would duplicate a real-world
                # action, so the cached result is reused instead.
                cache_key = _tool_cache_key(tool_name, tool_args)
                if cache_key in executed_this_turn:
                    result = executed_this_turn[cache_key]
                    logger.info(
                        f"[BRAIN] reusing result for already-executed tool {tool_name}"
                    )
                else:
                    result = self._execute_tool_call(tool_name, tool_args)
                    executed_this_turn[cache_key] = result

                StatusIndicator.tool_result(tool_name, result)

                assistant_calls.append({
                    "id": call.id, "name": tool_name, "arguments": tool_args,
                })
                results.append({"id": call.id, "name": tool_name, "result": result})

            turn_messages.append({
                "role": "assistant",
                "content": response.text or "",
                "tool_calls": assistant_calls,
            })
            turn_messages.append({"role": "tool", "results": results})

            response = session.send_message({"kind": "tool_results", "results": results})

        reply = (response.text or "").strip() or "Standing by, Boss."
        turn_messages.append({"role": "assistant", "content": reply})
        return reply


# ============================================================================
# Helpers
# ============================================================================

def _tool_cache_key(name: str, args: Dict[str, Any]) -> str:
    """Stable key identifying a tool invocation within one user turn.

    Side-effecting tools collapse to the tool name alone: within a single user
    message the same side effect must happen at most once, regardless of the
    arguments a re-issuing model chooses. Read-only tools keep their arguments
    in the key, so genuinely distinct calls (weather in two cities) still run.
    """
    if name in NON_IDEMPOTENT_TOOLS:
        return name
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
    except (TypeError, ValueError):
        return f"{name}:{sorted(args.items(), key=lambda kv: str(kv[0]))}"


def _now() -> float:
    import time
    return time.time()


def _friendly_error_message(exc: ProviderError) -> str:
    """Short, honest, user-facing text for a provider failure."""
    kind = getattr(exc, "kind", None)
    value = getattr(kind, "value", str(kind or ""))
    if value == "rate_limit":
        return "The AI provider's usage limit has been reached. Please try again later."
    if value == "server":
        return "The AI model is temporarily unavailable. Please try again shortly."
    if value == "timeout":
        return "The AI model took too long to respond. Please try again."
    if value == "network":
        return "Cannot reach the AI provider. Please check your connection."
    if value == "auth":
        return "The AI provider rejected the configured API key. Please check your .env file."
    if value == "invalid_request":
        return "The request was rejected by the AI provider."
    if value == "unsupported":
        return "The selected AI model does not support this request."
    return "I could not reach any available AI model just now. Please try again."


def _friendly_error_message_from_failures(failures: List[str]) -> str:
    """Summarise why every candidate failed."""
    kinds = set()
    for failure in failures:
        if ":" in failure:
            kinds.add(failure.split(":", 1)[1].strip().split(" ")[0])
    if "rate_limit" in kinds:
        return "The AI provider's usage limit has been reached. Please try again later."
    if "server" in kinds:
        return "All available AI models are temporarily unavailable. Please try again shortly."
    if "timeout" in kinds:
        return "The AI model took too long to respond. Please try again."
    return "I could not reach any available AI model just now. Please try again."


_NO_MODEL_MESSAGE = (
    "No AI model is currently available to handle that request. Please try again shortly."
)


class _UnconfiguredKind:
    """Marker kind for 'provider has no credentials'."""

    value = "invalid_request"
    name = "INVALID_REQUEST"


_UNCONFIGURED_KIND = _UnconfiguredKind()

# Kept for callers that imported the old helper.
_response_text = None