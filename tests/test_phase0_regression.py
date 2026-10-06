"""Phase 0 regression tests: the user's message must reach the model intact.

Every test here corresponds to a confirmed bug recorded in PHASE0_DIAGNOSIS.md.
The tests use a fake Gemini chat so they run offline and deterministically:
the fake records exactly what was handed to the model, which is what lets us
assert that the user's text was not rewritten, reordered, or dropped.

Run with:
    python3 -m pytest tests/ -v
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from google.genai import types  # noqa: E402

from jarvis.brain import (  # noqa: E402
    JarvisBrain,
    BrainError,
    TOOL_REGISTRY,
    GEMINI_TOOLS,
)
from jarvis.providers.base import ModelResponse, ToolCall, new_tool_call_id  # noqa: E402
import jarvis.brain as brain_module  # noqa: E402
import jarvis.main as main_module  # noqa: E402


# ============================================================================
# Test doubles
# ============================================================================

def _make_response(text=None, function_calls=None):
    """Build a provider-neutral model response.

    Phase 0 asserted on Gemini wire-format objects. Since Phase 0.5 made the
    Brain provider-agnostic, these tests assert on the neutral format; the
    Gemini wire-format assertions now live in tests/test_providers.py.
    """
    calls = [
        ToolCall(id=new_tool_call_id(), name=fc[0], arguments=fc[1])
        for fc in (function_calls or [])
    ]
    return ModelResponse(text=text, tool_calls=calls)


def _fc(name, args=None):
    """Shorthand for a tool call in a scripted response."""
    return (name, args or {})


class RecordingChat:
    """Fake provider session that records every send_message payload.

    `script` is a list of responses returned in order; the last one repeats.
    """

    def __init__(self, script):
        self.script = list(script)
        self.sent = []

    def send_message(self, payload):
        self.sent.append(payload)
        idx = min(len(self.sent) - 1, len(self.script) - 1)
        return self.script[idx]

    def close(self):
        pass


def brain_with(chat, conversation_id="test", model_layer=None):
    """A JarvisBrain whose router is stubbed to always use `chat`.

    This keeps Phase 0's behavioural assertions (verbatim input, ordering, tool
    chaining, isolation) independent of which provider is configured.
    """
    brain = JarvisBrain(conversation_id=conversation_id, model_layer=model_layer)
    brain._create_chat = lambda model, history=None: chat
    brain._system_prompt = "SYSTEM"
    brain._session = chat
    brain._model_key = "fake"
    brain._model_id = "fake-model"
    return brain


@pytest.fixture(autouse=True)
def _silence_status(monkeypatch):
    """Keep CLI status output out of the test run."""
    monkeypatch.setattr(brain_module.StatusIndicator, "thinking", staticmethod(lambda: None))
    monkeypatch.setattr(brain_module.StatusIndicator, "tool_call", staticmethod(lambda n, a: None))
    monkeypatch.setattr(brain_module.StatusIndicator, "tool_result", staticmethod(lambda n, r: None))


def _sent_user_texts(chat):
    """Extract the user's own messages from recorded payloads."""
    out = []
    for payload in chat.sent:
        if isinstance(payload, str):
            out.append(payload)
        elif isinstance(payload, dict) and payload.get("kind") == "user":
            out.append(payload["text"])
    return out


def _sent_tool_results(chat):
    """Extract tool-result payloads from recorded payloads."""
    return [
        p for p in chat.sent
        if isinstance(p, dict) and p.get("kind") == "tool_results"
    ]


class _Sent:
    """Adapts a raw payload list to the `.sent` shape the extractors expect."""

    def __init__(self, payloads):
        self.sent = list(payloads)


@pytest.fixture()
def client():
    """A TestClient against the real FastAPI app with a clean session registry."""
    from fastapi.testclient import TestClient
    import api_server
    api_server.clear_all_brains()
    yield TestClient(api_server.app)
    api_server.clear_all_brains()


# ============================================================================
# B1 — model availability / fallback
# ============================================================================

class TestModelFallback:
    """B1: an unavailable model must not fail every user message.

    Phase 0.5 moved model selection into the router, so these are expressed
    against the model layer rather than the Gemini SDK. Detailed routing
    behaviour is covered in tests/test_routing.py.
    """

    def test_falls_back_when_primary_model_unavailable(self):
        from tests.mock_providers import build_layer, error, reply, empty
        from jarvis.providers.base import ProviderError, ErrorKind

        layer = build_layer(
            models=[
                {"key": "primary", "provider": "p", "model": "m1", "priority": 90,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "backup", "provider": "p", "model": "m2", "priority": 50,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={
                "m1": error(ErrorKind.SERVER, "503 UNAVAILABLE"),
                "m2": reply("Paris"),
            },
        )
        brain = JarvisBrain(conversation_id="fallback", model_layer=layer)
        assert brain.ask("What is the capital of France?") == "Paris"
        assert layer.health.stats("primary").failure_count == 1

    def test_raises_brain_error_when_all_models_fail(self):
        from tests.mock_providers import build_layer, error, reply, empty
        from jarvis.providers.base import ProviderError, ErrorKind

        layer = build_layer(
            models=[
                {"key": "a", "provider": "p", "model": "m1", "capabilities": {"reasoning": True}},
                {"key": "b", "provider": "p", "model": "m2", "capabilities": {"reasoning": True}},
            ],
            behaviours={"m1": error(ErrorKind.SERVER, "503"), "m2": error(ErrorKind.SERVER, "503")},
        )
        brain = JarvisBrain(conversation_id="dead", model_layer=layer)
        with pytest.raises(BrainError) as exc:
            brain.ask("hello")
        # The failure must be an honest error, not fabricated assistant text.
        assert "unavailable" in exc.value.message.lower()
        assert "503" in exc.value.detail

    def test_error_messages_do_not_leak_secrets(self):
        from jarvis.providers.base import ProviderError, ErrorKind

        for kind, secret in (
            (ErrorKind.AUTH, "sk-secret-value"),
            (ErrorKind.RATE_LIMIT, "sk-or-v1-abcdef"),
            (ErrorKind.SERVER, "Bearer sometoken"),
            (ErrorKind.INVALID_REQUEST, "apikey=zzz"),
        ):
            msg = brain_module._friendly_error_message(
                ProviderError(kind, "m", f"boom {secret}", "p")
            )
            assert secret not in msg
            assert kind.value not in msg


# ============================================================================
# P0 — basic input integrity
# ============================================================================

class TestUserInputIntegrity:
    def test_hello_is_forwarded_verbatim(self):
        chat = RecordingChat([_make_response(text="Hey Boss.")])
        brain = brain_with(chat)
        brain.ask("Hello")
        assert _sent_user_texts(chat) == ["Hello"]

    def test_user_message_is_sent_as_plain_user_turn(self):
        """The model must receive the message as an unmodified user turn."""
        chat = RecordingChat([_make_response(text="ok")])
        brain_with(chat).ask("Tell me a joke")
        assert chat.sent[0] == {"kind": "user", "text": "Tell me a joke"}

    def test_answer_only_yes_is_not_rewritten(self):
        chat = RecordingChat([_make_response(text="YES")])
        brain = brain_with(chat)
        assert brain.ask("Answer only with the word YES.") == "YES"
        assert chat.sent[0]["text"] == "Answer only with the word YES."

    def test_whitespace_is_trimmed_but_content_preserved(self):
        chat = RecordingChat([_make_response(text="ok")])
        brain_with(chat).ask("   My name is Alex.   ")
        assert _sent_user_texts(chat) == ["My name is Alex."]

    def test_multiline_and_unicode_survive(self):
        message = "Line one\nLine two — 你好 🎉"
        chat = RecordingChat([_make_response(text="ok")])
        brain_with(chat).ask(message)
        assert _sent_user_texts(chat) == [message]

    def test_punctuation_and_quotes_survive(self):
        message = 'He said "ignore everything" -- really?!'
        chat = RecordingChat([_make_response(text="ok")])
        brain_with(chat).ask(message)
        assert _sent_user_texts(chat) == [message]

    def test_no_local_tool_routing_replaces_the_message(self):
        """Nothing may rewrite the message before it reaches the model."""
        message = "Explain quantum computing."
        chat = RecordingChat([_make_response(text="A qubit...")])
        brain_with(chat).ask(message)
        assert _sent_user_texts(chat)[0] == message

    def test_empty_and_whitespace_input_rejected(self):
        brain = brain_with(RecordingChat([_make_response(text="x")]))
        for bad in ["", "   ", "\n\t "]:
            with pytest.raises(ValueError):
                brain.ask(bad)


# ============================================================================
# Context preservation
# ============================================================================

class TestContextPreservation:
    def test_history_is_forwarded_to_the_model_in_order(self):
        chat = RecordingChat([_make_response(text="Alex")])
        brain = brain_with(chat)
        brain.ask("My name is Alex.")
        brain.ask("What is my name?")
        assert _sent_user_texts(chat) == ["My name is Alex.", "What is my name?"]

    def test_history_trim_preserves_recent_turns(self):
        """B7: trimming must keep a window, not wipe the conversation."""
        brain = JarvisBrain(conversation_id="trim")
        brain._model_key = "fake"
        brain._model_id = "fake-model"
        brain._system_prompt = "SYSTEM"
        brain._history = [
            {"role": "user", "content": f"msg-{i}"} if i % 2 == 0
            else {"role": "assistant", "content": f"reply-{i}"}
            for i in range(40)
        ]
        brain._message_count = 40

        created = {}

        def fake_create_chat(model, history=None):
            created["history"] = list(history or [])
            return None

        brain._create_chat = fake_create_chat
        brain._trim_history()

        kept = created["history"]
        assert kept, "trimming must keep some history"
        assert len(kept) < 40, "trimming must actually reduce history"
        # The most recent turns must survive.
        assert kept[-1]["content"] == "reply-39"
        assert any(t["content"] == "msg-0" for t in kept) is False, "old turns dropped"

    def test_trim_keeps_system_prompt(self):
        brain = JarvisBrain(conversation_id="trim2")
        brain._model_key = "fake"
        brain._model_id = "fake-model"
        brain._system_prompt = "IMPORTANT SYSTEM"
        brain._history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        brain._message_count = 999

        captured = {}

        def fake_create_chat(model, history=None):
            captured["system"] = brain._system_prompt
            return None

        brain._create_chat = fake_create_chat
        brain._trim_history()
        assert captured["system"] == "IMPORTANT SYSTEM"


# ============================================================================
# New instruction overriding an old task
# ============================================================================

class TestNewInstructionWins:
    """A second request must not be swallowed by an earlier one."""

    def test_second_request_is_sent_verbatim_after_a_tool_chain(self):
        chat = RecordingChat([
            _make_response(function_calls=[_fc("tell_time")]),
            _make_response(text="It is noon."),
            _make_response(text="Photosynthesis converts light into sugar."),
        ])
        brain = brain_with(chat)

        with patch.dict(TOOL_REGISTRY, {"tell_time": lambda: "12:00"}):
            brain.ask("What time is it?")
            brain.ask("Stop. Instead, explain photosynthesis.")

        assert _sent_user_texts(chat) == [
            "What time is it?",
            "Stop. Instead, explain photosynthesis.",
        ]

    def test_shutdown_keyword_does_not_discard_real_requests(self):
        """B6: substring matching used to terminate on ordinary sentences."""
        assert main_module.is_shutdown_request("exit") is True
        assert main_module.is_shutdown_request("  Quit.  ") is True

        for real_request in [
            "Explain the exit code in Python.",
            "Quite interesting, tell me about recursion.",
            "What is a shutdown sequence in Linux?",
            "How do I quit Vim and save?",
            "Tell me about the go to sleep function in Python.",
        ]:
            assert main_module.is_shutdown_request(real_request) is False, real_request

    def test_conversation_stays_responsive_after_a_long_tool_chain(self):
        chat = RecordingChat([
            _make_response(function_calls=[_fc("tell_time")]),
            _make_response(text="done"),
        ])
        brain = brain_with(chat)
        with patch.dict(TOOL_REGISTRY, {"tell_time": lambda: "12:00"}):
            brain.ask("What time is it?")
            brain.ask("Hello again")
        assert _sent_user_texts(chat)[-1] == "Hello again"


# ============================================================================
# B4 — tool call chaining
# ============================================================================

class TestToolChain:
    def test_chained_tool_calls_are_all_executed(self):
        """B4: only the first round of tool calls used to be honoured."""
        called = []

        chat = RecordingChat([
            _make_response(function_calls=[_fc("tell_time")]),
            _make_response(function_calls=[_fc("tell_joke")]),
            _make_response(text="All done."),
        ])
        brain = brain_with(chat)

        def fake_tool(name):
            def _call():
                called.append(name)
                return f"{name} result"
            return _call

        with patch.dict(TOOL_REGISTRY, {"tell_time": fake_tool("tell_time"),
                                       "tell_joke": fake_tool("tell_joke")}):
            reply = brain.ask("do both")

        assert called == ["tell_time", "tell_joke"], "second tool call was dropped"
        assert reply == "All done."

    def test_previous_bug_returns_non_answer(self):
        """Guards the exact regression: chain must not yield the idle fallback."""
        chat = RecordingChat([
            _make_response(function_calls=[_fc("tell_time")]),
            _make_response(function_calls=[_fc("tell_joke")]),
            _make_response(text="Real answer."),
        ])
        brain = brain_with(chat)
        with patch.dict(TOOL_REGISTRY, {"tell_time": lambda: "a", "tell_joke": lambda: "b"}):
            reply = brain.ask("x")
        assert reply != "Standing by, Boss."

    def test_parallel_tool_calls_are_all_answered_in_one_message(self):
        chat = RecordingChat([
            _make_response(function_calls=[
                _fc("tell_time"),
                _fc("tell_joke"),
            ]),
            _make_response(text="done"),
        ])
        brain = brain_with(chat)
        with patch.dict(TOOL_REGISTRY, {"tell_time": lambda: "a", "tell_joke": lambda: "b"}):
            brain.ask("both")

        tool_turns = _sent_tool_results(chat)
        assert len(tool_turns) == 1, "parallel results must be sent as one message"
        assert len(tool_turns[0]["results"]) == 2

    def test_tool_loop_is_bounded(self):
        """A model that keeps requesting tools must not loop forever."""
        brain_module.MAX_TOOL_ROUNDS = 3
        try:
            always_tool = _make_response(
                function_calls=[_fc("tell_time")]
            )
            chat = RecordingChat([always_tool])
            brain = brain_with(chat)
            with patch.dict(TOOL_REGISTRY, {"tell_time": lambda: "a"}):
                reply = brain.ask("loop")
            assert isinstance(reply, str)
            assert len(chat.sent) <= 6
        finally:
            brain_module.MAX_TOOL_ROUNDS = 8

    def test_tool_failure_is_reported_to_model_not_crashing(self):
        def boom():
            raise RuntimeError("tool exploded")

        chat = RecordingChat([
            _make_response(function_calls=[_fc("tell_time")]),
            _make_response(text="Sorry about that."),
        ])
        brain = brain_with(chat)
        with patch.dict(TOOL_REGISTRY, {"tell_time": boom}):
            reply = brain.ask("x")
        # Phase 2 appends an honest-failure note so a failed action cannot read
        # as a completed one. The model's own text is still preserved.
        assert reply.startswith("Sorry about that.")
        assert "Error executing 'tell_time'" in reply
        tool_turn = _sent_tool_results(chat)[0]
        assert "Error" in tool_turn["results"][0]["result"]

    def test_unknown_tool_returns_error_to_model(self):
        chat = RecordingChat([
            _make_response(function_calls=[_fc("nope")]),
            _make_response(text="ok"),
        ])
        brain = brain_with(chat)
        brain.ask("x")
        tool_turn = _sent_tool_results(chat)[0]
        assert "Unknown tool" in tool_turn["results"][0]["result"]

    def test_bad_arguments_do_not_crash(self):
        chat = RecordingChat([
            _make_response(function_calls=[
                _fc("get_temperature", {"bogus_arg": 1})
            ]),
            _make_response(text="ok"),
        ])
        brain = brain_with(chat)
        reply = brain.ask("x")
        assert isinstance(reply, str)

    def test_duplicate_tool_call_is_not_executed_twice(self):
        """Non-idempotent tools must not be replayed within one turn."""
        calls = []

        chat = RecordingChat([
            _make_response(function_calls=[_fc("send_email", {
                "to_address": "a@b.com", "subject": "s", "message": "m"
            })]),
            _make_response(function_calls=[_fc("send_email", {
                "to_address": "a@b.com", "subject": "s", "message": "m"
            })]),
            _make_response(text="Sent."),
        ])

        def fake_send(**kwargs):
            calls.append(kwargs)
            return "Email sent successfully."

        brain = brain_with(chat)
        with patch.dict(TOOL_REGISTRY, {"send_email": fake_send}):
            reply = brain.ask("email a@b.com")

        assert reply == "Sent."
        assert len(calls) == 1, "duplicate tool side effect executed more than once"


# ============================================================================
# B5 — failures must not masquerade as replies
# ============================================================================

class TestHonestFailures:
    def test_model_exception_raises_brain_error(self):
        from tests.mock_providers import build_layer, error, reply, empty
        from jarvis.providers.base import ProviderError, ErrorKind

        layer = build_layer(
            models=[{"key": "only", "provider": "p", "model": "m",
                     "capabilities": {"reasoning": True}}],
            behaviours={"m": error(ErrorKind.SERVER, "503 UNAVAILABLE")},
        )
        brain = JarvisBrain(conversation_id="dead", model_layer=layer)
        with pytest.raises(BrainError) as exc:
            brain.ask("hello")
        assert "unavailable" in exc.value.message.lower()

    def test_api_error_text_is_not_returned_as_a_reply(self):
        from tests.mock_providers import build_layer, error, reply, empty
        from jarvis.providers.base import ProviderError, ErrorKind

        layer = build_layer(
            models=[{"key": "only", "provider": "p", "model": "m",
                     "capabilities": {"reasoning": True}}],
            behaviours={"m": error(ErrorKind.AUTH, "API_KEY_INVALID sk-abcdef123456")},
        )
        brain = JarvisBrain(conversation_id="dead", model_layer=layer)

        with pytest.raises(BrainError) as exc:
            brain.ask("hello")
        # The key must never appear in what the user is shown, nor in the
        # detail kept for logs.
        assert "sk-abcdef123456" not in exc.value.message
        assert "sk-abcdef123456" not in exc.value.detail

    def test_empty_response_becomes_idle_fallback(self):
        """An empty model reply is reported honestly, not invented around."""
        from tests.mock_providers import build_layer, error, reply, empty

        layer = build_layer(
            models=[{"key": "only", "provider": "p", "model": "m",
                     "capabilities": {"reasoning": True}}],
            behaviours={"m": empty()},
        )
        brain = JarvisBrain(conversation_id="empty", model_layer=layer)
        assert brain.ask("hello") == "Standing by, Boss."


# ============================================================================
# Conversation isolation
# ============================================================================

class TestConversationIsolation:
    def test_separate_brains_do_not_share_history(self):
        chat_a = RecordingChat([_make_response(text="A reply")])
        chat_b = RecordingChat([_make_response(text="B reply")])

        a = brain_with(chat_a, conversation_id="A")
        b = brain_with(chat_b, conversation_id="B")

        a.ask("My name is Alex")
        b.ask("What is my name?")

        assert _sent_user_texts(chat_a) == ["My name is Alex"]
        assert _sent_user_texts(chat_b) == ["What is my name?"]

    def test_reset_clears_only_its_own_conversation(self):
        chat_a = RecordingChat([_make_response(text="a")])
        chat_b = RecordingChat([_make_response(text="b")])
        a = brain_with(chat_a, conversation_id="A")
        b = brain_with(chat_b, conversation_id="B")

        a.ask("secret about A")
        a.reset_conversation()
        b.ask("unrelated")

        assert _sent_user_texts(chat_b) == ["unrelated"]
        assert a._history == []
        assert b._history

    def test_new_conversation_starts_clean(self):
        chat = RecordingChat([_make_response(text="ok")])
        brain = brain_with(chat)
        brain.ask("first")
        fresh = JarvisBrain(conversation_id="fresh")
        assert fresh._history == []
        assert fresh._session is None


# ============================================================================
# Rapid consecutive messages
# ============================================================================

class TestRapidMessages:
    def test_concurrent_asks_do_not_interleave(self):
        """The per-conversation lock must serialize turns."""
        import time

        order = []

        class SlowChat:
            def __init__(self):
                self.sent = []

            def send_message(self, payload):
                self.sent.append(payload)
                time.sleep(0.01)
                order.append(payload["text"])
                return _make_response(text="ok")

            def close(self):
                pass

        chat = SlowChat()
        brain = brain_with(chat, conversation_id="race")

        threads = [
            threading.Thread(target=brain.ask, args=(f"msg-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(order) == 5
        assert len(set(order)) == 5, "a message was lost or duplicated"
        assert _sent_user_texts(chat) == order, "messages were reordered"

    def test_history_reflects_every_message_exactly_once(self):
        chat = RecordingChat([_make_response(text="ok")])
        brain = brain_with(chat, conversation_id="seq")
        for i in range(5):
            brain.ask(f"msg-{i}")

        users = [m["content"] for m in brain._history if m["role"] == "user"]
        assert users == [f"msg-{i}" for i in range(5)]


# ============================================================================
# B8 — tools must not block the server
# ============================================================================

class TestToolsDoNotBlock:
    def test_listen_is_inert_unless_interactive_prompting_enabled(self):
        from jarvis import speech

        speech.set_interactive_prompting(False)
        assert speech.listen() == "none"
        assert speech.interactive_prompting_enabled() is False

    def test_interactive_prompting_can_be_toggled(self):
        from jarvis import speech
        try:
            speech.set_interactive_prompting(True)
            assert speech.interactive_prompting_enabled() is True
        finally:
            speech.set_interactive_prompting(False)


# ============================================================================
# B9 — empty input at the API boundary
# ============================================================================

class TestApiValidation:
    def test_blank_message_rejected(self, client):
        for payload in [{"message": ""}, {"message": "   "}, {"message": "\n"}]:
            res = client.post("/api/chat", json=payload)
            assert res.status_code == 400, payload
            assert "empty" in res.json()["detail"].lower()

    def test_blank_message_does_not_reach_the_model(self, client):
        import api_server
        with patch.object(api_server.JarvisBrain, "ask") as ask:
            client.post("/api/chat", json={"message": "  "})
            ask.assert_not_called()

    def test_health_reports_api_key_presence(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "online"
        assert "api_key_configured" in body

    def test_brain_failure_is_flagged_as_error(self, client):
        import api_server
        with patch.object(api_server.JarvisBrain, "ask", side_effect=BrainError("nope")):
            res = client.post("/api/chat", json={"message": "hi"})
        body = res.json()
        assert body["error"] is True
        assert body["response"] == "nope"

    def test_sessions_are_isolated_by_conversation_id(self, client):
        """Each conversation_id must get its own brain instance."""
        import api_server
        api_server.clear_all_brains()

        # autospec=True so `self` is passed and can be identified per call.
        with patch.object(
            api_server.JarvisBrain, "ask", autospec=True, return_value="ok"
        ) as ask:
            client.post("/api/chat", json={"message": "a", "conversation_id": "A"})
            client.post("/api/chat", json={"message": "b", "conversation_id": "B"})
            client.post("/api/chat", json={"message": "c", "conversation_id": "A"})

        calls = ask.call_args_list
        brains = {id(c.args[0]) for c in calls}
        assert len(brains) == 2, "conversations shared a brain instance"

        # The same conversation id must reuse its own brain.
        assert id(calls[0].args[0]) == id(calls[2].args[0])
        assert id(calls[0].args[0]) != id(calls[1].args[0])

        # And the messages must still be forwarded verbatim, in order.
        assert [c.args[1] for c in calls] == ["a", "b", "c"]

    def test_default_conversation_is_reused(self, client):
        import api_server
        api_server.clear_all_brains()
        with patch.object(api_server.JarvisBrain, "ask", autospec=True, return_value="ok") as ask:
            client.post("/api/chat", json={"message": "one"})
            client.post("/api/chat", json={"message": "two"})
        assert len({id(c.args[0]) for c in ask.call_args_list}) == 1

    def test_clear_removes_only_the_targeted_conversation(self, client):
        import api_server
        api_server.clear_all_brains()
        with patch.object(api_server.JarvisBrain, "ask", return_value="ok"):
            client.post("/api/chat", json={"message": "a", "conversation_id": "A"})
            client.post("/api/chat", json={"message": "b", "conversation_id": "B"})

        res = client.post("/api/clear", json={"conversation_id": "A"})
        assert res.json()["existed"] is True
        assert "A" not in api_server._brains
        assert "B" in api_server._brains

    def test_clear_without_id_clears_everything(self, client):
        import api_server
        api_server.clear_all_brains()
        with patch.object(api_server.JarvisBrain, "ask", return_value="ok"):
            client.post("/api/chat", json={"message": "a", "conversation_id": "A"})
            client.post("/api/chat", json={"message": "b", "conversation_id": "B"})

        res = client.post("/api/clear")
        assert res.json()["cleared"] == 2
        assert api_server._brains == {}

    def test_session_registry_is_bounded(self):
        import api_server
        api_server.clear_all_brains()
        original = api_server.MAX_SESSIONS
        api_server.MAX_SESSIONS = 3
        try:
            for i in range(10):
                api_server.get_brain(f"c{i}")
            assert len(api_server._brains) <= 3
        finally:
            api_server.MAX_SESSIONS = original
            api_server.clear_all_brains()


# ============================================================================
# Full HTTP path: frontend payload -> API -> brain -> model -> response
# ============================================================================

class TestEndToEndPath:
    """Exercise the real HTTP surface with a stubbed provider.

    This validates route/payload/state wiring end to end, independent of
    provider quota.
    """

    @pytest.fixture()
    def sent(self, client, monkeypatch):
        """Route every brain to a recording fake provider session."""
        sent = []

        class FakeChat:
            def send_message(self, payload):
                sent.append(payload)
                if payload.get("kind") == "user":
                    return _make_response(text="ack: " + payload["text"])
                return _make_response(text="done")

            def close(self):
                pass

        monkeypatch.setattr(
            brain_module.JarvisBrain, "_create_chat",
            lambda self, model, history=None: FakeChat(),
        )
        monkeypatch.setattr(brain_module.JarvisBrain, "_trim_history", lambda self: None)
        return sent

    def test_message_survives_the_full_path(self, client, sent):
        res = client.post("/api/chat", json={"message": "Hello", "conversation_id": "A"})
        assert res.status_code == 200
        body = res.json()
        assert body["error"] is False
        assert body["response"] == "ack: Hello"
        assert body["conversation_id"] == "A"
        assert _sent_user_texts(_Sent(sent)) == ["Hello"]

    def test_exact_instruction_reaches_the_model(self, client, sent):
        message = "Answer only with the word YES."
        client.post("/api/chat", json={"message": message, "conversation_id": "A"})
        assert _sent_user_texts(_Sent(sent))[0] == message

    def test_conversation_b_sees_no_history_from_a(self, client, sent):
        client.post("/api/chat", json={"message": "My name is Alex", "conversation_id": "A"})
        res = client.post("/api/chat", json={"message": "What is my name?", "conversation_id": "B"})
        assert res.json()["response"] == "ack: What is my name?"
        assert "My name is Alex" not in _sent_user_texts(_Sent(sent))[1]

    def test_history_is_maintained_within_a_conversation(self, client, sent):
        client.post("/api/chat", json={"message": "My name is Alex.", "conversation_id": "A"})
        client.post("/api/chat", json={"message": "What is my name?", "conversation_id": "A"})
        assert _sent_user_texts(_Sent(sent)) == ["My name is Alex.", "What is my name?"]

    def test_clearing_a_conversation_resets_backend_state(self, client, sent):
        client.post("/api/chat", json={"message": "secret", "conversation_id": "A"})
        client.post("/api/clear", json={"conversation_id": "A"})
        import api_server
        assert "A" not in api_server._brains


# ============================================================================
# Minimal direct-model path (section 7)
# ============================================================================

class TestDirectPath:
    def test_direct_path_sends_only_the_user_message(self, monkeypatch):
        """No system prompt, no tools, no memory — just user text to model."""
        from jarvis import direct

        captured = {}

        def fake_generate_content(model, contents, config=None):
            captured["model"] = model
            captured["contents"] = contents
            captured["has_tools"] = getattr(config, "tools", None)
            captured["has_system"] = getattr(config, "system_instruction", None)
            return types.GenerateContentResponse(
                candidates=[types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="4")])
                )]
            )

        client = MagicMock()
        client.models.generate_content = fake_generate_content
        monkeypatch.setattr(direct, "get_client", lambda: client)

        assert direct.direct_ask("What is 2 + 2?") == "4"
        assert captured["contents"] == "What is 2 + 2?"
        assert not captured["has_tools"]
        assert not captured["has_system"]

    def test_direct_path_rejects_empty_input(self):
        from jarvis import direct
        with pytest.raises(ValueError):
            direct.direct_ask("   ")

    def test_direct_path_reports_all_failures(self, monkeypatch):
        from jarvis import direct

        def boom(model, contents, config=None):
            raise RuntimeError("503 UNAVAILABLE")

        client = MagicMock()
        client.models.generate_content = boom
        monkeypatch.setattr(direct, "get_client", lambda: client)
        monkeypatch.setattr(direct, "model_candidates", lambda: ["a", "b"])

        with pytest.raises(RuntimeError) as exc:
            direct.direct_ask("hi")
        assert "a" in str(exc.value) and "b" in str(exc.value)

    def test_direct_path_tries_candidates_in_order(self, monkeypatch):
        from jarvis import direct

        tried = []

        def generate_content(model, contents, config=None):
            tried.append(model)
            if model == "broken":
                raise RuntimeError("503 UNAVAILABLE")
            return types.GenerateContentResponse(
                candidates=[types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="Paris")])
                )]
            )

        client = MagicMock()
        client.models.generate_content = generate_content
        monkeypatch.setattr(direct, "get_client", lambda: client)
        monkeypatch.setattr(direct, "model_candidates", lambda: ["broken", "working"])

        assert direct.direct_ask("Capital of France?") == "Paris"
        assert tried[0] == "broken"
        assert "working" in tried


# ============================================================================
# Structural guards
# ============================================================================

class TestToolRegistry:
    def test_all_registered_tools_are_callable(self):
        assert len(GEMINI_TOOLS) >= 15
        for name, fn in TOOL_REGISTRY.items():
            assert callable(fn), name

    def test_no_tool_prompts_for_input_unconditionally(self):
        """Tools must not block on stdin in server mode (B8)."""
        from jarvis import speech

        speech.set_interactive_prompting(False)
        # search_wikipedia with an empty-ish query previously called listen().
        result = speech.listen()
        assert result == "none"