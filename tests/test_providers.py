"""Provider adapter tests.

Covers the two adapters and the provider-agnostic base contract:

* role/message translation between the neutral format and each wire format
* tool-call representation on both sides
* error classification and redaction
* the OpenAI-compatible tool-schema builder
* provider construction from configuration
* live provider probes, which skip cleanly without credentials
"""

import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from google.genai import types  # noqa: E402

from jarvis.providers import build_provider  # noqa: E402
from jarvis.providers.base import (  # noqa: E402
    Capabilities,
    ErrorKind,
    ModelProvider,
    ProviderError,
    ToolCall,
    assistant_message,
    classify_error,
    estimate_tokens,
    new_tool_call_id,
    tool_message,
    user_message,
)
from jarvis.providers.gemini import GeminiProvider, _to_gemini_content  # noqa: E402
from jarvis.providers.openai_compat import (  # noqa: E402
    OpenAICompatProvider,
    _function_to_schema,
)


# ============================================================================
# Neutral message format
# ============================================================================

class TestNeutralFormat:
    def test_message_helpers(self):
        assert user_message("hi") == {"role": "user", "content": "hi"}
        assert assistant_message("yo") == {"role": "assistant", "content": "yo", "tool_calls": []}
        msg = tool_message("id1", "tell_time", "12:00")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "id1"
        assert msg["name"] == "tell_time"

    def test_tool_call_ids_are_unique(self):
        ids = {new_tool_call_id() for _ in range(200)}
        assert len(ids) == 200

    def test_token_estimate_is_positive_and_scales(self):
        assert estimate_tokens([user_message("")]) == 0
        assert estimate_tokens([user_message("a" * 400)]) == 100
        assert estimate_tokens([user_message("a" * 800)]) > estimate_tokens(
            [user_message("a" * 400)]
        )


# ============================================================================
# Error classification and redaction
# ============================================================================

class TestErrorHandling:
    def test_auth_error_is_fatal_not_retryable(self):
        err = ProviderError(ErrorKind.AUTH, "m", "401", "p")
        assert err.retryable is False

    def test_rate_limit_and_server_are_retryable(self):
        assert ProviderError(ErrorKind.RATE_LIMIT, "m", "429", "p").retryable is True
        assert ProviderError(ErrorKind.SERVER, "m", "503", "p").retryable is True

    def test_invalid_request_is_not_retryable(self):
        """A 400 must not be retried blindly."""
        assert ProviderError(ErrorKind.INVALID_REQUEST, "m", "400", "p").retryable is False

    def test_timeout_error_classified(self):
        assert classify_error(TimeoutError("slow")) is ErrorKind.TIMEOUT

    def test_detail_is_redacted_at_construction(self):
        err = ProviderError(ErrorKind.SERVER, "m", "token sk-abcdef1234567890", "p")
        assert "sk-abcdef1234567890" not in err.safe_detail
        assert "sk-abcdef1234567890" not in str(err)

    def test_detail_is_length_bounded(self):
        err = ProviderError(ErrorKind.SERVER, "m", "x" * 10_000, "p")
        assert len(err.safe_detail) <= 200


# ============================================================================
# Gemini wire format
# ============================================================================

class TestGeminiTranslation:
    """Phase 0 asserted these wire-format details inline; they live here now."""

    def test_roles_are_mapped(self):
        contents = _to_gemini_content([
            user_message("hello"),
            assistant_message("hi"),
        ], types)

        assert [c.role for c in contents] == ["user", "model"]

    def test_assistant_tool_calls_become_function_call_parts(self):
        contents = _to_gemini_content([
            assistant_message(None, [
                {"id": "c1", "name": "tell_time", "arguments": {"a": 1}},
            ]),
        ], types)

        part = contents[0].parts[0]
        assert part.function_call.name == "tell_time"
        assert part.function_call.args == {"a": 1}
        assert part.function_call.id == "c1"

    def test_tool_results_become_user_function_response_parts(self):
        contents = _to_gemini_content([
            {"role": "tool", "results": [
                {"id": "c1", "name": "tell_time", "result": "12:00"},
            ]},
        ], types)

        # Gemini requires function responses on a user-role Content.
        assert contents[0].role == "user"
        response = contents[0].parts[0].function_response
        assert response.name == "tell_time"
        assert response.response == {"result": "12:00"}

    def test_string_arguments_are_parsed(self):
        contents = _to_gemini_content([
            assistant_message(None, [
                {"id": "c1", "name": "t", "arguments": '{"x": 2}'},
            ]),
        ], types)
        assert contents[0].parts[0].function_call.args == {"x": 2}

    def test_malformed_arguments_do_not_crash(self):
        contents = _to_gemini_content([
            assistant_message(None, [
                {"id": "c1", "name": "t", "arguments": "not json"},
            ]),
        ], types)
        assert contents[0].parts[0].function_call.args == {}

    def test_empty_messages_are_skipped(self):
        assert _to_gemini_content([user_message(""), assistant_message("")], types) == []

    def test_provider_requires_api_key(self):
        assert GeminiProvider(api_key="").is_configured() is False
        assert GeminiProvider(api_key="k").is_configured() is True

    def test_gemini_capabilities(self):
        caps = GeminiProvider(api_key="k").capabilities()
        assert caps.tool_calling is True
        assert caps.long_context is True

    def test_session_maps_gemini_response_to_neutral(self):
        provider = GeminiProvider(api_key="k")
        session = provider.open_session("m", "SYS", [], [])

        gemini_response = types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(
                role="model",
                parts=[
                    types.Part(text="hello"),
                    types.Part(function_call=types.FunctionCall(
                        id="c1", name="tell_time", args={}
                    )),
                ],
            ))]
        )
        # Exercise the response parser directly.
        parsed = session._to_response(gemini_response)
        assert parsed.text == "hello"
        assert parsed.has_tool_calls
        assert parsed.tool_calls[0].name == "tell_time"

    def test_session_handles_text_only_response(self):
        session = GeminiProvider(api_key="k").open_session("m", "SYS", [], [])
        response = types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(
                role="model", parts=[types.Part(text="just text")]
            ))]
        )
        parsed = session._to_response(response)
        assert parsed.text == "just text"
        assert not parsed.has_tool_calls

    def test_session_classifies_transport_errors(self):
        session = GeminiProvider(api_key="k").open_session("m", "SYS", [], [])

        class Boom:
            def send_message(self, content):
                raise RuntimeError("503 Service Unavailable")

        session._chat = Boom()
        with pytest.raises(ProviderError) as exc:
            session.send_message({"kind": "user", "text": "hi"})
        assert exc.value.kind is ErrorKind.SERVER
        assert exc.value.retryable is True

    def test_session_rejects_unknown_payload_kind(self):
        session = GeminiProvider(api_key="k").open_session("m", "SYS", [], [])
        with pytest.raises(ValueError):
            session.send_message({"kind": "nonsense"})


# ============================================================================
# OpenAI-compatible wire format
# ============================================================================

class TestOpenAICompat:
    def _provider(self):
        return OpenAICompatProvider(
            name="test", api_key="k", base_url="https://example.invalid/v1"
        )

    def test_requires_key_and_url(self):
        assert OpenAICompatProvider(name="t", api_key="", base_url="https://x").is_configured() is False
        assert OpenAICompatProvider(name="t", api_key="k", base_url="").is_configured() is False
        assert OpenAICompatProvider(name="t", api_key="k", base_url="https://x").is_configured() is True

    def test_system_prompt_becomes_leading_system_message(self):
        provider = self._provider()
        session = provider.open_session("m", "IDENTITY", [], [])
        assert session._messages[0] == {"role": "system", "content": "IDENTITY"}

    def test_user_text_is_added_verbatim(self):
        provider = self._provider()
        session = provider.open_session("m", "S", [], [])
        # Transport is stubbed: this test is about message construction, not HTTP.
        with patch.object(provider, "post_chat", return_value={"choices": [{"message": {"content": "YES"}}]}):
            session.send_message({"kind": "user", "text": "Answer only YES."})
        assert session._messages[-1] == {"role": "user", "content": "Answer only YES."}

    def test_history_is_seeded_in_order(self):
        provider = self._provider()
        session = provider.open_session(
            "m", "S", [],
            history=[user_message("first"), assistant_message("second")],
        )
        roles = [msg["role"] for msg in session._messages]
        assert roles == ["system", "user", "assistant"]

    def test_assistant_tool_calls_use_openai_shape(self):
        provider = self._provider()
        session = provider.open_session(
            "m", "S", [],
            history=[assistant_message(None, [
                {"id": "c1", "name": "tell_time", "arguments": {"a": 1}},
            ])],
        )
        call = session._messages[-1]["tool_calls"][0]
        assert call["type"] == "function"
        assert call["function"]["name"] == "tell_time"
        assert json.loads(call["function"]["arguments"]) == {"a": 1}

    def test_tool_results_become_one_message_each(self):
        provider = self._provider()
        session = provider.open_session("m", "S", [], [])
        with patch.object(provider, "post_chat", return_value={"choices": [{"message": {"content": "done"}}]}):
            session.send_message({"kind": "tool_results", "results": [
                {"id": "c1", "name": "a", "result": "1"},
                {"id": "c2", "name": "b", "result": "2"},
            ]})
        tool_msgs = [msg for msg in session._messages if msg["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "c1"
        assert tool_msgs[1]["content"] == "2"

    def test_response_parsing(self):
        provider = self._provider()
        session = provider.open_session("m", "S", [], [])

        data = {"choices": [{"message": {
            "content": "Paris",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "get_temperature", "arguments": '{"city":"Delhi"}'},
            }],
        }}]}
        parsed = session._parse(data)
        assert parsed.text == "Paris"
        assert parsed.tool_calls[0].name == "get_temperature"
        assert parsed.tool_calls[0].arguments == {"city": "Delhi"}

    def test_response_with_empty_content(self):
        provider = self._provider()
        session = provider.open_session("m", "S", [], [])
        parsed = session._parse({"choices": [{"message": {"content": ""}}]})
        assert parsed.text is None
        assert not parsed.has_tool_calls

    def test_malformed_tool_arguments_are_ignored(self):
        provider = self._provider()
        session = provider.open_session("m", "S", [], [])
        parsed = session._parse({"choices": [{"message": {
            "content": None,
            "tool_calls": [{"id": "c1", "function": {
                "name": "t", "arguments": "{not json"}}],
        }}]})
        assert parsed.tool_calls[0].arguments == {}

    def test_empty_choices_raises_provider_error(self):
        provider = self._provider()
        session = provider.open_session("m", "S", [], [])
        with pytest.raises(ProviderError):
            session._parse({"choices": []})

    def test_http_error_is_classified_and_redacted(self):
        provider = OpenAICompatProvider(
            name="test", api_key="sk-or-v1-supersecret123", base_url="https://example.invalid/v1"
        )
        error = urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions",
            429, "Too Many Requests", {}, None,
        )
        error.read = lambda: b'{"error":{"message":"rate limit"}}'

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(ProviderError) as exc:
                provider.post_chat({"model": "m", "messages": []}, "m")

        assert exc.value.kind is ErrorKind.RATE_LIMIT
        assert exc.value.status_code == 429
        assert "supersecret123" not in exc.value.safe_detail

    def test_auth_http_error_is_not_retryable(self):
        provider = self._provider()
        error = urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions", 401, "Unauthorized", {}, None
        )
        error.read = lambda: b'{"error":{"message":"invalid api key"}}'

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(ProviderError) as exc:
                provider.post_chat({"model": "m", "messages": []}, "m")

        assert exc.value.kind is ErrorKind.AUTH
        assert exc.value.retryable is False

    def test_server_http_error_is_retryable(self):
        provider = self._provider()
        error = urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions", 503, "Unavailable", {}, None
        )
        error.read = lambda: b"upstream down"

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(ProviderError) as exc:
                provider.post_chat({"model": "m", "messages": []}, "m")

        assert exc.value.kind is ErrorKind.SERVER
        assert exc.value.retryable is True

    def test_network_failure_is_classified(self):
        import socket
        provider = self._provider()
        for reason in (
            urllib.error.URLError("connection refused"),
            urllib.error.URLError(socket.gaierror("nodename nor servname provided")),
        ):
            with patch("urllib.request.urlopen", side_effect=reason):
                with pytest.raises(ProviderError) as exc:
                    provider.post_chat({"model": "m", "messages": []}, "m")
            assert exc.value.kind is ErrorKind.NETWORK, reason

    def test_timeout_is_classified(self):
        provider = self._provider()
        with patch("urllib.request.urlopen", side_effect=TimeoutError("slow")):
            with pytest.raises(ProviderError) as exc:
                provider.post_chat({"model": "m", "messages": []}, "m")
        assert exc.value.kind is ErrorKind.TIMEOUT

    def test_successful_post_returns_parsed_json(self):
        provider = self._provider()
        response = MagicMock()
        response.read.return_value = b'{"choices":[{"message":{"content":"hi"}}]}'
        response.__enter__ = lambda s: s
        response.__exit__ = lambda s, *a: False

        with patch("urllib.request.urlopen", return_value=response):
            data = provider.post_chat({"model": "m", "messages": []}, "m")
        assert data["choices"][0]["message"]["content"] == "hi"


class TestToolSchemaBuilder:
    def _provider(self):
        return OpenAICompatProvider(
            name="test", api_key="k", base_url="https://example.invalid/v1"
        )

    def test_schema_from_annotated_callable(self):
        def sample(city: str, days: int = 3, force: bool = False) -> str:
            """Get the weather.

            Longer description ignored.
            """
            return ""

        schema = _function_to_schema(sample)
        assert schema["function"]["name"] == "sample"
        assert schema["function"]["description"] == "Get the weather."
        params = schema["function"]["parameters"]
        assert params["properties"]["city"] == {"type": "string"}
        assert params["properties"]["days"] == {"type": "integer"}
        assert params["properties"]["force"] == {"type": "boolean"}
        assert params["required"] == ["city"]

    def test_schema_without_docstring(self):
        def bare(x: str) -> str:
            return ""
        assert _function_to_schema(bare)["function"]["description"] == "bare"

    def test_no_params_yields_empty_properties(self):
        def noparams() -> str:
            """Do nothing."""
            return ""
        schema = _function_to_schema(noparams)
        assert schema["function"]["parameters"]["properties"] == {}
        assert schema["function"]["parameters"]["required"] == []

    def test_builds_payload_from_brain_tools(self):
        import jarvis.brain as brain_module
        provider = self._provider()
        payload = provider.build_tools_payload(brain_module.GEMINI_TOOLS)
        assert len(payload) == len(brain_module.GEMINI_TOOLS)
        assert all(entry["type"] == "function" for entry in payload)
        assert {"name", "description", "parameters"} <= set(payload[0]["function"])

    def test_no_tools_yields_no_payload(self):
        assert self._provider().build_tools_payload([]) == []


# ============================================================================
# Provider construction from configuration
# ============================================================================

class TestProviderFactory:
    def test_builds_gemini_provider(self):
        provider = build_provider(
            "gemini",
            {"type": "gemini", "api_key_env": "MY_GEMINI_KEY", "default_model": "g"},
            env={"MY_GEMINI_KEY": "secret"},
        )
        assert isinstance(provider, GeminiProvider)
        assert provider.is_configured() is True

    def test_builds_openai_compatible_provider(self):
        provider = build_provider(
            "openrouter",
            {
                "type": "openai_compatible",
                "api_key_env": "MY_KEY",
                "base_url": "https://example.invalid/v1",
                "capabilities": {"tool_calling": True, "reasoning": True},
            },
            env={"MY_KEY": "secret"},
        )
        assert isinstance(provider, OpenAICompatProvider)
        assert provider.capabilities().tool_calling is True

    def test_missing_env_var_leaves_provider_unconfigured(self):
        provider = build_provider(
            "openrouter",
            {"type": "openai_compatible", "api_key_env": "ABSENT", "base_url": "https://x"},
            env={},
        )
        assert provider.is_configured() is False

    def test_unknown_provider_type_returns_none(self):
        assert build_provider("weird", {"type": "telepathy"}, env={}) is None

    def test_provider_name_is_not_hardcoded_in_routing(self):
        """Adding a provider must not require touching routing code."""
        import inspect
        import jarvis.router as router_module
        source = inspect.getsource(router_module)
        assert "gemini" not in source.lower(), "router is coupled to a vendor name"
        assert "openrouter" not in source.lower(), "router is coupled to a vendor name"


# ============================================================================
# Live providers (skipped without credentials or quota)
# ============================================================================

class TestLiveProviders:
    def test_gemini_live_round_trip(self):
        import os
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not configured")

        from jarvis.providers.base import ModelRequest
        provider = GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
        session = provider.open_session(
            "gemini-3.5-flash-lite", "You answer with just the number.", [], []
        )
        try:
            response = session.send_message({"kind": "user", "text": "What is 2+2?"})
        except ProviderError as exc:
            pytest.skip(f"provider unavailable: {exc.kind.value}")

        assert response.text
        assert "4" in response.text

    @pytest.mark.parametrize("provider_name,env_var", [
        ("openrouter", "OPENROUTER_API_KEY"),
        ("opencode_zen", "OPENCODE_ZEN_API_KEY"),
    ])
    def test_openai_compatible_probe(self, provider_name, env_var):
        import os
        import yaml
        from jarvis.config import PROVIDERS_CONFIG

        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
        block = config["providers"][provider_name]
        provider = build_provider(
            provider_name, block, env={env_var: os.getenv(env_var, "")}
        )
        if not provider.is_configured():
            pytest.skip(f"{env_var} not configured")
        # A probe must return a bool and never raise.
        assert isinstance(provider.probe(timeout=10.0), bool)

# ============================================================================
# Transport hygiene (real failures against live providers)
# ============================================================================

class TestOpenAICompatTransport:
    """Regression tests for two bugs that made every live OpenRouter /
    OpenCode Zen call fail while the unit tests stayed green."""

    def _provider(self):
        return OpenAICompatProvider(
            name="test", api_key="k", base_url="https://example.invalid/v1"
        )

    def test_requests_carry_a_user_agent(self):
        """Cloudflare rejects urllib's default UA with 'HTTP 403 error code:
        1010' (banned browser signature), which was classified as an auth
        failure and killed the whole request."""
        headers = self._provider()._headers()
        assert headers.get("User-Agent"), "no User-Agent sent to the provider"
        assert "Python-urllib" not in headers["User-Agent"]

    def test_caller_supplied_user_agent_wins(self):
        provider = OpenAICompatProvider(
            name="t", api_key="k", base_url="https://x", headers={"User-Agent": "custom/1"}
        )
        assert provider._headers()["User-Agent"] == "custom/1"

    def test_ssl_context_is_passed_to_urlopen(self):
        """Without a CA bundle urllib raised CERTIFICATE_VERIFY_FAILED on
        every HTTPS call, so the adapter could never reach a live endpoint."""
        import ssl

        from jarvis.providers.openai_compat import _ssl_context

        ctx = _ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.verify_mode == ssl.CERT_REQUIRED, "TLS verification was weakened"

        with patch("jarvis.providers.openai_compat.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b"{}"
            self._provider().post_chat(
                body={"model": "m", "messages": []}, model_id="m"
            )
        assert urlopen.call_count == 1
        assert isinstance(urlopen.call_args.kwargs.get("context"), ssl.SSLContext), (
            "urlopen called without an SSL context"
        )

    def test_probe_also_sends_headers_and_context(self):
        with patch("jarvis.providers.openai_compat.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b'{"data":[]}'
            assert self._provider().probe(timeout=5.0) is True
        request = urlopen.call_args.args[0]
        assert request.get_header("User-agent") or request.has_header("User-Agent")
        assert urlopen.call_args.kwargs.get("context") is not None
