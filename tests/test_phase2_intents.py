"""Phase 2 tests: deterministic local-first intent routing.

Every local command is proven to bypass providers: the mock layer raises an
unmistakable AssertionError on any provider contact. Tool side effects
(network, browser, files, TTS) are stubbed at the TOOL_REGISTRY slot, while
one test asserts the slots still point at the real brain wrappers.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jarvis.brain as brain_module  # noqa: E402
from jarvis.brain import TOOL_REGISTRY, JarvisBrain  # noqa: E402
from jarvis.intents import handle, match_only  # noqa: E402
from jarvis.logger import configure_logging  # noqa: E402
from tests.mock_providers import build_layer, reply  # noqa: E402


def _boom(_history, _payload):
    raise AssertionError("LLM SHOULD NOT HAVE BEEN CALLED")


def _blocking_layer():
    """A layer whose only model explodes if any provider call is attempted."""
    return build_layer(
        models=[{"key": "m1", "model": "m", "priority": 90,
                 "capabilities": {"reasoning": True, "tool_calling": True}}],
        behaviours={"m": _boom},
    )


@pytest.fixture()
def _quiet_logging():
    configure_logging(debug=False)
    yield
    configure_logging(debug=False)


@pytest.fixture()
def _stub_tools(monkeypatch):
    """Stub registry slots; record calls. Restore afterwards."""
    calls = []

    def _stub(name, result):
        def _fn(**kwargs):
            calls.append((name, kwargs))
            return result
        return _fn

    monkeypatch.setitem(TOOL_REGISTRY, "tell_time", _stub("tell_time", "08:42 PM"))
    monkeypatch.setitem(TOOL_REGISTRY, "lookup_dictionary",
                        _stub("lookup_dictionary", "lucid as a adjective: clear"))
    monkeypatch.setitem(TOOL_REGISTRY, "get_temperature", _stub("get_temperature", "32°C"))
    monkeypatch.setitem(TOOL_REGISTRY, "take_screenshot",
                        _stub("take_screenshot",
                              "Screenshot captured and saved to /tmp/s.png"))
    monkeypatch.setitem(TOOL_REGISTRY, "play_music",
                        _stub("play_music", "Playing 'yellow' on YouTube."))
    return calls


def _ask_blocked(text, monkeypatch=None):
    brain = JarvisBrain(model_layer=_blocking_layer())
    return brain, brain.ask(text)


# ============================================================================
# Pattern precision
# ============================================================================

class TestPatterns:
    @pytest.mark.parametrize("text", [
        "what time is it", "  WHAT TIME IS IT?  ", "what's the time",
        "tell me the time", "current time", "Hey Jarvis, what time is it",
    ])
    def test_time_variants(self, text):
        assert match_only(text) == "time"

    @pytest.mark.parametrize("text", [
        "define lucid", "what does lucid mean", "what do lucid mean",
        "meaning of lucid", "definition of lucid",
        "what is the meaning of lucid",
    ])
    def test_dictionary_variants(self, text):
        assert match_only(text) == "dictionary"

    @pytest.mark.parametrize("text", [
        "what's the weather", "what is the weather", "weather",
        "what's the weather today", "how's the weather in Mumbai",
        "weather in Mumbai",
    ])
    def test_weather_variants(self, text):
        assert match_only(text) == "weather"

    @pytest.mark.parametrize("text", ["take a screenshot", "capture my screen",
                                      "take a screenshot of the screen"])
    def test_screenshot_variants(self, text):
        assert match_only(text) == "screenshot"

    def test_media_play(self):
        assert match_only("play yellow by coldplay") == "media_play"

    @pytest.mark.parametrize("text", [
        "Write a short story about a robot living on Mars.",
        "Can you explain why the weather affects aircraft?",
        "Tell me about the history of WhatsApp.",
        "Write a poem about volume.",
        "Explain how weather affects climate.",
        "take a screenshot and email it to mom",
        "what time is it, and tell me a joke",
        "play", "define", "play music",
        "turn the volume up", "",
    ])
    def test_no_match_falls_through(self, text):
        assert match_only(text) is None

    @pytest.mark.parametrize("text", [
        "open WhatsApp", "Open whatsapp", "launch Safari", "Start Safari",
    ])
    def test_launch_app_matches(self, text):
        assert match_only(text) == "launch_app"


# ============================================================================
# Provider bypass: local commands never touch a provider
# ============================================================================

class TestProviderBypass:
    def test_time(self, _quiet_logging, _stub_tools):
        brain, response = _ask_blocked("what time is it")
        assert response == "The current time is 08:42 PM."
        assert ("tell_time", {}) in _stub_tools
        assert brain.last_model_key is None  # no model served this turn

    def test_dictionary_extracts_word(self, _quiet_logging, _stub_tools):
        _, response = _ask_blocked("define lucid")
        assert "lucid" in response
        assert ("lookup_dictionary", {"word": "lucid"}) in _stub_tools

    def test_weather_with_city(self, _quiet_logging, _stub_tools):
        _, response = _ask_blocked("weather in Mumbai")
        assert "Mumbai" in response and "32°C" in response
        assert ("get_temperature", {"city": "Mumbai"}) in _stub_tools

    def test_weather_default_city(self, _quiet_logging, _stub_tools):
        _, response = _ask_blocked("what's the weather")
        assert "Delhi" in response
        assert ("get_temperature", {"city": "Delhi"}) in _stub_tools

    def test_screenshot(self, _quiet_logging, _stub_tools):
        _, response = _ask_blocked("take a screenshot")
        assert "Screenshot captured" in response
        assert ("take_screenshot", {}) in _stub_tools

    def test_media_play(self, _quiet_logging, _stub_tools):
        _, response = _ask_blocked("play yellow by coldplay")
        assert response.startswith("Playing")
        assert ("play_music", {"song_name": "yellow by coldplay"}) in _stub_tools

    def test_local_turn_is_recorded(self, _quiet_logging, _stub_tools):
        brain, _ = _ask_blocked("what time is it")
        assert brain._history[-2:] == [
            {"role": "user", "content": "what time is it"},
            {"role": "assistant", "content": "The current time is 08:42 PM."},
        ]

    def test_registry_slots_are_real_wrappers(self):
        """Stubs above only shadow the slots; production still uses brain tools."""
        assert TOOL_REGISTRY["tell_time"] is brain_module.tell_time
        assert TOOL_REGISTRY["lookup_dictionary"] is brain_module.lookup_dictionary
        assert TOOL_REGISTRY["get_temperature"] is brain_module.get_temperature
        assert TOOL_REGISTRY["take_screenshot"] is brain_module.take_screenshot
        assert TOOL_REGISTRY["play_music"] is brain_module.play_music

    def test_honest_answer_when_tool_returns_empty(
            self, _quiet_logging, monkeypatch):
        monkeypatch.setitem(TOOL_REGISTRY, "tell_time", lambda: "")
        _, response = _ask_blocked("what time is it")
        assert "couldn't" in response  # honest, still no LLM call


# ============================================================================
# Fallback: unknown requests use the existing AI router untouched
# ============================================================================

class TestFallback:
    def test_unknown_request_reaches_provider(self, _quiet_logging):
        layer = build_layer(
            models=[{"key": "m1", "model": "m", "priority": 90,
                     "capabilities": {"reasoning": True, "tool_calling": True}}],
            behaviours={"m": reply("A robot story.")},
        )
        brain = JarvisBrain(model_layer=layer)
        assert brain.ask("Write a short story about a robot living on Mars.") == \
            "A robot story."
        assert brain.last_model_key == "m1"

    def test_router_result_contract(self):
        assert handle("take a screenshot", lambda n, a: "ok").matched is True
        assert handle("Write a poem.", lambda n, a: "ok").matched is False

    def test_ephemeral_turns_skip_local(self, _quiet_logging):
        """Perception turns keep existing behavior (provider serves them)."""
        layer = build_layer(
            models=[{"key": "m1", "model": "m", "priority": 90,
                     "capabilities": {"reasoning": True, "tool_calling": True}}],
            behaviours={"m": reply("perceived")},
        )
        brain = JarvisBrain(model_layer=layer)
        assert brain.ask("what time is it", ephemeral=True) == "perceived"


# ============================================================================
# Phase 1 logging stays intact
# ============================================================================

class TestLoggingSeparation:
    def test_normal_mode_terminal_stays_clean(
            self, _quiet_logging, _stub_tools, capsys):
        _ask_blocked("what time is it")
        out, err = capsys.readouterr()
        assert "Thinking" not in out and "intent" not in out.lower()
        assert err == ""

    def test_debug_mode_names_intent(self, _stub_tools, caplog):
        configure_logging(debug=True)
        try:
            with caplog.at_level("DEBUG", logger="jarvis"):
                _ask_blocked("define lucid")
            assert any("dictionary" in r.message and "skipped" in r.message
                       for r in caplog.records)
        finally:
            configure_logging(debug=False)
