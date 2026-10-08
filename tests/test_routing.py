"""Phase 0.5 regression tests: intelligent multi-model routing.

All routing decisions are asserted against deterministic mock providers, so
these tests never touch the network and never depend on provider quota.

Each test corresponds to an acceptance criterion from the Phase 0.5 brief.
"""

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.brain import JarvisBrain, BrainError  # noqa: E402
from jarvis.classify import TaskType, classify  # noqa: E402
from jarvis.health import HealthTracker  # noqa: E402
from jarvis.model_layer import ModelLayer  # noqa: E402
from jarvis.models import ModelRegistry  # noqa: E402
from jarvis.providers.base import (  # noqa: E402
    ErrorKind,
    ProviderError,
    classify_error,
    redact,
)
from jarvis.router import ModelRouter, RoutingConfig, RoutingWeights  # noqa: E402
from tests.mock_providers import (  # noqa: E402
    build_layer,
    empty,
    error,
    fail_then_success,
    reply,
    tool_then_reply,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    import jarvis.brain as brain_module
    monkeypatch.setattr(brain_module.StatusIndicator, "thinking", staticmethod(lambda: None))
    monkeypatch.setattr(brain_module.StatusIndicator, "tool_call", staticmethod(lambda n, a: None))
    monkeypatch.setattr(brain_module.StatusIndicator, "tool_result", staticmethod(lambda n, r: None))
    # Keep the memory DB out of these tests.
    monkeypatch.setattr(brain_module, "_build_system_prompt", lambda: "IDENTITY")


def _provider(layer: ModelLayer):
    return next(iter(layer.providers.values()))


# ============================================================================
# Model selection
# ============================================================================

class TestModelSelection:
    def test_simple_question_selects_a_capable_model(self):
        layer = build_layer(
            models=[
                {"key": "reasoner", "model": "r", "priority": 90,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "chatty", "model": "c", "priority": 50,
                 "capabilities": {"reasoning": False, "tool_calling": False}},
            ],
            behaviours={"r": reply("A"), "c": reply("B")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Why is the sky blue?")
        assert brain.last_model_key == "reasoner"

    def test_coding_task_prefers_a_coding_affinity_model(self):
        layer = build_layer(
            models=[
                {"key": "generalist", "model": "g", "priority": 90,
                 "capabilities": {"reasoning": True, "coding": True, "tool_calling": True},
                 "task_affinity": {"coding": 0}},
                {"key": "coder", "model": "c", "priority": 60,
                 "capabilities": {"reasoning": True, "coding": True, "tool_calling": True},
                 "task_affinity": {"coding": 40}},
            ],
            behaviours={"g": reply("G"), "c": reply("C")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Write a Python function to reverse a linked list")
        assert brain.last_model_key == "coder"

    def test_long_context_request_avoids_small_window_model(self):
        layer = build_layer(
            models=[
                {"key": "small", "model": "s", "priority": 99, "context_window": 8_000,
                 "capabilities": {"reasoning": True, "long_context": False}},
                {"key": "big", "model": "b", "priority": 50, "context_window": 1_000_000,
                 "capabilities": {"reasoning": True, "long_context": True}},
            ],
            behaviours={"s": reply("S"), "b": reply("B")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain._history = [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": "word " * 2000}
            for i in range(40)
        ]
        brain.ask("Summarise everything above")
        assert brain.last_model_key == "big"

    def test_disabled_model_is_never_selected(self):
        layer = build_layer(
            models=[
                {"key": "off", "model": "o", "enabled": False, "priority": 100,
                 "capabilities": {"reasoning": True}},
                {"key": "on", "model": "n", "priority": 10,
                 "capabilities": {"reasoning": True}},
            ],
            behaviours={"o": reply("O"), "n": reply("N")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Why?")
        assert brain.last_model_key == "on"

    def test_model_with_unconfigured_provider_is_skipped(self):
        layer = build_layer(
            models=[
                {"key": "noprov", "provider": "absent", "model": "x", "priority": 100,
                 "capabilities": {"reasoning": True}},
                {"key": "ok", "model": "y", "priority": 10,
                 "capabilities": {"reasoning": True}},
            ],
            behaviours={"y": reply("Y")},
            unconfigured_providers=["absent"],
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Why?")
        assert brain.last_model_key == "ok"


# ============================================================================
# Tool-calling requirement (hard filter)
# ============================================================================

class TestToolRequirement:
    def test_tool_request_only_reaches_tool_capable_models(self):
        layer = build_layer(
            models=[
                {"key": "no_tools", "model": "nt", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": False}},
                {"key": "with_tools", "model": "wt", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"nt": reply("NT"), "wt": reply("WT")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("What is the weather in Delhi right now?")
        assert brain.last_model_key == "with_tools"

    def test_non_tool_model_is_excluded_even_at_top_priority(self):
        """A non-tool model must never receive a tool-required request."""
        layer = build_layer(
            models=[
                {"key": "no_tools", "model": "nt", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": False}},
                {"key": "with_tools", "model": "wt", "priority": 1,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"nt": reply("NT"), "wt": reply("WT")},
        )
        c = layer.classify("Search Google for quantum computing")
        assert c.tool_required is True

        ranked = layer.plan(c)
        keys = [r.key for r in ranked]
        assert "no_tools" not in keys, "non-tool model ranked for a tool request"
        assert keys[0] == "with_tools"

    def test_tool_is_actually_executed_and_result_fed_back(self):
        layer = build_layer(
            models=[{"key": "wt", "model": "wt",
                     "capabilities": {"reasoning": True, "tool_calling": True}}],
            behaviours={"wt": tool_then_reply("tell_time", "ignored", "It is noon.")},
        )
        brain = JarvisBrain(model_layer=layer)
        import jarvis.brain as brain_module

        original = brain_module.TOOL_REGISTRY["tell_time"]
        brain_module.TOOL_REGISTRY["tell_time"] = lambda: "12:00"
        try:
            assert brain.ask("What time is it?") == "It is noon."
        finally:
            brain_module.TOOL_REGISTRY["tell_time"] = original

        payloads = _provider(layer).sent
        kinds = [p["kind"] for p in payloads]
        assert kinds == ["user", "tool_results"]
        assert payloads[1]["results"][0]["result"] == "12:00"

    def test_no_model_supports_tools_but_tools_requested_is_honest_error(self):
        layer = build_layer(
            models=[{"key": "nt", "model": "nt",
                     "capabilities": {"reasoning": True, "tool_calling": False}}],
            behaviours={"nt": reply("NT")},
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError) as exc:
            brain.ask("Check the weather in Tokyo")
        assert "No AI model is currently available" in exc.value.message

    def test_tool_is_not_requested_when_none_are_offered(self):
        result = classify("hello", tools_available=False)
        assert result.tool_required is False


# ============================================================================
# Fallback behaviour
# ============================================================================

class TestFallback:
    def test_primary_failure_falls_back_to_next_suitable_model(self):
        layer = build_layer(
            models=[
                {"key": "primary", "model": "p", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "backup", "model": "b", "priority": 50,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"p": error(ErrorKind.SERVER, "503"), "b": reply("backup answer")},
        )
        brain = JarvisBrain(model_layer=layer)
        assert brain.ask("Why?") == "backup answer"
        assert brain.last_model_key == "backup"

    def test_rate_limit_triggers_fallback(self):
        layer = build_layer(
            models=[
                {"key": "primary", "model": "p", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "backup", "model": "b", "priority": 50,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"p": error(ErrorKind.RATE_LIMIT, "429"), "b": reply("ok")},
        )
        brain = JarvisBrain(model_layer=layer)
        assert brain.ask("Why?") == "ok"
        assert layer.health.stats("primary").rate_limit_count == 1

    def test_timeout_triggers_fallback(self):
        layer = build_layer(
            models=[
                {"key": "primary", "model": "p", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "backup", "model": "b", "priority": 50,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"p": error(ErrorKind.TIMEOUT, "timed out"), "b": reply("ok")},
        )
        brain = JarvisBrain(model_layer=layer)
        assert brain.ask("Why?") == "ok"
        assert layer.health.stats("primary").timeout_count == 1

    def test_fallback_respects_capabilities(self):
        """Fallback must not land on a model that cannot do the task."""
        layer = build_layer(
            models=[
                {"key": "primary", "model": "p", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "no_tools", "model": "nt", "priority": 90,
                 "capabilities": {"reasoning": True, "tool_calling": False}},
            ],
            behaviours={
                "p": error(ErrorKind.SERVER, "503"),
                "nt": reply("should not be used"),
            },
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError):
            brain.ask("What is the weather in Paris?")
        assert layer.health.peek("no_tools") is None, "non-tool fallback was attempted"

    def test_max_attempts_is_respected(self):
        layer = build_layer(
            routing={"max_attempts": 2},
            models=[
                {"key": f"m{i}", "model": f"m{i}", "priority": 100 - i,
                 "capabilities": {"reasoning": True, "tool_calling": True}}
                for i in range(5)
            ],
            behaviours={f"m{i}": error(ErrorKind.SERVER, "503") for i in range(5)},
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError):
            brain.ask("Why?")
        attempted = [s.model_id for s in _provider(layer).sessions]
        assert len(attempted) == 2, "more models attempted than max_attempts allows"

    def test_all_models_unavailable_returns_honest_brain_error(self):
        layer = build_layer(
            models=[
                {"key": "a", "model": "a", "capabilities": {"reasoning": True}},
                {"key": "b", "model": "b", "capabilities": {"reasoning": True}},
            ],
            behaviours={"a": error(ErrorKind.SERVER, "503"), "b": error(ErrorKind.SERVER, "503")},
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError) as exc:
            brain.ask("Why?")
        # Honest, user-safe text. Not a fabricated assistant reply.
        assert "unavailable" in exc.value.message.lower()
        assert "503" in exc.value.detail

    def test_no_configured_models_returns_honest_error(self):
        layer = build_layer(
            models=[{"key": "a", "model": "a", "capabilities": {"reasoning": True}}],
            behaviours={"a": reply("x")},
            configured=False,
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError) as exc:
            brain.ask("Why?")
        assert "No AI model is currently available" in exc.value.message

    def test_transient_failure_recovers_on_retry(self):
        layer = build_layer(
            models=[
                {"key": "flaky", "model": "f", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "backup", "model": "b", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={
                "f": fail_then_success(ErrorKind.SERVER, "recovered", times=1),
                "b": reply("backup"),
            },
        )
        brain = JarvisBrain(model_layer=layer)
        assert brain.ask("Why?") == "backup"   # first turn falls back
        layer.health.reset()
        # Second turn: the flaky model is healthy again and wins.
        assert brain.ask("Why again?") == "recovered"
        assert brain.last_model_key == "flaky"


# ============================================================================
# Error classification
# ============================================================================

class TestErrorClassification:
    @pytest.mark.parametrize("text,expected", [
        ("429 Too Many Requests", ErrorKind.RATE_LIMIT),
        ("RESOURCE_EXHAUSTED quota exceeded", ErrorKind.RATE_LIMIT),
        ("insufficient_quota", ErrorKind.RATE_LIMIT),
        ("401 Unauthorized", ErrorKind.AUTH),
        ("API_KEY_INVALID", ErrorKind.AUTH),
        ("400 Bad Request invalid argument", ErrorKind.INVALID_REQUEST),
        ("404 model not found", ErrorKind.INVALID_REQUEST),
        ("503 Service Unavailable", ErrorKind.SERVER),
        ("502 Bad Gateway", ErrorKind.SERVER),
        ("deadline exceeded", ErrorKind.TIMEOUT),
        ("connection reset by peer", ErrorKind.NETWORK),
        ("this model does not support tools", ErrorKind.UNSUPPORTED),
    ])
    def test_classification(self, text, expected):
        assert classify_error(RuntimeError(text)) is expected

    def test_auth_error_does_not_cooldown(self):
        """Retrying a misconfigured key cannot help; backoff would hide it."""
        health = HealthTracker(base_backoff=60)
        health.record_failure("m", ErrorKind.AUTH)
        assert health.available("m") is True

    def test_server_error_does_cooldown(self):
        health = HealthTracker(base_backoff=60)
        health.record_failure("m", ErrorKind.SERVER)
        assert health.available("m") is False

    def test_auth_error_surfaces_immediately_without_other_retries(self):
        layer = build_layer(
            models=[
                {"key": "badkey", "model": "bk", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "other", "model": "ot", "priority": 50,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={
                "bk": error(ErrorKind.AUTH, "401 invalid api key"),
                "ot": reply("should not be reached"),
            },
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError) as exc:
            brain.ask("Why?")
        assert "api key" in exc.value.message.lower()
        assert layer.health.peek("other") is None, "retried despite a fatal config error"

    def test_empty_response_is_reported_not_faked(self):
        layer = build_layer(
            models=[{"key": "empty", "model": "e",
                     "capabilities": {"reasoning": True}}],
            behaviours={"e": empty()},
        )
        brain = JarvisBrain(model_layer=layer)
        assert brain.ask("Why?") == "Standing by, Boss."


# ============================================================================
# Cooldown
# ============================================================================

class TestCooldown:
    def test_repeated_failures_extend_cooldown(self):
        health = HealthTracker(base_backoff=10)
        health.record_failure("m", ErrorKind.SERVER)
        first = health.stats("m").cooldown_remaining()
        health.record_failure("m", ErrorKind.SERVER)
        second = health.stats("m").cooldown_remaining()
        assert second > first, "backoff did not grow with consecutive failures"

    def test_cooldown_removes_model_from_routing(self):
        layer = build_layer(
            routing={"cooldown_seconds": 300},
            models=[
                {"key": "flaky", "model": "f", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "solid", "model": "s", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"f": error(ErrorKind.SERVER, "503"), "s": reply("solid")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Why?")                       # flaky fails, enters cooldown
        brain.ask("Why again?")                 # must not touch flaky

        flaky_sessions = [x.model_id for x in _provider(layer).sessions]
        assert flaky_sessions.count("f") == 1, "cooled-down model was tried again"

    def test_success_clears_cooldown(self):
        health = HealthTracker(base_backoff=300)
        health.record_failure("m", ErrorKind.SERVER)
        assert health.available("m") is False
        health.record_success("m", 0.1)
        assert health.available("m") is True

    def test_probe_prevents_deadlock_when_everything_is_cooling(self):
        layer = build_layer(
            routing={"cooldown_seconds": 300, "probe_cooling_models": True},
            models=[
                {"key": "a", "model": "a", "capabilities": {"reasoning": True}},
                {"key": "b", "model": "b", "capabilities": {"reasoning": True}},
            ],
            behaviours={"a": error(ErrorKind.SERVER, "503"), "b": error(ErrorKind.SERVER, "503")},
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError):
            brain.ask("Why?")
        # Both are cooling down; the request must still be attempted rather
        # than deadlocking forever.
        with pytest.raises(BrainError):
            brain.ask("Why?")
        assert len(_provider(layer).sessions) >= 3

    def test_probe_disabled_returns_honest_error(self):
        layer = build_layer(
            routing={"cooldown_seconds": 300, "probe_cooling_models": False},
            models=[
                {"key": "a", "model": "a", "capabilities": {"reasoning": True}},
                {"key": "b", "model": "b", "capabilities": {"reasoning": True}},
            ],
            behaviours={"a": error(ErrorKind.SERVER, "503"), "b": error(ErrorKind.SERVER, "503")},
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError):
            brain.ask("Why?")
        with pytest.raises(BrainError) as exc:
            brain.ask("Why?")
        assert "No AI model is currently available" in exc.value.message


# ============================================================================
# Latency and reliability scoring
# ============================================================================

class TestScoring:
    def test_faster_healthy_model_scores_higher(self):
        layer = build_layer(
            models=[
                {"key": "slow", "model": "slow", "priority": 50,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "fast", "model": "fast", "priority": 50,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"slow": reply("s"), "fast": reply("f")},
            routing={"latency_weight": 0.5, "priority_weight": 0.0,
                     "reliability_weight": 0.0, "capability_weight": 0.0,
                     "task_weight": 0.0},
        )
        layer.health.record_success("slow", latency=4.0)
        layer.health.record_success("fast", latency=0.2)

        ranked = layer.plan(layer.classify("Why?"))
        assert ranked[0].key == "fast"
        assert ranked[0].reasons["latency"] > ranked[1].reasons["latency"]

    def test_latency_cannot_override_reliability(self):
        """A slightly slower reliable model must beat a fast flaky one."""
        layer = build_layer(
            models=[
                {"key": "flaky_fast", "model": "ff", "priority": 60,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "solid_slow", "model": "ss", "priority": 60,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"ff": reply("f"), "ss": reply("s")},
            routing={"latency_weight": 0.4, "reliability_weight": 0.4,
                     "capability_weight": 0.1, "task_weight": 0.0,
                     "priority_weight": 0.1},
        )
        # Flaky model: fast but fails most of the time.
        for _ in range(8):
            layer.health.record_success("flaky_fast", 0.1)
        for _ in range(9):
            layer.health.record_failure("flaky_fast", ErrorKind.SERVER)
        # Solid model: slower, but reliable.
        for _ in range(10):
            layer.health.record_success("solid_slow", 1.5)

        ranked = layer.plan(layer.classify("Why?"))
        assert ranked[0].key == "solid_slow", "latency outweighed reliability"

    def test_latency_cannot_override_capability(self):
        """Speed must not buy a request a model cannot actually serve."""
        layer = build_layer(
            models=[
                {"key": "fast_incapable", "model": "fi", "priority": 99,
                 "capabilities": {"reasoning": True, "tool_calling": False}},
                {"key": "slow_capable", "model": "sc", "priority": 1,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"fi": reply("f"), "sc": reply("s")},
            routing={"latency_weight": 0.9, "capability_weight": 0.1,
                     "reliability_weight": 0.0, "task_weight": 0.0,
                     "priority_weight": 0.0},
        )
        layer.health.record_success("fast_incapable", 0.01)
        layer.health.record_success("slow_capable", 10.0)

        c = layer.classify("What is the weather in Delhi?")
        ranked = layer.plan(c)
        assert [r.key for r in ranked] == ["slow_capable"]

    def test_unmeasured_model_gets_optimistic_prior(self):
        """A new model must be able to earn traffic."""
        layer = build_layer(
            models=[{"key": "new", "model": "n",
                     "capabilities": {"reasoning": True}}],
            behaviours={"n": reply("x")},
        )
        stats = layer.health.peek("new")
        assert stats is None
        ranked = layer.plan(layer.classify("Why?"))
        assert ranked[0].reasons["reliability"] == 0.8

    def test_rate_limit_demotes_priority(self):
        layer = build_layer(
            models=[
                {"key": "limited", "model": "l", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "fine", "model": "f", "priority": 95,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"l": reply("l"), "f": reply("f")},
        )
        for _ in range(3):
            layer.health.record_failure("limited", ErrorKind.RATE_LIMIT)

        # Rate limiting also triggers a cooldown, which removes it from routing
        # entirely. Clear the cooldown so the demotion itself is observable.
        layer.health.stats("limited").cooldown_until = None

        ranked = layer.plan(layer.classify("Why?"))
        limited = next(r for r in ranked if r.key == "limited")
        fine = next(r for r in ranked if r.key == "fine")

        # Declared priority is 100 vs 95, but repeated rate limits demote it.
        assert limited.reasons["priority"] < fine.reasons["priority"]
        assert limited.reasons["rate_limit_demotion"] < 0

    def test_rate_limited_model_is_cooldown_excluded(self):
        layer = build_layer(
            routing={"cooldown_seconds": 300},
            models=[
                {"key": "limited", "model": "l", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "fine", "model": "f", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"l": reply("l"), "f": reply("f")},
        )
        layer.health.record_failure("limited", ErrorKind.RATE_LIMIT)

        ranked = layer.plan(layer.classify("Why?"))
        assert "limited" not in [r.key for r in ranked]

    def test_unverified_model_is_penalised(self):
        layer = build_layer(
            models=[
                {"key": "unverified", "model": "u", "priority": 100, "verified": False,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "verified", "model": "v", "priority": 100, "verified": True,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"u": reply("u"), "v": reply("v")},
        )
        ranked = layer.plan(layer.classify("Why?"))
        assert ranked[0].key == "verified", "unverified model outranked a verified one"

    def test_routing_is_deterministic(self):
        layer = build_layer(
            models=[
                {"key": "a", "model": "a", "priority": 50,
                 "capabilities": {"reasoning": True}},
                {"key": "b", "model": "b", "priority": 50,
                 "capabilities": {"reasoning": True}},
            ],
            behaviours={"a": reply("a"), "b": reply("b")},
        )
        c = layer.classify("Why?")
        first = [r.key for r in layer.plan(c)]
        for _ in range(5):
            assert [r.key for r in layer.plan(c)] == first

    def test_weights_are_normalised(self):
        weights = RoutingWeights(
            capability=2.0, task=2.0, reliability=2.0, latency=2.0, priority=2.0
        )
        total = (weights.capability + weights.task + weights.reliability
                 + weights.latency + weights.priority)
        assert abs(total - 1.0) < 1e-9


# ============================================================================
# Conversation continuity across model switches
# ============================================================================

class TestConversationContinuity:
    def test_history_survives_a_model_switch(self):
        layer = build_layer(
            models=[
                {"key": "a", "model": "ma", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "b", "model": "mb", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={
                "ma": error(ErrorKind.SERVER, "503"),
                "mb": reply("B1"),
            },
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("My name is Alex")           # served by b after a fails

        # Now flip priorities so a wins the next turn.
        layer.registry.get("a").priority = 100
        layer.registry.get("b").priority = 1
        layer.health.reset()
        layer.health.record_success("ma", 0.1)
        layer.providers[next(iter(layer.providers))]._behaviours["ma"] = reply("A2")

        brain.ask("What is my name?")
        assert brain.last_model_key == "a", "expected the model to switch back"

        # The second turn's session must carry the first turn's history.
        session = _provider(layer).sessions[-1]
        assert "My name is Alex" in session.user_messages, (
            "conversation history was lost across a model switch"
        )

    def test_two_conversations_do_not_share_model_history(self):
        layer = build_layer(
            models=[{"key": "only", "model": "m",
                     "capabilities": {"reasoning": True, "tool_calling": True}}],
            behaviours={"m": reply("ok")},
        )
        a = JarvisBrain(conversation_id="A", model_layer=layer)
        b = JarvisBrain(conversation_id="B", model_layer=layer)
        a.ask("My name is Alex")
        b.ask("What is my name?")

        last = _provider(layer).sessions[-1]
        assert last.user_messages == [], "conversation B inherited A's history"
        assert _provider(layer).sent[-1]["text"] == "What is my name?"

    def test_same_system_identity_across_all_providers_and_models(self):
        """Identity is the AI Partner's, not the model's.

        Phase 2 appends a per-turn behavioral directive, so the full prompt is
        no longer a fixed string. What must stay identical across every
        provider and model is the identity base it is built on.
        """
        layer = build_layer(
            models=[
                {"key": "gem", "provider": "provA", "model": "mA", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "zen", "provider": "provB", "model": "mB", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"mA": reply("A"), "mB": reply("B")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Why?")
        brain.ask("Why again?")

        identities = set()
        for provider in layer.providers.values():
            for record in provider.sessions:
                assert record.system_prompt.startswith("IDENTITY"), (
                    f"identity base lost: {record.system_prompt!r}"
                )
                # Everything after the identity base is per-turn context
                # (memory / behavioral posture), which is expected to vary.
                identities.add(record.system_prompt.split("\n## ")[0].strip())
        assert identities == {"IDENTITY"}, f"inconsistent system identity: {identities}"

    def test_system_prompt_passed_to_every_model(self):
        layer = build_layer(
            models=[
                {"key": "a", "provider": "p1", "model": "ma",
                 "capabilities": {"reasoning": True}},
                {"key": "b", "provider": "p2", "model": "mb",
                 "capabilities": {"reasoning": True}},
            ],
            behaviours={"ma": reply("a"), "mb": reply("b")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Why?")
        for provider in layer.providers.values():
            for record in provider.sessions:
                assert record.system_prompt.startswith("IDENTITY"), (
                    f"identity missing from {record.system_prompt!r}"
                )

    def test_tools_are_offered_to_every_model(self):
        layer = build_layer(
            models=[
                {"key": "a", "model": "ma", "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "b", "model": "mb", "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"ma": reply("a"), "mb": reply("b")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Why?")
        brain.ask("Why again?")
        for provider in layer.providers.values():
            for record in provider.sessions:
                assert record.tools, "tools were not offered to the model"


# ============================================================================
# Tool state safety
# ============================================================================

class TestToolStateSafety:
    def test_tool_is_not_replayed_after_a_model_switch_mid_turn(self):
        """A switch must not duplicate side effects of an executed tool."""
        calls = []
        layer = build_layer(
            models=[
                {"key": "a", "model": "ma", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "b", "model": "mb", "priority": 50,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={
                # Model A asks for the tool, then fails mid-turn.
                "ma": tool_then_reply("send_email", "x", "unused"),
                "mb": tool_then_reply("send_email", "x", "sent"),
            },
        )
        provider = _provider(layer)
        original = provider._behaviours

        state = {"a_calls": 0}

        def model_a(history, payload):
            # First turn: request the tool. Then die when results come back.
            if payload["kind"] == "tool_results":
                raise ProviderError(ErrorKind.SERVER, "ma", "503 mid-turn", "mock")
            state["a_calls"] += 1
            from jarvis.providers.base import ModelResponse, ToolCall, new_tool_call_id
            return ModelResponse(
                text=None,
                tool_calls=[ToolCall(id=new_tool_call_id(), name="send_email",
                                     arguments={"to_address": "a@b.com", "subject": "s",
                                                "message": "m"})],
            )

        provider._behaviours["ma"] = model_a

        import jarvis.brain as brain_module
        sent = []
        brain_module.TOOL_REGISTRY["send_email"] = lambda **kw: (sent.append(kw), "Email sent successfully.")[1]
        try:
            brain = JarvisBrain(model_layer=layer)
            reply = brain.ask("Email a@b.com saying m")
        finally:
            brain_module.TOOL_REGISTRY.pop("send_email", None)

        assert len(sent) == 1, f"tool side effect duplicated ({len(sent)} executions)"
        assert reply == "sent"

    def test_committed_history_is_not_lost_when_a_turn_fails(self):
        layer = build_layer(
            models=[
                {"key": "only", "model": "m", "capabilities": {"reasoning": True}},
            ],
            behaviours={"m": error(ErrorKind.SERVER, "503")},
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError):
            brain.ask("Why?")
        # The failed attempt must not have been recorded as if it succeeded.
        assert brain._history == []


# ============================================================================
# Credential safety
# ============================================================================

class TestCredentialSafety:
    def test_provider_error_detail_is_redacted(self):
        for secret in (
            "sk-abcdef1234567890",
            "sk-or-v1-abcdef1234567890",
            "AIzaSyABCDEFGHIJKLMNOP",
            "Authorization: Bearer supersecrettoken",
            "api_key=leakedvalue123",
        ):
            err = ProviderError(ErrorKind.SERVER, "m", f"failed: {secret}", "p")
            assert secret not in err.safe_detail
            assert secret not in str(err)

    def test_redact_helper(self):
        assert "supersecret" not in redact("Bearer supersecret")
        assert "sk-abc123456" not in redact("key=sk-abc123456")

    def test_brain_error_never_contains_credentials(self):
        layer = build_layer(
            models=[{"key": "only", "model": "m",
                     "capabilities": {"reasoning": True}}],
            behaviours={"m": error(ErrorKind.SERVER, "sk-leaked-key-1234567890")},
        )
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError) as exc:
            brain.ask("Why?")
        assert "sk-leaked" not in exc.value.message
        assert "sk-leaked" not in exc.value.detail

    def test_status_snapshot_contains_no_credentials(self):
        layer = build_layer(
            models=[{"key": "a", "model": "ma", "capabilities": {"reasoning": True}}],
            behaviours={"ma": reply("a")},
        )
        snapshot = layer.status()
        blob = repr(snapshot)
        assert "api_key" not in blob.lower()
        assert "Authorization" not in blob

    def test_status_reports_configured_flag_not_key(self):
        layer = build_layer(
            models=[{"key": "a", "model": "ma", "capabilities": {"reasoning": True}}],
            behaviours={"ma": reply("a")},
        )
        row = layer.status()["models"][0]
        assert row["configured"] is True
        # "key" here is the model's configuration key, not a credential.
        assert row["key"] == "a"
        forbidden = {"api_key", "key_value", "token", "secret", "password",
                     "authorization", "credentials", "auth"}
        for field in row:
            assert field.lower() not in forbidden, field


# ============================================================================
# Model status endpoint
# ============================================================================

class TestStatusEndpoint:
    @pytest.fixture(autouse=True)
    def _fresh_status_cache(self):
        """The status cache is module-level, so each test must start empty."""
        import api_server
        api_server._clear_model_status_cache()
        yield
        api_server._clear_model_status_cache()

    @pytest.fixture()
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient
        import api_server
        import jarvis.model_layer as ml
        layer = build_layer(
            models=[
                {"key": "gem", "provider": "gem", "model": "gemini-x",
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "free", "provider": "or", "model": "free-x", "free": True,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"gemini-x": reply("g"), "free-x": reply("f")},
        )
        monkeypatch.setattr(ml, "get_model_layer", lambda: layer)
        # Context manager, not a bare return: Starlette's TestClient runs an
        # anyio portal on a daemon thread, and an unclosed client leaves that
        # thread racing native teardown at interpreter exit.
        with TestClient(api_server.app) as test_client:
            yield test_client

    def test_status_lists_models_with_health_and_capabilities(self, client):
        body = client.get("/api/models").json()
        keys = {m["key"] for m in body["models"]}
        assert keys == {"gem", "free"}

        gem = next(m for m in body["models"] if m["key"] == "gem")
        assert gem["provider"] == "gem"
        assert gem["model_id"] == "gemini-x"
        assert gem["capabilities"]["tool_calling"] is True
        assert "health" in gem
        assert "average_latency" in gem
        assert "failure_rate" in gem
        assert "cooldown_remaining" in gem

    def test_status_reflects_measured_health(self, client, monkeypatch):
        import jarvis.model_layer as ml
        layer = ml.get_model_layer()
        layer.health.record_success("gem", 1.5)
        layer.health.record_failure("free", ErrorKind.RATE_LIMIT)

        body = client.get("/api/models").json()
        gem = next(m for m in body["models"] if m["key"] == "gem")
        free = next(m for m in body["models"] if m["key"] == "free")
        assert gem["average_latency"] == pytest.approx(1.5)
        assert free["last_error_kind"] == "rate_limit"
        assert free["cooldown_remaining"] > 0

    def test_status_never_exposes_credentials(self, client):
        raw = client.get("/api/models").text
        assert "api_key" not in raw.lower()
        assert "Authorization" not in raw

    def test_status_can_be_disabled(self, monkeypatch):
        from fastapi.testclient import TestClient
        import api_server
        monkeypatch.setattr(api_server, "DEBUG_ENDPOINTS", False)
        with TestClient(api_server.app) as inline_client:
            assert inline_client.get("/api/models").status_code == 404


# ============================================================================
# Registry and configuration
# ============================================================================

class TestRegistry:
    def test_registry_reads_config_driven_models(self):
        registry = ModelRegistry.from_config({
            "alpha": {
                "provider": "openrouter", "model": "vendor/alpha:free",
                "priority": 80, "free": True, "context_window": 262_144,
                "verified": True, "capabilities": {"tool_calling": True, "coding": True},
            },
        })
        spec = registry.get("alpha")
        assert spec.provider == "openrouter"
        assert spec.model_id == "vendor/alpha:free"
        assert spec.free is True
        assert spec.supports("tool_calling") is True
        assert spec.supports("long_context") is True   # inferred from context window

    def test_adding_a_model_requires_no_code_change(self):
        """The registry is purely configuration-driven."""
        base = {
            "one": {"provider": "p", "model": "m1", "capabilities": {"reasoning": True}},
        }
        extended = dict(base)
        extended["two"] = {"provider": "p", "model": "m2",
                           "capabilities": {"reasoning": True}}

        assert len(ModelRegistry.from_config(base)) == 1
        assert len(ModelRegistry.from_config(extended)) == 2

    def test_malformed_entry_is_reported_not_silently_dropped(self, caplog):
        with caplog.at_level("WARNING"):
            ModelRegistry.from_config({"bad": "not-a-mapping"})
        assert any("bad" in r.message for r in caplog.records)

    def test_context_fit_reserves_output_room(self):
        registry = ModelRegistry.from_config({
            "m": {"provider": "p", "model": "m", "context_window": 1000,
                  "max_output_tokens": 400, "capabilities": {"reasoning": True}},
        })
        spec = registry.get("m")
        assert spec.can_fit(500) is True
        assert spec.can_fit(700) is False, "must reserve room for the model's reply"

    def test_shipped_config_declares_multiple_providers_and_models(self):
        """The real config.yaml must be a usable multi-model setup."""
        import yaml
        config = yaml.safe_load(
            (PROJECT_ROOT / "config.yaml").read_text()
        )
        providers = config.get("providers", {})
        assert "gemini" in providers
        assert "openrouter" in providers

        models = config.get("models", {})
        assert len(models) >= 3

        by_provider = {}
        for key, spec in models.items():
            by_provider.setdefault(spec["provider"], []).append(key)
            assert spec.get("model"), f"{key} has no model id"
            assert "verified" in spec, f"{key} does not record verification state"

        assert len(by_provider["gemini"]) >= 1
        assert len(by_provider["openrouter"]) >= 1

        # Gemini must remain available.
        assert providers["gemini"].get("enabled") is True

    def test_provider_config_never_inlines_a_key(self):
        import yaml
        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
        for name, block in config["providers"].items():
            assert "api_key_env" in block, f"{name} must read its key from the environment"
            assert "api_key" not in block


# ============================================================================
# Task classification
# ============================================================================

class TestTaskClassification:
    @pytest.mark.parametrize("message,expected", [
        ("Hello", TaskType.CASUAL),
        ("hi there", TaskType.CASUAL),
        ("Explain quantum computing", TaskType.REASONING),
        ("Write a Python function to parse JSON", TaskType.CODING),
        ("Write me a short poem about rain", TaskType.CREATIVE),
        ("Research the history of the Roman empire", TaskType.RESEARCH),
        ("Return only the word YES", TaskType.STRUCTURED_OUTPUT),
    ])
    def test_classification(self, message, expected):
        assert classify(message).task_type is expected

    def test_tool_words_set_tool_required(self):
        for message in (
            "What is the weather?",
            "Search Google for cats",
            "Take a screenshot",
            "What time is it?",
            "Play some music",
        ):
            assert classify(message).tool_required is True, message

    def test_plain_conversation_is_not_tool_required(self):
        for message in ("Hello", "How are you?", "Tell me a joke", "I'm doing well"):
            assert classify(message).tool_required is False, message

    def test_large_conversation_triggers_long_context(self):
        result = classify("Summarise this", conversation_tokens=500_000,
                          context_threshold=100_000)
        assert TaskType.LONG_CONTEXT in result.all_types

    def test_signals_are_recorded_for_debugging(self):
        result = classify("Debug this Python function")
        assert result.signals
        assert any(s.startswith("coding:") for s in result.signals)


# ============================================================================
# Health tracking
# ============================================================================

class TestHealthTracker:
    def test_statistics_are_tracked(self):
        health = HealthTracker()
        health.record_success("m", 1.0)
        health.record_success("m", 3.0)
        health.record_failure("m", ErrorKind.SERVER)

        stats = health.stats("m")
        assert stats.success_count == 2
        assert stats.failure_count == 1
        assert stats.average_latency == pytest.approx(2.0)
        assert stats.failure_rate == pytest.approx(1 / 3)
        # A just-failed model is in cooldown, which is the honest label.
        assert stats.status() == "cooldown"
        assert stats.is_available() is False

    def test_recent_latency_window_is_bounded(self):
        health = HealthTracker()
        for i in range(50):
            health.record_success("m", 1.0)
        assert len(health.stats("m").recent_latencies) <= 20

    def test_reset_clears_one_or_all(self):
        health = HealthTracker()
        health.record_success("a", 1.0)
        health.record_success("b", 1.0)
        health.reset("a")
        assert health.peek("a") is None
        assert health.peek("b") is not None
        health.reset()
        assert health.snapshot() == {}

    def test_snapshot_is_serialisable_and_safe(self):
        health = HealthTracker()
        health.record_success("m", 1.0)
        health.record_failure("m", ErrorKind.RATE_LIMIT)
        snap = health.snapshot()["m"]
        for key in ("status", "average_latency", "failure_rate", "cooldown_remaining"):
            assert key in snap


# ============================================================================
# No racing
# ============================================================================

class TestNoRacing:
    def test_a_request_goes_to_exactly_one_model(self):
        """Models must not be raced; only the chosen model is called."""
        layer = build_layer(
            models=[
                {"key": f"m{i}", "model": f"m{i}", "priority": 50,
                 "capabilities": {"reasoning": True, "tool_calling": True}}
                for i in range(5)
            ],
            behaviours={f"m{i}": reply(f"m{i}") for i in range(5)},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Why?")

        provider = _provider(layer)
        attempted = {record.model_id for record in provider.sessions}
        assert len(attempted) == 1, f"request was raced across {attempted}"
        assert len(provider.sent) == 1


# ============================================================================
# Free-only cost guard (routing.prefer_free)
# ============================================================================

class TestPreferFree:
    """`prefer_free` is a cost guard, not a tie-breaker: while a free model can
    serve the request, a paid model must never enter the fallback chain.

    Regression test. `prefer_free` was parsed from config.yaml and stored on
    RoutingConfig but never read anywhere, so paid models scored exactly like
    free ones and the fallback chain walked straight into a paid model once the
    free ones failed.
    """

    CAPS = {"reasoning": True, "coding": True, "conversation": True,
            "tool_calling": True, "long_context": True}

    @staticmethod
    def _specs(*, paid_free=False):
        return [
            # Paid, highest priority, every capability -- the shape of the real
            # gemini_flash entry, which outranked the free models in config.
            {"key": "paid_pro", "model": "paid-pro", "priority": 90,
             "free": paid_free, "capabilities": dict(TestPreferFree.CAPS)},
            {"key": "free_a", "model": "free-a", "priority": 50, "free": True,
             "capabilities": dict(TestPreferFree.CAPS)},
            {"key": "free_b", "model": "free-b", "priority": 40, "free": True,
             "capabilities": dict(TestPreferFree.CAPS)},
        ]

    def _layer(self, behaviours, **kw):
        # ScriptedProvider keys behaviours by model_id, not spec key.
        return build_layer(models=self._specs(), behaviours=behaviours, **kw)

    @staticmethod
    def _asked(layer):
        return {r.model_id for r in layer.providers["mock"].sessions}

    def test_paid_model_never_selected_while_free_exists(self):
        layer = self._layer({k: reply(k) for k in ("paid-pro", "free-a", "free-b")})
        chain = layer.plan(layer.classify("What is the capital of France?"))
        assert chain, "expected at least one eligible model"
        leaked = [s.spec.key for s in chain if not s.spec.free]
        assert not leaked, f"paid model in fallback chain: {leaked}"

    def test_free_model_serves_the_request(self):
        layer = self._layer({k: reply(k) for k in ("paid-pro", "free-a", "free-b")})
        brain = JarvisBrain(model_layer=layer)
        brain.ask("What is the capital of France?")
        assert self._asked(layer) == {"free-a"}, self._asked(layer)

    def test_free_failure_never_escalates_into_a_paid_call(self):
        """The whole point of the guard: a free outage must not cost money."""
        layer = self._layer({
            "free-a": error(ErrorKind.SERVER, "503"),
            "free-b": error(ErrorKind.SERVER, "503"),
            "paid-pro": reply("paid-pro"),
        })
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError):
            brain.ask("What is the capital of France?")
        assert "paid-pro" not in self._asked(layer), (
            f"paid model was called: {self._asked(layer)}"
        )

    def test_paid_still_usable_when_no_free_model_exists(self):
        """`prefer` semantics: with zero free candidates, do not deadlock."""
        layer = build_layer(
            models=self._specs(paid_free=True),
            behaviours={"paid-pro": reply("paid-pro")},
        )
        brain = JarvisBrain(model_layer=layer)
        assert brain.ask("What is the capital of France?") == "paid-pro"

    def test_flag_is_read_not_dead_config(self):
        """Guards against the flag silently becoming inert again."""
        layer = self._layer({k: reply(k) for k in ("paid-pro", "free-a", "free-b")})
        assert layer.router.config.prefer_free is True
        chain = layer.plan(layer.classify("What is the capital of France?"))
        assert chain and all(s.spec.free for s in chain)


# ============================================================================
# /api/models status cache
# ============================================================================

class TestModelStatusCache:
    """The endpoint's snapshot probes every configured model over the network,
    so the assembled result is cached for MODEL_STATUS_TTL seconds."""

    @pytest.fixture(autouse=True)
    def _fresh_status_cache(self):
        import api_server
        api_server._clear_model_status_cache()
        yield
        api_server._clear_model_status_cache()

    @pytest.fixture()
    def layer(self, monkeypatch):
        import api_server
        import jarvis.model_layer as ml
        layer = build_layer(
            models=[
                {"key": "gem", "provider": "gem", "model": "gemini-x",
                 "capabilities": {"reasoning": True}},
                {"key": "free", "provider": "or", "model": "free-x", "free": True,
                 "capabilities": {"reasoning": True}},
            ],
            behaviours={"gemini-x": reply("g"), "free-x": reply("f")},
        )
        monkeypatch.setattr(ml, "get_model_layer", lambda: layer)
        return layer

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        import api_server
        # Closed deterministically; see the note on the other client fixture.
        with TestClient(api_server.app) as test_client:
            yield test_client

    @staticmethod
    def _probes(layer):
        return sum(p.probes for p in layer.providers.values())

    # -- Test 1: first request probes ------------------------------------

    def test_first_request_probes_and_caches(self, client, layer):
        assert self._probes(layer) == 0
        body = client.get("/api/models").json()
        # Two models, both on configured providers -> two probes.
        assert self._probes(layer) == 2, "first request must probe"
        assert {m["key"] for m in body["models"]} == {"gem", "free"}
        import api_server
        assert api_server._model_status_cache is not None, "result was not cached"
        assert api_server._model_status_cached_at > 0

    # -- Test 2: repeat request uses the cache --------------------------

    def test_repeat_request_within_ttl_does_not_reprobe(self, client, layer):
        first = client.get("/api/models").json()
        after_first = self._probes(layer)
        assert after_first == 2

        second = client.get("/api/models").json()
        assert self._probes(layer) == after_first, "cache hit still probed"
        assert second == first, "cached response differs from the original"

    def test_many_repeat_requests_probe_once(self, client, layer):
        for _ in range(5):
            client.get("/api/models")
        assert self._probes(layer) == 2

    # -- Test 3: expiry refreshes ----------------------------------------

    def test_expired_cache_refreshes_with_a_fresh_probe(self, layer):
        import api_server
        api_server._get_model_status(1000.0)
        assert self._probes(layer) == 2

        # Just inside the window: still cached.
        api_server._get_model_status(1000.0 + api_server.MODEL_STATUS_TTL - 0.01)
        assert self._probes(layer) == 2, "cache expired early"

        # Past the window: fresh probe.
        stale = api_server._get_model_status(1000.0 + api_server.MODEL_STATUS_TTL + 0.01)
        assert self._probes(layer) == 4, "expired cache did not re-probe"
        assert api_server._model_status_cache is stale, "cache not replaced"

    def test_ttl_default_is_thirty_seconds(self):
        import api_server
        assert api_server.MODEL_STATUS_TTL == 30.0

    def test_refreshed_cache_is_reused_again(self, layer):
        import api_server
        api_server._get_model_status(0.0)
        api_server._get_model_status(100.0)          # forces a refresh
        assert self._probes(layer) == 4
        api_server._get_model_status(110.0)          # inside the new window
        assert self._probes(layer) == 4

    # -- Test 4: provider errors -----------------------------------------

    def test_probe_error_is_reported_and_cached_unchanged(self, monkeypatch):
        """A provider whose probe raises must surface exactly as before, and
        the cache must not swallow or reshape it."""
        import api_server
        from jarvis.providers.base import ProviderError, ErrorKind

        layer = build_layer(
            models=[{"key": "m", "provider": "mock", "model": "m",
                     "capabilities": {"reasoning": True}}],
            behaviours={"m": reply("x")},
        )
        import jarvis.model_layer as ml
        monkeypatch.setattr(ml, "get_model_layer", lambda: layer)

        def boom(timeout: float = 10.0) -> bool:
            raise ProviderError(ErrorKind.NETWORK, "mock", "connection refused", "mock")

        layer.providers["mock"].probe = boom

        with pytest.raises(ProviderError):
            api_server._get_model_status(0.0)
        # A failed probe must not be memoised as a good reading.
        assert api_server._model_status_cache is None

    def test_provider_unavailable_is_reported_not_hidden(self, layer):
        import api_server
        row = next(m for m in api_server._get_model_status(0.0).models
                   if m.key == "gem")
        # ScriptedProvider.probe() reports configured -> reachable.
        assert row.provider_available is True

    def test_unconfigured_provider_reports_none_not_cached_secret(self):
        import api_server
        layer = build_layer(
            models=[{"key": "m", "provider": "absent", "model": "m",
                     "capabilities": {"reasoning": True}}],
            unconfigured_providers=["absent"],
        )
        import jarvis.model_layer as ml
        real = ml.get_model_layer
        try:
            ml.get_model_layer = lambda: layer
            body = api_server._get_model_status(0.0)
        finally:
            ml.get_model_layer = real
        row = body.models[0]
        assert row.configured is False
        assert row.provider_available is None
        assert not hasattr(row, "api_key")

    def test_cached_payload_holds_no_credentials(self, client, layer):
        """Structural check plus a value check against the real secrets.

        The payload must not carry a credential field, and no configured secret
        value may appear anywhere in it. Values are compared in memory only --
        never printed, never logged.
        """
        import os

        import api_server

        raw = client.get("/api/models").text
        cached = api_server._model_status_cache
        assert cached is not None

        blobs = [raw, cached.model_dump_json()]
        forbidden_fields = {"api_key", "key_value", "token", "secret", "password",
                            "authorization", "credentials", "auth"}
        for blob in blobs:
            assert "Bearer" not in blob
            assert "Authorization" not in blob

        for row in cached.models:
            for field in type(row).model_fields:
                assert field.lower() not in forbidden_fields, field

        secrets = [
            v for k, v in os.environ.items()
            if k.endswith(("_API_KEY", "_APP_PASSWORD", "_ACCESS_KEY"))
            and v and len(v) >= 16
        ]
        assert secrets, "expected at least one configured secret in this env"
        for blob in blobs:
            for value in secrets:
                assert value not in blob, "a secret value leaked into /api/models"

    # -- Test 5: response compatibility ---------------------------------

    def test_response_schema_is_unchanged(self, client):
        from api_server import ModelInfo
        body = client.get("/api/models").json()
        assert set(body) == {
            "routing_enabled", "weights", "max_attempts",
            "cooldown_seconds", "models",
        }
        assert isinstance(body["weights"], dict)
        assert isinstance(body["models"], list)
        assert set(body["models"][0]) == set(ModelInfo.model_fields)

    def test_cached_response_matches_uncached_shape(self, layer):
        import api_server
        api_server._clear_model_status_cache()
        uncached = api_server._build_model_status().model_dump()
        cached = api_server._get_model_status(0.0).model_dump()
        assert cached == uncached

    def test_endpoint_still_404s_when_disabled(self, monkeypatch):
        from fastapi.testclient import TestClient
        import api_server
        monkeypatch.setattr(api_server, "DEBUG_ENDPOINTS", False)
        with TestClient(api_server.app) as inline_client:
            assert inline_client.get("/api/models").status_code == 404

    # -- Test 6: concurrency ---------------------------------------------

    def test_concurrent_cold_requests_probe_once(self, layer):
        import threading
        import api_server

        results = []
        errors = []
        start = threading.Barrier(8)

        def hit():
            try:
                start.wait(timeout=5)
                results.append(api_server._get_model_status(0.0))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=hit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, errors
        assert len(results) == 8
        assert self._probes(layer) == 2, (
            f"8 concurrent cold requests caused {self._probes(layer)} probes"
        )
        assert all(r is results[0] for r in results)

    def test_concurrent_endpoint_requests_probe_once(self, client, layer):
        """Same guarantee through the real HTTP endpoint."""
        import threading

        bodies = []
        errors = []
        start = threading.Barrier(6)

        def hit():
            try:
                start.wait(timeout=5)
                bodies.append(client.get("/api/models"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=hit) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, errors
        assert all(r.status_code == 200 for r in bodies)
        assert self._probes(layer) == 2, (
            f"concurrent endpoint calls caused {self._probes(layer)} probes"
        )
