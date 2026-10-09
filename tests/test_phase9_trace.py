"""Phase 9 tests: turn tracing measures without changing behavior.

Deterministic: fake clocks, scripted mock providers, stubbed tools. No
network, no microphone, no model weights. Exact durations are never
asserted -- only structure, statuses, isolation, and silence.
"""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.brain import JarvisBrain  # noqa: E402
from jarvis.logger import configure_logging  # noqa: E402
from jarvis.providers.base import ErrorKind, ModelResponse  # noqa: E402
from jarvis.trace import (  # noqa: E402
    TurnTrace,
    current_trace,
    new_trace,
    span_of,
    use_trace,
)
from tests.mock_providers import build_layer, error, reply, tool_then_reply  # noqa: E402


@pytest.fixture()
def _quiet():
    configure_logging(debug=False)
    yield
    configure_logging(debug=False)


def _layer(behaviours, **kwargs):
    return build_layer(
        models=[{"key": "m1", "model": "m", "priority": 90,
                 "capabilities": {"reasoning": True, "tool_calling": True}}],
        behaviours=behaviours, **kwargs,
    )


class TestTimer:
    def test_monotonic_clock_used(self, _quiet):
        ticks = iter([100.0, 100.5])
        trace = TurnTrace(clock=lambda: next(ticks))
        with trace.span("work"):
            pass
        assert trace.spans[0].ms == pytest.approx(500.0)

    def test_failed_span_keeps_duration_and_status(self, _quiet):
        trace = TurnTrace()
        with pytest.raises(RuntimeError):
            with trace.span("work"):
                raise RuntimeError("boom")
        assert trace.spans[0].status == "fail"
        assert trace.spans[0].ms >= 0.0

    def test_cancel_open_does_not_touch_finished(self, _quiet):
        trace = TurnTrace()
        with trace.span("done"):
            pass
        trace.cancel_open("stale")
        assert trace.spans[0].status == "ok"
        # A later turn starts clean.
        assert TurnTrace().spans == []


class TestCorrelation:
    def test_trace_ids_unique_per_turn(self, _quiet):
        assert new_trace().trace_id != new_trace().trace_id

    def test_nested_ask_reuses_enclosing_trace(self, _quiet):
        outer = new_trace()
        layer = _layer({"m": reply("hi")})
        with use_trace(outer):
            JarvisBrain(model_layer=layer).ask("Why is the sky blue?")
        assert current_trace() is None  # binding restored
        names = [s.name for s in outer.spans]
        assert "classify" in names and "provider_request" in names

    def test_threads_do_not_share_trace(self, _quiet):
        seen = []
        barrier = threading.Barrier(4)

        def _go():
            barrier.wait()
            with use_trace(new_trace()):
                time.sleep(0.01)
                seen.append(current_trace().trace_id)

        threads = [threading.Thread(target=_go) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(seen)) == 4


class TestBrainStages:
    def test_local_command_has_no_provider_span(self, _quiet, monkeypatch):
        import jarvis.brain as brain_module
        monkeypatch.setitem(brain_module.TOOL_REGISTRY, "tell_time",
                            lambda: "08:42 PM")
        layer = _layer({"m": reply("unused")})
        trace = new_trace()
        with use_trace(trace):
            out = JarvisBrain(model_layer=layer).ask("what time is it")
        assert "08:42 PM" in out
        names = [s.name for s in trace.spans]
        assert "intent_route" in names
        assert "provider_request" not in names
        assert trace._finished is False  # owner (voice loop) finishes

    def test_owned_trace_finishes_local(self, _quiet, monkeypatch, caplog):
        import jarvis.brain as brain_module
        monkeypatch.setitem(brain_module.TOOL_REGISTRY, "tell_time",
                            lambda: "08:42 PM")
        layer = _layer({"m": reply("unused")})
        with caplog.at_level("DEBUG", logger="jarvis"):
            out = JarvisBrain(model_layer=layer).ask("what time is it")
        assert "08:42 PM" in out
        assert any("route=local" in r.message for r in caplog.records)

    def test_provider_span_carries_model(self, _quiet):
        layer = _layer({"m": reply("hello there")})
        trace = new_trace()
        with use_trace(trace):
            JarvisBrain(model_layer=layer).ask("Why is the sky blue?")
        spans = [s for s in trace.spans if s.name == "provider_request"]
        assert len(spans) >= 1
        assert spans[0].attrs["model"] == "m1"
        assert spans[0].status == "ok"

    def test_fallback_attempts_each_get_a_span(self, _quiet):
        layer = build_layer(
            models=[
                {"key": "p", "model": "p", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "b", "model": "b", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"p": error(ErrorKind.SERVER, "503"),
                         "b": reply("backup ok")},
        )
        trace = new_trace()
        with use_trace(trace):
            assert JarvisBrain(model_layer=layer).ask("Why?") == "backup ok"
        models = [s.attrs["model"] for s in trace.spans
                  if s.name == "provider_request"]
        assert models == ["p", "b"]  # one span per attempt, no double count

    def test_tool_span_names_tool_not_args(self, _quiet):
        from tests.mock_providers import ToolCall, new_tool_call_id
        from jarvis.providers.base import ModelResponse as MR

        def _script(history, payload):
            if payload.get("kind") == "tool_results":
                return MR(text="done")
            return MR(text=None, tool_calls=[ToolCall(
                id=new_tool_call_id(), name="get_system_time", arguments={})])

        layer = _layer({"m": _script})
        trace = new_trace()
        with use_trace(trace):
            JarvisBrain(model_layer=layer).ask("check my inbox")
        tools = [s for s in trace.spans if s.name == "tool_execute"]
        assert [t.attrs["tool"] for t in tools] == ["get_system_time"]
        assert all("arguments" not in t.attrs for t in tools)


class TestTTSReadiness:
    def test_warm_model_reports_ready_not_loading(self, _quiet, monkeypatch):
        from jarvis import speech
        model = MagicMock()
        import numpy as np
        model.generate.return_value = np.zeros((1, 2400), dtype=np.float32)
        monkeypatch.setattr(speech, "_get_chatterbox_model", lambda: model)
        speech._chatterbox_model = None
        speech._chatterbox_conds_key = None
        speech._chatterbox_warmed = False
        try:
            trace = new_trace()
            with use_trace(trace):
                speech._synthesize_chatterbox("Hello.")
                speech._synthesize_chatterbox("Again.")
            ready = [s for s in trace.spans if s.name == "tts_ready"]
            assert len(ready) == 2
            assert model.prepare_conditionals.call_count == 1
            assert not any(s.name == "tts_load" for s in trace.spans)
        finally:
            speech._chatterbox_model = None
            speech._chatterbox_conds_key = None
            speech._chatterbox_warmed = False


class TestOutput:
    def test_normal_output_clean(self, _quiet, monkeypatch, capsys):
        import jarvis.brain as brain_module
        monkeypatch.setitem(brain_module.TOOL_REGISTRY, "tell_time",
                            lambda: "08:42 PM")
        layer = _layer({"m": reply("unused")})
        JarvisBrain(model_layer=layer).ask("what time is it")
        out, err = capsys.readouterr()
        assert "trace" not in out and "ms" not in out and err == ""

    def test_debug_shows_compact_trace(self, _quiet, monkeypatch, caplog):
        import jarvis.brain as brain_module
        monkeypatch.setitem(brain_module.TOOL_REGISTRY, "tell_time",
                            lambda: "08:42 PM")
        layer = _layer({"m": reply("unused")})
        configure_logging(debug=True)
        try:
            with caplog.at_level("DEBUG", logger="jarvis"):
                JarvisBrain(model_layer=layer).ask("what time is it")
            assert any(r.message.startswith("trace turn_") and "route=local" in r.message
                       for r in caplog.records)
        finally:
            configure_logging(debug=False)

    def test_no_duplicate_handlers_or_events(self, _quiet):
        import logging
        from jarvis.logger import logger
        import logging.handlers as lh
        consoles = [h for h in logger.handlers
                    if isinstance(h, logging.StreamHandler)
                    and not isinstance(h, lh.RotatingFileHandler)]
        assert len(consoles) == 1
