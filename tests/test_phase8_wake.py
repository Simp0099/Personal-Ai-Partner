"""Phase 8 tests: multi-phrase wake configuration, honest model coverage.

Deterministic throughout: scripted fake engines, no microphone, no ONNX.
Real acoustic numbers live in docs/wake_phrases.md (scripts/wake_eval.py).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.config import (  # noqa: E402
    WAKE_WORD_MODEL,
    WAKE_WORD_PHRASES,
    WAKE_WORD_THRESHOLD,
)
from jarvis.conversation import State  # noqa: E402
from jarvis.voice_loop import VoiceLoop  # noqa: E402
from jarvis.wake_word import WakeWordEngine, known_wake_models  # noqa: E402


class FakeMultiEngine(WakeWordEngine):
    """Scripted per-phrase scores; load/close counted; no ONNX."""

    def __init__(self, scores=None, phrases=None, loaded_ok=True):
        super().__init__(phrases=phrases or [
            {"phrase": "hey jarvis", "model": "hey_jarvis", "threshold": 0.5},
            {"phrase": "jarvis", "model": "jarvis_custom", "threshold": 0.6},
        ])
        self._script = list(scores or [])
        self.loaded_ok = loaded_ok
        self.loads = 0
        self.closes = 0

    def load(self):
        self.loads += 1
        return self.loaded_ok

    def scores(self, pcm):
        return dict(self._script.pop(0)) if self._script else {}

    def score(self, pcm):
        return self.scores(pcm).get(self.phrase, 0.0)

    def close(self):
        self.closes += 1
        super().close()


class FakeSource:
    def __init__(self):
        self.opened = 0
        self.is_open = False

    def open(self):
        self.opened += 1
        self.is_open = True

    def read(self):
        import time
        time.sleep(0.005)
        return None

    def close(self):
        self.is_open = False

    def status(self):
        return {}


class FakeVAD:
    def __init__(self):
        self.frames = 0

    def push(self, pcm, barge_in=False):
        self.frames += 1
        return None

    def reset(self):
        pass


def _pcm():
    import numpy as np
    return (np.zeros(1280, dtype=np.int16)).tobytes()


def _loop(engine, **kwargs):
    from jarvis.audio import MicrophoneStream
    kwargs.setdefault("wake_enabled", True)
    kwargs.setdefault("follow_up_window", 0.0)
    kwargs.setdefault("microphone", MicrophoneStream(source=FakeSource()))
    kwargs.setdefault("respond", lambda t: "ok")
    return VoiceLoop(wake_engine=engine, **kwargs)


class TestPhraseCoverage:
    def test_hey_jarvis_active_by_default(self):
        assert WAKE_WORD_MODEL == "hey_jarvis"
        assert WAKE_WORD_PHRASES[0]["model"] == "hey_jarvis"
        assert WAKE_WORD_PHRASES[0]["enabled"] is True

    def test_bare_jarvis_disabled_without_verified_model(self):
        assert "hey_jarvis" in known_wake_models()
        assert "jarvis" not in known_wake_models()
        by_phrase = {p["phrase"]: p for p in WAKE_WORD_PHRASES}
        assert by_phrase.get("jarvis", {}).get("enabled", False) is False

    def test_candidates_not_active(self):
        phrases = [p["phrase"] for p in WAKE_WORD_PHRASES if p.get("enabled")]
        assert "you there" not in phrases
        assert "wake up" not in phrases
        assert "hello" not in phrases


class TestConfigParsing:
    def test_multi_phrase_parsed(self):
        from jarvis.config import _validated_wake_phrases
        out = _validated_wake_phrases([
            {"phrase": "hey jarvis", "model": "hey_jarvis", "threshold": 0.5},
            {"phrase": "jarvis", "model": "jarvis_custom", "threshold": 0.6,
             "enabled": False},
        ], "hey_jarvis", 0.5)
        assert [(p["phrase"], p["model"]) for p in out] == [("hey jarvis", "hey_jarvis")]

    def test_enabled_invalid_dropped(self):
        from jarvis.config import _validated_wake_phrases
        out = _validated_wake_phrases([
            {"phrase": "x", "model": "", "threshold": 0.5},
            {"phrase": "y", "model": "hey_jarvis", "threshold": 9.0},
            {"phrase": "hey jarvis", "model": "hey_jarvis", "threshold": 0.5},
        ], "hey_jarvis", 0.5)
        assert [p["phrase"] for p in out] == ["hey jarvis"]

    def test_explicit_disable_all_respected(self):
        from jarvis.config import _validated_wake_phrases
        out = _validated_wake_phrases([
            {"phrase": "hey jarvis", "model": "hey_jarvis",
             "threshold": 0.5, "enabled": False},
        ], "hey_jarvis", 0.5)
        assert out == []


class TestEngine:
    def test_single_model_session_for_all_phrases(self):
        import openwakeword
        import openwakeword.model as owm
        with patch.object(openwakeword, "MODELS",
                          {"hey_jarvis": {}, "jarvis_custom": {}}):
            with patch.object(owm, "Model") as model_cls:
                assert WakeWordEngine(phrases=[
                    {"phrase": "hey jarvis", "model": "hey_jarvis", "threshold": 0.5},
                    {"phrase": "jarvis", "model": "jarvis_custom", "threshold": 0.6},
                ]).load() is True
                model_cls.assert_called_once()
                assert model_cls.call_args.kwargs["wakeword_models"] == [
                    "hey_jarvis", "jarvis_custom"]

    def test_unknown_model_fails_clearly(self):
        import openwakeword.model as owm
        with patch.object(owm, "Model", side_effect=FileNotFoundError("nope")):
            engine = WakeWordEngine(phrases=[
                {"phrase": "nope", "model": "nope", "threshold": 0.5}])
            assert engine.load() is False
            assert engine.loaded is False

    def test_per_phrase_thresholds(self):
        engine = FakeMultiEngine()
        assert engine.thresholds == {"hey jarvis": 0.5, "jarvis": 0.6}
        assert engine.threshold == 0.5  # primary unchanged
        assert engine.score(b"") == 0.0  # unloaded → silent

    def test_legacy_single_phrase_compat(self):
        engine = WakeWordEngine()
        assert engine.model == "hey_jarvis"
        assert engine.threshold == WAKE_WORD_THRESHOLD


class TestSharedPipeline:
    def test_one_utterance_one_turn(self):
        engine = FakeMultiEngine(scores=[
            {"hey jarvis": 0.9, "jarvis": 0.8},  # both fire, same chunk
            {"hey jarvis": 0.9, "jarvis": 0.8},  # overlapping chunk
        ])
        loop = _loop(engine)
        before = loop.wake_events
        loop._on_frame(_pcm())
        loop._on_frame(_pcm())
        assert loop.wake_events == before + 1
        assert loop._wake_rejections >= 1

    def test_second_phrase_fires_when_first_quiet(self):
        engine = FakeMultiEngine(scores=[{"hey jarvis": 0.1, "jarvis": 0.9}])
        loop = _loop(engine)
        before = loop.wake_events
        loop._on_frame(_pcm())
        assert loop.wake_events == before + 1

    def test_vad_still_receives_frames(self):
        engine = FakeMultiEngine(scores=[{"hey jarvis": 0.95}])
        loop = _loop(engine)
        vad = FakeVAD()
        loop.vad = vad
        loop._on_frame(_pcm())
        assert vad.frames == 1

    def test_microphone_opened_once(self):
        from jarvis.audio import MicrophoneStream
        source = FakeSource()
        loop = _loop(FakeMultiEngine(), microphone=MicrophoneStream(source=source))
        assert loop.microphone.start() is True
        loop.microphone.stop()
        assert source.opened == 1

    def test_wake_drives_listening_lifecycle(self):
        loop = _loop(FakeMultiEngine())
        assert loop.machine.state is State.IDLE
        loop.on_wake()
        assert loop.machine.state is State.LISTENING

    def test_shutdown_cleans_up(self):
        engine = FakeMultiEngine()
        loop = _loop(engine)
        assert loop.start() is True
        loop.stop()
        assert engine.closes >= 1
        assert loop.machine.state is State.IDLE
        assert loop.is_running() is False

    def test_init_failure_disables_voice_safely(self):
        from jarvis.audio import MicrophoneStream
        loop = _loop(FakeMultiEngine(loaded_ok=False),
                     microphone=MicrophoneStream(source=FakeSource()))
        assert loop.start() is True
        assert loop.wake_enabled is False
        loop.stop()


class TestLogging:
    def test_normal_output_clean(self, capsys):
        engine = FakeMultiEngine(scores=[{"hey jarvis": 0.95}])
        loop = _loop(engine)
        loop.vad = FakeVAD()
        loop._on_frame(_pcm())
        out, err = capsys.readouterr()
        assert "hey jarvis" not in out and "score" not in out

    def test_debug_names_phrase(self, caplog):
        from jarvis.logger import configure_logging
        configure_logging(debug=True)
        try:
            engine = FakeMultiEngine(scores=[{"hey jarvis": 0.95}])
            loop = _loop(engine)
            loop.vad = FakeVAD()
            with caplog.at_level("DEBUG", logger="jarvis"):
                loop._on_frame(_pcm())
            assert any("hey jarvis" in r.message for r in caplog.records)
        finally:
            configure_logging(debug=False)
