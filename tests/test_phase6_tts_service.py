"""Phase 6 tests: persistent warm Chatterbox service.

The real model is never loaded here (see scripts/tts_benchmark.py for live
numbers). A fake model proves lifecycle semantics: load-once, prepare-once,
warm-once, inference context, failure handling, and concurrent-init safety.
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import speech  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_service():
    speech._chatterbox_model = None
    speech._chatterbox_conds_key = None
    speech._chatterbox_warmed = False
    yield
    speech._chatterbox_model = None
    speech._chatterbox_conds_key = None
    speech._chatterbox_warmed = False


def _fake_model():
    model = MagicMock()
    model.sr = 24000
    model.generate.return_value = np.zeros((1, 2400), dtype=np.float32)
    return model


def _inference_active() -> bool:
    import torch
    probe = getattr(torch, "is_inference_mode_enabled", None)
    return bool(probe()) if probe else not torch.is_grad_enabled()


def _real_generate_calls(model):
    """Synthesis generations, excluding the warm-up call."""
    return [c for c in model.generate.call_args_list
            if c[0] and c[0][0] != speech._CHATTERBOX_WARMUP_TEXT]


class TestModelReuse:
    def test_model_constructed_once_across_syntheses(self):
        model = _fake_model()
        with patch.dict(sys.modules, {"chatterbox": MagicMock(),
                                      "chatterbox.tts": MagicMock()}):
            with patch.object(speech, "_get_chatterbox_model",
                              wraps=speech._get_chatterbox_model) as get:
                with patch("chatterbox.tts.ChatterboxTTS.from_pretrained",
                           return_value=model) as load:
                    for _ in range(3):
                        speech._synthesize_chatterbox("Hello.")
        assert load.call_count == 1
        assert get.call_count >= 3  # asked often, built once

    def test_repeated_ensure_is_safe(self):
        model = _fake_model()
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            first = speech.ensure_chatterbox_ready()
            second = speech.ensure_chatterbox_ready()
        assert first is second is model

    def test_concurrent_init_builds_once(self):
        model = _fake_model()
        built = []
        real_pretrained = lambda **kw: (built.append(1), model)[1]  # noqa: E731
        with patch.dict(sys.modules, {"chatterbox": MagicMock(),
                                      "chatterbox.tts": MagicMock()}):
            with patch("chatterbox.tts.ChatterboxTTS.from_pretrained",
                       side_effect=real_pretrained):
                threads = [threading.Thread(target=speech._get_chatterbox_model)
                           for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        assert len(built) == 1


class TestReferenceCache:
    def test_prepare_once_across_syntheses(self):
        model = _fake_model()
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            speech._synthesize_chatterbox("One.")
            speech._synthesize_chatterbox("Two.")
            first = speech.prepare_chatterbox_reference(model)
            second = speech.prepare_chatterbox_reference(model)
        assert model.prepare_conditionals.call_count == 1
        assert first > 0.0 or first == 0.0  # timed, non-negative
        assert second == 0.0  # cache reused

    def test_approved_voice_preserved(self):
        model = _fake_model()
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            speech._synthesize_chatterbox("Hello.")
        prepared = model.prepare_conditionals.call_args[0][0]
        assert prepared == str(PROJECT_ROOT / "chatterbox_emotion_test.wav")

    def test_generation_params_unchanged(self):
        from jarvis.config import (CHATTERBOX_CFG_WEIGHT, CHATTERBOX_EXAGGERATION,
                                   CHATTERBOX_TEMPERATURE)
        model = _fake_model()
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            speech._synthesize_chatterbox("Hello.")
        kwargs = _real_generate_calls(model)[0].kwargs
        assert kwargs["exaggeration"] == CHATTERBOX_EXAGGERATION == 0.5
        assert kwargs["cfg_weight"] == CHATTERBOX_CFG_WEIGHT == 0.5
        assert kwargs["temperature"] == CHATTERBOX_TEMPERATURE == 0.8


class TestWarmUp:
    def test_warm_up_runs_once(self):
        model = _fake_model()
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            speech._synthesize_chatterbox("Hello.")
            speech._synthesize_chatterbox("Again.")
        warms = [c for c in model.generate.call_args_list
                 if c[0] and c[0][0] == speech._CHATTERBOX_WARMUP_TEXT]
        assert len(warms) == 1
        assert len(_real_generate_calls(model)) == 2

    def test_failed_warm_up_does_not_block_speech(self, caplog):
        model = _fake_model()
        model.generate.side_effect = [RuntimeError("mps hiccup"),
                                      np.zeros((1, 2400), dtype=np.float32)]
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            with caplog.at_level("WARNING", logger="jarvis"):
                audio = speech._synthesize_chatterbox("Hello.")
        assert audio.shape == (2400,)
        assert any("warm-up failed" in r.message for r in caplog.records)
        assert speech._chatterbox_warmed is False  # retried next time


class TestInferenceContext:
    def test_generate_runs_under_inference_mode(self):
        model = _fake_model()
        seen = []

        def _recording_generate(text, **kwargs):
            seen.append(_inference_active())
            return np.zeros((1, 2400), dtype=np.float32)

        model.generate.side_effect = _recording_generate
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            speech._synthesize_chatterbox("Hello.")
        assert seen and all(seen)


class TestFailures:
    def test_generation_failure_reported_not_silent(self, caplog):
        model = _fake_model()
        model.generate.side_effect = RuntimeError("cuda oom")
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            with caplog.at_level("WARNING", logger="jarvis"):
                with pytest.raises(speech.TTSEngineError) as exc:
                    speech._synthesize_chatterbox("Hello.")
        assert "cuda oom" in str(exc.value)

    def test_service_recoverable_after_failure(self):
        model = _fake_model()
        model.generate.side_effect = RuntimeError("boom")
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            with pytest.raises(speech.TTSEngineError):
                speech._synthesize_chatterbox("Hello.")
            model.generate.side_effect = None
            model.generate.return_value = np.zeros((1, 2400), dtype=np.float32)
            assert speech._synthesize_chatterbox("Hello.").shape == (2400,)

    def test_init_failure_marks_nothing_ready(self):
        with patch.dict(sys.modules, {"chatterbox": MagicMock(),
                                      "chatterbox.tts": MagicMock()}):
            with patch("chatterbox.tts.ChatterboxTTS.from_pretrained",
                       side_effect=RuntimeError("no weights")):
                with pytest.raises(speech.TTSEngineError):
                    speech.ensure_chatterbox_ready()
        assert speech._chatterbox_model is None
        assert speech._chatterbox_warmed is False
        assert speech._chatterbox_conds_key is None
