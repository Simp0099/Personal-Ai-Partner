"""Tests for the Chatterbox TTS engine.

Covers configuration, reference-audio resolution, model lifecycle, synthesis
format, and the assistant -> TTS seam. The expensive Chatterbox model is stubbed;
a real-inference smoke test lives in scripts/tts_smoke_test.py.

Run with:
    python3 -m pytest tests/test_tts_chatterbox.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Pre-import torch (when present) OUTSIDE any mock block. Several tests below
# use patch.dict(sys.modules, ...) which evicts every module imported inside
# the block on exit — re-executing torch's native init segfaults. Importing it
# once here keeps it resident. Absence is fine: speech.py falls back to "cpu".
try:
    import torch  # noqa: F401
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import speech  # noqa: E402
from jarvis.config import CHATTERBOX_REFERENCE_AUDIO, TTS_ENGINE  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_chatterbox_cache():
    """Keep the module-level model cache from leaking between tests."""
    speech._chatterbox_model = None
    speech._chatterbox_conds_key = None
    speech._chatterbox_warmed = False
    yield
    speech._chatterbox_model = None
    speech._chatterbox_conds_key = None
    speech._chatterbox_warmed = False


def _fake_model():
    """A ChatterboxTTS stand-in that returns a plausible waveform."""
    model = MagicMock()
    model.sr = 24000
    model.generate.return_value = np.zeros((1, 2400), dtype=np.float32)
    return model


class TestChatterboxConfiguration:
    """1. Chatterbox is the active/default engine."""

    def test_chatterbox_is_default_engine(self):
        assert TTS_ENGINE == "chatterbox"

    def test_reference_audio_points_at_emotion_test_wav(self):
        assert CHATTERBOX_REFERENCE_AUDIO == "chatterbox_emotion_test.wav"

    def test_no_hardcoded_absolute_path(self):
        """The macOS-specific path must not leak into the source."""
        source = (PROJECT_ROOT / "jarvis" / "speech.py").read_text()
        assert "/Users/" not in source

    def test_no_alternate_reference_assets_are_referenced(self):
        """Only the approved WAV may be named anywhere in the TTS source."""
        source = (PROJECT_ROOT / "jarvis" / "speech.py").read_text()
        assert "jarvis_dominant" not in source
        assert "chatterbox_reference.mp3" not in source
        assert "turbo" not in source.lower()

    def test_no_silent_kokoro_fallback(self):
        """Kokoro must never be reached from the Chatterbox branch."""
        source = (PROJECT_ROOT / "jarvis" / "speech.py").read_text()
        chatterbox_branch = source.split('if TTS_ENGINE == "chatterbox":')[1]
        chatterbox_branch = chatterbox_branch.split("elif")[0]
        assert "_speak_kokoro" not in chatterbox_branch

    def test_config_has_single_reference_key(self):
        """One authoritative reference setting in config.yaml."""
        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
        reference_keys = [k for k in config["speech"] if "reference" in k]
        assert reference_keys == ["chatterbox_reference_audio"]

    def test_no_env_var_can_override_the_reference(self):
        """The reference comes only from config.yaml, so it cannot drift silently."""
        source = (PROJECT_ROOT / "jarvis" / "config.py").read_text()
        assert 'CHATTERBOX_REFERENCE_AUDIO = _SPEECH.get("chatterbox_reference_audio"' in source
        assert not any(
            "os.getenv" in line for line in source.splitlines()
            if "CHATTERBOX_REFERENCE_AUDIO" in line
        )

    def test_alternate_assets_exist_but_are_unreferenced(self):
        """Leftover voice files sit in the repo root; nothing points at them."""
        repo_root_files = [p.name for p in PROJECT_ROOT.glob("*.wav")]
        assert "chatterbox_emotion_test.wav" in repo_root_files
        source = (PROJECT_ROOT / "jarvis" / "speech.py").read_text()
        for name in repo_root_files:
            if name != "chatterbox_emotion_test.wav":
                assert name not in source


class TestReferenceResolution:
    """2. Reference path resolution. 3. Missing reference audio."""

    def test_resolves_relative_to_project_root(self):
        resolved = speech._resolve_chatterbox_reference()
        assert resolved == PROJECT_ROOT / "chatterbox_emotion_test.wav"
        assert resolved.is_file(), "reference WAV must ship with the repository"

    def test_absolute_path_is_used_as_is(self, tmp_path):
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF")
        with patch.object(speech, "CHATTERBOX_REFERENCE_AUDIO", str(wav)):
            assert speech._resolve_chatterbox_reference() == wav

    def test_missing_reference_raises_clear_error(self, tmp_path):
        with patch.object(speech, "CHATTERBOX_REFERENCE_AUDIO", str(tmp_path / "nope.wav")):
            with pytest.raises(speech.TTSEngineError) as exc:
                speech._resolve_chatterbox_reference()
        assert "reference audio not found" in str(exc.value)
        assert "nope.wav" in str(exc.value)


class TestModelLifecycle:
    """4. Model initialization. 5. Model reuse across requests."""

    def test_model_initialized_with_device(self):
        model = _fake_model()
        with patch.dict(sys.modules, {"chatterbox": MagicMock(), "chatterbox.tts": MagicMock()}):
            with patch("chatterbox.tts.ChatterboxTTS.from_pretrained", return_value=model) as load:
                speech._get_chatterbox_model()

        load.assert_called_once()
        assert load.call_args.kwargs["device"] == speech._chatterbox_device()

    def test_missing_reference_fails_before_loading_weights(self):
        """A missing reference must not trigger a model download first."""
        with patch.dict(sys.modules, {"chatterbox": MagicMock(), "chatterbox.tts": MagicMock()}):
            with patch("chatterbox.tts.ChatterboxTTS.from_pretrained") as load:
                with patch.object(speech, "CHATTERBOX_REFERENCE_AUDIO", "does_not_exist.wav"):
                    with pytest.raises(speech.TTSEngineError):
                        speech._get_chatterbox_model()
        assert load.call_count == 0

    def test_model_is_reused_not_reloaded(self):
        model = _fake_model()
        with patch.dict(sys.modules, {"chatterbox": MagicMock(), "chatterbox.tts": MagicMock()}):
            with patch("chatterbox.tts.ChatterboxTTS.from_pretrained", return_value=model) as load:
                speech._get_chatterbox_model()
                speech._get_chatterbox_model()
                speech._get_chatterbox_model()

        assert load.call_count == 1

    def test_missing_dependency_names_the_package(self):
        speech._chatterbox_model = None
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("chatterbox"):
                raise ImportError("no module named chatterbox")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", fake_import):
            with pytest.raises(speech.TTSEngineError) as exc:
                speech._get_chatterbox_model()
        assert "chatterbox-tts" in str(exc.value)

    def test_unreadable_reference_names_the_file(self):
        """A reference the engine cannot read must not be swallowed."""
        model = _fake_model()
        model.generate.side_effect = RuntimeError("bad wav")
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            with pytest.raises(speech.TTSEngineError) as exc:
                speech._synthesize_chatterbox("Hello.")
        assert "chatterbox_emotion_test.wav" in str(exc.value)
        assert "bad wav" in str(exc.value)


class TestSynthesis:
    """6. Synthesis calls. 7. Audio format matches the playback layer."""

    def test_resolved_reference_reaches_chatterbox_generate(self):
        """The approved WAV is prepared once; generations reuse it (Phase 6).

        Per-call `audio_prompt_path` was the old contract: the library
        re-preprocessed the reference on every request. The service now
        prepares conditionals once and generates without the argument.
        """
        model = _fake_model()
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            speech._synthesize_chatterbox("Hello. I am online.")

        model.prepare_conditionals.assert_called_once()
        prepared = model.prepare_conditionals.call_args[0][0]
        assert prepared == str(PROJECT_ROOT / "chatterbox_emotion_test.wav")
        assert Path(prepared).is_file()
        for call in model.generate.call_args_list:
            assert "audio_prompt_path" not in call.kwargs

    def test_every_request_uses_the_approved_reference(self):
        """Repeated speech prepares once and never drifts voices (Phase 6)."""
        model = _fake_model()
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            for _ in range(3):
                speech._synthesize_chatterbox("Hello.")
        assert model.prepare_conditionals.call_count == 1
        prepared = model.prepare_conditionals.call_args[0][0]
        assert prepared == str(PROJECT_ROOT / "chatterbox_emotion_test.wav")

    def test_voice_parameters_are_unchanged(self):
        """This task locks the reference, not the synthesis parameters."""
        from jarvis.config import (
            CHATTERBOX_CFG_WEIGHT,
            CHATTERBOX_EXAGGERATION,
            CHATTERBOX_TEMPERATURE,
        )

        model = _fake_model()
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            speech._synthesize_chatterbox("Hello.")

        kwargs = model.generate.call_args.kwargs
        assert kwargs["exaggeration"] == CHATTERBOX_EXAGGERATION == 0.5
        assert kwargs["cfg_weight"] == CHATTERBOX_CFG_WEIGHT == 0.5
        assert kwargs["temperature"] == CHATTERBOX_TEMPERATURE == 0.8
        assert model.generate.call_args[0][0] == "Hello."

    def test_turbo_model_is_never_constructed(self):
        """Only the standard ChatterboxTTS is used."""
        source = (PROJECT_ROOT / "jarvis" / "speech.py").read_text()
        assert "ChatterboxMultilingual" not in source
        assert "ChatterboxTurbo" not in source
        assert "from chatterbox.tts import ChatterboxTTS" in source

    def test_audio_shape_dtype_and_sample_rate(self):
        """Playback uses sounddevice.play(numpy_1d, samplerate)."""
        model = _fake_model()
        model.generate.return_value = np.zeros((1, 48000), dtype=np.float64)
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            audio = speech._synthesize_chatterbox("Hello.")

        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert audio.dtype == np.float32
        assert audio.shape == (48000,)
        assert model.sr == 24000

    def test_playback_reaches_sounddevice_from_a_real_reference(self):
        """End-to-end seam: assistant text -> prepared ref -> sd.play."""
        model = _fake_model()
        sd = MagicMock()
        with patch.dict(sys.modules, {"sounddevice": sd}):
            with patch.object(speech, "_get_chatterbox_model", return_value=model):
                speech.speak("Hey Boss... I'm here. What are we doing?")

        prepared = model.prepare_conditionals.call_args[0][0]
        assert prepared == str(PROJECT_ROOT / "chatterbox_emotion_test.wav")
        sd.play.assert_called_once()
        sd.wait.assert_called_once()

    def test_synthesis_failure_is_wrapped(self):
        model = _fake_model()
        model.generate.side_effect = RuntimeError("cuda oom")
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            with pytest.raises(speech.TTSEngineError) as exc:
                speech._synthesize_chatterbox("Hello.")
        assert "cuda oom" in str(exc.value)

class TestAssistantIntegration:
    """8. The assistant response -> TTS seam still routes through Chatterbox."""

    def test_speak_uses_chatterbox(self):
        with patch.object(speech, "TTS_ENGINE", "chatterbox"), \
             patch.object(speech, "_speak_chatterbox", return_value=True) as cb, \
             patch.object(speech, "_speak_kokoro") as kokoro, \
             patch.object(speech, "_speak_pyttsx3") as pyttsx:
            speech.speak("Assistant response")

        assert cb.call_count == 1
        assert kokoro.call_count == 0
        assert pyttsx.call_count == 0

    def test_speak_console_fallback_only(self):
        """Default failure mode is console text, never a different voice."""
        with patch.object(speech, "TTS_ENGINE", "chatterbox"), \
             patch.object(speech, "_speak_chatterbox", return_value=False), \
             patch.object(speech, "_speak_pyttsx3") as pyttsx:
            speech.speak("Assistant response")
        assert pyttsx.call_count == 0

    def test_explicit_pyttsx3_fallback_is_configurable(self):
        with patch.object(speech, "TTS_ENGINE", "chatterbox"), \
             patch.object(speech, "CHATTERBOX_FALLBACK_ENGINE", "pyttsx3"), \
             patch.object(speech, "_speak_chatterbox", return_value=False), \
             patch.object(speech, "_speak_pyttsx3", return_value=True) as pyttsx:
            speech.speak("Assistant response")
        assert pyttsx.call_count == 1

    def test_kokoro_engine_still_selectable_when_explicit(self):
        """Legacy path is preserved but must be opt-in."""
        with patch.object(speech, "TTS_ENGINE", "kokoro"), \
             patch.object(speech, "_speak_kokoro", return_value=True) as kokoro:
            speech.speak("Assistant response")
        assert kokoro.call_count == 1

    def test_playback_receives_ndarray_and_sample_rate(self):
        """_speak_chatterbox hands sounddevice the array plus the model rate."""
        model = _fake_model()
        sd = MagicMock()
        with patch.object(speech, "_get_chatterbox_model", return_value=model):
            with patch.dict(sys.modules, {"sounddevice": sd}):
                assert speech._speak_chatterbox("Hello.") is True

        sd.play.assert_called_once()
        played, rate = sd.play.call_args[0]
        assert isinstance(played, np.ndarray)
        assert rate == 24000
        sd.wait.assert_called_once()

    def test_main_loop_still_uses_speak_seam(self):
        source = (PROJECT_ROOT / "jarvis" / "main.py").read_text()
        assert "from jarvis.speech import speak" in source
        assert "chatterbox" not in source.lower()
