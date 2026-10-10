"""Fast local TTS engine (`say`) selection and behavior.

Real `say` calls (macOS, ~1 s) prove the path; failures and threading use
mocks. Chatterbox behavior is covered by test_tts_chatterbox.py.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import speech  # noqa: E402
from jarvis.config import SAY_VOICE, TTS_ENGINE  # noqa: E402


class TestSelection:
    def test_kokoro_is_default_engine(self):
        assert TTS_ENGINE == "kokoro"
        assert SAY_VOICE == "Samantha"

    def test_dispatcher_honors_engine(self, monkeypatch):
        say = MagicMock(return_value=np.zeros(100, dtype=np.float32))
        cb = MagicMock(return_value=np.zeros(100, dtype=np.float32))
        monkeypatch.setattr(speech, "_synthesize_say", say)
        monkeypatch.setattr(speech, "_synthesize_chatterbox", cb)
        monkeypatch.setattr(speech, "TTS_ENGINE", "say")
        speech.synthesize_for_engine("hi")
        assert say.call_count == 1 and cb.call_count == 0
        monkeypatch.setattr(speech, "TTS_ENGINE", "chatterbox")
        speech.synthesize_for_engine("hi")
        assert cb.call_count == 1

    def test_player_default_uses_configured_engine(self, monkeypatch):
        from jarvis.speech_pipeline import SpeechPlayer
        say = MagicMock(return_value=np.zeros(16000, dtype=np.float32))
        monkeypatch.setattr(speech, "_synthesize_say", say)
        monkeypatch.setattr(speech, "TTS_ENGINE", "say")
        player = SpeechPlayer()
        audio = player.synthesize("hi", None)
        assert audio.dtype == np.float32 and audio.ndim == 1
        assert say.call_count == 1

    def test_kokoro_synthesis_uses_pipeline_sample_rate_and_float32(self, monkeypatch):
        class Pipeline:
            sample_rate = 24000

            def __call__(self, text, voice, speed):
                assert text == "Hello."
                assert voice == speech.KOKORO_VOICE
                assert speed == speech.KOKORO_SPEED
                yield "Hello.", None, np.linspace(-0.5, 0.5, 24000, dtype=np.float32)

        monkeypatch.setattr(speech, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(speech, "_get_kokoro_pipeline", lambda: Pipeline())
        audio = speech.synthesize_for_engine("Hello.")
        assert audio.dtype == np.float32
        assert len(audio) == 16000  # 24 kHz one-second input converted to 16 kHz
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) <= 1.0

    def test_kokoro_resampling_antialiases_frequencies_above_output_nyquist(self, monkeypatch):
        source_rate = 24000
        samples = np.sin(2 * np.pi * 10000 * np.arange(source_rate) / source_rate)

        class Pipeline:
            sample_rate = source_rate

            def __call__(self, *args, **kwargs):
                yield "Hello", None, samples.astype(np.float32)

        monkeypatch.setattr(speech, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(speech, "_get_kokoro_pipeline", lambda: Pipeline())
        audio = speech.synthesize_for_engine("Hello.")
        assert len(audio) == 16000
        assert np.sqrt(np.mean(audio ** 2)) < 0.01

    def test_kokoro_unavailable_is_a_visible_synthesis_error(self, monkeypatch):
        monkeypatch.setattr(speech, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(speech, "_get_kokoro_pipeline", lambda: False)
        with pytest.raises(speech.TTSEngineError, match="Install `kokoro>=0.9.4`"):
            speech.synthesize_for_engine("Hello.")

    def test_unsupported_tts_engine_is_rejected(self, monkeypatch):
        monkeypatch.setattr(speech, "TTS_ENGINE", "mystery")
        monkeypatch.setattr(speech, "_synthesize_chatterbox",
                            lambda *args, **kwargs: pytest.fail("unexpected fallback"))
        with pytest.raises(speech.TTSEngineError, match="Unsupported TTS engine"):
            speech.synthesize_for_engine("Hello.")

    @pytest.mark.parametrize("samples", [np.array([], dtype=np.float32),
                                           np.array([np.nan], dtype=np.float32),
                                           np.array([1.1], dtype=np.float32)])
    def test_synthesis_rejects_empty_or_invalid_waveforms(self, monkeypatch, samples):
        monkeypatch.setattr(speech, "TTS_ENGINE", "kokoro")

        class Pipeline:
            sample_rate = 16000

            def __call__(self, *args, **kwargs):
                yield "Hello", None, samples

        monkeypatch.setattr(speech, "_get_kokoro_pipeline", lambda: Pipeline())
        with pytest.raises(speech.TTSEngineError):
            speech.synthesize_for_engine("Hello.")


class TestSaySynthesis:
    @staticmethod
    def _mock_say(monkeypatch, *, samples=22050):
        import soundfile

        monkeypatch.setattr("subprocess.run", MagicMock())
        monkeypatch.setattr(
            soundfile, "read",
            lambda *args, **kwargs: (np.ones((samples, 1), dtype=np.float32), 22050),
        )

    def test_returns_16k_float32(self):
        with patch("subprocess.run"), patch("soundfile.read", return_value=(
            np.ones((22050, 1), dtype=np.float32), 22050
        )):
            audio = speech._synthesize_say("Hi.")
        assert isinstance(audio, np.ndarray) and audio.ndim == 1
        assert audio.dtype == np.float32 and len(audio) > 1000

    def test_empty_text_rejected_without_subprocess(self):
        with patch("subprocess.run", autospec=True) as mocked:
            with pytest.raises(speech.TTSEngineError):
                speech._synthesize_say("   ")
        mocked.assert_not_called()

    def test_missing_binary_is_clean_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("no say")):
            with pytest.raises(speech.TTSEngineError, match="not found"):
                speech._synthesize_say("Hi.")

    def test_say_launch_uses_posix_spawn_from_threaded_audio_process(self, monkeypatch):
        import subprocess
        import threading

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/true")
        spawned = []
        posix_spawn = subprocess.Popen._posix_spawn

        def observe_spawn(self, *args, **kwargs):
            spawned.append(True)
            return posix_spawn(self, *args, **kwargs)

        monkeypatch.setattr(subprocess.Popen, "_posix_spawn", observe_spawn)
        monkeypatch.setattr(
            subprocess, "_fork_exec",
            lambda *args, **kwargs: pytest.fail("unsafe fork path was selected"),
        )
        errors = []

        def launch():
            try:
                speech._run_say([], check=True, capture_output=True)
            except Exception as exc:  # noqa: BLE001 - surface worker failures
                errors.append(exc)

        worker = threading.Thread(target=launch, name="synthetic-voice-turn")
        worker.start()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert not errors
        assert spawned == [True]

    def test_temp_files_cleaned(self, monkeypatch):
        import glob
        self._mock_say(monkeypatch)
        speech._synthesize_say("Hi.")
        assert glob.glob(str(Path(tempfile.gettempdir()) / "jarvis-say-*.aiff")) == []

    def test_concurrent_synthesis(self, monkeypatch):
        import threading
        self._mock_say(monkeypatch)
        out, errors = [], []

        def _go():
            try:
                out.append(speech._synthesize_say("Hi there."))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_go) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors and len(out) == 3
        assert all(a.dtype == np.float32 and a.ndim == 1 for a in out)

    def test_trace_spans(self, monkeypatch):
        from jarvis.trace import new_trace, use_trace
        self._mock_say(monkeypatch)
        trace = new_trace()
        with use_trace(trace):
            speech._synthesize_say("Hi.")
        names = [s.name for s in trace.spans]
        assert "tts_generate" in names
        assert "tts_ready" not in names  # stateless engine: nothing to ready

    def test_empty_output_is_a_clear_engine_error(self):
        with patch("subprocess.run"), patch("soundfile.read", return_value=(
            np.zeros((0, 1), dtype=np.float32), 22050
        )), pytest.raises(speech.TTSEngineError, match="empty audio"):
            speech._synthesize_say("Hi.")


class TestSpeakFallback:
    def test_speak_uses_say_engine(self, monkeypatch, capsys):
        monkeypatch.setattr(speech, "TTS_ENGINE", "say")
        monkeypatch.setattr(speech, "_speak_say", lambda text: True)
        speech.speak("Hello.")
        out, _ = capsys.readouterr()
        assert "[JARVIS]: Hello." in out

    def test_say_failure_falls_back_to_console(self, monkeypatch, capsys):
        monkeypatch.setattr(speech, "TTS_ENGINE", "say")
        monkeypatch.setattr(speech, "_speak_say", lambda text: False)
        speech.speak("Hello.")  # must not raise; text already shown
        out, _ = capsys.readouterr()
        assert "[JARVIS]: Hello." in out
