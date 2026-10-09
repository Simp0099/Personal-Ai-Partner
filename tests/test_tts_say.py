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
    def test_say_is_default_engine(self):
        assert TTS_ENGINE == "say"  # changed from chatterbox: speed (see docs)
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
        assert player.synthesize("hi", None) is say.return_value
        assert say.call_count == 1


class TestSaySynthesis:
    def test_returns_16k_float32(self):
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

    def test_temp_files_cleaned(self):
        import glob
        speech._synthesize_say("Hi.")
        assert glob.glob(str(Path(tempfile.gettempdir()) / "jarvis-say-*.aiff")) == []

    def test_concurrent_synthesis(self):
        import threading
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

    def test_trace_spans(self):
        from jarvis.trace import new_trace, use_trace
        trace = new_trace()
        with use_trace(trace):
            speech._synthesize_say("Hi.")
        names = [s.name for s in trace.spans]
        assert "tts_generate" in names
        assert "tts_ready" not in names  # stateless engine: nothing to ready


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
