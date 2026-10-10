from types import SimpleNamespace
from unittest.mock import patch

import start


def _layer(configured):
    provider = SimpleNamespace(is_configured=lambda: configured)
    model = SimpleNamespace(key="openrouter-model", provider="openrouter")
    return SimpleNamespace(
        registry=SimpleNamespace(enabled=lambda: [model]),
        providers={"openrouter": provider},
    )


def test_startup_accepts_configured_non_gemini_provider(capsys):
    with patch("jarvis.model_layer.ModelLayer.from_config", return_value=_layer(True)):
        assert start.check_environment() is True
    assert "1 usable" in capsys.readouterr().out


def test_startup_rejects_when_no_enabled_provider_is_configured(capsys):
    with patch("jarvis.model_layer.ModelLayer.from_config", return_value=_layer(False)):
        assert start.check_environment() is False
    assert "No enabled model" in capsys.readouterr().out


def test_disabled_voice_falls_back_to_text_without_starting_audio(monkeypatch):
    import jarvis.main as main

    start_voice = patch.object(main, "start_voice", return_value=None)
    text_mode = patch.object(main, "run_text_mode")
    with start_voice as voice, text_mode as typed:
        main.run_wake_word_mode()
    voice.assert_called_once()
    typed.assert_called_once()
