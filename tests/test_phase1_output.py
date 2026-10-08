"""Phase 1 regression tests: user output separated from diagnostics.

Normal mode keeps the terminal to user-facing output only; the "jarvis"
logger's console handler stays silent and the rotating file keeps DEBUG+.
--debug lowers the console handler to DEBUG. All provider interaction here
uses deterministic mock providers: no network, no quota, no secrets.
"""

import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jarvis.brain as brain_module  # noqa: E402
from jarvis.brain import BrainError, JarvisBrain  # noqa: E402
from jarvis.logger import configure_logging, debug_mode, logger  # noqa: E402
from jarvis.providers.base import ErrorKind, ProviderError  # noqa: E402
from tests.mock_providers import build_layer, error, reply  # noqa: E402


def _single_model_layer(behaviour, key="m1", model="m"):
    return build_layer(
        models=[{"key": key, "model": model, "priority": 90,
                 "capabilities": {"reasoning": True, "tool_calling": True}}],
        behaviours={model: behaviour},
    )


@pytest.fixture()
def _output_state():
    """Snapshot global logging/CLI state; restore after each test."""
    saved_handlers = list(logger.handlers)
    import logging.handlers as _lh
    saved_console_level = None
    for h in saved_handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, _lh.RotatingFileHandler):
            saved_console_level = h.level
    saved_debug_input = brain_module.DEBUG_INPUT
    import jarvis.logger as logger_module
    saved_debug = logger_module._DEBUG
    yield
    for h in list(logger.handlers):
        if h not in saved_handlers:
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    for h in saved_handlers:
        if h not in logger.handlers:
            logger.addHandler(h)
    if saved_console_level is not None:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, _lh.RotatingFileHandler):
                h.setLevel(saved_console_level)
    brain_module.DEBUG_INPUT = saved_debug_input
    logger_module._DEBUG = saved_debug


def _brain_answer(text="It is 12:00."):
    layer = _single_model_layer(reply(text))
    brain = JarvisBrain(model_layer=layer)
    return brain


DIAGNOSTIC_TOKENS = ("[Thinking", "[Tool]", "[ROUTING]", "[MODEL OUTPUT]",
                      "rate_limit", "429", "Traceback", "INFO")


class TestNormalMode:
    def test_success_shows_no_diagnostics(self, _output_state, capsys):
        configure_logging(debug=False)
        brain = _brain_answer()
        assert brain.ask("What time is it?") == "It is 12:00."
        out, err = capsys.readouterr()
        for token in DIAGNOSTIC_TOKENS + ("provider", "model"):
            assert token not in out and token not in err

    def test_provider_error_stays_concise(self, _output_state, capsys):
        configure_logging(debug=False)
        layer = _single_model_layer(
            error(ErrorKind.RATE_LIMIT, "HTTP 429 Too Many Requests: quota hit"))
        brain = JarvisBrain(model_layer=layer)
        with pytest.raises(BrainError) as excinfo:
            brain.ask("What time is it?")
        assert "429" not in excinfo.value.message  # user text stays concise
        assert "429" in excinfo.value.detail  # raw detail preserved for logs
        out, err = capsys.readouterr()
        assert "429" not in out and "429" not in err
        assert "Traceback" not in out and "Traceback" not in err

    def test_text_mode_error_path_logs_detail_prints_concise(
            self, _output_state, tmp_path, capsys):
        """Mirror jarvis/main.py text-mode handling: detail logged, user brief."""
        configure_logging(debug=False, log_file=tmp_path / "phase1.log")
        try:
            raise BrainError("The AI provider's usage limit has been reached. "
                             "Please try again later.",
                             detail="m1: rate_limit (HTTP 429 quota hit)")
        except BrainError as e:
            logger.error(f"Model unavailable: {e.detail or e}")
            print(f"[Error]: {e.message}")
        for h in logger.handlers:
            h.flush()
        out, _ = capsys.readouterr()
        assert "429" not in out  # terminal stays clean
        assert "[Error]:" in out  # user still told
        logged = (tmp_path / "phase1.log").read_text()
        assert "429" in logged  # file keeps the raw detail


class TestDebugMode:
    def test_debug_shows_diagnostics_and_response(self, _output_state, capsys):
        configure_logging(debug=True)
        brain = _brain_answer()
        assert brain.ask("What time is it?") == "It is 12:00."
        out, _ = capsys.readouterr()
        assert "[Thinking" in out  # StatusIndicator diagnostics visible
        assert "429" not in out

    def test_debug_flag_wires_cli(self, _output_state, monkeypatch, capsys):
        import start
        monkeypatch.setattr(sys, "argv", ["jarvis", "--status", "--debug"])
        start.main()
        import jarvis.logger as logger_module
        assert logger_module.debug_mode() is True
        assert brain_module.DEBUG_INPUT is True
        monkeypatch.setattr(sys, "argv", ["jarvis", "--status"])
        start.main()
        assert logger_module.debug_mode() is False
        capsys.readouterr()


class TestLogFile:
    def test_diagnostics_persist_while_terminal_stays_clean(
            self, _output_state, tmp_path, capsys):
        log_file = tmp_path / "phase1.log"
        configure_logging(debug=False, log_file=log_file)
        brain = _brain_answer()
        brain.ask("What time is it?")
        for h in logger.handlers:
            h.flush()
        out, err = capsys.readouterr()
        assert "[Thinking" not in out and "[Thinking" not in err
        assert "[Thinking" in log_file.read_text()


class TestNoDuplicateHandlers:
    def test_repeated_configuration_keeps_single_handlers(
            self, _output_state, tmp_path, capsys):
        import logging.handlers as _lh
        configure_logging(debug=True)
        configure_logging(debug=False)
        configure_logging(debug=True, log_file=tmp_path / "a.log")
        configure_logging(debug=True, log_file=tmp_path / "a.log")
        consoles = [h for h in logger.handlers
                    if isinstance(h, logging.StreamHandler)
                    and not isinstance(h, _lh.RotatingFileHandler)]
        files = [h for h in logger.handlers if isinstance(h, _lh.RotatingFileHandler)]
        assert len(consoles) == 1 and len(files) == 1
        logger.info("phase1-once-marker")
        for h in logger.handlers:
            h.flush()
        out, _ = capsys.readouterr()
        assert out.count("phase1-once-marker") == 1


class TestSecretRedaction:
    def test_key_shaped_detail_never_leaves_redacted(self, _output_state, tmp_path):
        configure_logging(debug=True, log_file=tmp_path / "phase1.log")
        err = ProviderError(ErrorKind.AUTH, "m",
                            detail="rejected key sk-or-v1-SECRETXYZ123 here",
                            provider="mock")
        assert "SECRETXYZ123" not in err.safe_detail
        assert "[REDACTED]" in err.safe_detail
        logger.error(f"auth failure: {err.safe_detail}")
        for h in logger.handlers:
            h.flush()
        logged = (tmp_path / "phase1.log").read_text()
        assert "SECRETXYZ123" not in logged


class TestRoutingUnchanged:
    def test_fallback_still_serves_backup(self, _output_state, capsys):
        """Phase 1 must not alter routing: a failed first model falls back."""
        from tests.mock_providers import fail_then_success
        layer = build_layer(
            models=[
                {"key": "primary", "model": "m1", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "backup", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"m1": error(ErrorKind.SERVER, "503"),
                        "m2": reply("backup ok")},
        )
        brain = JarvisBrain(model_layer=layer)
        configure_logging(debug=False)
        assert brain.ask("Why is the sky blue?") == "backup ok"
        assert brain.last_model_key == "backup"
        capsys.readouterr()
