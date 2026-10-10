"""Phase 3 tests: safe allowlisted application launching.

Providers are boom-mocked: ANY provider contact raises AssertionError, so
every test below proves the local path. subprocess is mocked to prove
argument-list execution without a shell and without touching the real OS.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jarvis.config as config_module  # noqa: E402
from jarvis.brain import TOOL_REGISTRY, JarvisBrain  # noqa: E402
import jarvis.brain as brain_module  # noqa: E402
from jarvis.intents import handle  # noqa: E402
from jarvis.logger import configure_logging  # noqa: E402
from jarvis.tools import apps  # noqa: E402
from tests.mock_providers import build_layer, reply  # noqa: E402


def _boom(_history, _payload):
    raise AssertionError("Provider must not be called for local app launch")


def _blocking_layer():
    return build_layer(
        models=[{"key": "m1", "model": "m", "priority": 90,
                 "capabilities": {"reasoning": True, "tool_calling": True}}],
        behaviours={"m": _boom},
    )


@pytest.fixture()
def _quiet():
    configure_logging(debug=False)
    yield
    configure_logging(debug=False)


@pytest.fixture()
def _allowlist(monkeypatch):
    """Allowlist with two apps; restored afterwards."""
    monkeypatch.setattr(config_module, "ALLOWED_APPS", ["WhatsApp", "Safari"])
    monkeypatch.setattr(apps, "ALLOWED_APPS", ["WhatsApp", "Safari"])
    return ["WhatsApp", "Safari"]


def _ask_blocked(text):
    brain = JarvisBrain(model_layer=_blocking_layer())
    return brain, brain.ask(text)


class _Done(subprocess.CompletedProcess):
    def __init__(self, rc=0, err=""):
        super().__init__(args=["open"], returncode=rc, stdout="", stderr=err)


# ============================================================================
# 19.1 allowlisted success — subprocess, args, no provider
# ============================================================================

class TestAllowlistedLaunch:
    def test_success(self, _quiet, _allowlist, monkeypatch):
        seen = []
        monkeypatch.setattr(apps.subprocess, "run",
                            lambda argv, **kw: seen.append((argv, kw)) or _Done())
        _, response = _ask_blocked("Open WhatsApp")
        assert response == "WhatsApp is now open."
        assert len(seen) == 1
        argv, kwargs = seen[0]
        assert argv == ["open", "-a", "WhatsApp"]
        assert kwargs.get("shell") is not True and "shell" not in kwargs

    def test_case_normalization(self, _quiet, _allowlist, monkeypatch):
        monkeypatch.setattr(apps.subprocess, "run", lambda argv, **kw: _Done())
        _, response = _ask_blocked("open whatsapp")
        assert response == "WhatsApp is now open."

    def test_registry_slot_is_real_wrapper(self):
        assert TOOL_REGISTRY["launch_app"] is brain_module.launch_app


# ============================================================================
# 19.3/19.4/19.7 failures — local errors, no subprocess/provider where due
# ============================================================================

class TestLocalFailures:
    def test_unknown_app_rejected(self, _quiet, _allowlist, monkeypatch):
        calls = []
        monkeypatch.setattr(apps.subprocess, "run",
                            lambda argv, **kw: calls.append(argv) or _Done())
        _, response = _ask_blocked("Open RandomApp")
        assert calls == []
        assert "RandomApp" in response and "approved" in response

    def test_not_installed_is_local_error(self, _quiet, _allowlist, monkeypatch):
        monkeypatch.setattr(apps.subprocess, "run",
                            lambda argv, **kw: _Done(rc=1, err="No such app"))
        _, response = _ask_blocked("Launch Safari")
        assert "isn't installed" in response

    def test_empty_input_rejected(self, _quiet, _allowlist, monkeypatch):
        calls = []
        monkeypatch.setattr(apps.subprocess, "run",
                            lambda argv, **kw: calls.append(argv) or _Done())
        assert "need an application name" in apps.launch_app("")
        assert "need an application name" in apps.launch_app("   ")
        assert "need an application name" in apps.launch_app(None)
        assert calls == []

    def test_missing_or_empty_allowlist_denies_all(
            self, _quiet, monkeypatch):
        calls = []
        monkeypatch.setattr(apps.subprocess, "run",
                            lambda argv, **kw: calls.append(argv) or _Done())
        for empty in ([], None, "WhatsApp", [123, None]):
            monkeypatch.setattr(apps, "ALLOWED_APPS", empty)
            response = apps.launch_app("WhatsApp")
            assert "approved" in response or "need an application" in response
        assert calls == []


# ============================================================================
# 19.5 shell safety + 19.6 injection
# ============================================================================

class TestShellSafety:
    def test_argv_list_no_shell(self, _quiet, _allowlist, monkeypatch):
        seen = []

        def _spy(argv, **kwargs):
            seen.append((argv, kwargs))
            assert isinstance(argv, list)
            assert "shell" not in kwargs
            return _Done()

        monkeypatch.setattr(apps.subprocess, "run", _spy)
        apps.launch_app("Safari")
        assert seen[0][0][:2] == ["open", "-a"]

    @pytest.mark.parametrize("evil", [
        "WhatsApp; rm -rf ~", "WhatsApp && echo hacked", "$(whoami)",
        "`whoami`", "../../something", "WhatsApp | cat /etc/passwd",
    ])
    def test_injection_rejected_at_allowlist(
            self, _quiet, _allowlist, monkeypatch, evil):
        calls = []
        monkeypatch.setattr(apps.subprocess, "run",
                            lambda argv, **kw: calls.append(argv) or _Done())
        _, response = _ask_blocked(f"open {evil}")
        assert calls == []  # never reached the OS
        assert "approved" in response

    def test_even_allowlisted_evil_runs_without_shell(
            self, _quiet, monkeypatch):
        """Belt and braces: if it IS allowlisted, arg-list still can't run it."""
        weird = "Weird; rm -rf ~"
        monkeypatch.setattr(apps, "ALLOWED_APPS", [weird])
        seen = []
        monkeypatch.setattr(apps.subprocess, "run",
                            lambda argv, **kw: seen.append((argv, kw)) or _Done())
        apps.launch_app(weird)
        argv, kwargs = seen[0]
        assert argv == ["open", "-a", weird] and "shell" not in kwargs


# ============================================================================
# Config validation
# ============================================================================

class TestAllowlistConfig:
    def test_validation(self):
        v = config_module._validated_app_allowlist
        assert v(["A", "b", "", "  ", None, 7, "A"]) == ["A", "b"]
        assert v("nope") == [] and v(None) == [] and v({}) == []
        assert v([]) == []

    def test_empty_by_default(self):
        assert config_module.ALLOWED_APPS == []


# ============================================================================
# Logging separation (Phase 1 intact)
# ============================================================================

class TestLogging:
    def test_normal_terminal_clean(self, _quiet, _allowlist, monkeypatch, capsys):
        monkeypatch.setattr(apps.subprocess, "run", lambda argv, **kw: _Done())
        _ask_blocked("Open WhatsApp")
        out, err = capsys.readouterr()
        assert out == "" and err == ""

    def test_debug_names_launch(self, _allowlist, monkeypatch, caplog):
        configure_logging(debug=True)
        try:
            monkeypatch.setattr(apps.subprocess, "run", lambda argv, **kw: _Done())
            with caplog.at_level("DEBUG", logger="jarvis"):
                _ask_blocked("Launch Safari")
            assert any("launch_app" in r.message and "skipped" in r.message
                       for r in caplog.records)
        finally:
            configure_logging(debug=False)
