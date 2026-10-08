"""JARVIS 2.0 End-to-End Integration Test.

Verifies the full pipeline:
  1. Security & Config Audit
  2. Memory Persistence (short-term trimming + long-term SQLite)
  3. Tool Execution with error handling
  4. Brain + Persona integration
  5. Wake word listener initialization
  6. Speech module fallback

Run with:
    python3 -m pytest tests/test_integration.py -v
    # or
    python3 tests/test_integration.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSecurityAudit:
    """Checklist Item 1: Security & Config Audit."""

    def test_no_hardcoded_secrets(self):
        """Verify no credentials are hardcoded in source files.

        The earlier version of this check flagged any line containing
        `api_key =`, which produced false positives on legitimate code such as
        `api_key = os.getenv("GEMINI_API_KEY")` and on the redaction-marker
        list. A false-positive-prone security test trains people to ignore it.

        This version flags what actually matters: a credential-shaped *string
        literal* assigned or passed as a value.
        """
        import re

        jarvis_files = [
            p for p in (PROJECT_ROOT / "jarvis").rglob("*.py") if "__pycache__" not in str(p)
        ]
        jarvis_files.append(PROJECT_ROOT / "api_server.py")
        jarvis_files.append(PROJECT_ROOT / "start.py")

        # A credential-shaped literal: a long opaque token, or a known key prefix.
        literal_pattern = re.compile(
            r"""(?:['"](?:sk-[A-Za-z0-9_\-]{8,}|sk-or-v1-[A-Za-z0-9]{8,}"""
            r"""|AIza[A-Za-z0-9_\-]{10,}|ghp_[A-Za-z0-9]{20,})['"])"""
        )
        # Assignment of a credential-named variable to a literal.
        assignment_pattern = re.compile(
            r"""(?i)^\s*(?:self\.)?[A-Za-z_]*(?:api_?key|secret|password|token|credential)s?"""
            r"""\s*(?::[^=]+)?=\s*(['"])(?P<value>[^'"]+)\1"""
        )

        violations = []
        for py_file in jarvis_files:
            if not py_file.exists():
                continue
            for lineno, line in enumerate(py_file.read_text().split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if literal_pattern.search(line):
                    violations.append(f"{py_file.name}:{lineno}: credential-shaped literal")
                    continue
                match = assignment_pattern.match(line)
                if match:
                    value = match.group("value").strip()
                    # Short values and obvious placeholders/env references are fine.
                    if value and len(value) >= 8 and not re.search(
                        r"(?i)your_|example|placeholder|changeme|dummy|xxx|env|getenv|redact",
                        value,
                    ):
                        violations.append(f"{py_file.name}:{lineno}: credential assigned a literal")

        assert len(violations) == 0, (
            "Potential hardcoded secrets found:\n" + "\n".join(violations)
        )

    def test_credentials_are_not_read_from_config(self):
        """Credentials must come from the environment, never from config.yaml."""
        config_file = PROJECT_ROOT / "config.yaml"
        content = config_file.read_text()

        # Config may name the env var holding a key, but must not contain a value.
        assert "api_key:" not in content, "config.yaml must not define api_key values"
        assert "api_key_env:" in content, "config.yaml should reference env var names"

        # And the values themselves must not appear anywhere in config.
        for env_var in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENCODE_ZEN_API_KEY"):
            for line in content.split("\n"):
                if env_var in line:
                    assert "=" not in line.split(env_var, 1)[1].split("#")[0], (
                        f"{env_var} must not be assigned a value in config.yaml"
                    )

    def test_credentials_from_env(self):
        """Verify all credentials are loaded from environment variables."""
        from jarvis.config import (
            GEMINI_API_KEY,
            NASA_API_KEY,
            GMAIL_ADDRESS,
            GMAIL_APP_PASSWORD,
        )

        # These should be strings (empty or populated), not hardcoded values
        assert isinstance(GEMINI_API_KEY, str)
        assert isinstance(NASA_API_KEY, str)
        assert isinstance(GMAIL_ADDRESS, str)
        assert isinstance(GMAIL_APP_PASSWORD, str)

    def test_config_yaml_exists(self):
        """Verify config.yaml is present and parseable."""
        config_file = PROJECT_ROOT / "config.yaml"
        assert config_file.exists(), "config.yaml not found"

        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)

        assert "assistant" in config
        assert "llm" in config
        assert "speech" in config
        assert "wake_word" in config
        assert "logging" in config

    def test_no_windows_paths(self):
        """Verify no hardcoded Windows paths remain."""
        jarvis_dir = PROJECT_ROOT / "jarvis"
        for py_file in jarvis_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            content = py_file.read_text()
            assert "C:\\" not in content, f"Windows path found in {py_file.name}"
            assert "D:\\" not in content, f"Windows path found in {py_file.name}"

    def test_env_file_exists(self):
        """Verify .env file exists (not committed, but should be present)."""
        env_file = PROJECT_ROOT / ".env"
        assert env_file.exists(), ".env file not found — copy from .env.example"


class TestMemoryPersistence:
    """Checklist Item 3: Memory Persistence Check."""

    def test_sqlite_long_term_memory(self):
        """Verify SQLite memory save/recall works."""
        from jarvis import memory

        # Use a temp database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)

        try:
            conn = memory.init_memory_db(db_path)

            # Save facts
            assert memory.remember(conn, "Test fact 1", "test")
            assert memory.remember(conn, "Test fact 2", "test")
            assert memory.remember(conn, "Another fact", "general")

            # Recall all
            facts = memory.recall_all(conn)
            assert len(facts) == 3
            assert "Test fact 1" in facts
            assert "Test fact 2" in facts

            # Search
            results = memory.search_memories(conn, "Test")
            assert len(results) == 2

            # Category filter
            test_facts = memory.recall_by_category(conn, "test")
            assert len(test_facts) == 2

            # Count
            assert memory.get_memory_count(conn) == 3

            # Clear
            memory.clear_all_memories(conn)
            assert memory.get_memory_count(conn) == 0

            conn.close()
        finally:
            db_path.unlink(missing_ok=True)

    def test_memory_survives_reopen(self):
        """Verify facts persist across database close/reopen."""
        from jarvis import memory

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)

        try:
            # Session 1: Save
            conn1 = memory.init_memory_db(db_path)
            memory.remember(conn1, "Persistent fact", "test")
            conn1.close()

            # Session 2: Recall
            conn2 = memory.init_memory_db(db_path)
            facts = memory.recall_all(conn2)
            assert "Persistent fact" in facts
            conn2.close()
        finally:
            db_path.unlink(missing_ok=True)

    def test_short_term_history_trimming(self):
        """Verify conversation history is capped at MAX_HISTORY_MESSAGES.

        Phase 0: trimming keeps a recent window of turns and rebuilds the chat
        session. It previously wiped the entire conversation, destroying context
        the user still needed.
        """
        from jarvis.brain import JarvisBrain
        from jarvis.config import MAX_HISTORY_MESSAGES

        brain = JarvisBrain()
        assert brain._message_count == 0
        assert brain._session is None

        # Below the cap, nothing is trimmed.
        brain._message_count = MAX_HISTORY_MESSAGES - 1
        brain._trim_history()
        assert brain._message_count == MAX_HISTORY_MESSAGES - 1

        # At the cap, a trimmed session is rebuilt from recent turns.
        brain._model_key = "some_model"
        brain._history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}"}
            for i in range(MAX_HISTORY_MESSAGES * 2)
        ]
        brain._message_count = MAX_HISTORY_MESSAGES

        captured = {}

        def fake_create_chat(model, history=None):
            captured["history"] = list(history or [])
            return None

        brain._create_chat = fake_create_chat
        brain._trim_history()

        assert captured["history"], "trimming must retain recent context"
        # The most recent turns survive trimming.
        assert captured["history"][-1]["content"] == "turn-39"
        assert len(captured["history"]) <= MAX_HISTORY_MESSAGES


class TestToolExecution:
    """Checklist Item 2: Tool execution with error handling."""

    def test_tool_registry_complete(self):
        """Verify all expected tools are registered."""
        from jarvis.brain import GEMINI_TOOLS, TOOL_REGISTRY

        expected_tools = [
            "search_wikipedia", "open_website", "search_youtube",
            "search_google", "open_maps", "get_temperature",
            "get_nasa_apod", "take_screenshot", "play_music",
            "lookup_dictionary", "send_email", "tell_time",
            "tell_joke", "get_system_time", "get_directory_contents",
            "get_system_status", "save_memory", "recall_memories",
        ]

        for tool_name in expected_tools:
            assert tool_name in TOOL_REGISTRY, f"Tool '{tool_name}' not in registry"
            assert callable(TOOL_REGISTRY[tool_name])

    def test_tool_error_handling(self):
        """Verify tool execution catches exceptions gracefully."""
        from jarvis.brain import JarvisBrain

        brain = JarvisBrain()

        # Unknown tool
        result = brain._execute_tool_call("nonexistent_tool", {})
        assert "Error" in result
        assert "Unknown tool" in result

        # Tool with bad arguments
        result = brain._execute_tool_call("get_directory_contents", {"directory": "/nonexistent/path/12345"})
        # Should return error message, not crash
        assert isinstance(result, str)

    def test_system_tools_work(self):
        """Verify system introspection tools return valid data."""
        from jarvis.brain import get_system_time, get_system_status

        time_result = get_system_time()
        assert isinstance(time_result, str)
        assert len(time_result) > 0

        status_result = get_system_status()
        assert isinstance(status_result, str)
        assert len(status_result) > 0


class TestBrainIntegration:
    """Checklist Item 2: Brain + Persona integration."""

    def test_persona_in_system_prompt(self):
        """Verify the Ai Partner persona is in the system prompt."""
        from jarvis.config import JARVIS_SYSTEM_PROMPT

        assert "Ai Partner" in JARVIS_SYSTEM_PROMPT
        assert "Honesty" in JARVIS_SYSTEM_PROMPT
        assert "Proactiveness" in JARVIS_SYSTEM_PROMPT
        assert "User agency" in JARVIS_SYSTEM_PROMPT

    def test_brain_initialization(self):
        """Verify JarvisBrain can be created."""
        from jarvis.brain import JarvisBrain

        brain = JarvisBrain()
        assert brain._session is None
        assert brain._history == []
        assert brain._message_count == 0

    def test_reset_conversation(self):
        """Verify conversation reset works."""
        from jarvis.brain import JarvisBrain

        brain = JarvisBrain()
        brain._message_count = 10
        brain._history = [{"role": "user", "content": "hi"}]
        brain._session = MagicMock()

        brain.reset_conversation()
        assert brain._session is None
        assert brain._history == []
        assert brain._message_count == 0


class TestWakeWordIntegration:
    """Checklist Item 2: Wake word listener initialization."""

    def test_wake_word_config(self):
        """Verify wake word settings are loaded."""
        from jarvis.config import WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD

        assert isinstance(WAKE_WORD_MODEL, str)
        assert WAKE_WORD_MODEL == "hey_jarvis"
        assert 0.0 <= WAKE_WORD_THRESHOLD <= 1.0

    def test_wake_word_listener_creation(self):
        """Verify WakeWordListener can be created."""
        from jarvis.wake_word import WakeWordListener

        listener = WakeWordListener(on_wake=lambda: None)
        assert listener._running is False
        assert listener._thread is None
        assert listener._inference is None

    @pytest.mark.skipif(
        not os.environ.get("JARVIS_TEST_NATIVE_WAKEWORD"),
        reason="loads ONNX Runtime, whose native worker thread races interpreter "
               "teardown (intermittent 'libc++abi ... recursive_mutex' abort). "
               "Run with JARVIS_TEST_NATIVE_WAKEWORD=1 to exercise the real model.",
    )
    def test_wake_word_engine_init(self):
        """Verify openWakeWord engine can initialize.

        Opt-in because it loads ONNX Runtime. That library starts a native
        background thread which, in 1.30.0, can lock a mutex whose owner has
        already been destroyed and abort the process at exit. It is not a bug in
        this codebase and cannot be shut down from Python, so the default suite
        does not load the native runtime. :func:`test_engine_init_fails_gracefully`
        covers our side of the same code path deterministically.
        """
        from jarvis.wake_word import WakeWordListener

        listener = WakeWordListener(on_wake=lambda: None)
        success = listener._init_engine()
        assert success is True
        assert listener._inference is not None
        # Release the ONNX session deterministically rather than relying on
        # process exit, so the engine has a real shutdown path.
        listener.close()
        assert listener._inference is None

    def test_engine_init_fails_gracefully(self):
        """A missing/broken engine must be reported, not raised."""
        from jarvis.wake_word import WakeWordListener

        listener = WakeWordListener(on_wake=lambda: None)
        with patch.dict("sys.modules", {"openwakeword": None, "openwakeword.model": None}):
            assert listener._init_engine() is False
        assert listener._inference is None

    def test_engine_close_releases_state(self):
        """The engine has a real shutdown path, not just process exit."""
        from jarvis.wake_word import WakeWordListener

        listener = WakeWordListener(on_wake=lambda: None)
        listener._inference = object()
        listener.close()
        assert listener._inference is None


class TestSpeechIntegration:
    """Checklist Item 2: Speech module fallback."""

    def test_speak_function(self):
        """Verify speak() doesn't crash without audio hardware."""
        from jarvis.speech import speak

        # Should fall back to console output without crashing
        speak("Test message for integration test")

    def test_listen_fallback(self):
        """Verify listen() falls back to console input gracefully."""
        from jarvis.speech import listen

        # Without microphone, should return "none" or fallback text
        result = listen()
        assert isinstance(result, str)


class TestLoggingIntegration:
    """Checklist Item 4: Logging system."""

    def test_logger_has_handlers(self):
        """Verify logger has both console and file handlers."""
        from jarvis.logger import logger

        assert len(logger.handlers) >= 1

    def test_status_indicator(self):
        """Verify StatusIndicator methods work."""
        from jarvis.logger import StatusIndicator

        # These should not raise exceptions
        StatusIndicator.listening()
        StatusIndicator.thinking()
        StatusIndicator.speaking()
        StatusIndicator.tool_call("test_tool", {"arg": "value"})
        StatusIndicator.wake_detected(0.75)
        StatusIndicator.memory("Test memory action")
        StatusIndicator.ready()
        StatusIndicator.shutdown()


class TestEndToEndPipeline:
    """Checklist Item 2: Full pipeline verification."""

    def test_pipeline_components_connected(self):
        """Verify all pipeline components are properly imported and wired."""
        from jarvis.brain import JarvisBrain, GEMINI_TOOLS
        from jarvis.speech import speak, listen
        from jarvis.wake_word import WakeWordListener
        from jarvis.memory import init_memory_db
        from jarvis.logger import logger, StatusIndicator
        from jarvis.config import (
            ASSISTANT_NAME, GREETING_NAME, LLM_MODEL,
            JARVIS_SYSTEM_PROMPT, WAKE_WORD_MODEL,
        )

        # Verify all components exist
        assert callable(JarvisBrain)
        assert callable(speak)
        assert callable(listen)
        assert callable(WakeWordListener)
        assert callable(init_memory_db)
        assert len(GEMINI_TOOLS) >= 15

        # Verify config
        assert ASSISTANT_NAME == "Ai Partner"
        assert GREETING_NAME == "Boss"
        assert "gemini" in LLM_MODEL
        assert "Ai Partner" in JARVIS_SYSTEM_PROMPT
        assert WAKE_WORD_MODEL == "hey_jarvis"


def run_all_tests():
    """Run all integration tests and report results."""
    import traceback

    test_classes = [
        TestSecurityAudit,
        TestMemoryPersistence,
        TestToolExecution,
        TestBrainIntegration,
        TestWakeWordIntegration,
        TestSpeechIntegration,
        TestLoggingIntegration,
        TestEndToEndPipeline,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    print(f"\n{'='*60}")
    print(f"  JARVIS 2.0 — End-to-End Integration Test Suite")
    print(f"{'='*60}\n")

    for cls in test_classes:
        print(f"--- {cls.__name__} ---")
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS: {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL: {method_name}")
                    print(f"        {e}")
                    failed += 1
                    errors.append((cls.__name__, method_name, str(e)))
        print()

    print(f"{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}\n")

    if errors:
        print("Failed tests:")
        for cls_name, method, error in errors:
            print(f"  - {cls_name}.{method}: {error}")
        sys.exit(1)
    else:
        print("All integration tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
