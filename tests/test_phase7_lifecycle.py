"""Phase 7 tests: persistent assistant state machine.

Offline throughout: no microphone, no model, no TTS weights. A fake speech
model proves TTS persistence across IDLE; a turn driver mirroring
run_text_mode proves failure recovery through the real lifecycle helper.
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.conversation import (  # noqa: E402
    AssistantState,
    AssistantStateMachine,
    ConversationMachine,
    InvalidTransition,
    State,
    detailed_to_lifecycle,
    get_assistant_machine,
)
from jarvis.main import _to  # noqa: E402


def _fresh():
    return AssistantStateMachine()


def _drive_turn(machine, ask, present, user_input="hello"):
    """One text-mode turn through the real lifecycle helper."""
    _to(machine, AssistantState.LISTENING, "awaiting input")
    _to(machine, AssistantState.PROCESSING, "request accepted")
    try:
        response = ask(user_input)
    except Exception:
        machine.recover("processing error")
        raise
    _to(machine, AssistantState.SPEAKING, "response ready")
    try:
        present(response)
    except Exception:
        machine.recover("speaking error")
        raise
    machine.transition_to(AssistantState.IDLE, "turn complete")
    return response


class TestStates:
    def test_starts_idle(self):
        assert _fresh().state is AssistantState.IDLE

    def test_normal_lifecycle(self):
        machine = _fresh()
        for state in (AssistantState.LISTENING, AssistantState.PROCESSING,
                      AssistantState.SPEAKING, AssistantState.IDLE):
            machine.transition_to(state, "test")
        assert machine.state is AssistantState.IDLE

    def test_cancelled_listening_returns_to_idle(self):
        machine = _fresh()
        machine.transition_to(AssistantState.LISTENING, "test")
        machine.transition_to(AssistantState.IDLE, "cancelled")
        assert machine.state is AssistantState.IDLE

    def test_text_only_completion_skips_speaking(self):
        machine = _fresh()
        machine.transition_to(AssistantState.LISTENING, "test")
        machine.transition_to(AssistantState.PROCESSING, "test")
        machine.transition_to(AssistantState.IDLE, "no speech needed")
        assert machine.state is AssistantState.IDLE

    def test_invalid_transition_raises_and_holds(self):
        machine = _fresh()
        with pytest.raises(InvalidTransition):
            machine.transition_to(AssistantState.SPEAKING, "skipped ahead")
        assert machine.state is AssistantState.IDLE

    def test_same_state_is_idempotent(self):
        machine = _fresh()
        assert machine.transition_to(AssistantState.IDLE, "again") is True


class TestFailureRecovery:
    def test_tool_failure_recovers(self):
        machine = _fresh()

        def _boom(_query):
            raise RuntimeError("tool exploded")

        with pytest.raises(RuntimeError):
            _drive_turn(machine, _boom, lambda _r: None)
        assert machine.state is AssistantState.IDLE

    def test_provider_failure_recovers(self):
        from jarvis.brain import BrainError
        machine = _fresh()

        def _fail(_query):
            raise BrainError("unavailable", detail="m: server (503)")

        with pytest.raises(BrainError):
            _drive_turn(machine, _fail, lambda _r: None)
        assert machine.state is AssistantState.IDLE

    def test_tts_failure_leaves_speaking(self):
        machine = _fresh()

        def _present(_response):
            raise RuntimeError("playback gone")

        with pytest.raises(RuntimeError):
            _drive_turn(machine, lambda _q: "hi", _present)
        assert machine.state is AssistantState.IDLE

    def test_repeated_requests_end_idle(self):
        machine = _fresh()
        for _ in range(3):
            _drive_turn(machine, lambda _q: "ok", lambda _r: None)
        assert machine.state is AssistantState.IDLE


class TestLogging:
    def test_transitions_logged_at_debug(self, caplog):
        machine = _fresh()
        with caplog.at_level("DEBUG", logger="jarvis"):
            machine.transition_to(AssistantState.LISTENING, "test")
        assert any("idle -> listening" in r.message for r in caplog.records)

    def test_normal_output_clean(self, capsys):
        machine = _fresh()
        _drive_turn(machine, lambda _q: "ok", lambda _r: None)
        out, err = capsys.readouterr()
        assert out == "" and err == ""


class TestPersistentTTS:
    def test_idle_never_unloads_chatterbox(self):
        from jarvis import speech
        speech._chatterbox_model = None
        speech._chatterbox_conds_key = None
        speech._chatterbox_warmed = False
        try:
            model = MagicMock()
            model.sr = 24000
            import numpy as np
            model.generate.return_value = np.zeros((1, 2400), dtype=np.float32)
            machine = _fresh()
            with patch.dict(sys.modules, {"chatterbox": MagicMock(),
                                          "chatterbox.tts": MagicMock()}):
                with patch("chatterbox.tts.ChatterboxTTS.from_pretrained",
                           return_value=model) as load:
                    for _ in range(2):  # two full speak/idle cycles
                        _drive_turn(machine,
                                    lambda _q: "hi",
                                    lambda _r: speech._synthesize_chatterbox("hi"))
                        assert machine.state is AssistantState.IDLE
            assert load.call_count == 1
            assert model.prepare_conditionals.call_count == 1
            assert speech._chatterbox_model is model
            assert speech._chatterbox_warmed is True
        finally:
            speech._chatterbox_model = None
            speech._chatterbox_conds_key = None
            speech._chatterbox_warmed = False

    def test_lifecycle_module_owns_no_tts(self):
        source = (PROJECT_ROOT / "jarvis" / "conversation.py").read_text()
        assert "chatterbox" not in source.lower()
        assert "prepare_conditionals" not in source


class TestVoiceMirror:
    def test_full_voice_sequence_mirrors(self):
        machine = _fresh()
        for detailed in (State.LISTENING, State.TRANSCRIBING, State.THINKING,
                         State.SPEAKING, State.FOLLOW_UP, State.IDLE):
            assert machine.observe(detailed, "voice loop") is True
        assert machine.state is AssistantState.IDLE

    def test_error_and_interrupt_mirror_to_idle(self):
        machine = _fresh()
        machine.transition_to(AssistantState.LISTENING, "test")
        machine.transition_to(AssistantState.PROCESSING, "test")
        assert machine.observe(State.ERROR, "failed") is True
        assert machine.state is AssistantState.IDLE

    def test_observe_never_raises(self):
        machine = _fresh()
        machine.transition_to(AssistantState.LISTENING, "test")
        machine.transition_to(AssistantState.PROCESSING, "test")
        machine.transition_to(AssistantState.SPEAKING, "test")
        # FOLLOW_UP maps to IDLE; SPEAKING -> IDLE is legal, fine either way.
        assert machine.observe(State.FOLLOW_UP, "done") in (True, False)
        assert machine.state in tuple(AssistantState)

    def test_voice_loop_wires_mirror(self):
        from jarvis.voice_loop import VoiceLoop
        assistant = _fresh()
        detailed = ConversationMachine()
        loop = VoiceLoop(machine=detailed, assistant=assistant)
        assert loop._assistant is assistant
        detailed.transition(State.LISTENING, reason="test")
        detailed.transition(State.TRANSCRIBING, reason="test")
        assert assistant.state is AssistantState.PROCESSING

    def test_voice_loop_default_unchanged(self):
        from jarvis.voice_loop import VoiceLoop
        loop = VoiceLoop()
        assert loop._assistant is None
        assert len(loop.machine._listeners) == 1  # only the HUD forwarder


class TestConcurrency:
    def test_concurrent_observe_stays_valid(self):
        machine = _fresh()
        states = [State.LISTENING, State.THINKING, State.SPEAKING,
                  State.FOLLOW_UP, State.IDLE]
        barrier = threading.Barrier(8)

        def _hammer(k):
            barrier.wait()
            for _ in range(50):
                machine.observe(states[k % len(states)], "hammer")

        threads = [threading.Thread(target=_hammer, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert machine.state in tuple(AssistantState)

    def test_singleton_shared(self):
        assert get_assistant_machine() is get_assistant_machine()
