"""Phase 6 tests: proactive partner — knowing when talking is worth it.

Behavior, not execution: the default answer is NO, and every gate that fails
must hold the silence. Same observation + different context = different
decision.
"""

import ast
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import proactive as pro  # noqa: E402
from jarvis.conversation import ConversationMachine, State  # noqa: E402
from jarvis.proactive import (  # noqa: E402
    CLEARLY_AVAILABLE,
    PROBABLY_AVAILABLE,
    UNAVAILABLE,
    UNKNOWN,
    Candidate,
    ContextSnapshot,
    ObservationEngine,
    ProactiveDecisionEngine,
    ProactiveOrchestrator,
    style_hint,
)
from jarvis.tone import ConversationState, Mood  # noqa: E402
from jarvis.vision import OBSERVATION, INFERENCE, Observation, VisualContext  # noqa: E402


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def obs(*texts, kind=OBSERVATION):
    return [Observation(text=t, kind=kind) for t in texts]


def engine(**kw):
    kw.setdefault("clock", FakeClock())
    return ObservationEngine(**kw), kw["clock"]


def orch(**kw):
    clock = kw.pop("clock", None) or FakeClock()
    kw.setdefault("clock", clock)
    kw.setdefault("cooldown_s", 300.0)
    kw.setdefault("dedup_s", 600.0)
    kw.setdefault("max_per_window", 3)
    kw.setdefault("window_s", 3600.0)
    kw.setdefault("quiet_enabled", False)
    return ProactiveOrchestrator(**kw), clock


def good_candidate(reason=pro.USER_RETURNED, conf=0.9):
    return Candidate(reason=reason, confidence=conf, text=reason, repeat_count=2)


def good_snap(**kw):
    base = dict(user_availability=CLEARLY_AVAILABLE, proactive_enabled=True)
    base.update(kw)
    return ContextSnapshot(**base)


# A. initialization / config ------------------------------------------------

class TestInit:
    def test_defaults_come_from_config(self):
        from jarvis import config
        assert config.PROACTIVE_ENABLED is True
        assert config.PROACTIVE_COOLDOWN_S > 0
        assert config.PROACTIVE_DEDUP_S > 0
        assert config.PROACTIVE_MAX_PER_WINDOW >= 1
        assert 0.0 <= config.PROACTIVE_CONFIDENCE_THRESHOLD <= 1.0
        assert config.PROACTIVE_MIN_REPEAT >= 1

    def test_config_block_has_expected_keys(self):
        import yaml
        block = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())["proactive"]
        assert block["enabled"] is True
        assert block["global_cooldown_seconds"] > 0
        assert block["max_interactions_per_window"] >= 1
        assert block["quiet_hours"]["enabled"] is False

    def test_disabled_flag_fully_disables(self):
        eng = ProactiveDecisionEngine()
        r = eng.decide(good_candidate(), good_snap(proactive_enabled=False))
        assert r.should_speak is False and r.reason == "proactive_disabled"


# B. default silence ----------------------------------------------------------

class TestDefaultSilence:
    def test_no_candidate_is_no(self):
        assert ProactiveDecisionEngine().decide(None, good_snap()).reason == "no_observation"

    def test_empty_visual_context_yields_no_candidate(self):
        eng, _ = engine()
        assert eng.note([]) is None
        assert eng.note(None) is None


# C. meaningfulness ------------------------------------------------------------

class TestMeaningfulness:
    def test_person_present_alone_is_not_a_candidate(self):
        eng, _ = engine()
        assert eng.note(obs("A person is sitting at a desk.")) is None
        assert eng.note(obs("A person is sitting at a desk.")) is None

    def test_lighting_change_is_not_a_candidate(self):
        eng, _ = engine()
        assert eng.note(obs("The light level changed slightly.")) is None
        assert eng.note(obs("The light level changed slightly.")) is None

    def test_user_returned_becomes_candidate(self):
        eng, _ = engine()
        assert eng.note(obs("The user returned and is back at the desk.")) is None
        c = eng.note(obs("The user returned and is back at the desk."))
        assert c is not None and c.reason == pro.USER_RETURNED

    def test_glasses_absent_becomes_candidate(self):
        eng, _ = engine()
        assert eng.note(obs("observed: glasses are absent from the face.")) is None
        c = eng.note(obs("observed: glasses are absent from the face."))
        assert c is not None and c.reason == pro.GLASSES_ABSENT

    def test_hedged_lines_can_never_become_candidates(self):
        """The anti-hallucination gate: inferred:/unclear: lines are ineligible."""
        eng, _ = engine()
        for _ in range(4):
            assert eng.note(obs("The user returned to the desk.", kind=INFERENCE)) is None


# D. persistence ----------------------------------------------------------------

class TestPersistence:
    def test_one_frame_never_triggers(self):
        eng, _ = engine(min_repeat=2)
        assert eng.note(obs("The user returned and is back at the desk.")) is None

    def test_gap_breaks_the_streak(self):
        eng, _ = engine(min_repeat=2)
        eng.note(obs("The user returned and is back at the desk."))
        eng.note(obs("A person is sitting at a desk."))  # meaningless gap
        assert eng.note(obs("The user returned and is back at the desk.")) is None


# E/F/G/H/I/J/K/L. gates ----------------------------------------------------------

class TestGates:
    def test_low_confidence_is_no(self):
        r = ProactiveDecisionEngine().decide(good_candidate(conf=0.1), good_snap())
        assert (r.should_speak, r.reason) == (False, "low_confidence")

    @pytest.mark.parametrize("avail", [UNKNOWN, UNAVAILABLE])
    def test_unavailable_user_is_no(self, avail):
        r = ProactiveDecisionEngine().decide(good_candidate(), good_snap(user_availability=avail))
        assert (r.should_speak, r.reason) == (False, "user_unavailable")

    def test_active_conversation_is_no(self):
        r = ProactiveDecisionEngine().decide(good_candidate(), good_snap(active_conversation=True))
        assert (r.should_speak, r.reason) == (False, "active_conversation")

    def test_user_speaking_is_no(self):
        r = ProactiveDecisionEngine().decide(good_candidate(), good_snap(user_speaking=True))
        assert (r.should_speak, r.reason) == (False, "active_conversation")

    def test_wake_word_wins(self):
        r = ProactiveDecisionEngine().decide(good_candidate(), good_snap(wake_active=True))
        assert (r.should_speak, r.reason) == (False, "wake_word_active")

    def test_assistant_speaking_is_no(self):
        r = ProactiveDecisionEngine().decide(good_candidate(), good_snap(assistant_speaking=True))
        assert (r.should_speak, r.reason) == (False, "assistant_speaking")

    def test_cooldown_is_no(self):
        r = ProactiveDecisionEngine().decide(good_candidate(), good_snap(cooldown_remaining=120.0))
        assert (r.should_speak, r.reason) == (False, "cooldown_active")

    def test_duplicate_is_no(self):
        r = ProactiveDecisionEngine().decide(
            good_candidate(), good_snap(recently_spoken=(pro.USER_RETURNED,)))
        assert (r.should_speak, r.reason) == (False, "duplicate")

    def test_suppression_is_no(self):
        assert ProactiveDecisionEngine().decide(
            good_candidate(), good_snap(suppressed=True)).reason == "suppressed"
        assert ProactiveDecisionEngine().decide(
            good_candidate(), good_snap(quiet_hours=True)).reason == "suppressed"

    def test_all_clear_is_yes_with_directive(self):
        r = ProactiveDecisionEngine().decide(good_candidate(), good_snap())
        assert r.should_speak is True
        assert "## Proactive Context" in r.directive
        assert pro.USER_RETURNED in r.directive


# M. rate limiting / K. cooldown / L. dedup via orchestrator --------------------------

class TestContinuity:
    def test_second_event_inside_cooldown_is_held(self):
        o, clock = orch(cooldown_s=300.0)
        assert o.maybe_proactive(good_candidate(), good_snap(), lambda d: "Hey.") == "Hey."
        clock.advance(60.0)
        assert o.maybe_proactive(good_candidate(), good_snap(), lambda d: "Hey.") is None

    def test_same_reason_inside_dedup_is_held(self):
        o, clock = orch(cooldown_s=0.0, dedup_s=600.0)
        assert o.maybe_proactive(good_candidate(), good_snap(), lambda d: "Hey.") == "Hey."
        clock.advance(60.0)
        assert o.maybe_proactive(good_candidate(), good_snap(), lambda d: "Hey.") is None

    def test_rolling_limit_enforced(self):
        o, clock = orch(cooldown_s=0.0, dedup_s=0.0, max_per_window=2, window_s=3600.0)
        reasons = [pro.USER_RETURNED, pro.GLASSES_ABSENT, pro.PROLONGED_WORK]
        assert o.maybe_proactive(good_candidate(reasons[0]), good_snap(), lambda d: "a") == "a"
        clock.advance(1.0)
        assert o.maybe_proactive(good_candidate(reasons[1]), good_snap(), lambda d: "b") == "b"
        clock.advance(1.0)
        assert o.maybe_proactive(good_candidate(reasons[2]), good_snap(), lambda d: "c") is None

    def test_quiet_hours_hold(self):
        o, _ = orch(quiet_enabled=True, quiet_start=22, quiet_end=8,
                    hour=lambda: 23)
        assert o.maybe_proactive(good_candidate(), good_snap(), lambda d: "Hey.") is None
        assert o.quiet_active() is True

    def test_history_is_ephemeral_and_resettable(self):
        o, _ = orch()
        o.maybe_proactive(good_candidate(), good_snap(), lambda d: "Hey.")
        assert o.status()["interactions_in_window"] == 1
        o.reset()
        assert o.status()["interactions_in_window"] == 0
        assert o.cooldown_remaining() == 0.0


# O. follow-up continuity / V+W. fail closed -------------------------------------

class TestGeneration:
    def test_generate_called_once_with_directive(self):
        o, _ = orch()
        seen = []
        out = o.maybe_proactive(good_candidate(), good_snap(),
                                lambda d: seen.append(d) or "Hey.")
        assert out == "Hey." and len(seen) == 1
        assert "## Proactive Context" in seen[0]

    def test_generate_failure_is_silent_single_attempt(self):
        o, _ = orch()
        calls = []
        def boom(directive):
            calls.append(directive)
            raise RuntimeError("provider 400")
        assert o.maybe_proactive(good_candidate(), good_snap(), boom) is None
        assert len(calls) == 1  # no autonomous retry loops

    def test_empty_reply_is_silence(self):
        o, _ = orch()
        assert o.maybe_proactive(good_candidate(), good_snap(), lambda d: "  ") is None


# Q. vision integration / R. isolation --------------------------------------------

class TestVision:
    def test_visual_context_can_produce_candidate(self):
        ctx = VisualContext(ttl=600.0, max_observations=6)
        eng, _ = engine()
        ctx.update(obs("The user returned and is back at the desk."))
        assert eng.note(ctx.observations()) is None
        ctx.update(obs("The user returned and is back at the desk."))
        c = eng.note(ctx.observations())
        assert c is not None and c.reason == pro.USER_RETURNED

    def test_camera_cannot_reach_speech(self):
        """Structural: no mic/TTS/state-machine path exists in this module."""
        tree = ast.parse(Path(pro.__file__).read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for banned in ("MicrophoneStream", "SpeechPlayer", "synthesize",
                       "ConversationMachine", "transition", "begin_turn",
                       "image_part", "VideoCapture"):
            assert banned not in names, f"proactive can reach {banned}"

    def test_no_camera_emotion_path(self):
        src = Path(pro.__file__).read_text().lower()
        for banned in ("tired", "sadness", "angry", "depress", "biometric",
                       "face recognition", "gaze", "emotion recognition"):
            assert banned not in src


# S. tone ---------------------------------------------------------------------------

class TestTone:
    def test_frustrated_user_gets_restraint_hint(self):
        s = ConversationState(user_state="frustrated")
        assert "restrained" in style_hint(s)

    def test_playful_gets_light_hint(self):
        assert "light" in style_hint(ConversationState(mood=Mood.PLAYFUL))

    def test_directive_leaks_no_internals(self):
        r = ProactiveDecisionEngine().decide(good_candidate(), good_snap())
        blob = r.directive.lower()
        for banned in ("confidence", "0.9", "cooldown", "availability",
                       "conversationmachine", "gate"):
            assert banned not in blob

    def test_tone_module_untouched_by_proactive(self):
        import inspect
        assert "proactive" not in inspect.getsource(
            sys.modules["jarvis.tone"]).lower()


# T/U. memory boundary -----------------------------------------------------------------

class TestMemoryBoundary:
    def test_proactive_module_has_no_store(self):
        import inspect
        src = inspect.getsource(pro)
        for banned in ("sqlite", "CREATE TABLE", "INSERT", "memory.recall",
                       "remember", "save_memory", ".db"):
            assert banned not in src

    def test_no_database_file_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        o, _ = orch()
        o.maybe_proactive(good_candidate(), good_snap(), lambda d: "Hey.")
        assert list(tmp_path.glob("*.db")) == []


# X. multimodal routing regression -------------------------------------------------------

class TestMultimodalRegression:
    def test_image_then_text_stays_on_vision_model(self):
        from tests.mock_providers import build_layer, reply
        from jarvis.brain import JarvisBrain
        from jarvis.providers.base import image_part
        png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        layer = build_layer(
            models=[
                {"key": "blind", "provider": "p1", "model": "m1", "priority": 100,
                 "free": True, "capabilities": {"reasoning": True, "vision": False}},
                {"key": "sees", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": reply("blind"), "m2": reply("saw it")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("what is this?", images=[image_part(png)])
        assert brain.ask("and what does the error say?") == "saw it"
        assert brain.last_model_key == "sees"


# Y/P. voice regression + interruption ----------------------------------------------------

class TestVoiceIntegration:
    def _loop(self):
        from jarvis.audio import MicrophoneStream
        from jarvis.speech_pipeline import SpeechPlayer
        from jarvis.voice_loop import VoiceLoop

        class Source:
            is_open = False

            def open(self):
                self.is_open = True

            def read(self):
                time.sleep(0.004)
                return None

            def close(self):
                self.is_open = False

        class ASR:
            def transcribe(self, pcm, sample_rate=16000):
                return "what time is it"

        return VoiceLoop(
            microphone=MicrophoneStream(source=Source()),
            player=SpeechPlayer(
                synthesize=lambda t: np.zeros(8000, dtype=np.float32),
                play=lambda d, r: time.sleep(0.002),
                stop_playback=lambda: None,
                chunk_ms=100,
            ),
            transcriber=ASR(),
            respond=lambda t: "It is twelve.",
            wake_enabled=False,
            follow_up_window=0.2,
        )

    def test_check_proactive_never_moves_the_machine(self):
        loop = self._loop()
        before = loop.machine.state
        loop.check_proactive(good_candidate())  # YES or NO, never a transition
        assert loop.machine.state is before

    def test_speaking_machine_holds_proactive(self):
        loop = self._loop()
        loop.machine.transition(State.LISTENING)
        loop.machine.transition(State.TRANSCRIBING, reason="t")
        loop.machine.transition(State.THINKING, reason="t")
        loop.machine.transition(State.SPEAKING, reason="t")
        assert loop.check_proactive(good_candidate()) is None

    def test_idle_machine_can_approve(self):
        loop = self._loop()
        # IDLE + probably_available + no cooldown + no dup -> YES
        # check_proactive builds probably_available for non-INTERRUPTABLE states
        result = loop.check_proactive(good_candidate())
        assert result is not None and result.should_speak is True


# Z. tone regression (same input, different context) ----------------------------------------

class TestSameInputDifferentContext:
    def test_candidate_decision_depends_on_context(self):
        eng = ProactiveDecisionEngine()
        c = good_candidate()
        assert eng.decide(c, good_snap(user_availability=UNAVAILABLE)).should_speak is False
        assert eng.decide(c, good_snap(user_availability=CLEARLY_AVAILABLE)).should_speak is True
        assert eng.decide(c, good_snap(active_conversation=True)).should_speak is False
        assert eng.decide(c, good_snap(cooldown_remaining=10.0)).should_speak is False
        assert eng.decide(c, good_snap(proactive_enabled=False)).should_speak is False


# AA. shutdown: no threads --------------------------------------------------------------------

class TestShutdown:
    def test_no_threads_started(self):
        before = threading.active_count()
        for _ in range(50):
            ObservationEngine().note(obs("A person is sitting at a desk."))
            ProactiveOrchestrator().maybe_proactive(None, good_snap(), lambda d: "x")
        assert threading.active_count() <= before
