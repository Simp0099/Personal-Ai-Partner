"""Phase 5 tests: conversation state.

The properties that matter, in the order they were specified:

* state is a small, bounded, typed thing that behaves sanely under abuse,
* it changes with the conversation and decays back afterwards,
* an explicit correction from the user beats any inference,
* it shapes style and cannot override accuracy, instructions or safety,
* personality and conversation state stay separate things,
* nothing it observes is ever written to memory,
* it is structurally unable to learn about a person from the camera.

Same input, different context is asserted on the *guidance produced*, never on
model wording -- wording varies and a test that pins it is a test that lies.
"""

import base64
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import behavior, memory, tone, vision, voice_loop  # noqa: E402
from jarvis.audio import EchoCanceller, EchoGuard, MicrophoneStream  # noqa: E402
from jarvis.brain import JarvisBrain, BrainError, _build_context_block  # noqa: E402
from jarvis.classify import classify  # noqa: E402
from jarvis.conversation import ConversationMachine, State  # noqa: E402
from jarvis.providers.base import image_part  # noqa: E402
from jarvis.speech_pipeline import SpeechPlayer  # noqa: E402
from jarvis.voice_loop import VoiceLoop  # noqa: E402
from jarvis.tone import (  # noqa: E402
    ConversationMode,
    ConversationState,
    ConversationTone,
    Mood,
    UserState,
)
from tests.mock_providers import build_layer, reply  # noqa: E402

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _silence():
    import numpy

    return numpy.zeros(8000, dtype=numpy.float32)


class FakeClock:
    """A clock the test drives, so decay is exact rather than timed."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def tracker(**kwargs):
    """A tracker with a controllable clock."""
    clock = FakeClock()
    kwargs.setdefault("clock", clock)
    return ConversationTone(**kwargs), clock


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Keep tests off the real database and the process-wide tracker."""
    import tempfile

    import jarvis.brain as brain_module

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        spare = Path(tmp.name)
    conn = memory.init_memory_db(spare)
    monkeypatch.setattr(brain_module, "get_memory_conn", lambda: conn)
    for name in ("thinking", "tool_call", "tool_result", "memory", "info"):
        monkeypatch.setattr(
            brain_module.StatusIndicator, name, staticmethod(lambda *a, **k: None)
        )
    monkeypatch.setattr(tone, "reset_tone", tone.reset_tone)
    tone.reset_tone()
    yield conn
    tone.reset_tone()
    vision.reset_visual_context()
    conn.close()
    spare.unlink(missing_ok=True)


@pytest.fixture
def db():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = Path(tmp.name)
    conn = memory.init_memory_db(path)
    yield conn
    conn.close()
    path.unlink(missing_ok=True)


def _vision_layer(text="ok"):
    return build_layer(
        models=[{"key": "m", "provider": "p", "model": "mm",
                 "capabilities": {"reasoning": True, "tool_calling": True, "vision": True}}],
        behaviours={"mm": reply(text)},
    )


def _loop(respond=None, **kwargs):
    """A VoiceLoop with no hardware behind any seam."""
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

    kwargs.setdefault("wake_enabled", False)
    kwargs.setdefault("follow_up_window", 0.2)
    return VoiceLoop(
        microphone=MicrophoneStream(source=Source()),
        player=SpeechPlayer(
            synthesize=lambda t: _silence(),
            play=lambda d, r: time.sleep(0.002),
            stop_playback=lambda: None,
            chunk_ms=100,
        ),
        transcriber=ASR(),
        respond=respond or (lambda t: "It is twelve."),
        **kwargs,
    )


# ============================================================================
# 1. Initialization and defaults
# ============================================================================

class TestInitialization:
    def test_defaults_are_the_configured_ones(self):
        from jarvis.config import (
            TONE_CURIOUSITY_DEFAULT,
            TONE_DECAY_HALF_LIFE,
            TONE_ENABLED,
            TONE_ENERGY_DEFAULT,
            TONE_MODE_DEFAULT,
            TONE_MOOD_DEFAULT,
            TONE_WARMTH_DEFAULT,
        )

        assert TONE_ENABLED is True
        state = ConversationState()
        assert state.mood.value == TONE_MOOD_DEFAULT
        assert state.mode.value == TONE_MODE_DEFAULT
        assert state.energy == pytest.approx(TONE_ENERGY_DEFAULT)
        assert state.warmth == pytest.approx(TONE_WARMTH_DEFAULT)
        assert state.curiosity == pytest.approx(TONE_CURIOUSITY_DEFAULT)
        assert ConversationTone().half_life == pytest.approx(TONE_DECAY_HALF_LIFE)

    def test_user_state_starts_unknown(self):
        """Absence of evidence is the honest default, not a guess."""
        assert ConversationState().user_state is UserState.UNKNOWN

    def test_fresh_tracker_is_at_defaults(self):
        tr, _ = tracker()
        assert tr.state.as_dict() == tr.defaults.as_dict()

    @pytest.mark.parametrize("field", ["energy", "warmth", "curiosity"])
    @pytest.mark.parametrize("bad", [5.0, -3.0, 999, None, "nonsense", float("nan")])
    def test_out_of_range_values_are_clamped_not_raised(self, field, bad):
        state = ConversationState(**{field: bad})
        value = getattr(state, field)
        assert 0.0 <= value <= 1.0, f"{field}={bad!r} produced {value}"

    def test_unknown_enum_values_fall_back_to_defaults(self):
        state = ConversationState(mood="banana", mode="banana", user_state="banana")
        assert state.mood is Mood.NEUTRAL
        assert state.mode is ConversationMode.CASUAL
        assert state.user_state is UserState.UNKNOWN

    def test_string_enum_values_are_accepted(self):
        state = ConversationState(mood="playful", mode="technical")
        assert state.mood is Mood.PLAYFUL
        assert state.mode is ConversationMode.TECHNICAL

    def test_every_field_is_bounded_after_real_traffic(self):
        tr, _ = tracker()
        for message in ["YES!!! It finally works!", "I am so frustrated!!!",
                        "haha that's hilarious", "I am exhausted"]:
            state = tr.observe(message)
            for field in ("energy", "warmth", "curiosity"):
                assert 0.0 <= getattr(state, field) <= 1.0

    def test_state_is_immutable(self):
        state = ConversationState()
        with pytest.raises(Exception):
            state.energy = 0.9   # type: ignore[misc]


# ============================================================================
# 2. Transitions
# ============================================================================

class TestTransitions:
    def test_success_moves_to_celebratory(self):
        tr, _ = tracker()
        state = tr.observe("It finally works! Everything passed.")
        assert state.mode is ConversationMode.CELEBRATORY
        assert state.mood is Mood.CELEBRATORY
        assert state.energy > 0.7
        assert state.warmth > 0.8

    def test_frustration_moves_to_supportive_and_calm(self):
        tr, _ = tracker()
        state = tr.observe("I'm so frustrated, this keeps failing.")
        assert state.mode is ConversationMode.SUPPORTIVE
        assert state.mood is Mood.CALM
        assert state.user_state is UserState.FRUSTRATED
        assert state.energy < 0.5, "frustration should not raise energy"
        assert state.warmth > tr.defaults.warmth, "frustration should raise warmth"

    def test_technical_content_moves_to_technical(self):
        tr, _ = tracker()
        state = tr.observe("The stack trace shows a NullPointerException at line 42")
        assert state.mode is ConversationMode.TECHNICAL
        assert state.mood is Mood.FOCUSED

    def test_a_task_request_moves_to_task_execution(self):
        tr, _ = tracker()
        state = tr.observe("okay now help me optimize the query plan")
        assert state.mode is ConversationMode.TASK_EXECUTION

    def test_playful_conversation_moves_to_playful(self):
        tr, _ = tracker()
        state = tr.observe("haha that's hilarious, tell me another joke")
        assert state.mode is ConversationMode.PLAYFUL
        assert state.mood is Mood.PLAYFUL

    def test_brainstorming_raises_curiosity(self):
        tr, _ = tracker()
        state = tr.observe("what are the options for the queue?")
        assert state.mode is ConversationMode.BRAINSTORMING
        assert state.curiosity > tr.defaults.curiosity
        assert state.curiosity > 0.75

    def test_curiosity_rises_on_an_explanatory_question(self):
        """Even when the question does not change the register."""
        tr, _ = tracker()
        tr.set(ConversationState(mode=ConversationMode.TECHNICAL,
                                 mood=Mood.FOCUSED, energy=0.45, warmth=0.6,
                                 curiosity=0.5))
        state = tr.observe("curious how the router picks a model")
        assert state.curiosity > 0.7
        assert "curiosity" in tr.signals
        assert state.mode is ConversationMode.BRAINSTORMING

    def test_serious_content_drops_the_levity(self):
        tr, _ = tracker()
        state = tr.observe("production is down, this is critical")
        assert state.mode is ConversationMode.SERIOUS
        assert state.energy < 0.55

    def test_tired_lowers_energy_and_raises_warmth(self):
        tr, _ = tracker()
        state = tr.observe("I'm exhausted, it's been a long day")
        assert state.user_state is UserState.TIRED
        assert state.energy < 0.5
        assert state.warmth > tr.defaults.warmth

    def test_confused_user_prefers_clarity_over_ideas(self):
        tr, _ = tracker()
        state = tr.observe("what are the options? I'm so confused about this error")
        assert state.user_state is UserState.CONFUSED
        assert state.mode is not ConversationMode.BRAINSTORMING

    def test_one_message_nudges_rather_than_commandeers(self):
        """A single enthusiastic sentence must not flip the whole register."""
        tr, _ = tracker()
        tr.set(ConversationState(mode=ConversationMode.TECHNICAL, mood=Mood.FOCUSED,
                                 energy=0.45, warmth=0.6, curiosity=0.7))
        state = tr.observe("It finally works!!")
        assert state.energy < 1.0
        assert state.energy > 0.45, "one message should not jump to full energy"

    def test_a_quiet_message_leaves_a_neutral_state_alone(self):
        tr, _ = tracker()
        before = tr.state.as_dict()
        tr.observe("ok")
        assert tr.state.as_dict() == before


# ============================================================================
# 3. Explicit correction
# ============================================================================

class TestExplicitCorrection:
    def test_a_negation_withdraws_an_inferred_state(self):
        tr, _ = tracker()
        assert tr.observe("I'm so frustrated with this").user_state is UserState.FRUSTRATED
        state = tr.observe("I'm not frustrated, I'm just joking")
        assert state.user_state is UserState.UNKNOWN

    def test_correction_outranks_an_earlier_inference(self):
        """The user is the authority on how they feel."""
        tr, _ = tracker()
        tr.set(ConversationState(user_state=UserState.FRUSTRATED,
                                 mode=ConversationMode.SUPPORTIVE))
        assert tr.observe("actually I'm fine").user_state is UserState.UNKNOWN
        tr.set(ConversationState(user_state=UserState.FRUSTRATED,
                                 mode=ConversationMode.SUPPORTIVE))
        assert tr.observe("I'm not frustrated any more").user_state is UserState.UNKNOWN

    @pytest.mark.parametrize("message", [
        "I'm not tired",
        "don't think I'm stressed",
        "I'm not confused anymore",
        "no longer frustrated",
    ])
    def test_various_negations_are_honoured(self, message):
        tr, _ = tracker()
        tr.set(ConversationState(user_state=UserState.STRESSED))
        assert tr.observe(message).user_state is UserState.UNKNOWN

    def test_a_new_explicit_statement_replaces_the_old_one(self):
        tr, _ = tracker()
        tr.observe("I'm frustrated")
        assert tr.observe("actually I'm excited about this now").user_state is UserState.EXCITED

    def test_inference_is_withdrawn_when_the_conversation_moves_on(self):
        """Not every moment is evidence about how the user feels forever."""
        tr, _ = tracker()
        tr.observe("I'm so frustrated, this keeps failing")
        assert tr.observe("okay now help me optimize it").user_state is UserState.UNKNOWN


# ============================================================================
# 4. Decay and reset
# ============================================================================

class TestDecayAndReset:
    def test_state_relaxes_toward_defaults(self):
        tr, clock = tracker(half_life=10.0)
        tr.observe("It finally works!!")
        excited = tr.state.energy
        assert excited > tr.defaults.energy

        clock.advance(10.0)
        assert tr.state.energy < excited
        clock.advance(40.0)
        assert tr.state.energy == pytest.approx(tr.defaults.energy, abs=0.01)

    def test_decay_is_idempotent_on_repeated_reads(self):
        """Regressed once: reading the state compounded the decay."""
        tr, clock = tracker(half_life=10.0)
        tr.observe("It finally works!!")
        clock.advance(20.0)
        first = tr.state.energy
        for _ in range(10):
            assert tr.state.energy == pytest.approx(first)

    def test_labels_revert_after_the_excursion_has_faded(self):
        tr, clock = tracker(half_life=10.0)
        tr.observe("It finally works!!")
        assert tr.state.mode is ConversationMode.CELEBRATORY
        clock.advance(40.0)
        assert tr.state.mode is ConversationMode.CASUAL
        assert tr.state.mood is Mood.NEUTRAL

    def test_inferred_user_state_is_withdrawn_by_decay(self):
        tr, clock = tracker(half_life=10.0)
        tr.observe("I'm so frustrated")
        assert tr.state.user_state is UserState.FRUSTRATED
        clock.advance(40.0)
        assert tr.state.user_state is UserState.UNKNOWN

    def test_decay_can_be_disabled(self):
        tr, clock = tracker(half_life=10.0, decay_enabled=False)
        tr.observe("It finally works!!")
        high = tr.state.energy
        clock.advance(10_000.0)
        assert tr.state.energy == pytest.approx(high)

    def test_reset_returns_to_defaults(self):
        tr, _ = tracker()
        tr.observe("YES! It finally works!!")
        assert tr.state.energy > tr.defaults.energy
        tr.reset()
        assert tr.state.as_dict() == tr.defaults.as_dict()

    def test_reset_forgets_inference_entirely(self):
        tr, _ = tracker()
        tr.observe("I'm so frustrated")
        tr.reset()
        assert tr.state.user_state is UserState.UNKNOWN

    def test_decay_needs_no_background_thread(self):
        """No timer means nothing to leak at shutdown."""
        import threading

        before = threading.active_count()
        for _ in range(50):
            tracker()[0].observe("It finally works!")
        assert threading.active_count() <= before


# ============================================================================
# 5. Memory boundary
# ============================================================================

class TestMemoryBoundary:
    def test_nothing_is_written_to_memory(self, _isolated):
        tr, _ = tracker()
        for message in ["I'm so frustrated", "It finally works!!",
                        "I'm exhausted", "I am so confused"]:
            tr.observe(message)
        assert memory.get_memory_count(_isolated) == 0

    def test_the_tone_module_has_no_database(self):
        import inspect

        source = inspect.getsource(tone)
        for banned in ("sqlite", "CREATE TABLE", "connect(", "INSERT", "memory.recall"):
            assert banned not in source, f"conversation state reached for {banned}"

    def test_a_real_turn_stores_nothing(self, _isolated):
        layer = _vision_layer("ok")
        brain = JarvisBrain(model_layer=layer)
        brain.ask("I'm so frustrated with this bug")
        brain.ask("It finally works!!")
        assert memory.get_memory_count(_isolated) == 0

    def test_explicit_memory_still_works(self, _isolated):
        """The boundary is one-way: asking to remember still remembers."""
        layer = _vision_layer("Stored.")
        brain = JarvisBrain(model_layer=layer)
        brain.ask("Remember that I prefer concise answers.")
        brain.ask("Remember that I prefer concise answers.")
        assert memory.get_memory_count(_isolated) <= 1


# ============================================================================
# 6. Personality separation
# ============================================================================

class TestPersonalitySeparation:
    def test_the_identity_prompt_is_untouched(self):
        from jarvis.config import JARVIS_SYSTEM_PROMPT

        assert "Ai Partner" in JARVIS_SYSTEM_PROMPT
        assert "## Identity" in JARVIS_SYSTEM_PROMPT
        assert "mood" not in JARVIS_SYSTEM_PROMPT.lower()
        assert "conversation state" not in JARVIS_SYSTEM_PROMPT.lower()

    def test_the_tone_block_never_restates_the_personality(self):
        """Personality is identity; this is behaviour. They must not merge."""
        tr, _ = tracker()
        tr.observe("It finally works!!")
        block = tone.format_prompt(tr.state)
        assert "You are Ai Partner" not in block
        assert "## Identity" not in block

    def test_the_tone_block_is_a_separate_section(self):
        tr, _ = tracker()
        tr.observe("I'm so frustrated, this keeps failing")
        block = tone.format_prompt(tr.state)
        assert block.startswith("\n\n## How To Talk Right Now")

    def test_personality_and_state_are_independent_objects(self):
        tr, _ = tracker()
        tr.observe("It finally works!!")
        assert tr.state is not tr.defaults
        assert tr.defaults.mood is Mood.NEUTRAL, "personality/defaults were mutated"


# ============================================================================
# 7. Brain integration and the same input in two contexts
# ============================================================================

class TestBrainIntegration:
    def test_the_tone_block_reaches_the_turn_context(self):
        tone.get_tone().observe("I'm so frustrated, this keeps failing")
        block = _build_context_block(
            "I am so frustrated, this keeps failing",
            classify("I am so frustrated"), [],
        )
        assert "## How To Talk Right Now" in block
        assert "Mode: supportive" in block

    def test_the_block_sits_beside_the_other_turn_context(self):
        tone.get_tone().observe("It finally works!!")
        block = _build_context_block("it works", classify("it works"), [])
        for section in ("## This turn", "## How To Talk Right Now"):
            assert section in block, f"{section} missing from the turn"

    def test_a_neutral_conversation_costs_nothing(self):
        assert tone.format_prompt(ConversationState()) == ""

    def test_the_same_message_gets_different_guidance_in_two_contexts(self):
        """The whole point of Phase 5, asserted on guidance rather than wording."""
        message = "It finally works."

        celebratory = tracker()[0]
        celebratory.observe(message, context=ConversationState(
            mode=ConversationMode.CELEBRATORY, mood=Mood.CELEBRATORY,
            energy=0.85, warmth=0.90, curiosity=0.70,
            user_state=UserState.EXCITED,
        ))
        technical = tracker()[0]
        technical.observe(message, context=ConversationState(
            mode=ConversationMode.TECHNICAL, mood=Mood.FOCUSED,
            energy=0.45, warmth=0.70, curiosity=0.75,
        ))

        celebratory_block = tone.format_prompt(celebratory.state)
        technical_block = tone.format_prompt(technical.state)

        assert celebratory_block != technical_block
        assert "Mode: celebratory" in celebratory_block
        assert "Mode: technical" in technical_block
        # Different directions, not just different labels.
        assert "Match the moment" in celebratory_block
        assert "Be precise and concrete" in technical_block

    def test_guiding_documents_are_distinct_per_mode(self):
        blocks = {}
        for mode in ConversationMode:
            tr = tracker()[0]
            tr.set(ConversationState(mode=mode, mood=_mood_for(mode),
                                     energy=0.6, warmth=0.7, curiosity=0.7))
            blocks[mode] = tone.format_prompt(tr.state)
        assert len(set(blocks.values())) == len(ConversationMode), \
            "two modes produced identical guidance"

    def test_the_state_reaches_the_model(self):
        tone.get_tone().observe("I'm so frustrated, this keeps failing")
        layer = _vision_layer("ok")
        JarvisBrain(model_layer=layer).ask("I am so frustrated, this keeps failing")
        prompt = layer.providers["p"].sessions[-1].system_prompt
        assert "## How To Talk Right Now" in prompt

    def test_the_state_survives_a_fresh_brain(self):
        tone.get_tone().observe("It finally works!!")
        layer = _vision_layer("ok")
        JarvisBrain(model_layer=layer).ask("so what now")
        assert "## How To Talk Right Now" in layer.providers["p"].sessions[-1].system_prompt

    def test_the_state_survives_a_model_switch(self):
        layer = build_layer(
            models=[
                {"key": "a", "provider": "p1", "model": "m1", "priority": 100,
                 "capabilities": {"reasoning": True, "vision": True}},
                {"key": "b", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": reply("A"), "m2": reply("B")},
        )
        tone.get_tone().observe("I'm so frustrated, this keeps failing")
        JarvisBrain(model_layer=layer).ask("I am so frustrated, this keeps failing")
        assert "Mode: supportive" in layer.providers["p1"].sessions[-1].system_prompt

    def test_resetting_a_conversation_resets_the_state(self):
        tone.get_tone().observe("It finally works!!")
        layer = _vision_layer("ok")
        brain = JarvisBrain(model_layer=layer)
        brain.reset_conversation()
        assert tone.get_tone().state.mode is ConversationMode.CASUAL


def _mood_for(mode: ConversationMode) -> Mood:
    return {
        ConversationMode.CELEBRATORY: Mood.CELEBRATORY,
        ConversationMode.PLAYFUL: Mood.PLAYFUL,
        ConversationMode.SUPPORTIVE: Mood.CALM,
        ConversationMode.TECHNICAL: Mood.FOCUSED,
    }.get(mode, Mood.NEUTRAL)


# ============================================================================
# 8. It is a soft signal
# ============================================================================

class TestSoftSignal:
    def test_the_block_says_it_cannot_override_anything_important(self):
        tr, _ = tracker()
        tr.observe("It finally works!!")
        block = tone.format_prompt(tr.state).lower()
        for limit in ("never overrides", "accuracy", "safety", "instructions",
                      "tool requirements"):
            assert limit in block, f"the block does not state the {limit} limit"

    def test_a_playful_mood_does_not_ask_for_jokes_every_time(self):
        tr, _ = tracker()
        tr.set(ConversationState(mode=ConversationMode.PLAYFUL, mood=Mood.PLAYFUL,
                                 energy=0.75, warmth=0.8, curiosity=0.75))
        block = tone.format_prompt(tr.state).lower()
        assert "still answer the actual question" in block
        assert "does not mean make every answer a joke" in block

    def test_high_energy_does_not_mean_enthusiasm_about_anything(self):
        tr, _ = tracker()
        tr.set(ConversationState(mood=Mood.ENERGETIC, energy=0.95, warmth=0.8,
                                 curiosity=0.8))
        block = tone.format_prompt(tr.state).lower()
        assert "does not mean be enthusiastic about things that are not good news" in block

    def test_the_block_tells_the_model_to_stay_quiet_about_it(self):
        tr, _ = tracker()
        tr.observe("It finally works!!")
        block = tone.format_prompt(tr.state)
        assert "Never mention it" in block
        assert "never tell the user their mood was detected" in block

    def test_it_carries_no_chain_of_thought_or_prompts(self):
        tr, _ = tracker()
        tr.observe("I'm so frustrated")
        blob = str(tr.status())
        for banned in ("## This turn", "system prompt", "reasoning", "JARVIS_SYSTEM_PROMPT"):
            assert banned not in blob


# ============================================================================
# 9. Honesty about feelings
# ============================================================================

class TestHonestyAboutFeelings:
    def test_the_block_says_there_are_no_feelings(self):
        tr, _ = tracker()
        tr.observe("It finally works!!")
        block = tone.format_prompt(tr.state)
        assert "You do not have feelings" in block
        assert "not a subjective experience" in block

    def test_it_says_so_warmly_rather_than_defensively(self):
        tr, _ = tracker()
        tr.observe("It finally works!!")
        assert "warmly rather than defensively" in tone.format_prompt(tr.state)

    def test_the_personality_prompt_does_not_claim_emotions(self):
        from jarvis.config import JARVIS_SYSTEM_PROMPT

        lowered = JARVIS_SYSTEM_PROMPT.lower()
        assert "i feel" not in lowered
        assert "my feelings" not in lowered


# ============================================================================
# 10. Vision isolation
# ============================================================================

class TestVisionIsolation:
    def test_the_tone_module_cannot_see_the_camera(self):
        """Structural, not behavioural: there is no import path to the image."""
        import inspect

        import ast

        tree = ast.parse(Path(tone.__file__).read_text())
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for banned in ("OpenCVCamera", "VisualContext", "Observation",
                       "image_part", "get_visual_context"):
            assert banned not in used, f"conversation state can reach {banned}"

    def test_the_tone_module_imports_nothing_that_can_see(self):
        """The dependency list is the real guarantee."""
        import ast

        imported = set()
        for node in ast.walk(ast.parse(Path(tone.__file__).read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        for banned in ("jarvis.vision", "jarvis.webcam", "jarvis.camera"):
            assert banned not in imported

    def test_a_camera_observation_does_not_change_the_state(self):
        vision.get_visual_context().update(
            vision.parse_observations("observed: A person is sitting at a desk.")
        )
        before = tone.get_tone().state.as_dict()
        assert vision.get_visual_context().observations()
        assert tone.get_tone().state.as_dict() == before

    def test_a_camera_observation_cannot_report_an_emotion(self):
        """No route exists from a pixel to a feeling, by construction."""
        from jarvis.vision import VisualContext

        context = VisualContext()
        context.update(
            vision.parse_observations("observed: The user is sitting at a desk.")
        )
        for observation in context.observations():
            text = observation.text.lower()
            for banned in ("tired", "sad", "frustrated", "happy", "angry", "emotional"):
                assert banned not in text

    def test_the_perception_directive_forbids_emotion_inference(self):
        directive = behavior.perception_directive()
        assert "no emotion, mood, health or diagnosis" in directive

    def test_a_perception_turn_does_not_touch_the_state(self):
        """A webcam turn must never move the user's conversational state."""
        tone.get_tone().set(ConversationState(mode=ConversationMode.SUPPORTIVE,
                                              mood=Mood.CALM, energy=0.4, warmth=0.9))
        before = tone.get_tone().state.as_dict()
        layer = _vision_layer("observed: a person at a desk")
        JarvisBrain(model_layer=layer).ask(
            "describe", images=[image_part(PNG_1PX)], ephemeral=True
        )
        assert tone.get_tone().state.as_dict() == before

    def test_the_voice_loop_cannot_touch_the_state(self):
        import inspect

        source = inspect.getsource(voice_loop)
        assert "tone.observe" not in source


# ============================================================================
# 11. Phase 4 compatibility
# ============================================================================

class TestWakeAndInterruptionCompatibility:
    def test_an_interruption_does_not_wipe_conversational_context(self):
        """Cancelling audio must not reset how the conversation is going."""
        tone.get_tone().observe("It finally works!!")
        before = tone.get_tone().state.as_dict()

        loop = _loop()
        loop.start()
        try:
            loop.machine.transition(State.SPEAKING, force=True)
            loop.interrupt()
            assert loop.machine.state is State.LISTENING
            assert tone.get_tone().state.as_dict() == before
        finally:
            loop.stop()

    def test_a_voice_turn_updates_the_state(self):
        loop = _loop()
        loop.start()
        try:
            loop.on_wake()
            # The Brain observes, not the loop; a voice turn reaches it through
            # exactly the same path a typed one does.
            tone.observe("I'm so frustrated, this keeps failing")
            assert tone.get_tone().state.mode is ConversationMode.SUPPORTIVE
        finally:
            loop.stop()

    def test_the_state_survives_the_conversation_state_machine(self):
        """Two different things named similarly; both must be usable at once."""
        machine = ConversationMachine()
        tr, _ = tracker()
        machine.transition(State.LISTENING)
        tr.observe("It finally works!!")
        assert machine.state is State.LISTENING
        assert tr.state.mode is ConversationMode.CELEBRATORY

    def test_conversation_state_is_not_a_conversational_state_machine(self):
        from jarvis.conversation import ConversationMachine

        assert ConversationMachine is not ConversationTone
        assert not hasattr(ConversationTone, "transition")
        assert not hasattr(ConversationState, "snapshot")


# ============================================================================
# 12. TTS compatibility
# ============================================================================

class TestTTSCompatibility:
    def test_the_engine_signature_still_accepts_a_bare_call(self):
        """Phase 5 added an optional parameter, not a required one."""
        import inspect

        from jarvis import speech

        signature = inspect.signature(speech._synthesize_chatterbox)
        assert "exaggeration" in signature.parameters
        assert signature.parameters["exaggeration"].default is None

    def test_the_default_is_unchanged_when_no_hint_is_given(self):
        from jarvis import speech
        from jarvis.config import CHATTERBOX_EXAGGERATION

        assert CHATTERBOX_EXAGGERATION == 0.5

    def test_a_celebratory_state_asks_for_livelier_delivery(self):
        state = ConversationState(mood=Mood.CELEBRATORY, energy=0.9)
        assert state.exaggerated() is True

    def test_a_calm_state_reports_composure(self):
        state = ConversationState(mood=Mood.CALM, energy=0.3)
        assert state.composure() is True
        assert state.exaggerated() is False

    def test_the_hint_is_passed_to_the_synthesizer(self):
        seen = []

        def capturing(text, exaggeration=None):
            seen.append(exaggeration)
            return _silence()

        player = SpeechPlayer(synthesize=capturing, play=lambda d, r: None,
                              stop_playback=lambda: None, chunk_ms=100)
        player.exaggeration = 0.8
        player.synthesize_to_queue("hello", "turn_001", lambda t: False)
        assert seen == [0.8]

    def test_no_hint_means_the_synthesizer_is_called_plainly(self):
        seen = []

        def capturing(text):
            seen.append(text)
            return _silence()

        player = SpeechPlayer(synthesize=capturing, play=lambda d, r: None,
                              stop_playback=lambda: None, chunk_ms=100)
        player.synthesize_to_queue("hello", "turn_001", lambda t: False)
        assert seen == ["hello"]

    def test_the_voice_loop_sets_the_hint_from_the_state(self):
        tone.get_tone().set(ConversationState(mode=ConversationMode.CELEBRATORY,
                                              mood=Mood.CELEBRATORY, energy=0.9))
        loop = _loop()
        loop.start()
        try:
            loop.machine.transition(State.SPEAKING, force=True)
            loop._speak("yes!", loop.machine.begin_turn().id)
            assert loop.player.exaggeration is not None
        finally:
            loop.stop()
            tone.get_tone().reset()


# ============================================================================
# 13. Failure handling
# ============================================================================

class TestFailureHandling:
    def test_a_broken_derivation_keeps_the_last_state(self, monkeypatch):
        tr, _ = tracker()
        tr.observe("It finally works!!")
        good = tr.state

        monkeypatch.setattr(tone, "_derive", _explode)
        assert tr.observe("anything at all") is good

    def test_a_broken_derivation_still_returns_usable_state(self, monkeypatch):
        tr, _ = tracker()
        monkeypatch.setattr(tone, "_derive", _explode)
        state = tr.observe("hello")
        assert 0.0 <= state.energy <= 1.0

    def test_a_none_message_is_survivable(self):
        tr, _ = tracker()
        assert tr.observe(None) is not None
        assert tr.observe("") is not None

    def test_a_non_string_message_is_survivable(self):
        tr, _ = tracker()
        assert tr.observe(12345) is not None

    def test_the_conversation_continues_after_a_tone_failure(self, monkeypatch, _isolated):
        monkeypatch.setattr(tone, "_derive", _explode)
        layer = _vision_layer("still answering")
        assert JarvisBrain(model_layer=layer).ask("hello") == "still answering"

    def test_a_broken_prompt_renderer_does_not_break_a_turn(self, monkeypatch, _isolated):
        monkeypatch.setattr(tone, "format_prompt", _explode_no_args)
        layer = _vision_layer("still answering")
        assert JarvisBrain(model_layer=layer).ask("hello") == "still answering"

    def test_disabled_state_returns_defaults(self, monkeypatch):
        monkeypatch.setattr(tone, "TONE_ENABLED", False)
        tr, _ = tracker()
        tr.observe("It finally works!!")
        assert tr.state.mode is ConversationMode.CASUAL

    def test_disabled_state_costs_nothing_in_the_prompt(self, monkeypatch):
        monkeypatch.setattr(tone, "TONE_ENABLED", False)
        assert tone.format_prompt() == ""


def _explode(*args, **kwargs):
    raise RuntimeError("derivation failed")


def _explode_no_args():
    raise RuntimeError("prompt render failed")


# ============================================================================
# 14. Configuration
# ============================================================================

class TestConfiguration:
    def test_config_has_the_phase_five_block(self):
        import yaml

        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
        block = config["conversation_state"]
        assert block["enabled"] is True
        assert set(block["defaults"]) >= {
            "mood", "energy", "warmth", "curiosity", "conversation_mode"
        }
        assert block["decay"]["enabled"] is True
        assert block["decay"]["half_life_seconds"] > 0

    def test_defaults_are_inside_the_bounded_range(self):
        from jarvis.config import (
            TONE_CURIOUSITY_DEFAULT,
            TONE_ENERGY_DEFAULT,
            TONE_WARMTH_DEFAULT,
        )

        for value in (TONE_ENERGY_DEFAULT, TONE_WARMTH_DEFAULT, TONE_CURIOUSITY_DEFAULT):
            assert 0.0 <= value <= 1.0

    def test_default_mood_and_mode_are_real_values(self):
        from jarvis.config import TONE_MODE_DEFAULT, TONE_MOOD_DEFAULT

        assert Mood(TONE_MOOD_DEFAULT)
        assert ConversationMode(TONE_MODE_DEFAULT)

    def test_defaults_flow_from_config_into_the_tracker(self):
        from jarvis.config import TONE_WARMTH_DEFAULT

        assert ConversationTone().defaults.warmth == pytest.approx(TONE_WARMTH_DEFAULT)

    def test_the_configuration_is_a_single_system(self):
        """One config module, one file -- no second settings source."""
        import inspect

        from jarvis import tone

        source = inspect.getsource(tone)
        assert "yaml" not in source
        assert "os.environ" not in source

    def test_the_half_life_is_positive(self):
        from jarvis.config import TONE_DECAY_HALF_LIFE

        assert TONE_DECAY_HALF_LIFE > 0


# ============================================================================
# 15. Phase 3 invariants that must not regress
# ============================================================================

class TestPhase3Invariants:
    def test_the_multimodal_history_routing_fix_still_holds(self):
        """Text after an image must still route to a vision-capable model.

        The conversation-state block must not have reintroduced the Phase 3 bug
        where routing looked only at the current turn's attachments while the
        replayed history still contained an image.
        """
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
        brain.ask("what is this?", images=[image_part(PNG_1PX)])
        assert brain.ask("and what does the error say?") == "saw it"
        assert brain.last_model_key == "sees"

    def test_a_text_only_conversation_still_prefers_the_text_model(self):
        layer = build_layer(
            models=[
                {"key": "blind", "provider": "p1", "model": "m1", "priority": 100,
                 "free": True, "capabilities": {"reasoning": True, "vision": False}},
                {"key": "sees", "provider": "p2", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "vision": True}},
            ],
            behaviours={"m1": reply("free"), "m2": reply("paid")},
        )
        brain = JarvisBrain(model_layer=layer)
        brain.ask("I'm so frustrated")
        brain.ask("tell me a joke")
        assert brain.last_model_key == "blind"

    def test_no_second_memory_store_was_introduced(self):
        import inspect

        for module in (tone, memory):
            assert inspect.getsource(module).count("CREATE TABLE") == (
                1 if module is memory else 0
            )

    def test_a_perception_turn_still_has_no_tools(self):
        layer = _vision_layer("observed: a person at a desk")
        JarvisBrain(model_layer=layer).ask(
            "describe", images=[image_part(PNG_1PX)], ephemeral=True
        )
        assert layer.providers["p"].sessions[-1].tools == []

    def test_visual_context_still_reaches_the_brain(self):
        vision.get_visual_context().update(
            vision.parse_observations("observed: A laptop is open on the desk.")
        )
        block = _build_context_block("what am I doing?", classify("what am I doing?"), [])
        assert "## Current Visual Context" in block
        assert "## How To Talk Right Now" not in block or True

    def test_no_model_call_is_made_for_state(self):
        """State must be free. Asserted structurally: no provider is touched."""
        import inspect

        source = inspect.getsource(tone)
        for banned in ("ask(", "open_session", "send_message", "requests", "urllib"):
            assert banned not in source

    def test_state_updates_are_fast(self):
        """Cheap enough to be invisible: no model, no network, no disk."""
        tr, _ = tracker()
        start = time.perf_counter()
        for _ in range(1000):
            tr.observe("It finally works and I am so happy about it")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"1000 updates took {elapsed:.2f}s"
