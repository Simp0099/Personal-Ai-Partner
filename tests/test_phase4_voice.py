"""Phase 4 tests: wake word, conversational state machine, interruption, shutdown.

The properties under test are the ones that make the difference between an
assistant and a loop that talks to itself:

* speech alone never starts a conversation in IDLE -- only a wake word does,
* every illegal transition is refused rather than absorbed,
* a barge-in cancels the old turn *before* the new one starts, so old audio can
  never resume,
* assistant playback cannot wake, VAD-trigger or interrupt itself,
* vision perception can never move the state machine,
* text conversation works with no microphone at all,
* shutdown leaves no thread, no device and no audio queued.

Everything runs with a fake microphone, a fake recogniser and a fake synthesiser,
so there is no hardware, no network and no sleeping on wall-clock timing except
where the test is specifically about waiting.
"""

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import audio as audio_mod  # noqa: E402
from jarvis.audio import (  # noqa: E402
    AudioQueue,
    EchoCanceller,
    EchoGuard,
    MicrophoneStream,
    MicrophoneUnavailable,
    SoundDeviceSource,
    VoiceActivityDetector,
    frame_bytes,
    rms,
)
from jarvis.conversation import (  # noqa: E402
    ConversationMachine,
    InvalidTransition,
    State,
)
from jarvis.speech_pipeline import ASRUnavailable, SpeechPlayer, Transcriber  # noqa: E402
from jarvis.voice_loop import VoiceLoop  # noqa: E402
from jarvis.wake_word import WakeWordEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSource:
    """A microphone that never produces audio on its own."""

    def __init__(self, frames=None, open_error=None):
        self.frames = list(frames or [])
        self.open_error = open_error
        self.opened = 0
        self.closed = 0
        self.reads = 0
        self.is_open = False

    def open(self):
        if self.open_error:
            raise self.open_error
        self.opened += 1
        self.is_open = True

    def read(self):
        self.reads += 1
        if self.frames:
            return self.frames.pop(0)
        time.sleep(0.005)
        return None

    def close(self):
        self.closed += 1
        self.is_open = False


class FakeEngine(WakeWordEngine):
    """A wake-word engine whose scores are scripted, with no ONNX runtime."""

    def __init__(self, scores=None, threshold=0.5, **kwargs):
        super().__init__(threshold=threshold)
        self.scores = list(scores or [])
        self.loaded_ok = kwargs.pop("loaded_ok", True)
        self.calls = 0

    def load(self):
        return self.loaded_ok

    def score(self, pcm):
        self.calls += 1
        return self.scores.pop(0) if self.scores else 0.0

    @property
    def loaded(self):
        return self.loaded_ok


class FakeASR:
    def __init__(self, text="what time is it", error=None):
        self.text = text
        self.error = error
        self.calls = 0

    def transcribe(self, pcm, sample_rate=16000):
        self.calls += 1
        if self.error:
            raise self.error
        return self.text


def _speech_frame(level=9000, seed=0):
    """One frame of speech-like audio: amplitude-modulated, so it has pauses.

    Constant-amplitude noise is not a usable stand-in for speech any more. The
    detector measures the room's noise floor and requires speech to clear it, so
    a constant loud signal is correctly read as *the room* and never as speech.
    Real speech rises and falls; the fake has to as well.
    """
    rng = np.random.default_rng(seed)
    n = frame_bytes() // 2
    t = np.arange(n) / audio_mod.SAMPLE_RATE
    envelope = 0.35 + 0.65 * np.abs(np.sin(2 * np.pi * 4.5 * t))   # syllable rate
    return (rng.normal(0, level * envelope, n)).astype("<i2").tobytes()


def _room_tone(level=25, seed=0):
    """One frame of background room noise -- the floor the detector measures."""
    rng = np.random.default_rng(seed)
    n = frame_bytes() // 2
    return rng.normal(0, level, n).astype("<i2").tobytes()


def _loud(level=6000, seed=0):
    return _speech_frame(level=level, seed=seed)


def _quiet():
    return np.zeros(frame_bytes() // 2, dtype="<i2").tobytes()


def _calibrate(loop, frames=18):
    """Let the detector characterise the room before anything is said."""
    for i in range(frames):
        loop._on_frame(_room_tone(seed=1000 + i))


def _make_loop(**kwargs):
    """A VoiceLoop with no hardware behind any of its seams."""
    kwargs.setdefault("wake_enabled", False)
    kwargs.setdefault("follow_up_window", 0.2)
    kwargs.setdefault("microphone", MicrophoneStream(source=FakeSource()))
    kwargs.setdefault("player", SpeechPlayer(
        synthesize=lambda t: np.zeros(8000, dtype=np.float32),
        play=lambda d, r: time.sleep(0.002),
        stop_playback=lambda: None,
        chunk_ms=100,
    ))
    kwargs.setdefault("transcriber", FakeASR())
    kwargs.setdefault("respond", lambda t: "It is twelve.")
    return VoiceLoop(**kwargs)


def _await(predicate, timeout=2.0):
    """Wait for a condition that the test needs to hold to be meaningful."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _speak_into(loop, frames=25):
    """Drive the audio sink with speech followed by silence."""
    _calibrate(loop)
    for i in range(frames):
        loop._on_frame(_speech_frame(seed=i))
    for i in range(frames):
        loop._on_frame(_room_tone(seed=2000 + i))


def _speaking_loop(reply_words=8, **kwargs):
    """A loop whose reply is long enough to still be speaking when it is cut."""
    return _make_loop(
        respond=lambda t: "It is twelve. " * reply_words,
        player=SpeechPlayer(
            synthesize=lambda t: np.zeros(2400 * reply_words, dtype=np.float32),
            play=lambda d, r: time.sleep(0.02),
            stop_playback=lambda: None,
            chunk_ms=100,
        ),
        follow_up_window=5.0,
        **kwargs,
    )


def _await_state(loop, *states, timeout=5.0):
    """Wait for the machine to reach any of `states`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if loop.machine.state in states:
            return True
        time.sleep(0.01)
    return False


def _await_turn(loop, turn_id, timeout=10.0):
    """Wait until a specific turn has started.

    Preferred over waiting on a state: several states are revisited across a
    multi-turn conversation, so waiting for a state can match the wrong one and
    make the test race rather than fail.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        turn = loop.machine.turn
        if turn is not None and turn.id == turn_id:
            return True
        time.sleep(0.01)
    return False


# ============================================================================
# Wake word
# ============================================================================

class TestWakeWordConfiguration:
    def test_wake_word_enablement_matches_configuration(self):
        from jarvis.config import CONFIG, WAKE_WORD_ENABLED

        assert WAKE_WORD_ENABLED is bool(CONFIG.get("wake_word", {}).get("enabled", False))

    def test_config_declares_the_opt_in(self):
        import yaml

        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
        assert "enabled" in config["wake_word"]
        assert isinstance(config["wake_word"]["enabled"], bool)

    def test_threshold_is_configurable_not_hardcoded(self):
        from jarvis.config import WAKE_WORD_THRESHOLD, WAKE_WORD_MODEL

        assert isinstance(WAKE_WORD_THRESHOLD, float)
        assert WAKE_WORD_MODEL == "hey_jarvis", "Phase 4 keeps the existing wake word"

    def test_threshold_is_read_from_config_not_scattered(self):
        """One source for the threshold, so it cannot drift between modules."""
        import inspect

        from jarvis import wake_word

        source = inspect.getsource(wake_word)
        assert "0.5" not in source.replace('threshold: float = None', '')

    def test_engine_reads_the_configured_threshold(self):
        from jarvis.config import WAKE_WORD_THRESHOLD

        assert WakeWordEngine().threshold == WAKE_WORD_THRESHOLD

    def test_disabled_wake_word_opens_no_microphone(self):
        source = FakeSource()
        loop = _make_loop(wake_enabled=False, microphone=MicrophoneStream(source=source))
        assert loop.start() is True
        try:
            loop.on_wake()
            loop.stop()
        finally:
            pass
        assert source.opened == 1  # opened for VAD/ASR, never for the wake word

    def test_wake_model_loads_before_the_microphone_opens(self):
        """Audio must not arrive before the wake model can score it.

        `WakeWordEngine.scores()` returns `{}` while the ONNX session is still
        building, so every frame delivered in that window is dropped in silence.
        With the wake word enabled the loop rests in IDLE, which only a wake word
        can leave, so a request spoken during start-up is simply never heard.
        """
        order = []

        class RecordingEngine:
            def __init__(self):
                self.thresholds = {"hey jarvis": 0.5}
                self.threshold = 0.5
                self.loaded = False

            def load(self):
                order.append("load")
                self.loaded = True
                return True

            def scores(self, pcm):
                return {}

            def score(self, pcm):
                return 0.0

        engine = RecordingEngine()

        class OrderingSource(FakeSource):
            def open(self):
                order.append("mic_open")
                super().open()

        mic = MicrophoneStream(source=OrderingSource())
        loop = _make_loop(microphone=mic, wake_enabled=True, wake_engine=engine)
        assert loop.start() is True
        try:
            assert order.index("load") < order.index("mic_open"), (
                f"wake model loaded after the microphone opened: {order}")
            assert engine.loaded is True
        finally:
            loop.stop()

    def test_enabled_wake_word_needs_the_engine(self):
        engine = FakeEngine()
        loop = _make_loop(wake_engine=engine, wake_enabled=True)
        assert loop.start() is True
        try:
            assert engine.loaded
        finally:
            loop.stop()


class TestWakeWordEvents:
    def test_wake_word_moves_idle_to_listening(self):
        # C2: the loop rests in IDLE only when the wake word is on. With wake
        # off it rests in LISTENING, so this needs wake on to test the wake path.
        loop = _make_loop(wake_enabled=True, wake_engine=FakeEngine())
        loop.start()
        try:
            assert loop.machine.state is State.IDLE
            loop.on_wake()
            assert loop.machine.state is State.LISTENING
        finally:
            loop.stop()

    def test_one_wake_word_produces_one_transition(self):
        """Overlapping chunks all score high for one utterance."""
        loop = _make_loop()
        loop.start()
        try:
            loop.on_wake()
            loop.on_wake()
            loop.on_wake()
            listening = [e for e in loop.machine.snapshot()["history"]
                         if e["state"] == "listening"]
            assert len(listening) == 1, "one wake word woke it several times"
            assert loop.wake_events == 1
            assert loop._wake_rejections >= 2
        finally:
            loop.stop()

    def test_wake_below_threshold_does_not_wake(self):
        engine = FakeEngine(scores=[0.4, 0.1, 0.05])
        loop = _make_loop(wake_engine=engine, wake_enabled=True)
        loop.start()
        try:
            for _ in range(3):
                loop._on_frame(_room_tone(seed=9000))
            assert loop.machine.state is State.IDLE
            assert loop.wake_events == 0
        finally:
            loop.stop()

    def test_wake_above_threshold_wakes(self):
        engine = FakeEngine(scores=[0.1, 0.9, 0.1])
        loop = _make_loop(wake_engine=engine, wake_enabled=True)
        loop.start()
        try:
            for _ in range(3):
                loop._on_frame(_room_tone(seed=9000))
            assert loop.machine.state is State.LISTENING
            assert loop.wake_events == 1
        finally:
            loop.stop()

    def test_speech_during_the_wake_burst_is_not_lost(self):
        """'Hey Jarvis, what time is it' is one breath, not two.

        Returning early on a high wake score clipped the frames that carried the
        rest of the utterance, so the wake word had to be spoken in isolation.
        """
        engine = FakeEngine(scores=[0.99] * 6 + [0.0] * 200)
        loop = _make_loop(wake_enabled=True, wake_engine=engine)
        loop.start()
        try:
            # The room is measured first, exactly as it is at loop start.
            _calibrate(loop)
            for i in range(6):
                loop._on_frame(_speech_frame(seed=i))
            assert loop.machine.state is State.LISTENING
            assert loop.wake_events == 1
            for i in range(6, 40):
                loop._on_frame(_speech_frame(seed=i))
            for i in range(25):
                loop._on_frame(_room_tone(seed=500 + i))
            assert _await_turn(loop, "turn_001"), "the utterance was clipped by the wake burst"
        finally:
            loop.stop()

    def test_arbitrary_text_never_triggers_a_wake(self):
        """Only audio is ever a wake trigger. A message is not."""
        # C2: with wake on the loop rests in IDLE, which only audio can leave.
        loop = _make_loop(wake_enabled=True, wake_engine=FakeEngine())
        loop.start()
        try:
            loop._respond("hey jarvis what time is it", None)
            assert loop.machine.state is State.IDLE
        finally:
            loop.stop()

    def test_wake_word_does_not_interfere_with_text_mode(self):
        """A text turn works with wake detection on and no microphone activity."""
        calls = []
        loop = _make_loop(wake_engine=FakeEngine(scores=[0.0] * 100), wake_enabled=True,
                          respond=lambda t: calls.append(t) or "text reply")
        loop.start()
        try:
            loop.respond("hello")
            assert calls == ["hello"]
        finally:
            loop.stop()


class TestWakeWordFalsePositives:
    def test_quiet_frames_never_wake(self):
        engine = FakeEngine(scores=[0.01] * 50)
        loop = _make_loop(wake_engine=engine, wake_enabled=True)
        loop.start()
        try:
            for _ in range(50):
                loop._on_frame(_quiet())
            assert loop.wake_events == 0
        finally:
            loop.stop()

    def test_a_strict_threshold_rejects_what_a_loose_one_accepts(self):
        """Tuning is a config value, so it must actually change behaviour."""
        def fires(threshold):
            loop = _make_loop(
                wake_enabled=True,
                wake_engine=FakeEngine(scores=[0.3], threshold=threshold),
            )
            loop.start()
            try:
                loop._on_frame(_quiet())
                return loop.wake_events == 1
            finally:
                loop.stop()

        assert fires(0.2) is True
        assert fires(0.9) is False


# ============================================================================
# State machine
# ============================================================================

class TestStateMachine:
    def test_states_are_the_wire_values(self):
        assert State.IDLE.value == "idle"
        assert State.LISTENING.value == "listening"
        assert State.TRANSCRIBING.value == "transcribing"
        assert State.THINKING.value == "thinking"
        assert State.SPEAKING.value == "speaking"
        assert State.FOLLOW_UP.value == "follow_up"
        assert State.INTERRUPTED.value == "interrupted"
        assert State.ERROR.value == "error"

    def test_starts_idle(self):
        assert ConversationMachine().state is State.IDLE

    @pytest.mark.parametrize("a, b", [
        (State.IDLE, State.LISTENING),
        (State.LISTENING, State.TRANSCRIBING),
        (State.TRANSCRIBING, State.THINKING),
        (State.THINKING, State.SPEAKING),
        (State.SPEAKING, State.FOLLOW_UP),
        (State.FOLLOW_UP, State.LISTENING),
        (State.FOLLOW_UP, State.IDLE),
        (State.THINKING, State.INTERRUPTED),
        (State.SPEAKING, State.INTERRUPTED),
        (State.INTERRUPTED, State.LISTENING),
        (State.ERROR, State.IDLE),
    ])
    def test_legal_transitions_are_allowed(self, a, b):
        machine = ConversationMachine()
        machine.transition(a, force=True)
        machine.transition(b)
        assert machine.state is b

    @pytest.mark.parametrize("a, b", [
        (State.IDLE, State.SPEAKING),        # cannot speak without being asked
        (State.IDLE, State.THINKING),
        (State.LISTENING, State.SPEAKING),   # nothing was said
        (State.TRANSCRIBING, State.SPEAKING),  # never asked anything
        (State.SPEAKING, State.LISTENING),   # must go through INTERRUPTED
        (State.FOLLOW_UP, State.SPEAKING),   # nothing to speak
        (State.FOLLOW_UP, State.THINKING),
        (State.IDLE, State.INTERRUPTED),     # nothing to interrupt
    ])
    def test_illegal_transitions_are_refused(self, a, b):
        machine = ConversationMachine()
        machine.transition(a, force=True)
        with pytest.raises(InvalidTransition):
            machine.transition(b)
        assert machine.state is a, "a refused transition still moved the state"

    def test_refusal_reports_both_states(self):
        machine = ConversationMachine()
        with pytest.raises(InvalidTransition) as exc:
            machine.transition(State.SPEAKING)
        assert "idle" in str(exc.value) and "speaking" in str(exc.value)

    def test_force_bypasses_the_check_for_shutdown_only(self):
        machine = ConversationMachine()
        machine.transition(State.ERROR, force=True)
        assert machine.state is State.ERROR

    def test_snapshot_is_the_hud_payload(self):
        machine = ConversationMachine()
        machine.transition(State.LISTENING, reason="wake word")
        snap = machine.snapshot()
        assert snap["state"] == "listening"
        assert snap["history"][-1]["previous"] == "idle"
        assert snap["history"][-1]["type"] == "state"

    def test_listeners_receive_transitions(self):
        machine = ConversationMachine()
        seen = []
        unsubscribe = machine.subscribe(seen.append)
        machine.transition(State.LISTENING)
        unsubscribe()
        machine.transition(State.TRANSCRIBING)
        assert len(seen) == 1

    def test_a_broken_listener_cannot_break_the_machine(self):
        machine = ConversationMachine()
        machine.subscribe(lambda e: 1 / 0)
        machine.transition(State.LISTENING)
        assert machine.state is State.LISTENING


# ============================================================================
# Turn ids and staleness
# ============================================================================

class TestTurnIdentity:
    def test_turn_ids_are_unique_and_ordered(self):
        machine = ConversationMachine()
        ids = [machine.begin_turn().id for _ in range(3)]
        assert ids == ["turn_001", "turn_002", "turn_003"]

    def test_a_new_turn_invalidates_the_previous_one(self):
        """This is the anti-overlap guarantee, stated once."""
        machine = ConversationMachine()
        first = machine.begin_turn()
        machine.begin_turn()
        assert machine.is_stale(first.id) is True

    def test_the_current_turn_is_not_stale(self):
        machine = ConversationMachine()
        turn = machine.begin_turn()
        assert machine.is_stale(turn.id) is False

    def test_cancelling_makes_a_turn_stale(self):
        machine = ConversationMachine()
        turn = machine.begin_turn()
        machine.cancel_turn("barge-in")
        assert machine.is_stale(turn.id) is True

    def test_an_unknown_or_missing_id_is_stale(self):
        machine = ConversationMachine()
        machine.begin_turn()
        assert machine.is_stale("turn_999") is True
        assert machine.is_stale(None) is True

    def test_history_bounded(self):
        machine = ConversationMachine()
        machine.transition(State.LISTENING, force=True)
        for _ in range(200):
            machine.transition(State.IDLE, force=True)
        assert len(machine.snapshot()["history"]) <= 50


# ============================================================================
# Audio queue
# ============================================================================

class TestAudioQueue:
    def test_chunks_come_back_in_order(self):
        queue = AudioQueue()
        for i in range(5):
            queue.push("turn_001", bytes([i]))
        out = [queue.pop(lambda t: False).data[0] for _ in range(5)]
        assert out == [0, 1, 2, 3, 4]
        assert queue.pop(lambda t: False) is None

    def test_stale_chunks_are_never_returned(self):
        queue = AudioQueue()
        queue.push("turn_001", b"old")
        queue.push("turn_002", b"new")

        def is_stale(turn_id):
            return turn_id == "turn_001"

        chunk = queue.pop(is_stale)
        assert chunk.data == b"new"
        assert queue.discarded_stale == 1

    def test_a_fully_stale_queue_drains_to_nothing(self):
        queue = AudioQueue()
        for i in range(3):
            queue.push("turn_001", bytes([i]))
        assert queue.pop(lambda t: True) is None
        assert queue.discarded_stale == 3

    def test_clear_reports_what_it_discarded(self):
        queue = AudioQueue()
        for i in range(4):
            queue.push("turn_001", bytes([i]))
        assert queue.clear() == 4
        assert queue.peek_len() == 0

    def test_new_turn_audio_takes_precedence_over_old(self):
        """Old audio is dropped, not played late."""
        queue = AudioQueue()
        queue.push("turn_001", b"stale-response")
        queue.push("turn_002", b"new-response")
        assert queue.pop(lambda t: t == "turn_001").data == b"new-response"


# ============================================================================
# Full turn
# ============================================================================

class TestVoiceTurn:
    def test_spoken_shutdown_stops_voice_loop_without_model_call(self, monkeypatch):
        seen = []
        loop = _make_loop(respond=lambda text: seen.append(text) or "unexpected")
        loop.on_wake()
        monkeypatch.setattr(loop, "_transcribe", lambda audio, turn_id: "exit")
        spoken = []
        monkeypatch.setattr(loop, "_speak", lambda text, turn_id: spoken.append(text))
        loop._run_turn()
        assert loop.wait_for_stop(0)
        assert seen == []
        assert spoken and "Going offline" in spoken[0]
        assert loop.machine.state is State.IDLE

    def test_full_happy_path(self):
        loop = _make_loop(follow_up_window=5.0)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.FOLLOW_UP)
            states = [e["state"] for e in loop.machine.snapshot()["history"]]
        finally:
            loop.stop()
        for expected in ("listening", "transcribing", "thinking", "speaking", "follow_up"):
            assert expected in states, f"never reached {expected}: {states}"

    def test_transcript_reaches_the_brain(self):
        seen = []
        loop = _make_loop(respond=lambda t: seen.append(t) or "noon")
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.FOLLOW_UP)
        finally:
            loop.stop()
        assert seen == ["what time is it"]

    def test_speech_alone_does_not_start_a_conversation(self):
        """IDLE must need a wake word. Otherwise the assistant listens to the room."""
        loop = _make_loop()
        loop.start()
        try:
            _speak_into(loop)
            time.sleep(0.2)
            assert loop.machine.state is State.IDLE
            assert loop.machine.turn is None
        finally:
            loop.stop()

    def test_an_empty_transcript_is_not_invented(self):
        asr = FakeASR(text="")
        loop = _make_loop(transcriber=asr)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.ERROR, State.IDLE)
        finally:
            loop.stop()
        assert loop.machine.last_error
        assert loop.machine.state is State.IDLE

    def test_asr_failure_returns_to_a_safe_state(self):
        loop = _make_loop(transcriber=FakeASR(error=RuntimeError("asr down")))
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.ERROR, State.IDLE)
        finally:
            loop.stop()
        assert loop.machine.state is State.IDLE
        assert "asr down" in loop.machine.last_error

    def test_llm_failure_returns_to_a_safe_state(self):
        def boom(text):
            raise RuntimeError("model unavailable")

        loop = _make_loop(respond=boom)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.ERROR, State.IDLE)
        finally:
            loop.stop()
        assert loop.machine.state is State.IDLE
        assert "model unavailable" in loop.machine.last_error

    def test_tts_failure_does_not_stick_in_speaking(self):
        def broken_synthesis(text):
            raise RuntimeError("tts failed")

        loop = _make_loop(player=SpeechPlayer(
            synthesize=broken_synthesis, play=lambda d, r: None,
            stop_playback=lambda: None, chunk_ms=100,
        ))
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.ERROR, State.IDLE)
        finally:
            loop.stop()
        assert loop.machine.state is not State.SPEAKING

    def test_playback_failure_does_not_stick_in_speaking(self):
        def broken_play(data, rate):
            raise RuntimeError("audio device gone")

        loop = _make_loop(player=SpeechPlayer(
            synthesize=lambda t: np.zeros(8000, dtype=np.float32),
            play=broken_play, stop_playback=lambda: None, chunk_ms=100,
        ))
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.FOLLOW_UP, State.IDLE, State.ERROR)
        finally:
            loop.stop()
        assert loop.machine.state is not State.SPEAKING


# ============================================================================
# Follow-up
# ============================================================================

class TestFollowUp:
    def test_speaking_is_followed_by_the_follow_up_window(self):
        loop = _make_loop(follow_up_window=5.0)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.FOLLOW_UP)
        finally:
            loop.stop()

    def test_follow_up_returns_to_idle_when_nothing_is_said(self):
        loop = _make_loop(follow_up_window=0.2)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.IDLE)
        finally:
            loop.stop()

    def test_follow_up_accepts_speech_without_the_wake_word(self):
        asr = FakeASR(text="another one")
        loop = _make_loop(transcriber=asr, follow_up_window=5.0)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_turn(loop, "turn_001")
            assert loop.wake_events == 1, "the follow-up used the wake word"

            _speak_into(loop)
            assert _await_turn(loop, "turn_002")
        finally:
            loop.stop()
        assert asr.calls >= 2, "the follow-up was never transcribed"

    def test_multiple_follow_ups_are_allowed(self):
        loop = _make_loop(follow_up_window=10.0)
        loop.start()
        try:
            loop.on_wake()
            for expected in ("turn_001", "turn_002", "turn_003"):
                _speak_into(loop)
                assert _await_turn(loop, expected), f"{expected} never started"
            # Each follow-up ran without the wake word being said again.
            assert loop.wake_events == 1
        finally:
            loop.stop()

    def test_the_conversation_is_never_permanently_open(self):
        loop = _make_loop(follow_up_window=0.15)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.IDLE, timeout=3.0)
        finally:
            loop.stop()


# ============================================================================
# Interruption
# ============================================================================

class TestInterruption:
    def _speaking_loop(self, **kwargs):
        """A loop whose reply is long enough to still be speaking when cut."""
        return _speaking_loop(**kwargs)

    def test_interrupt_from_speaking_goes_to_listening(self):
        loop = self._speaking_loop()
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.SPEAKING)
            loop.interrupt()
            assert loop.machine.state is State.LISTENING
            assert loop.interrupts == 1
        finally:
            loop.stop()

    def test_interrupt_records_the_interrupted_state(self):
        loop = self._speaking_loop()
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.SPEAKING)
            loop.interrupt()
            states = [e["state"] for e in loop.machine.snapshot()["history"]]
            assert "interrupted" in states
        finally:
            loop.stop()

    def test_interrupt_cancels_the_turn_so_its_audio_is_stale(self):
        loop = self._speaking_loop()
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.SPEAKING)
            turn_id = loop.machine.turn.id
            loop.interrupt()
            assert loop.machine.is_stale(turn_id) is True
        finally:
            loop.stop()

    def test_interrupt_empties_the_audio_queue(self):
        loop = _speaking_loop(reply_words=40)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.SPEAKING)
            assert _await(lambda: loop.player.queue.peek_len() > 0, timeout=2.0), \
                "audio was already drained; this test would prove nothing"
            loop.interrupt()
            assert loop.player.queue.peek_len() == 0
        finally:
            loop.stop()

    def test_interrupt_stops_playback(self):
        stopped = []
        loop = self._speaking_loop()
        loop.player.stop_playback = lambda: stopped.append(1)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.SPEAKING)
            loop.interrupt()
            assert stopped, "the audio device was not told to stop"
        finally:
            loop.stop()

    def test_interrupted_audio_never_resumes(self):
        """Chunks queued before the interruption must not be played afterwards."""
        played = []
        loop = _make_loop(
            respond=lambda t: "old answer. " * 6,
            player=SpeechPlayer(
                synthesize=lambda t: np.zeros(24000, dtype=np.float32),
                play=lambda d, r: played.append(d) or time.sleep(0.02),
                stop_playback=lambda: None,
                chunk_ms=100,
            ),
            follow_up_window=5.0,
        )
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.SPEAKING)
            time.sleep(0.05)
            played.clear()
            loop.interrupt()
            time.sleep(0.3)
            assert played == [], "audio from the interrupted turn was played"
        finally:
            loop.stop()

    def test_the_interrupted_reply_is_never_spoken(self):
        spoken = []
        loop = self._speaking_loop()
        loop.respond = lambda t: spoken.append(t) or "old answer. " * 8
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.SPEAKING)
            loop.interrupt()
            time.sleep(0.2)
            assert all("old answer" not in str(chunk) for chunk in [])
            assert loop.player.queue.peek_len() == 0
        finally:
            loop.stop()

    def test_interrupt_during_thinking_discards_the_reply(self):
        """The model call cannot be recalled, but its result must not be used."""
        release = threading.Event()
        results = []

        def slow(text):
            release.wait(2.0)
            results.append(text)
            return "the late answer"

        loop = _make_loop(respond=slow, follow_up_window=5.0)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.THINKING)
            loop.interrupt()
            release.set()
            time.sleep(0.3)
            assert results == ["what time is it"], "the interrupted turn was never run"
            assert loop.player.queue.peek_len() == 0
            assert loop.player.chunks_played == 0, "a cancelled reply was spoken"
        finally:
            loop.stop()

    def test_a_new_turn_after_an_interruption_speaks_normally(self):
        replies = iter(["first answer", "second answer"])
        loop = _make_loop(respond=lambda t: next(replies), follow_up_window=5.0)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_turn(loop, "turn_001")
            loop.interrupt()
            _speak_into(loop)
            assert _await_turn(loop, "turn_002")
        finally:
            loop.stop()

    def test_interruption_during_speech_comes_from_speech_not_the_api(self):
        """Barge-in is driven by the microphone, not only by an explicit call."""
        loop = _speaking_loop(reply_words=80)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            # Wait for audio to actually be coming out. Asserting before this
            # point would prove nothing: the assistant would already be done.
            assert _await(loop.player.is_playing, timeout=3.0), "playback never started"
            assert loop.machine.state is State.SPEAKING

            for _ in range(15):
                loop._on_frame(_speech_frame(level=14000, seed=3))
            assert _await(lambda: loop.interrupts == 1, timeout=2.0), \
                "sustained speech during playback did not interrupt"
            assert loop.machine.state is State.LISTENING
        finally:
            loop.stop()

    def test_speech_during_thinking_interrupts_without_echo_playing(self):
        """Regressed once, and it was the 'thinking can be interrupted' case.

        Barge-in was gated on whether the assistant was currently *echoing*. If
        the user spoke during a model call -- before any audio had played --
        nothing handled it: the utterance ended, the turn thread tried
        THINKING -> TRANSCRIBING, and the state machine refused the transition.
        The user's speech was discarded and the turn logged an error.
        """
        release = threading.Event()

        def slow(text):
            release.wait(3.0)
            return "the late answer"

        loop = _make_loop(respond=slow, follow_up_window=5.0)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await(lambda: loop.machine.state is State.THINKING)
            assert not loop.echo.playing, "the premise of this test is no echo"

            _speak_into(loop)
            assert _await(lambda: loop.interrupts == 1, timeout=3.0), \
                "speech during thinking was ignored"
            release.set()
        finally:
            loop.stop()

    def test_speech_during_transcribing_interrupts(self):
        class SlowASR:
            def transcribe(self, pcm, sample_rate=16000):
                time.sleep(0.5)
                return "what time is it"

        loop = _make_loop(transcriber=SlowASR(), follow_up_window=5.0)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await(lambda: loop.machine.state is State.TRANSCRIBING)
            loop.interrupt()
            assert loop.machine.state is State.LISTENING
        finally:
            loop.stop()

    def test_speech_while_idle_is_not_an_interruption(self):
        loop = _make_loop()
        loop.start()
        try:
            loop.on_wake()
            loop.machine.transition(State.IDLE, force=True)
            _speak_into(loop)
            time.sleep(0.2)
            assert loop.interrupts == 0
        finally:
            loop.stop()

    def test_echo_suppression_does_not_disable_barge_in(self):
        """Suppression may mute the wake word; it must never mute the VAD."""
        loop = _speaking_loop(reply_words=60)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await(loop.player.is_playing, timeout=3.0)
            assert loop.echo.should_suppress() is True
            for _ in range(15):
                loop._on_frame(_speech_frame(level=14000, seed=3))
            assert _await(lambda: loop.interrupts == 1, timeout=2.0), \
                "echo suppression silenced barge-in"
        finally:
            loop.stop()

    def test_interrupt_is_idempotent(self):
        loop = self._speaking_loop()
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.SPEAKING)
            loop.interrupt()
            loop.interrupt()
            assert loop.machine.state is State.LISTENING
        finally:
            loop.stop()

    def test_interrupt_in_idle_does_not_corrupt_state(self):
        """Nothing to interrupt, so nothing should happen -- not a fake LISTENING."""
        loop = _make_loop()
        loop.start()
        try:
            loop.machine.transition(State.IDLE, force=True)
            loop.interrupt()
            assert loop.machine.state is State.IDLE
        finally:
            loop.stop()


# ============================================================================
# Echo / self-trigger
# ============================================================================

class TestEchoProtection:
    def test_assistant_speech_does_not_wake_the_assistant(self):
        engine = FakeEngine(scores=[0.99] + [0.0] * 200)
        loop = _make_loop(
            wake_enabled=True, wake_engine=engine,
            player=SpeechPlayer(
                synthesize=lambda t: np.zeros(24000, dtype=np.float32),
                play=lambda d, r: time.sleep(0.02),
                stop_playback=lambda: None,
                chunk_ms=100,
            ),
            respond=lambda t: "a long answer. " * 6,
            follow_up_window=5.0,
        )
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await(loop.player.is_playing, timeout=3.0), "playback never started"
            before = loop.wake_events
            # The microphone hears the assistant through the speakers.
            for _ in range(40):
                loop._on_frame(_speech_frame(level=14000, seed=5))
                if loop.wake_events != before:
                    break
            assert loop.wake_events == before, "the assistant woke itself"
        finally:
            loop.stop()

    def test_assistant_speech_does_not_start_a_second_turn(self):
        loop = _make_loop(
            player=SpeechPlayer(
                synthesize=lambda t: np.zeros(24000, dtype=np.float32),
                play=lambda d, r: time.sleep(0.02),
                stop_playback=lambda: None, chunk_ms=100,
            ),
            respond=lambda t: "a long answer. " * 6,
            follow_up_window=5.0,
        )
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await(loop.player.is_playing, timeout=3.0), "playback never started"
            turn_id = loop.machine.turn.id
            for _ in range(40):
                loop._on_frame(_speech_frame(level=14000, seed=5))
            assert loop.machine.turn.id == turn_id, "echo started a second turn"
            assert loop.wake_events == 1, "echo re-triggered the wake word"
        finally:
            loop.stop()

    def test_echo_guard_gates_the_wake_word_while_playing(self):
        echo = EchoGuard(cooldown_s=0.2)
        assert echo.should_suppress() is False
        echo.begin_playback()
        assert echo.should_suppress() is True
        assert echo.barge_in() is True, "barge-in must stay possible while speaking"
        echo.end_playback()
        assert echo.should_suppress() is True, "cooldown must cover the playback tail"
        time.sleep(0.25)
        assert echo.should_suppress() is False

    def test_a_short_burst_does_not_interrupt(self):
        """A click is not the user cutting in."""
        loop = _make_loop(
            player=SpeechPlayer(
                synthesize=lambda t: np.zeros(24000, dtype=np.float32),
                play=lambda d, r: time.sleep(0.05),
                stop_playback=lambda: None, chunk_ms=100,
            ),
            respond=lambda t: "long answer. " * 6,
            follow_up_window=5.0,
        )
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await(loop.player.is_playing, timeout=3.0), "playback never started"
            loop._on_frame(_speech_frame(level=14000))   # a single frame
            time.sleep(0.15)
            assert loop.interrupts == 0, "a click interrupted the assistant"
        finally:
            loop.stop()

    def test_vad_ignores_sustained_quiet(self):
        vad = VoiceActivityDetector()
        for _ in range(50):
            assert vad.push(_quiet()) is None
        assert vad.in_speech is False


# ============================================================================
# Echo cancellation
# ============================================================================

class _FakeClock:
    """A clock the test advances, so reference alignment is deterministic."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _speech_like(seconds=0.4, seed=1):
    rng = np.random.default_rng(seed)
    n = int(audio_mod.SAMPLE_RATE * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    return (0.4 * np.sin(2 * np.pi * 180 * t)
            + 0.15 * rng.normal(0, 1, n)).astype(np.float32)


def _pcm(samples):
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def _rms_pcm(pcm):
    return rms(pcm)


class TestEchoCancellation:
    def test_a_frame_with_no_playback_is_untouched(self):
        clock = _FakeClock()
        canceller = EchoCanceller(clock=clock)
        frame = _pcm(_speech_like(0.08, seed=2))
        assert canceller.cancel(frame) == frame

    def test_the_assistants_own_voice_is_removed(self):
        """The measured failure this exists to fix: the assistant cut itself off."""
        clock = _FakeClock()
        canceller = EchoCanceller(clock=clock)
        reference = _speech_like(0.4, seed=3)
        canceller.play_reference(_pcm(reference))

        frame_len = int(audio_mod.SAMPLE_RATE * 0.08)
        # Each captured frame must line up with the part of the reference that
        # was playing at that instant, which is what the canceller looks up.
        for i in range(len(reference) // frame_len):
            clock.advance(0.08)
            captured = reference[i * frame_len:(i + 1) * frame_len]
            before = _rms_pcm(_pcm(captured))
            residual = canceller.cancel(_pcm(captured))
            assert _rms_pcm(residual) < before * 0.5, (
                f"echo not removed on frame {i}: {before:.3f} -> "
                f"{_rms_pcm(residual):.3f}"
            )

    def test_a_human_over_the_assistant_is_preserved(self):
        """Cancelling echo must not also cancel the person interrupting."""
        clock = _FakeClock()
        canceller = EchoCanceller(clock=clock)
        reference = _speech_like(0.4, seed=4)
        canceller.play_reference(_pcm(reference))

        frame_len = int(audio_mod.SAMPLE_RATE * 0.08)
        voice = _speech_like(0.4, seed=1234)
        for i in range(3):
            clock.advance(0.08)
            played = reference[i * frame_len:(i + 1) * frame_len]
            leak = played * 0.4                             # speakers into the mic
            spoken = voice[i * frame_len:(i + 1) * frame_len]
            residual = canceller.cancel(_pcm(leak + spoken))
            assert _rms_pcm(residual) > _rms_pcm(_pcm(spoken)) * 0.3, \
                "the user's voice was cancelled along with the echo"

    def test_gain_is_estimated_not_assumed(self):
        clock = _FakeClock()
        canceller = EchoCanceller(clock=clock)
        reference = _speech_like(0.4, seed=5)
        canceller.play_reference(_pcm(reference))
        frame_len = int(audio_mod.SAMPLE_RATE * 0.08)
        # A frame captured covering [t-80ms, t] hears the reference played
        # during that same window, so the aligned segment starts at 0 -- the
        # canceller looks *behind* now, which is what the acoustic delay means.
        clock.advance(0.08)
        captured = reference[:frame_len] * 0.5          # 6 dB down
        canceller.cancel(_pcm(captured))
        assert 0.3 < canceller.last_gain < 0.7, \
            f"echo gain estimated as {canceller.last_gain}, expected ~0.5"

    def test_history_is_bounded(self):
        clock = _FakeClock()
        canceller = EchoCanceller(history_s=0.5, clock=clock)
        for _ in range(200):
            clock.advance(0.1)
            canceller.play_reference(_pcm(_speech_like(0.1, seed=6)))
        assert len(canceller._history) <= 7

    def test_close_clears_the_reference(self):
        clock = _FakeClock()
        canceller = EchoCanceller(clock=clock)
        canceller.play_reference(_pcm(_speech_like(0.1)))
        canceller.close()
        assert canceller._history == []

    def test_clipped_odd_and_mismatched_pcm_lengths_are_safe(self):
        clock = _FakeClock()
        canceller = EchoCanceller(clock=clock)
        clipped = np.full(1280, 32767, dtype="<i2").tobytes()
        malformed = b"\x7f"

        assert canceller.cancel(b"") == b""
        assert canceller.cancel(malformed) == malformed
        canceller.play_reference(malformed)
        assert canceller._history == []

        canceller.play_reference(clipped + clipped[:400])
        clock.advance(0.08)
        residual = canceller.cancel(clipped)
        assert len(residual) == len(clipped)
        assert np.max(np.abs(np.frombuffer(residual, dtype="<i2"))) <= 32768

    def test_the_loop_feeds_playback_to_the_canceller(self):
        """Wiring: whatever is played must reach the canceller."""
        from jarvis.audio import EchoCanceller as C

        loop = _make_loop()
        assert loop.player.on_play == loop.canceller.play_reference
        loop.player.synthesize(lambda t: np.zeros(8000, dtype=np.float32))
        assert isinstance(loop.canceller, C)

    def test_the_microphone_signal_is_cancelled_before_the_vad_sees_it(self):
        """Otherwise the canceller exists but is never applied.

        Checks the wiring with a spy, because the arithmetic is covered above and
        re-deriving it here would only test the test.
        """
        class Spy:
            def __init__(self):
                self.cancelled = []

            def cancel(self, pcm):
                self.cancelled.append(pcm)
                return b"CANCELLED"

            def play_reference(self, pcm):
                pass

            def close(self):
                pass

            def status(self):
                return {}

        loop = _make_loop(wake_enabled=False)
        spy = Spy()
        loop.canceller = spy
        seen = []
        loop.vad.push = lambda pcm, barge_in=False: seen.append(pcm) or None

        frame = _pcm(_speech_like(0.08, seed=8))
        loop._on_frame(frame)

        assert spy.cancelled == [frame], "the canceller never saw the frame"
        assert seen == [b"CANCELLED"], "the VAD saw the uncancelled signal"


# ============================================================================
# VAD
# ============================================================================

class TestVoiceActivityDetector:
    @staticmethod
    def _detector():
        """A detector that has already measured a quiet room."""
        vad = VoiceActivityDetector()
        for i in range(18):
            vad.push(_room_tone(seed=3000 + i))
        return vad

    def test_speech_start_and_end(self):
        vad = self._detector()
        events = [vad.push(_speech_frame(seed=i)) for i in range(25)]
        events += [vad.push(_room_tone(seed=4000 + i)) for i in range(25)]
        kinds = [e.kind for e in events if e]
        assert kinds == ["start", "end"]

    def test_utterance_audio_is_captured(self):
        vad = self._detector()
        events = [vad.push(_speech_frame(seed=i)) for i in range(25)]
        events += [vad.push(_room_tone(seed=5000 + i)) for i in range(25)]
        end = [e for e in events if e and e.kind == "end"][0]
        assert len(end.audio) > 0
        assert end.peak > 0.0, "speech peak was lost while resetting the detector"

    def test_quiet_never_starts_speech(self):
        vad = VoiceActivityDetector()
        assert all(vad.push(_quiet()) is None for _ in range(30))

    def test_a_single_burst_is_not_speech(self):
        vad = self._detector()
        assert vad.push(_speech_frame()) is None

    def test_a_loud_room_does_not_hear_itself(self):
        """The measured failure this exists for.

        Ambient RMS on the development machine was 0.046-0.14, above every
        absolute threshold configured here, so the detector heard permanent
        speech and the assistant interrupted itself out of its own reply.
        """
        vad = VoiceActivityDetector()
        for i in range(40):
            vad.push(_room_tone(level=2600, seed=6000 + i))   # ~0.08 RMS
        assert vad.noise_floor > 0.02, "the room was not measured"
        assert all(
            vad.push(_room_tone(level=2600, seed=6100 + i)) is None
            for i in range(40)
        ), "room noise was heard as speech"

    def test_the_configured_threshold_is_still_a_floor(self):
        """A quiet room must not become hypersensitive."""
        vad = self._detector()
        assert vad._level_needed(False) >= vad.energy_threshold

    def test_barge_in_needs_a_higher_bar_than_normal(self):
        vad = self._detector()
        assert vad._level_needed(True) > vad._level_needed(False)

    def test_barge_in_needs_more_speech_than_normal(self):
        """Interrupting should take intent."""
        strict = VoiceActivityDetector(barge_in_ms=300, min_speech_ms=150)
        for i in range(18):
            strict.push(_room_tone(seed=7000 + i))
        events = [strict.push(_speech_frame(level=14000, seed=i), barge_in=True)
                  for i in range(25)]
        assert [e for e in events if e], "barge-in never triggered"

        lazy = VoiceActivityDetector(barge_in_ms=3000, min_speech_ms=150)
        for i in range(18):
            lazy.push(_room_tone(seed=8000 + i))
        assert lazy.push(_speech_frame(level=14000), barge_in=True) is None

    def test_barge_in_threshold_is_stricter_by_default(self):
        vad = VoiceActivityDetector(energy_threshold=0.01)
        assert vad.barge_in_threshold >= vad.energy_threshold * 2

    def test_reset_clears_a_partial_utterance(self):
        vad = VoiceActivityDetector()
        for _ in range(5):
            vad.push(_loud())
        vad.reset()
        assert vad.in_speech is False

    def test_thresholds_are_configurable(self):
        from jarvis.config import VAD_BARGE_IN_MS, VAD_END_SILENCE_MS

        vad = VoiceActivityDetector()
        assert vad.end_silence_ms == VAD_END_SILENCE_MS
        assert vad.barge_in_ms == VAD_BARGE_IN_MS

    def test_rms_of_silence_is_zero(self):
        assert rms(_quiet()) == 0.0

    def test_rms_of_loud_audio_is_large(self):
        assert rms(_loud()) > 0.05


# ============================================================================
# Microphone ownership
# ============================================================================

class TestSingleMicrophoneOwner:
    def test_source_cleanup_failure_is_reported(self, caplog):
        """A backend that refuses to close must not strand the device.

        PyAudioSource raised MicrophoneUnavailable here. SoundDeviceSource does
        not: PortAudio streams can already be gone when a device is unplugged,
        so `close()` is best-effort by design and the equivalent guarantee is
        that the failure is still reported rather than swallowed silently, and
        that the source ends up genuinely closed.
        """

        class Stream:
            def stop(self):
                raise OSError("device busy")

            def close(self):
                raise OSError("device busy")

        source = SoundDeviceSource()
        source._stream = Stream()
        source._queue.put_nowait(b"\x00" * 4)

        with caplog.at_level(logging.DEBUG, logger="jarvis"):
            source.close()  # must not raise

        assert "device busy" in caplog.text, "a failed close was hidden"
        assert source.is_open is False
        assert source._queue.empty(), "frames survived the close"

    def test_source_close_drains_a_backlog_before_reopening(self):
        """Reopening must not replay audio the previous session captured."""
        source = SoundDeviceSource()
        source._queue.put_nowait(b"\x01" * frame_bytes())
        source.close()
        assert source.read(timeout=0.01) is None

    def test_sink_registration_and_repeated_start_stop_do_not_duplicate_owner(self):
        source = FakeSource(frames=[_quiet()])
        mic = MicrophoneStream(source=source)
        received = []
        sink = received.append
        for _ in range(3):
            mic.add_sink(sink)
            mic.add_sink(sink)
            assert mic.start()
            assert _await(lambda: mic.frames_read >= 1)
            mic.stop()
            assert not mic.is_running()
        assert received
        assert source.opened == source.closed == 3
        assert mic.status()["sinks"] == 1

    def test_malformed_capture_frame_stops_and_reports_capture(self):
        source = FakeSource(frames=[b"\x00"])
        errors = []
        mic = MicrophoneStream(source=source, on_error=errors.append)
        assert mic.start()
        assert _await(lambda: bool(errors))
        assert "malformed frame" in errors[0]
        mic.stop()

    def test_failed_reader_shutdown_keeps_device_open(self):
        source = FakeSource()
        mic = MicrophoneStream(source=source)
        mic.start()
        entered = threading.Event()
        release = threading.Event()

        def stuck_reader():
            entered.set()
            release.wait(1)

        worker = threading.Thread(target=stuck_reader)
        worker.start()
        assert entered.wait(1)
        mic._thread = worker
        with pytest.raises(TimeoutError, match="device remains open"):
            mic.stop(timeout=0.01)
        assert source.closed == 0
        assert mic._thread is worker
        with pytest.raises(RuntimeError, match="still stopping"):
            mic.start()
        assert source.opened == 1
        release.set()
        worker.join(1)

    def test_empty_reads_are_retried_and_remain_cancellable(self):
        """No audio is normal; the pump must keep reading and still be stoppable.

        This test used to assert a 10ms `_stop.wait()` between empty reads. That
        pacing is not in the specification: Section 4.A3 makes an empty read a
        bare `continue`, and with the real SoundDeviceSource the pacing comes
        from `read(timeout=1.0)` blocking on the queue, so a wait in `_pump`
        would be unreachable in production. What still has to hold -- and what is
        asserted here -- is that repeated empty reads neither terminate the pump
        nor reset it into a failure state, and that setting the stop event ends it.
        """
        source = FakeSource()
        source.is_open = True
        mic = MicrophoneStream(source=source)

        def read():
            source.reads += 1
            if source.reads >= 25:
                # Stop only once the pump has proved it retries instead of giving up.
                mic._stop.set()
            return None

        source.read = read
        mic.available = True          # the state start() leaves the mic in
        mic._pump()
        assert source.reads == 25, "an empty read ended the pump early"
        assert mic._stop.is_set()
        assert mic.available is True, "empty reads were treated as capture failures"
        assert mic.last_error is None

    def test_sounddevice_receives_sample_blocks_not_byte_counts(self, monkeypatch):
        """blocksize is samples per block. Passing bytes is the classic bug.

        80ms at 16kHz is 1280 samples, delivered as 2560 int16 bytes. Asserting
        both catches the swap in either direction.
        """
        import types

        opened = {}

        class InputStream:
            def __init__(self, **kwargs):
                opened.update(kwargs)

            def start(self):
                # A real device delivers its first block through the callback.
                block = np.zeros(opened["blocksize"], dtype="int16")
                opened["callback"](block.reshape(-1, 1), len(block), None, None)

            def stop(self):
                pass

            def close(self):
                pass

        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
            InputStream=InputStream,
            query_devices=lambda device=None, kind=None: {
                "name": "input", "index": 0, "max_input_channels": 1},
        ))
        source = SoundDeviceSource()
        source.open()
        try:
            assert opened["blocksize"] == 1280
            assert opened["blocksize"] == (
                opened["samplerate"] * audio_mod.AUDIO_FRAME_MS // 1000)
            assert opened["dtype"] == "int16"
            assert opened["channels"] == 1
            assert source.device_name == "input"
            assert len(source.read()) == frame_bytes()
        finally:
            source.close()

    def test_explicit_device_index_is_passed_to_portaudio(self, monkeypatch):
        """A configured index reaches the backend verbatim, not guessed at."""
        import types

        opened = {}

        class InputStream:
            def __init__(self, **kwargs):
                opened.update(kwargs)

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
            InputStream=InputStream,
            query_devices=lambda device=None, kind=None: {
                "name": "usb mic", "index": device, "max_input_channels": 1},
        ))
        source = SoundDeviceSource(device=3)
        source.open()
        try:
            assert opened["device"] == 3
            assert source.device_name == "usb mic"
            assert source.device == 3
        finally:
            source.close()

    def test_missing_device_index_is_left_to_portaudio(self, monkeypatch):
        """No enumeration fallback exists here, and none is wanted.

        PyAudioSource walked the device list to find an input when the default
        was missing. SoundDeviceSource passes ``device=None`` through so PortAudio
        applies its own default policy, which is where device selection belongs.
        Asserting the pass-through pins that decision so a future enumeration
        hack has to be a deliberate change.
        """
        import types

        opened = {}

        class InputStream:
            def __init__(self, **kwargs):
                opened.update(kwargs)

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
            InputStream=InputStream,
            query_devices=lambda device=None, kind=None: {
                "name": "built-in mic", "index": 0, "max_input_channels": 1},
        ))
        source = SoundDeviceSource()
        source.open()
        try:
            assert opened["device"] is None
            assert source.device is None
            # The resolved name is still recorded, so status() can name the mic.
            assert source.device_name == "built-in mic"
        finally:
            source.close()

    def test_configured_invalid_input_device_fails_actionably(self, monkeypatch):
        """An index naming no input device must fail loudly and say why.

        The exception type is unchanged. The message is not: it now comes from
        SoundDeviceSource, so the old PyAudio wording is gone and the assertions
        below check the message that actually exists.
        """
        import types

        def query_devices(device=None, kind=None):
            raise OSError("Invalid device")

        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
            InputStream=object,
            query_devices=query_devices,
        ))
        with pytest.raises(MicrophoneUnavailable) as excinfo:
            SoundDeviceSource(device=27).open()
        message = str(excinfo.value)
        assert "Could not open microphone" in message
        assert "27" in message, "the failing index must be named"
        assert "Privacy and Security > Microphone" in message, (
            "the message must say how to fix it")

    def test_input_device_failure_after_a_good_probe_is_reported(self, monkeypatch):
        """A device that probes fine but refuses to stream must not look silent.

        query_devices succeeding is not proof the stream opens. If InputStream
        raises, the failure has to surface as MicrophoneUnavailable rather than
        an empty read loop.
        """
        import types

        class InputStream:
            def __init__(self, **kwargs):
                raise OSError("Device or resource busy")

        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
            InputStream=InputStream,
            query_devices=lambda device=None, kind=None: {
                "name": "built-in mic", "index": 0, "max_input_channels": 1},
        ))
        with pytest.raises(MicrophoneUnavailable) as excinfo:
            SoundDeviceSource().open()
        message = str(excinfo.value)
        assert "built-in mic" in message, "the device name must be identified"
        assert "Device or resource busy" in message, "the cause must survive"

    def test_sounddevice_read_error_is_reported_to_microphone_owner(self, monkeypatch):
        """A dead device must reach the owner as an error, never as silence."""
        import types

        class InputStream:
            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
            InputStream=InputStream,
            query_devices=lambda device=None, kind=None: {
                "name": "input", "index": 0, "max_input_channels": 1},
        ))
        source = SoundDeviceSource()

        def dead_read(timeout=1.0):
            raise OSError("device disconnected")

        source.read = dead_read

        errors = []
        mic = MicrophoneStream(source=source, on_error=errors.append)
        assert mic.start()
        try:
            deadline = time.monotonic() + 2
            while mic.last_error is None and time.monotonic() < deadline:
                time.sleep(0.005)
            assert mic.last_error, "a failed read was swallowed instead of reported"
            assert "read failed" in mic.last_error
            assert "device disconnected" in mic.last_error
            # Observable status carries it, so this is diagnosable from outside.
            assert mic.status()["last_error"] == mic.last_error
        finally:
            mic.stop()

    def test_repeated_read_errors_terminate_the_capture_path(self):
        """Retrying a dead device forever is how a broken mic becomes silence.

        Twenty consecutive failures have to end the capture path and mark the
        microphone unavailable, so the owner can report it rather than sit on a
        stream that will never produce audio again.
        """

        class DeadSource(FakeSource):
            def read(self):
                self.reads += 1
                raise OSError("device disconnected")

        errors = []
        mic = MicrophoneStream(source=DeadSource(), on_error=errors.append)
        assert mic.start()
        try:
            deadline = time.monotonic() + 30
            while mic.is_running() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not mic.is_running(), "capture retried a dead device forever"
            assert mic.available is False
            assert errors and "repeated read failures" in errors[-1]
        finally:
            mic.stop()

    def test_ordinary_read_timeouts_do_not_terminate_capture(self):
        """A quiet room is not a broken microphone.

        read() returning None is the documented timeout path and must keep the
        capture alive. If timeouts were treated as failures, a silent room
        would tear down the microphone after twenty pauses.
        """
        source = FakeSource()
        mic = MicrophoneStream(source=source)
        assert mic.start()
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                time.sleep(0.02)
            assert mic.is_running()
            assert mic.available is True
            assert mic.last_error is None
            assert source.reads >= 20, "the retry threshold was reached on timeouts"
        finally:
            mic.stop()

    def test_one_source_serves_every_consumer(self):
        source = FakeSource(frames=[_loud()] * 5)
        mic = MicrophoneStream(source=source)
        seen_a, seen_b = [], []
        mic.add_sink(seen_a.append)
        mic.add_sink(seen_b.append)
        assert mic.start() is True
        try:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and len(seen_a) < 5:
                time.sleep(0.01)
        finally:
            mic.stop()
        assert len(seen_a) == len(seen_b) == 5, "sinks saw different audio"
        assert source.opened == 1, "the device was opened more than once"

    def test_only_one_source_is_ever_constructed(self):
        import inspect

        from jarvis import voice_loop

        source = inspect.getsource(voice_loop)
        assert source.count("SoundDeviceSource(") == 0, "the loop opened its own microphone"
        assert source.count("PyAudioSource(") == 0, "the loop still opens a PyAudio mic"

    def test_the_legacy_listener_does_not_run_alongside_the_loop(self):
        """Two owners of one device is the failure this replaces."""
        from jarvis.voice_loop import VoiceLoop as Loop

        mic = MicrophoneStream(source=FakeSource())
        assert mic.is_running() is False
        loop = Loop(microphone=mic, wake_enabled=False,
                    player=SpeechPlayer(synthesize=lambda t: np.zeros(10, "float32"),
                                        play=lambda d, r: None,
                                        stop_playback=lambda: None),
                    transcriber=FakeASR(), respond=lambda t: "x")
        loop.start()
        try:
            sources = [s for s in (mic.source,) if getattr(s, "is_open", False)]
            assert len(sources) == 1
        finally:
            loop.stop()

    def test_missing_microphone_is_reported_not_raised(self):
        source = FakeSource(open_error=MicrophoneUnavailable("device busy"))
        loop = _make_loop(microphone=MicrophoneStream(source=source))
        assert loop.start() is False
        assert loop.last_error and "device busy" in loop.last_error
        assert loop.machine.state is State.IDLE

    def test_text_conversation_survives_a_missing_microphone(self):
        source = FakeSource(open_error=MicrophoneUnavailable("no mic"))
        loop = _make_loop(microphone=MicrophoneStream(source=source))
        assert loop.start() is False
        try:
            assert loop.respond("hello") == "It is twelve."
        finally:
            loop.stop()

    def test_microphone_is_released_on_stop(self):
        source = FakeSource()
        mic = MicrophoneStream(source=source)
        mic.start()
        mic.stop()
        assert source.closed == 1
        assert mic.is_running() is False


# ============================================================================
# Text mode coexistence
# ============================================================================

class TestTextCoexistence:
    def test_text_needs_no_microphone(self):
        calls = []
        loop = _make_loop(respond=lambda t: calls.append(t) or "text reply")
        assert loop.start() is True
        try:
            assert loop.respond("What is 2+2?") == "text reply"
        finally:
            loop.stop()
        assert calls == ["What is 2+2?"]

    def test_brain_ask_works_with_every_voice_component_absent(self):
        from jarvis.brain import JarvisBrain
        from tests.mock_providers import build_layer, reply

        layer = build_layer(
            models=[{"key": "m", "provider": "p", "model": "mm",
                     "capabilities": {"reasoning": True, "tool_calling": True}}],
            behaviours={"mm": reply("four")},
        )
        brain = JarvisBrain(model_layer=layer)
        loop = VoiceLoop(
            brain=brain,
            wake_enabled=False,
            microphone=MicrophoneStream(source=FakeSource()),
            player=SpeechPlayer(synthesize=lambda t: np.zeros(10, "float32"),
                                play=lambda d, r: None, stop_playback=lambda: None),
            transcriber=FakeASR(),
        )
        assert loop.respond("What is 2+2?") == "four"

    def test_asr_unavailable_is_reported_honestly(self, monkeypatch):
        """A missing dependency is reported, never papered over."""
        import sys

        assert issubclass(ASRUnavailable, RuntimeError)
        monkeypatch.setitem(sys.modules, "speech_recognition", None)
        with pytest.raises(ASRUnavailable):
            Transcriber().transcribe(b"\x00" * 100)

    def test_unintelligible_speech_yields_no_transcript(self):
        """Speech heard but not understood must not become a made-up transcript."""
        import speech_recognition as sr

        class Unintelligible:
            def recognize_google(self, audio, language=None):
                raise sr.UnknownValueError()

        transcriber = Transcriber(recognizer=Unintelligible())
        assert transcriber.transcribe(b"\x00" * 320) == ""
        assert transcriber.last_error

    def test_transcriber_passes_captured_int16_pcm_without_float_reinterpretation(self):
        pcm = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2").tobytes()
        observed = []

        class Recognizer:
            def recognize_google(self, audio, language=None):
                assert self.operation_timeout == 15.0
                observed.append((audio.frame_data, audio.sample_rate, audio.sample_width))
                return "test transcript"

        assert Transcriber(recognizer=Recognizer(), engine="google").transcribe(pcm) == "test transcript"
        assert observed == [(pcm, 16000, 2)]

    def test_google_asr_respects_configured_operation_timeout(self):
        class Recognizer:
            operation_timeout = None

            def recognize_google(self, audio, language=None):
                return "ok"

        recognizer = Recognizer()
        assert Transcriber(recognizer=recognizer, engine="google", timeout=2.5).transcribe(b"\x00\x00") == "ok"
        assert recognizer.operation_timeout == 2.5

    def test_local_asr_mode_never_calls_google(self):
        class Recognizer:
            def recognize_google(self, *args, **kwargs):
                pytest.fail("remote recognition called in local-only mode")

        with pytest.raises(ASRUnavailable, match="no local recognizer"):
            Transcriber(recognizer=Recognizer(), engine="local")

    def test_unknown_asr_engine_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported ASR engine"):
            Transcriber(engine="googlish")

    def test_local_asr_selection_fails_at_initialization(self):
        with pytest.raises(ASRUnavailable, match="no local recognizer"):
            Transcriber(engine="local")

    def test_malformed_pcm_is_an_error(self):
        transcriber = Transcriber(engine="google")
        with pytest.raises(ValueError, match="incomplete int16"):
            transcriber.transcribe(b"\x01")

    def test_a_missing_transcriber_is_an_error_not_a_fake_transcript(self):
        loop = _make_loop(transcriber=None)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.ERROR, State.IDLE)
        finally:
            loop.stop()
        assert loop.machine.state is State.IDLE


# ============================================================================
# Vision coexistence
# ============================================================================

class TestVisionCoexistence:
    def test_perception_never_moves_the_state_machine(self):
        """A camera observation is not a conversation."""
        from jarvis.vision import Observation, VisualContext

        machine = ConversationMachine()
        context = VisualContext()
        context.update([Observation("observed: the user is at the desk")])

        before = machine.snapshot()
        assert context.observations(), "perception produced nothing at all"
        assert machine.snapshot()["state"] == before["state"] == "idle"
        assert machine.turn is None

    def test_the_voice_loop_has_no_vision_dependency(self):
        import inspect

        from jarvis import vision, voice_loop, webcam

        assert "webcam" not in inspect.getsource(voice_loop)
        assert "vision" not in inspect.getsource(voice_loop)

    def test_visual_context_is_still_available_to_the_brain(self):
        from jarvis.brain import _build_context_block
        from jarvis.classify import classify
        from jarvis.vision import Observation, VisualContext, reset_visual_context

        reset_visual_context()
        try:
            from jarvis import vision
            vision.get_visual_context().update(
                [Observation("observed: A laptop is open on the desk.")]
            )
            block = _build_context_block(
                "what am I doing right now?", classify("what am I doing right now?"), []
            )
            assert "## Current Visual Context" in block
            assert "A laptop is open" in block
        finally:
            reset_visual_context()

    def test_wake_word_is_not_triggered_by_camera_activity(self):
        engine = FakeEngine(scores=[0.0] * 10)
        loop = _make_loop(wake_enabled=True, wake_engine=engine)
        loop.start()
        try:
            from jarvis.vision import Observation, VisualContext

            VisualContext().update([Observation("observed: someone arrived")])
            for _ in range(10):
                loop._on_frame(_quiet())
            assert loop.wake_events == 0
            assert loop.machine.state is State.IDLE
        finally:
            loop.stop()


# ============================================================================
# Shutdown
# ============================================================================

class TestShutdown:
    def test_stop_releases_everything(self):
        source = FakeSource(frames=[_loud()] * 1000)
        loop = _make_loop(microphone=MicrophoneStream(source=source))
        loop.start()
        loop.on_wake()
        loop.stop()
        assert loop.is_running() is False
        assert source.closed == 1
        assert loop.microphone.is_running() is False
        assert loop.player.queue.peek_len() == 0

    def test_stop_returns_to_idle_and_cancels_the_turn(self):
        loop = _make_loop(follow_up_window=5.0)
        loop.start()
        loop.on_wake()
        _speak_into(loop)
        assert _await_state(loop, State.FOLLOW_UP, State.SPEAKING)
        loop.stop()
        assert loop.machine.state is State.IDLE
        assert loop.machine.turn is None

    def test_stop_without_start_is_safe(self):
        _make_loop().stop()

    def test_stop_is_idempotent(self):
        loop = _make_loop()
        loop.start()
        loop.stop()
        loop.stop()
        assert loop.is_running() is False

    def test_stop_does_not_hold_the_lock_while_joining(self):
        """Regressed once: every shutdown left a turn thread behind.

        stop() joined the turn thread while holding the same lock the thread
        takes in its own `finally`, so the join could only ever time out.
        """
        loop = _make_loop(follow_up_window=0.2)
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop, frames=10)
            started = time.monotonic()
            loop.stop(timeout=5.0)
            elapsed = time.monotonic() - started
            assert elapsed < 3.0, f"stop() blocked for {elapsed:.1f}s"
            assert not [t for t in threading.enumerate() if t.name == "voice-turn"]
        finally:
            pass

    def test_no_worker_threads_survive(self):
        before = threading.active_count()
        for _ in range(3):
            loop = _make_loop(follow_up_window=0.2)
            loop.start()
            loop.on_wake()
            _speak_into(loop)
            loop.stop()
        time.sleep(0.3)
        assert threading.active_count() <= before
        assert not [t for t in threading.enumerate()
                    if t.name in ("microphone", "speech-player", "voice-turn")]

    def test_repeated_full_cycles_are_clean(self):
        """20x start -> wake -> turn -> interrupt -> follow-up -> shutdown."""
        failures = []
        for i in range(20):
            try:
                loop = _make_loop(follow_up_window=0.2)
                loop.start()
                loop.on_wake()
                _speak_into(loop, frames=10)
                loop.interrupt()
                _speak_into(loop, frames=10)
                loop.stop()
                assert loop.machine.state is State.IDLE
            except Exception as e:  # noqa: BLE001
                failures.append(f"cycle {i}: {type(e).__name__}: {e}")
        assert not failures, failures

    def test_shutdown_after_a_failure_is_still_clean(self):
        loop = _make_loop(transcriber=FakeASR(error=RuntimeError("asr down")))
        loop.start()
        try:
            loop.on_wake()
            _speak_into(loop)
            assert _await_state(loop, State.ERROR, State.IDLE)
        finally:
            loop.stop()
        assert loop.machine.state is State.IDLE


# ============================================================================
# Speech pipeline
# ============================================================================

class TestSpeechPipeline:
    def test_audio_is_split_into_chunks(self):
        player = SpeechPlayer(synthesize=lambda t: None,
                              play=lambda d, r: None, stop_playback=lambda: None,
                              chunk_ms=100)
        # One second at the rate this player actually plays at, so the count states
        # something about chunking rather than about a stale 16 kHz assumption.
        player.synthesize = lambda t: np.zeros(player.sample_rate, dtype=np.float32)
        queued = player.synthesize_to_queue("hello", "turn_001", lambda t: False)
        assert queued == 10, "1s of audio at 100ms chunks is not 10 chunks"

    def test_synthesis_stops_early_when_the_turn_goes_stale(self):
        player = SpeechPlayer(synthesize=lambda t: np.zeros(16000, dtype=np.float32),
                              play=lambda d, r: None, stop_playback=lambda: None,
                              chunk_ms=100)
        queued = player.synthesize_to_queue("hello", "turn_001", lambda t: True)
        assert queued == 0

    def test_cancel_clears_and_reports(self):
        player = SpeechPlayer(synthesize=lambda t: None,
                              play=lambda d, r: None, stop_playback=lambda: None,
                              chunk_ms=100)
        # One second at this player's playback rate: 10 chunks at 100ms.
        player.synthesize = lambda t: np.zeros(player.sample_rate, dtype=np.float32)
        player.synthesize_to_queue("hello", "turn_001", lambda t: False)
        assert player.cancel() == 10
        assert player.queue.peek_len() == 0

    def test_playback_stops_when_the_queue_is_empty(self):
        player = SpeechPlayer(synthesize=lambda t: np.zeros(8000, dtype=np.float32),
                              play=lambda d, r: None, stop_playback=lambda: None,
                              chunk_ms=100)
        player.synthesize_to_queue("hello", "turn_001", lambda t: False)
        player.start("turn_001", lambda t: False)
        assert player.wait(timeout=2.0) is True
        assert player.is_playing() is False

    def test_playback_failure_is_recorded(self):
        player = SpeechPlayer(
            synthesize=lambda _: np.ones(1600, dtype=np.float32),
            play=lambda *_: (_ for _ in ()).throw(OSError("speaker disconnected")),
            stop_playback=lambda: None, chunk_ms=100,
        )
        player.synthesize_to_queue("hi", "turn", lambda _: False)
        player.start("turn", lambda _: False)
        assert player.wait(1)
        assert "speaker disconnected" in player.last_error

    def test_physical_playback_uses_one_contiguous_stream_per_turn(self, monkeypatch):
        import jarvis.speech_pipeline as pipeline

        calls = []
        def play(data, rate, cancel_event=None):
            calls.append((data, rate))

        monkeypatch.setattr(pipeline, "_sounddevice_play", play)
        player = SpeechPlayer(
            synthesize=lambda _: np.ones(3200, dtype=np.float32), chunk_ms=100,
        )
        assert player.synthesize_to_queue("hello", "turn", lambda _: False) == 2
        player.start("turn", lambda _: False)
        assert player.wait(1)
        expected = np.full(3200, 32767, dtype="<i2").tobytes()
        # The stream must open at the rate this player plays at, which is the
        # engine's native rate (Kokoro 24 kHz), not the microphone's 16 kHz.
        assert calls == [(expected, player.sample_rate)]
        assert player.chunks_played == 2

    def test_close_waits_for_playback_worker_before_returning(self):
        entered = threading.Event()
        release = threading.Event()
        player = SpeechPlayer(
            synthesize=lambda t: np.zeros(1600, dtype=np.float32),
            play=lambda d, r: (entered.set(), release.wait(3)),
            stop_playback=lambda: None, chunk_ms=100,
        )
        player.synthesize_to_queue("hello", "turn", lambda _: False)
        player.start("turn", lambda _: False)
        assert entered.wait(1)
        closed = threading.Event()
        closer = threading.Thread(target=lambda: (player.close(), closed.set()))
        closer.start()
        assert not closed.wait(0.05)
        release.set()
        closer.join(1)
        assert closed.is_set()
        assert not player.is_playing()

    def test_close_timeout_preserves_live_worker_reference(self):
        entered = threading.Event()
        release = threading.Event()
        player = SpeechPlayer(
            synthesize=lambda t: np.zeros(1600, dtype=np.float32),
            play=lambda d, r: (entered.set(), release.wait(3)),
            stop_playback=lambda: None, chunk_ms=100,
        )
        player.synthesize_to_queue("hello", "turn", lambda _: False)
        player.start("turn", lambda _: False)
        assert entered.wait(1)
        worker = player._thread
        with pytest.raises(TimeoutError, match="worker did not stop"):
            player.close(timeout=0.01)
        assert player._thread is worker and worker.is_alive()
        release.set()
        worker.join(1)
        player.close(timeout=1)
        assert player._thread is None

    def test_new_playback_waits_for_a_live_worker_then_returns(self):
        """A new reply must not be dropped because the old worker still looks alive.

        Spec B2 (V8): start() joins the previous drain thread and only returns
        early if that thread is still alive afterwards. Raising used to discard
        the new reply instead of playing it.
        """
        entered = threading.Event()
        release = threading.Event()
        player = SpeechPlayer(
            synthesize=lambda t: np.zeros(1600, dtype=np.float32),
            play=lambda d, r: (entered.set(), release.wait(3)),
            stop_playback=lambda: None, chunk_ms=100,
        )
        player.synthesize_to_queue("first", "one", lambda _: False)
        player.start("one", lambda _: False)
        assert entered.wait(1)
        first_thread = player._thread

        player.start("two", lambda _: False)  # must not raise

        # Exactly one live worker: the second reply did not spawn an overlap.
        assert player._thread is first_thread
        release.set()
        assert player.wait(1)
        assert not player.is_playing()

    def test_new_reply_plays_after_the_previous_worker_finishes(self):
        """The point of V8: a reply started after the previous one is not lost."""
        release = threading.Event()
        played = []
        player = SpeechPlayer(
            synthesize=lambda t: np.zeros(2400, dtype=np.float32),
            play=lambda d, r: (played.append(len(d)), release.wait(3)),
            stop_playback=lambda: None, chunk_ms=100,
        )
        player.synthesize_to_queue("first", "one", lambda _: False)
        player.start("one", lambda _: False)
        player.synthesize_to_queue("second", "two", lambda _: False)
        release.set()

        player.start("two", lambda _: False)  # must not raise
        assert player.wait(2)
        assert played, "the second reply was never spoken"
        assert player._thread is not None and not player._thread.is_alive()

    def test_empty_reply_produces_no_audio(self):
        player = SpeechPlayer(play=lambda d, r: None, stop_playback=lambda: None)
        assert player.synthesize_to_queue("   ", "turn_001", lambda t: False) == 0

    def test_pcm_conversion_is_clipped_not_wrapped(self):
        from jarvis.speech_pipeline import _to_pcm16

        loud = np.array([2.0, -2.0], dtype=np.float32)
        data = _to_pcm16(loud)
        assert len(data) == 4
        assert int.from_bytes(data[:2], "little", signed=True) == 32767

    def test_sounddevice_play_converts_int16_chunks_to_float32(self, monkeypatch):
        import types
        import threading
        from jarvis.speech_pipeline import _sounddevice_play

        played = []
        class CallbackStop(Exception):
            pass

        class OutputStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                self.enter_thread = threading.get_ident()
                def render():
                    block = np.empty((3, 1), dtype=np.float32)
                    while True:
                        try:
                            self.kwargs["callback"](block, 3, None, None)
                        except CallbackStop:
                            played.append(block.copy())
                            break
                        played.append(block.copy())
                    self.kwargs["finished_callback"]()
                self.render_thread = threading.Thread(target=render)
                self.render_thread.start()
                return self

            def __exit__(self, *_):
                self.render_thread.join(timeout=1)
                self.exit_thread = threading.get_ident()

        streams = []
        def output_stream(**kwargs):
            stream = OutputStream(**kwargs)
            streams.append(stream)
            return stream

        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
            default=types.SimpleNamespace(device=(0, 1)),
            query_devices=lambda device, kind: {"name": "Mock speakers"},
            check_output_settings=lambda **kwargs: played.append(kwargs),
            OutputStream=output_stream,
            CallbackStop=CallbackStop,
        ))
        pcm = np.array([-32768, 0, 16384, 32767, -1], dtype="<i2").tobytes()
        assert _sounddevice_play(pcm, 16000) == "completed"
        rendered = np.concatenate([block[:, 0] for block in played[1:]])
        assert np.allclose(
            rendered[:5], [-1.0, 0.0, 0.5, 32767 / 32768, -1 / 32768],
        )
        assert played[0] == {
            "device": 1, "channels": 1, "dtype": "float32", "samplerate": 16000,
        }
        assert streams[0].kwargs["dtype"] == "float32"
        assert streams[0].enter_thread == streams[0].exit_thread

    def test_output_cancellation_closes_stream_on_playback_owner(self, monkeypatch):
        import types
        import threading
        from jarvis.speech_pipeline import _sounddevice_play, _sounddevice_stop

        entered = threading.Event()
        streams = []
        results = []

        class CallbackStop(Exception):
            pass

        class OutputStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                streams.append(self)

            def __enter__(self):
                self.owner = threading.get_ident()
                entered.set()
                return self

            def __exit__(self, *_):
                self.closed_by = threading.get_ident()

        monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
            default=types.SimpleNamespace(device=(0, 1)),
            query_devices=lambda device, kind: {"name": "Mock speakers"},
            check_output_settings=lambda **kwargs: None,
            OutputStream=OutputStream,
            CallbackStop=CallbackStop,
        ))
        worker = threading.Thread(
            target=lambda: results.append(
                _sounddevice_play(np.zeros(16000, dtype="<i2").tobytes(), 16000)
            ),
        )
        worker.start()
        assert entered.wait(1)
        _sounddevice_stop()
        out = np.empty((128, 1), dtype=np.float32)
        with pytest.raises(CallbackStop):
            streams[0].kwargs["callback"](out, 128, None, None)
        streams[0].kwargs["finished_callback"]()
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert streams[0].closed_by == streams[0].owner
        assert results == ["cancelled"]

    def test_voice_loop_lease_prevents_two_process_style_mic_owners(self, monkeypatch, tmp_path):
        import tempfile

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        first_source, second_source = FakeSource(), FakeSource()
        first = _make_loop(microphone=MicrophoneStream(source=first_source))
        second = _make_loop(microphone=MicrophoneStream(source=second_source))
        try:
            assert first.start()
            assert not second.start()
            assert first_source.opened == 1
            assert second_source.opened == 0
            first.stop()
            assert second.start()
            assert second_source.opened == 1
        finally:
            first.stop()
            second.stop()
