#!/usr/bin/env python3
"""Phase 5 live verification.

Every conversational scenario below runs the *real* Brain against the *real*
configured model, so what is checked is the prompt the model actually received
and the state it actually produced — not a mock's idea of either.

What is asserted is state and guidance, never model wording. Wording varies
between runs and providers, and a test that pins it is a test that lies.

Usage:
    python3 scripts/phase5_live_check.py
"""

import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import memory, tone, vision  # noqa: E402
from jarvis.brain import JarvisBrain, BrainError  # noqa: E402
from jarvis.tone import (  # noqa: E402
    ConversationMode,
    ConversationState,
    ConversationTone,
    Mood,
    UserState,
    format_prompt,
)

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"--- {name} ---\nDetail: {detail}\nResult: {'PASS' if ok else 'FAIL'}\n", flush=True)


def skip(name, detail):
    RESULTS.append((name, None, detail))
    print(f"--- {name} ---\nDetail: {detail}\nResult: SKIPPED\n", flush=True)


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}\n", flush=True)


# ---------------------------------------------------------------------------
# A prompt-recording brain, so we can see what the model really received
# ---------------------------------------------------------------------------

class Recorder:
    """Wraps a real Brain and keeps the prompt and reply of every turn."""

    def __init__(self, brain):
        self.brain = brain
        self.turns = []
        self.errors = []

    def ask(self, text, images=None, attempts=6, **kwargs):
        """One real turn, retried on a transient provider failure.

        The free providers this project routes to occasionally return a 400
        upstream, which is classified non-retryable and aborts the turn. That is
        correct behaviour for a user turn and wrong for a verification run, so
        the retry lives here rather than in the product.
        """
        error = None
        captured = []
        original = self.brain._create_chat

        def capturing(model, history=None):
            # The turn context is captured here, at the moment it is attached to
            # a session, rather than read back afterwards. A retried attempt must
            # not be able to overwrite what an earlier one actually sent.
            captured.append(self.brain._turn_context or "")
            return original(model, history=history)

        self.brain._create_chat = capturing
        try:
            for attempt in range(attempts):
                try:
                    reply = self.brain.ask(text, images=images, **kwargs)
                    break
                except BrainError as e:
                    error = e
                    self.errors.append(f"{text[:30]!r}: {e.message}")
                    if attempt == attempts - 1:
                        # Recorded rather than raised: one flaky free provider
                        # must not abort thirty checks that do not need it.
                        reply = ""
                    else:
                        time.sleep(2.0 * (attempt + 1))
            else:  # pragma: no cover - unreachable
                reply = ""
        finally:
            self.brain._create_chat = original
        # `_turn_context` is exactly what was appended to the identity prompt for
        # this turn. Read from the Brain rather than from a provider: the real
        # adapters keep no session history to inspect, and what matters is what
        # the Brain handed them.
        self.turns.append({
            "text": text,
            "prompt": captured[0] if captured else "",
            "reply": reply,
            "error": str(error.message) if error is not None else None,
        })
        return reply


def guidance_or_skip(turn, phrase, name):
    """Check that a real turn carried the expected guidance.

    Skips rather than fails when the turn never completed, so a provider outage
    is never reported as a defect in conversation state.
    """
    if turn.get("error"):
        skip(name, f"the turn did not complete ({turn['error']}); nothing to inspect")
        return False
    block = guidance_for(turn)
    ok = phrase in block
    check(name, ok,
          f"guidance block ({len(block)} chars): "
          f"{[line for line in block.splitlines() if phrase.split()[0].lower() in line.lower()][:1] or block[:120]!r}; "
          f"prompt={len(turn['prompt'])} chars, error={turn.get('error')}")
    return ok


def guidance_for(turn):
    """The behavioural block the model was actually given."""
    prompt = turn["prompt"]
    marker = "\n\n## How To Talk Right Now"
    if marker not in prompt:
        return ""
    # Skip past the marker before splitting on section headers: the block
    # *starts* with that delimiter, so splitting the slice from position zero
    # returns an empty string every time.
    body = prompt[prompt.index(marker) + len(marker):]
    return body.split("\n\n## ")[0] if body.strip() else ""


def state_line(state):
    return (f"mode={state.mode.value} mood={state.mood.value} "
            f"energy={state.energy:.2f} warmth={state.warmth:.2f} "
            f"curiosity={state.curiosity:.2f} user={state.user_state.value}")


# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Phase 5 Live Verification")
    print("=" * 60 + "\n", flush=True)

    brain = Recorder(JarvisBrain(conversation_id="phase5-live"))

    # Probe before anything else. If no model can answer at all, say that once
    # rather than reporting thirty unrelated failures.
    if not brain.ask("Say ok."):
        for title in ("casual", "celebration", "frustration", "technical",
                      "playful", "recovery", "decay", "same input two contexts",
                      "correction", "soft signal", "honesty", "vision",
                      "memory", "performance", "shutdown"):
            skip(f"provider unavailable ({title})",
                 "no configured model answered after 3 attempts: "
                 f"{brain.errors[0] if brain.errors else 'unknown error'}")
        _summarise()
        return 1

    # ------------------------------------------------------------------
    section("TEST 1 — CASUAL (baseline behaviour)")
    # ------------------------------------------------------------------
    tone.reset_tone()
    reply = brain.ask("Hey, what have you been up to?")
    state = tone.get_tone().state
    check("T1 — a casual turn works normally",
          bool(reply.strip()) and state.mode is ConversationMode.CASUAL,
          f"{state_line(state)}; reply: {reply[:70]!r}")
    check("T1b — a neutral conversation costs no prompt tokens",
          "## How To Talk Right Now" not in brain.turns[-1]["prompt"],
          "no behavioural block was added to the prompt")

    # ------------------------------------------------------------------
    section("TEST 2 — CELEBRATION")
    # ------------------------------------------------------------------
    tone.reset_tone()
    reply = brain.ask("YES! It finally works, all the tests pass!")
    state = tone.get_tone().state
    guidance = guidance_for(brain.turns[-1])
    check("T2 — success moves to celebratory",
          state.mode is ConversationMode.CELEBRATORY and state.energy > 0.6,
          f"{state_line(state)}")
    guidance_or_skip(brain.turns[-1], "Match the moment",
                     "T2b — the model was told to match the moment")
    check("T2c — warmth stayed high rather than spiking",
          0.75 <= state.warmth <= 1.0, f"warmth={state.warmth:.2f}")

    # ------------------------------------------------------------------
    section("TEST 3 — FRUSTRATION")
    # ------------------------------------------------------------------
    tone.reset_tone()
    reply = brain.ask("I've been debugging this for three hours and I'm so frustrated, it still doesn't work")
    state = tone.get_tone().state
    guidance = guidance_for(brain.turns[-1])
    check("T3 — frustration moves to supportive",
          state.mode is ConversationMode.SUPPORTIVE
          and state.user_state is UserState.FRUSTRATED,
          f"{state_line(state)}")
    check("T3b — energy is moderate and warmth high",
          state.energy < 0.5 and state.warmth > 0.75,
          f"energy={state.energy:.2f} warmth={state.warmth:.2f}")
    guidance_or_skip(brain.turns[-1], "do not dwell",
                     "T3c — dwelling on it was ruled out")

    # ------------------------------------------------------------------
    section("TEST 4 — TECHNICAL")
    # ------------------------------------------------------------------
    tone.reset_tone()
    reply = brain.ask("The stack trace shows a NullPointerException at line 42")
    state = tone.get_tone().state
    guidance = guidance_for(brain.turns[-1])
    check("T4 — technical content moves to technical",
          state.mode is ConversationMode.TECHNICAL, state_line(state))
    guidance_or_skip(brain.turns[-1], "Be precise and concrete",
                     "T4b — the model was told to be precise and avoid padding")
    check("T4c — a real answer came back", bool(reply.strip()),
          f"reply: {reply[:90]!r}")

    # ------------------------------------------------------------------
    section("TEST 5 — PLAYFUL, AND NO CONTAMINATION")
    # ------------------------------------------------------------------
    tone.reset_tone()
    brain.ask("haha that's hilarious, tell me another joke")
    playful = tone.get_tone().state
    check("T5 — playful conversation moves to playful",
          playful.mode is ConversationMode.PLAYFUL, state_line(playful))
    reply = brain.ask("now, what's the capital of France?")
    after = tone.get_tone().state
    laughter = ("haha", "lol", "😂", "🤣", "joke")
    check("T5b — a factual answer stays factual under a playful state",
          "paris" in reply.lower()
          and not any(marker in reply.lower() for marker in laughter),
          f"reply: {reply.strip()[:110]!r}; state still {after.mode.value}")

    # ------------------------------------------------------------------
    section("TEST 6 — RECOVERY (energetic -> serious technical)")
    # ------------------------------------------------------------------
    tone.reset_tone()
    brain.ask("YES! It finally works!!")
    excited = tone.get_tone().state
    brain.ask("production is down, this is critical, fix it now")
    recovered = tone.get_tone().state
    check("T6 — an energetic conversation adapts to a serious task",
          excited.mode is ConversationMode.CELEBRATORY
          and recovered.mode is ConversationMode.SERIOUS
          and recovered.energy < excited.energy,
          f"{excited.mode.value}(e={excited.energy:.2f}) -> "
          f"{recovered.mode.value}(e={recovered.energy:.2f})")

    # ------------------------------------------------------------------
    section("TEST 7 — DECAY AND RESET")
    # ------------------------------------------------------------------
    tracker = ConversationTone(half_life=10.0)
    tracker.observe("It finally works!!")
    high = tracker.state.energy
    tracker._last_update -= 40.0
    decayed = tracker.state
    check("T7 — temporary state decays back toward the defaults",
          decayed.energy < high and decayed.mode is ConversationMode.CASUAL,
          f"energy {high:.2f} -> {decayed.energy:.2f}, mode -> {decayed.mode.value}")
    tracker.observe("I'm so frustrated")
    tracker.reset()
    check("T7b — reset returns to defaults", tracker.state.as_dict() == tracker.defaults.as_dict(),
          f"reset state: {tracker.state.as_dict()}")

    # ------------------------------------------------------------------
    section("TEST 8 — SAME INPUT, TWO CONTEXTS")
    # ------------------------------------------------------------------
    message = "It finally works."
    a = ConversationTone()
    a.observe(message, context=ConversationState(
        mode=ConversationMode.CELEBRATORY, mood=Mood.CELEBRATORY,
        energy=0.85, warmth=0.90, curiosity=0.70, user_state=UserState.EXCITED))
    b = ConversationTone()
    b.observe(message, context=ConversationState(
        mode=ConversationMode.TECHNICAL, mood=Mood.FOCUSED,
        energy=0.45, warmth=0.70, curiosity=0.75))
    block_a, block_b = format_prompt(a.state), format_prompt(b.state)
    check("T8 — the same sentence yields different guidance",
          block_a != block_b
          and "Match the moment" in block_a
          and "Be precise and concrete" in block_b,
          f"celebratory: {[ln for ln in block_a.splitlines() if 'Mode:' in ln][0]!r} "
          f"-> {[ln for ln in block_a.splitlines() if 'Match the moment' in ln][0].strip()!r}; "
          f"technical: {[ln for ln in block_b.splitlines() if 'Mode:' in ln][0]!r} "
          f"-> {[ln for ln in block_b.splitlines() if 'precise' in ln][0].strip()!r}")

    # ------------------------------------------------------------------
    section("TEST 9 — EXPLICIT CORRECTION")
    # ------------------------------------------------------------------
    tone.reset_tone()
    brain.ask("I'm so frustrated, this keeps failing")
    frustrated = tone.get_tone().state.user_state
    brain.ask("I'm not frustrated, I'm just joking")
    corrected = tone.get_tone().state.user_state
    check("T9 — an explicit correction withdraws the inference",
          frustrated is UserState.FRUSTRATED and corrected is UserState.UNKNOWN,
          f"{frustrated.value} -> {corrected.value}")

    # ------------------------------------------------------------------
    section("TEST 10 — SOFT SIGNAL (tone cannot override judgment)")
    # ------------------------------------------------------------------
    playful_state = ConversationState(mode=ConversationMode.PLAYFUL, mood=Mood.PLAYFUL,
                                      energy=0.9, warmth=0.85, curiosity=0.9)
    block = format_prompt(playful_state).lower()
    check("T10 — the block states its own limits",
          "never overrides" in block and "accuracy wins" in block,
          "the model is told accuracy, instructions, safety and tools outrank tone")
    reply = brain.ask("Is 2+2 equal to 5? Answer with just the number.")
    check("T10b — accuracy is unaffected by a playful state",
          reply.strip().startswith("4"), f"reply: {reply.strip()[:40]!r}")

    # ------------------------------------------------------------------
    section("TEST 11 — HONESTY ABOUT FEELINGS")
    # ------------------------------------------------------------------
    reply = brain.ask("Are you actually feeling excited right now?")
    lowered = reply.lower()
    honest = any(word in lowered for word in
                 ("do not have feelings", "don't have feelings", "not feelings",
                  "not conscious", "no feelings", "don't actually feel",
                  "not actually feel", "conversational tone", "tone setting"))
    check("T11 — the assistant does not claim subjective feelings",
          honest,
          f"reply: {reply[:170]!r}")

    # ------------------------------------------------------------------
    section("TEST 12 — VISION INDEPENDENCE")
    # ------------------------------------------------------------------
    tone.reset_tone()
    before = tone.get_tone().state.as_dict()
    vision.get_visual_context().update(
        vision.parse_observations("observed: A person is sitting at a desk."))
    layer_reply = brain.ask("What can you currently see?")
    after = tone.get_tone().state.as_dict()
    check("T12 — visual context never moves the state",
          before == after and vision.get_visual_context().observations(),
          f"visual context has {len(vision.get_visual_context().observations())} "
          f"observation(s); state unchanged")
    check("T12b — the visual context is still used for the answer",
          "desk" in layer_reply.lower(), f"reply: {layer_reply[:90]!r}")
    vision.reset_visual_context()

    # ------------------------------------------------------------------
    section("TEST 13 — MEMORY BOUNDARY")
    # ------------------------------------------------------------------
    import tempfile

    import jarvis.brain as brain_module

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        scratch = Path(tmp.name)
    conn = memory.init_memory_db(scratch)
    original = brain_module.get_memory_conn
    brain_module.get_memory_conn = lambda: conn
    try:
        for message in ("I'm so exhausted today", "I'm so frustrated again",
                        "YES! It finally works!!", "I'm really confused about this"):
            brain.ask(message)
        stored = memory.recall_all(conn)
        check("T13 — nothing about a mood was written to memory",
              stored == [], f"{len(stored)} memories after four emotional turns")
        brain.ask("Remember that I prefer concise answers.")
        check("T13b — explicit memory still works",
              any("concise" in f for f in memory.recall_all(conn)),
              f"stored: {memory.recall_all(conn)}")
    finally:
        brain_module.get_memory_conn = original
        conn.close()
        scratch.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    section("TEST 14 — PERFORMANCE")
    # ------------------------------------------------------------------
    tracker = ConversationTone()
    start = time.perf_counter()
    for _ in range(10_000):
        tracker.observe("It finally works and I am so happy about it")
    elapsed = time.perf_counter() - start
    check("T14 — state updates are effectively free",
          elapsed < 5.0,
          f"10,000 updates in {elapsed:.3f}s ({elapsed / 10_000 * 1e6:.1f}us each)")

    before_threads = threading.active_count()
    for _ in range(50):
        ConversationTone().observe("hello")
    check("T14b — no background process was started",
          threading.active_count() <= before_threads,
          "conversation state needs no timer, no thread, no model call")

    # ------------------------------------------------------------------
    section("TEST 15 — SHUTDOWN")
    # ------------------------------------------------------------------
    from jarvis.voice_loop import VoiceLoop
    from jarvis.audio import MicrophoneStream
    from jarvis.speech_pipeline import SpeechPlayer
    import numpy

    class _Source:
        is_open = False

        def open(self):
            self.is_open = True

        def read(self):
            time.sleep(0.003)
            return None

        def close(self):
            self.is_open = False

    class _ASR:
        def transcribe(self, pcm, sample_rate=16000):
            return "what time is it"

    tone.get_tone().set(ConversationState(mode=ConversationMode.CELEBRATORY,
                                          mood=Mood.CELEBRATORY, energy=0.9))
    failures, leaked = [], []
    for i in range(20):
        loop = VoiceLoop(
            microphone=MicrophoneStream(source=_Source()),
            player=SpeechPlayer(synthesize=lambda t: numpy.zeros(24000, dtype="float32"),
                                play=lambda d, r: time.sleep(0.004),
                                stop_playback=lambda: None, chunk_ms=100),
            transcriber=_ASR(), wake_enabled=False, respond=lambda t: "an answer",
            follow_up_window=0.2,
        )
        try:
            loop.start()
            loop.on_wake()
            tone.observe("It finally works!!")
            loop.interrupt()
            loop.stop()
        except Exception as e:  # noqa: BLE001
            failures.append(f"cycle {i}: {type(e).__name__}: {e}")
        leaked += [t.name for t in threading.enumerate()
                   if t.name in ("microphone", "speech-player", "voice-turn")]
    check("T15 — repeated voice shutdown cycles are clean",
          not failures and not leaked,
          f"{20 - len(failures)}/20 clean, leaked threads: {sorted(set(leaked)) or 'none'}")
    check("T15b — conversational state survived all of it",
          tone.get_tone().state.mode is ConversationMode.CELEBRATORY,
          state_line(tone.get_tone().state))
    tone.reset_tone()

    # ------------------------------------------------------------------
    return _summarise()


def _summarise():
    passed = sum(1 for _, ok, _ in RESULTS if ok is True)
    failed = sum(1 for _, ok, _ in RESULTS if ok is False)
    skipped = sum(1 for _, ok, _ in RESULTS if ok is None)
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60 + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())