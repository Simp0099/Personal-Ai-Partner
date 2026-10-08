"""JARVIS 2.0 — Main Entry Point (Phase 8: Production Polish).

Orchestrates wake-word-activated voice interaction powered by Google Gemini.
Uses openWakeWord for local, offline wake-word detection.

Phase 8 features:
  - Robust error handling with try/except blocks
  - Rotating file logging (jarvis.log)
  - Centralized configuration via config.yaml
  - CLI status indicators ([Listening...], [Thinking, Boss...], [Speaking...])

Usage:
    python3 -m jarvis.main              # Wake word mode (default)
    python3 -m jarvis.main --text       # Text mode (skip wake word)
    python3 -m jarvis.main --test       # Run automated verification
    python3 -m jarvis.main --no-wake    # Voice mode without wake word
"""

import sys
import random
import datetime
from jarvis.config import ASSISTANT_NAME, GREETING_NAME, LLM_MODEL
from jarvis.speech import speak, listen, set_interactive_prompting
from jarvis.brain import JarvisBrain, BrainError
import jarvis.brain as jarvis_brain
from jarvis.logger import logger, StatusIndicator

# Phrases that mean "shut down". Matched against the whole normalized utterance
# only. The previous substring check ("exit" in query) terminated the process on
# ordinary requests such as "Explain the exit code in Python".
SHUTDOWN_PHRASES = frozenset({
    "you need a break",
    "go to sleep",
    "exit",
    "quit",
    "shutdown",
    "shut down",
    "goodbye",
    "bye",
})


def is_shutdown_request(query: str) -> bool:
    """Return True only if the entire message is a shutdown command.

    Exact whole-string comparison keeps ordinary sentences that merely contain
    a trigger word from discarding the user's actual request.
    """
    if not query:
        return False
    normalized = " ".join(query.lower().strip().rstrip(".!?").split())
    return normalized in SHUTDOWN_PHRASES


def start_perception(brain) -> object:
    """Start webcam perception if config.yaml enables it.

    Explicitly opt-in: with ``vision.webcam.enabled: false`` this returns None
    and the camera is never opened. Returns the started instance so the caller
    can stop it in a ``finally`` — that is what releases the device.
    """
    from jarvis.webcam import start_webcam_perception

    return start_webcam_perception(brain)


def stop_perception() -> None:
    """Release the camera, if one was opened. Safe to call unconditionally."""
    from jarvis.webcam import stop_webcam_perception

    stop_webcam_perception()


def start_voice(brain) -> object:
    """Start the voice conversation loop if config.yaml enables it.

    Returns None when the wake word is disabled -- the microphone is never
    opened, and text conversation is unaffected. Voice is a pathway, not the
    brain.
    """
    from jarvis.voice_loop import start_voice_loop

    return start_voice_loop(brain)


def stop_voice() -> None:
    """Cancel any turn and release the microphone. Safe to call unconditionally."""
    from jarvis.voice_loop import stop_voice_loop

    stop_voice_loop()


def wish_me() -> None:
    """Time-sensitive greeting sequence."""
    hour = int(datetime.datetime.now().hour)

    if 0 <= hour < 12:
        greetings = [
            f"Good Morning {GREETING_NAME}!",
            f"What's up {GREETING_NAME}!",
            f"Ready for some work, {GREETING_NAME}!",
        ]
    elif 12 <= hour < 18:
        greetings = [
            f"Good Afternoon {GREETING_NAME}!",
            f"Welcome back, {GREETING_NAME}!",
            f"How was your day, {GREETING_NAME}!",
        ]
    else:
        greetings = [
            f"Good Evening {GREETING_NAME}!",
            f"Welcome back, {GREETING_NAME}!",
            f"At your service, {GREETING_NAME}!",
        ]

    speak(random.choice(greetings))
    speak(f"{ASSISTANT_NAME} is online and running on {LLM_MODEL}.")


def run_assistant() -> None:
    """Core command loop driven by Google Gemini brain (voice mode, no wake word)."""
    set_interactive_prompting(True)
    wish_me()
    brain = JarvisBrain()
    start_perception(brain)
    voice = start_voice(brain)

    try:
        while True:
            query = listen().strip()

            if not query or query.lower() == "none":
                continue

            # Clean shutdown triggers (whole-utterance match only)
            if is_shutdown_request(query):
                StatusIndicator.shutdown()
                speak(f"Going offline. You can call me anytime, {GREETING_NAME}!")
                break

            # Process through Gemini LLM Brain
            try:
                response = brain.ask(query)
                speak(response)
            except BrainError as e:
                logger.error(f"Model unavailable: {e.detail or e}")
                speak(e.message)
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                speak("I encountered an issue processing that command, Boss.")
    finally:
        stop_voice()
        stop_perception()


def run_wake_word_mode() -> None:
    """Wake word activated mode using openWakeWord.

    Continuously listens for the wake word in the background.
    When detected, activates the assistant to listen for a command.
    """
    from jarvis.wake_word import WakeWordListener

    set_interactive_prompting(True)
    brain = JarvisBrain()

    def on_wake():
        """Called when wake word is detected."""
        speak("Yes, Boss?")
        query = listen().strip()

        if not query or query.lower() == "none":
            return

        if is_shutdown_request(query):
            StatusIndicator.shutdown()
            speak(f"Going offline. You can call me anytime, {GREETING_NAME}!")
            sys.exit(0)

        try:
            response = brain.ask(query)
            speak(response)
        except BrainError as e:
            logger.error(f"Model unavailable: {e.detail or e}")
            speak(e.message)
        except Exception as e:
            logger.error(f"Error processing wake command: {e}", exc_info=True)
            speak("I encountered an issue processing that command, Boss.")

    print(f"\n{'='*60}")
    print(f"  {ASSISTANT_NAME} 2.0 — Wake Word Mode")
    print(f"  Model: {LLM_MODEL}")
    print(f"{'='*60}")
    print(f"\nSay 'hey jarvis' to activate. Type Ctrl+C to exit.\n")

    listener = WakeWordListener(on_wake=on_wake)
    listener.start()
    start_perception(brain)
    voice = start_voice(brain)

    try:
        # Keep main thread alive while listener runs in background
        while listener.is_listening():
            import time
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Session ended by user.")
        listener.stop()
        sys.exit(0)
    finally:
        # Release audio and camera before the process goes down. Freeing a
        # native handle while another thread still holds it is exactly what the
        # previous shutdown race looked like.
        stop_voice()
        stop_perception()


def run_text_mode() -> None:
    """Text-based interaction mode for testing tool calling without voice."""
    print(f"\n{'='*60}")
    print(f"  {ASSISTANT_NAME} 2.0 — Text Mode (Tool Calling + Memory Test)")
    print(f"  Model: {LLM_MODEL}")
    print(f"{'='*60}")
    print(f"\nType your commands. Try things like:")
    print(f'  - "What time is it?"')
    print(f'  - "Show me the system status"')
    print(f'  - "List files in the current directory"')
    print(f'  - "Tell me a joke"')
    print(f'  - "Remember that my favorite color is blue"')
    print(f'  - "What do you know about me?"')
    print(f'\nType "exit" or "quit" to end the session.\n')

    set_interactive_prompting(True)
    brain = JarvisBrain()
    start_perception(brain)
    start_voice(brain)

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GREETING_NAME}, signing off. Goodbye!")
            break

        if not user_input:
            continue

        if is_shutdown_request(user_input):
            print(f"\n{ASSISTANT_NAME}: Going offline. You can call me anytime, {GREETING_NAME}!")
            break

        try:
            response = brain.ask(user_input)
            print(f"\n{ASSISTANT_NAME}: {response}\n")
        except BrainError as e:
            logger.error(f"Model unavailable: {e.detail or e}")
            print(f"[Error]: {e.message}")
        except Exception as e:
            logger.error(f"Error in text mode: {e}", exc_info=True)
            print("[Error]: I encountered an issue processing that command, Boss.")

    stop_voice()
    stop_perception()


def run_verification_test() -> None:
    """Automated verification of the conversation and routing pipeline.

    Deterministic checks run against an in-process probe provider, so this
    command verifies *our* logic rather than a particular model's mood. A live
    model call is also made to confirm the configured credentials work, and is
    reported separately because provider availability is not something this
    repository controls.

    Checks:
      1. The user's message reaches the model verbatim.
      2. Tool results are fed back to the model and the loop terminates.
      3. A failing model falls back to the next candidate.
      4. A tool-required request only reaches a tool-capable model.
      5. All models failing produces an honest error, not a fake reply.
      6. Empty input is rejected.
      7. The model layer loads and reports status.
      8. Live: the minimal direct-model path answers.
    """
    from jarvis.providers.base import (
        ErrorKind, ModelResponse, ProviderError, ToolCall, new_tool_call_id,
    )

    print(f"\n{'='*60}")
    print(f"  {ASSISTANT_NAME} 2.0 — Pipeline Verification")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0

    def check(name, fn):
        nonlocal passed, failed
        print(f"--- {name} ---")
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            logger.error(f"{name} raised: {e}", exc_info=True)
            ok, detail = False, f"{type(e).__name__}: {e}"
        print(f"Detail: {detail}")
        print("Result: " + ("PASS\n" if ok else "FAIL\n"))
        if ok:
            passed += 1
        else:
            failed += 1

    # ------------------------------------------------------------------
    # Deterministic pipeline checks (no network, no provider dependency)
    # ------------------------------------------------------------------

    def _brain_with_probe(responder):
        """A brain whose every turn is served by a recording probe session."""
        probe = _ProbeChat(responder)
        brain = JarvisBrain(conversation_id="verify")
        brain._create_chat = lambda model, history=None: probe
        brain._system_prompt = "IDENTITY"
        brain._probe = probe
        return brain

    def check_verbatim():
        def responder(index, payload):
            return ModelResponse(text="ack")

        brain = _brain_with_probe(responder)
        brain.ask("Tell me about the exit code in Python.")
        got = [p["text"] for p in brain._probe.sent if p.get("kind") == "user"]
        expected = ["Tell me about the exit code in Python."]
        return got == expected, f"model received {got!r}"

    def check_tool_loop():
        calls = []

        def responder(index, payload):
            if payload.get("kind") == "tool_results":
                return ModelResponse(text="It is 12:00.")
            return ModelResponse(text=None, tool_calls=[ToolCall(
                id=new_tool_call_id(), name="get_system_time", arguments={}
            )])

        brain = _brain_with_probe(responder)
        original = jarvis_brain.TOOL_REGISTRY["get_system_time"]
        jarvis_brain.TOOL_REGISTRY["get_system_time"] = lambda: (calls.append(1), "12:00")[1]
        try:
            reply = brain.ask("What time is it?")
        finally:
            jarvis_brain.TOOL_REGISTRY["get_system_time"] = original

        kinds = [p.get("kind") for p in brain._probe.sent]
        ok = len(calls) == 1 and kinds == ["user", "tool_results"] and reply == "It is 12:00."
        return ok, f"tool calls={len(calls)} turns={kinds} reply={reply!r}"

    def check_fallback():
        from tests.mock_providers import build_layer, error, reply as scripted_reply
        layer = build_layer(
            models=[
                {"key": "primary", "model": "m1", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
                {"key": "backup", "model": "m2", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"m1": error(ErrorKind.SERVER, "503"), "m2": scripted_reply("backup ok")},
        )
        brain = JarvisBrain(conversation_id="verify-fallback", model_layer=layer)
        reply = brain.ask("Why is the sky blue?")
        return reply == "backup ok", f"served by {brain.last_model_key!r}: {reply!r}"

    def check_tool_filter():
        from tests.mock_providers import build_layer, reply as scripted_reply
        layer = build_layer(
            models=[
                {"key": "no_tools", "model": "nt", "priority": 100,
                 "capabilities": {"reasoning": True, "tool_calling": False}},
                {"key": "with_tools", "model": "wt", "priority": 10,
                 "capabilities": {"reasoning": True, "tool_calling": True}},
            ],
            behaviours={"nt": scripted_reply("nt"), "wt": scripted_reply("wt")},
        )
        brain = JarvisBrain(conversation_id="verify-tool", model_layer=layer)
        brain.ask("What is the weather in Delhi?")
        return brain.last_model_key == "with_tools", f"selected {brain.last_model_key!r}"

    def check_honest_failure():
        from tests.mock_providers import build_layer, error
        layer = build_layer(
            models=[{"key": "only", "model": "m",
                     "capabilities": {"reasoning": True}}],
            behaviours={"m": error(ErrorKind.SERVER, "503")},
        )
        brain = JarvisBrain(conversation_id="verify-fail", model_layer=layer)
        try:
            brain.ask("Why?")
        except BrainError as e:
            return ("unavailable" in e.message.lower()), f"honest error: {e.message!r}"
        return False, "a fake assistant reply was returned instead of an error"

    def check_empty_input():
        brain = JarvisBrain(conversation_id="verify-empty")
        try:
            brain.ask("   ")
        except ValueError:
            return True, "blank input rejected"
        return False, "blank input was accepted"

    def check_model_layer():
        from jarvis.model_layer import get_model_layer
        layer = get_model_layer()
        status = layer.status()
        usable = [m for m in status["models"] if m["enabled"] and m["configured"]]
        return len(status["models"]) > 0, (
            f"{len(status['models'])} models configured, {len(usable)} with credentials"
        )

    def check_direct_live():
        from jarvis.direct import direct_ask
        reply = direct_ask("What is 2 + 2? Reply with just the number.")
        return "4" in reply, f"live reply: {reply[:80]!r}"

    check("User input forwarded verbatim", check_verbatim)
    check("Tool result fed back and loop terminates", check_tool_loop)
    check("Failed model falls back", check_fallback)
    check("Tool request only reaches tool-capable model", check_tool_filter)
    check("All models failing gives honest error", check_honest_failure)
    check("Empty input rejected", check_empty_input)
    check("Model layer loads", check_model_layer)

    # Live check: last, because provider availability is outside our control and
    # it should not mask a failure in the deterministic checks above.
    try:
        check("Minimal direct-model path (live)", check_direct_live)
    except BrainError as e:
        print("--- Minimal direct-model path (live) ---")
        print(f"Detail: {e.message}")
        print("Result: SKIPPED (no provider currently available)\n")

    total = passed + failed
    print(f"{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed out of {total} checks")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)


class _ProbeChat:
    """Minimal provider session that records payloads and replays a responder."""

    def __init__(self, responder):
        self.responder = responder
        self.sent = []

    def send_message(self, payload):
        self.sent.append(payload)
        return self.responder(len(self.sent), payload)

    def close(self):
        pass



if __name__ == "__main__":
    args = sys.argv[1:]

    if "--test" in args:
        run_verification_test()
    elif "--text" in args:
        run_text_mode()
    elif "--no-wake" in args:
        try:
            run_assistant()
        except KeyboardInterrupt:
            logger.info("Session ended by user.")
            sys.exit(0)
    else:
        # Default: wake word mode
        try:
            run_wake_word_mode()
        except KeyboardInterrupt:
            logger.info("Session ended by user.")
            sys.exit(0)
