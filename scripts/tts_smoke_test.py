"""Real Chatterbox smoke test through the application's TTS path (not a unit test).

Traces and proves the live path:

    assistant text
      -> jarvis.speech.speak()
      -> _synthesize_chatterbox()
      -> ChatterboxTTS.generate(audio_prompt_path=<resolved reference>)
      -> sounddevice.play()

It records the real `audio_prompt_path` Chatterbox receives and fails if it is
not the approved reference voice.

Run:
    python3 scripts/tts_smoke_test.py
    python3 scripts/tts_smoke_test.py --no-play    # generate + verify, no audio
"""

import argparse
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis import speech  # noqa: E402
from jarvis.config import CHATTERBOX_REFERENCE_AUDIO  # noqa: E402

SENTENCE = "Hey Boss... I'm here. What are we doing?"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-play", action="store_true", help="skip audio playback")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "chatterbox_smoke.wav"))
    args = parser.parse_args()

    expected = speech._resolve_chatterbox_reference()
    print(f"Configured reference : {CHATTERBOX_REFERENCE_AUDIO}")
    print(f"Resolved reference   : {expected}")
    print(f"Engine               : {speech.TTS_ENGINE}")

    # Record what actually reaches the Chatterbox generation call.
    from chatterbox.tts import ChatterboxTTS

    seen = {}
    real_generate = ChatterboxTTS.generate

    def spy(self, text, **kwargs):
        seen["text"] = text
        seen["audio_prompt_path"] = kwargs.get("audio_prompt_path")
        return real_generate(self, text, **kwargs)

    ChatterboxTTS.generate = spy
    played = {}
    try:
        import sounddevice

        real_play = sounddevice.play
        sounddevice.play = lambda data, sr=None: played.update(
            samples=len(data), sample_rate=sr, data=data
        )
    except ImportError:
        real_play = None

    try:
        # The real application entry point: text in, audio out.
        speech.speak(SENTENCE)
    finally:
        ChatterboxTTS.generate = real_generate
        if real_play is not None:
            sounddevice.play = real_play

    prompt = Path(seen.get("audio_prompt_path") or "")
    assert seen.get("audio_prompt_path"), "Chatterbox received no audio_prompt_path"
    assert prompt.samefile(expected), f"Wrong reference reached Chatterbox: {prompt}"
    assert seen.get("text") == SENTENCE, f"Unexpected text: {seen.get('text')!r}"
    print(f"audio_prompt_path    : {prompt} (matches approved reference)")
    print(f"Text to Chatterbox   : {seen['text']!r}")

    audio = played.get("data")
    if audio is None:
        print("Playback layer: skipped (--no-play)")
        return 0

    sample_rate = played["sample_rate"]
    print(f"Playback             : {played['samples']} samples @ {sample_rate} Hz "
          f"({played['samples'] / sample_rate:.2f}s) via sounddevice.play")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((audio * 32767).astype("int16").tobytes())
    print(f"Wrote {out}")
    print("SMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
