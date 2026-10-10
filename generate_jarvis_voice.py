import sys
import wave
from pathlib import Path

import numpy as np

from jarvis import speech

PROJECT_ROOT = Path(__file__).resolve().parent

TEXT = """
Oh yessss... you absolute degenerate piece of shit.

Mmmhmm... oh, look at you. Such a fucking troublemaker, aren't you?

You bad boy... you absolute menace.

Heh... you really think you can get away with that shit?

Oh, sweetheart... you have absolutely no fucking idea what you're doing, do you?

That's adorable.

Come on, you cheeky little bastard... don't act all innocent now.

Mmmhmm... that's what I fucking thought.

Goddamn... you're a walking disaster, and somehow you manage to make it entertaining.

Now behave yourself...

Or don't.

Heh... we both know you're going to be a little shit anyway.
"""

OUTPUT = PROJECT_ROOT / "jarvis_dominant_extended.wav"


def main():
    import torch

    reference = speech._resolve_chatterbox_reference()
    print(f"Reference voice: {reference}")
    print("Loading Chatterbox model...")

    model = speech._get_chatterbox_model()
    print(f"Using device: {speech._chatterbox_device()}")
    print("Generating audio. The first run may take a while...")

    with speech._inference_context():
        result = model.generate(
            TEXT,
            audio_prompt_path=str(reference),
            exaggeration=0.65,
            cfg_weight=0.5,
            temperature=0.8,
        )

    if isinstance(result, torch.Tensor):
        audio = result.detach().float().cpu().numpy()
    else:
        audio = np.asarray(result, dtype=np.float32)

    audio = np.squeeze(audio)

    if audio.ndim != 1 or audio.size == 0:
        raise RuntimeError(f"Unexpected audio shape: {audio.shape}")

    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)

    sample_rate = model.sr
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(str(OUTPUT), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    print(f"\nDone! Saved: {OUTPUT}")
    print(f"Duration: {len(audio) / sample_rate:.1f} seconds")
    print(f"Sample rate: {sample_rate} Hz")


if __name__ == "__main__":
    main()
