#!/usr/bin/env python3
"""Render a test phrase with Kokoro, report timing and pauses, and write a WAV file."""
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jarvis import tts  # noqa: E402
PHRASE = "Hello there... how are you doing today? Good. Let us get started."
t0 = time.perf_counter()
audio = tts.synthesize(PHRASE)
cold = time.perf_counter() - t0
t0 = time.perf_counter()
audio = tts.synthesize(PHRASE)
warm = time.perf_counter() - t0
seconds = len(audio) / tts.KOKORO_RATE
print(f"cold start {cold:.2f}s, warm {warm:.2f}s, audio {seconds:.2f}s, real-time factor {warm / seconds:.2f}")
quiet = np.abs(audio) < 1e-3
pauses, run = [], 0
for is_quiet in quiet:
    if is_quiet:
        run += 1
    else:
        if run / tts.KOKORO_RATE > 0.2:
            pauses.append(round(run / tts.KOKORO_RATE, 2))
        run = 0
print("pauses longer than 0.2s (seconds):", pauses)
out = ROOT / "data" / "kokoro_check.wav"
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(out), audio, tts.KOKORO_RATE)
print(f"wrote {out}")
