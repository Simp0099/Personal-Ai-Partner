#!/usr/bin/env python3
"""Record four seconds from the default microphone and report the level per half second."""

import numpy as np
import sounddevice as sd

RATE = 16000
SECONDS = 4

info = sd.query_devices(kind="input")
print(f"Input device: {info['name']} ({info['max_input_channels']} channels)")
print("Speak now...")
audio = sd.rec(int(RATE * SECONDS), samplerate=RATE, channels=1, dtype="float32")
sd.wait()
x = audio[:, 0]
for start in range(0, len(x), RATE // 2):
    seg = x[start:start + RATE // 2]
    print(f"{start / RATE:4.1f}s  rms={np.sqrt(np.mean(seg ** 2)):.4f}")
if np.max(np.abs(x)) < 1e-4:
    print("SILENT. Check the macOS microphone permission (spec section 3.3).")
    raise SystemExit(1)
print("OK. Audio is arriving.")