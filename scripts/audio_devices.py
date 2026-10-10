#!/usr/bin/env python3
"""Enumerate PortAudio devices and show configured/default selections."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from jarvis.config import AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE

    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in, default_out = sd.default.device
    except Exception as exc:
        print(f"FAIL: PortAudio device enumeration failed: {exc}")
        print("Check the audio backend and macOS Microphone permission, then retry.")
        return 1

    print(f"Default input={default_in}, output={default_out}")
    print(f"Configured input={AUDIO_INPUT_DEVICE!r}, output={AUDIO_OUTPUT_DEVICE!r}")
    usable_in = usable_out = 0
    for index, device in enumerate(devices):
        inputs = int(device["max_input_channels"])
        outputs = int(device["max_output_channels"])
        usable_in += inputs > 0
        usable_out += outputs > 0
        input_16k = output_16k = "n/a"
        try:
            if inputs:
                sd.check_input_settings(device=index, channels=1, dtype="int16", samplerate=16000)
                input_16k = "yes"
        except Exception:
            input_16k = "no"
        try:
            if outputs:
                sd.check_output_settings(device=index, channels=1, dtype="float32", samplerate=16000)
                output_16k = "yes"
        except Exception:
            output_16k = "no"
        print(f"{index}: {device['name']} | input_channels={inputs} "
              f"output_channels={outputs} default_rate={device['default_samplerate']} "
              f"mono_int16_16k_input={input_16k} mono_float32_16k_output={output_16k}")
    if not devices or not (usable_in or usable_out):
        print("BLOCKED: PortAudio exposed no usable input or output devices to this process.")
        print("Connect/select a device and check macOS Privacy & Security permissions.")
        return 2
    if default_in < 0 and usable_in:
        print("NOTE: input exists but no default is selected; configure conversation.input_device.")
    if default_out < 0 and usable_out:
        print("NOTE: output exists but no default is selected; configure conversation.output_device.")
    if not usable_in:
        print("BLOCKED: no input-capable device is visible (hardware and permission denial can look alike).")
    if not usable_out:
        print("BLOCKED: no output-capable device is visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
