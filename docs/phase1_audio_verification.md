# Phase 1 audio verification

Run from the repository root. Use the existing virtual environment; do not run
the install step without approving model/dependency downloads.

## A. Static and unit checks

```bash
.venv/bin/python -m pytest tests/test_phase4_voice.py tests/test_phase8_wake.py tests/test_wake_listener_lifecycle.py tests/test_startup.py tests/test_tts_say.py tests/test_tts_chatterbox.py tests/test_phase6_tts_service.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/python start.py --test
.venv/bin/python start.py --status
git diff --check
.venv/bin/python -m compileall -q jarvis scripts/audio_devices.py scripts/mic_check.py scripts/tts_smoke_test.py
```

## B–C. Device enumeration and microphone

```bash
.venv/bin/python scripts/audio_devices.py
.venv/bin/python scripts/mic_check.py --seconds 5
```

The first command reports PortAudio-visible input/output devices, channel
capabilities, default rate, and defaults. If devices exist without defaults, put
the selected current device IDs in `conversation.input_device` and
`conversation.output_device` in `config.yaml`. IDs may change after reconnecting.
Speak during the capture interval; compare peak and mean RMS.

## D–F. Kokoro synthesis, playback, and ASR

The current Kokoro project documents `kokoro>=0.9.4`, `soundfile`, English
phonemization through Misaki/espeak-ng, `KPipeline(lang_code='a')`, voice
`am_adam`, and 24 kHz waveform output. Kokoro downloads the model and voice pack
on first use. On macOS, install dependencies only after reviewing the package
and system runtime needs; then run:

```bash
.venv/bin/python scripts/tts_smoke_test.py --no-play  # offline only after assets are cached
.venv/bin/python scripts/tts_smoke_test.py            # opens selected output device
```

`--no-play` performs real synthesis and returns nonzero if the package, model,
voice, phonemizer, or waveform validation fails. It can still access the network
on the first invocation if assets are missing; do not run until that is approved.
The second command requires a functioning speaker/output device.

ASR is remote only when `speech.asr_engine: google` is configured. To verify a
spoken phrase without recording it to disk, use the normal voice session below;
the request is sent to Google's SpeechRecognition endpoint. Local ASR is not
implemented and fails during initialization.

## G–H. User-present live voice acceptance

```bash
.venv/bin/python start.py --no-wake
```

Speak five separate turns and confirm each reply is audible. Interrupt one
reply by speaking while it plays, then press Ctrl+C during capture and during a
later reply. Restart the command for repeated start/stop checks and inspect for
PortAudio errors or surviving `microphone`, `speech-player`, and `voice-turn`
threads. For wake mode, run `.venv/bin/python start.py` and say the configured
wake phrase before each new conversation. These steps require the user and
working hardware; automated tests are not evidence of live success.
