# TTS Latency Optimization

## Root cause

Chatterbox warm generation medians ~54–67 s per short reply (Phase 5/6).
The `27/1000` progress figure is an autoregressive step budget, not audio:
`t3.inference` breaks on the stop-speech token, so short replies already stop
early (~25 steps) — the `max_new_tokens=1000` cap (hardcoded in
chatterbox-tts 0.1.7's `generate()`, marked `# TODO: use the value in config`)
is only a worst-case bound, NOT the short-reply bottleneck. It is unreachable
through the installed `generate()` signature, so no supported knob shortens
short replies; per-step AR rate + fixed overheads dominate. Not changed.

## Engine comparison (measured, this machine)

| Engine | Warm short reply | Init | Voice | Deps/downloads |
|---|---|---|---|---|
| Chatterbox 0.1.7 (MPS) | ~54–67 s median | ~20–34 s once | cloned (`chatterbox_emotion_test.wav`) | installed, cached weights |
| macOS `say` / Samantha | **~1.06 s median** (min 1.05, max 1.12, RTF ~0.5–0.7) | none (~960 ms fixed process cost + ~100 ms text) | system voice, NOT the clone | OS-bundled, offline |
| espeak-ng | ~14 ms | none | robotic | installed (not wired; documented alternative) |
| kokoro / kokoro-onnx | n/a | n/a | n/a | `kokoro` not installed; `kokoro_onnx` present but NO model file exists offline → requires download → rejected per policy |

Detailed `say` numbers: `scripts/tts_latency.py` (4 phrases × 5 warm trials;
cold == warm by design). `say "Hi."` ≈ 960 ms proves the ~1 s floor is fixed
process/voice-load cost, not text length.

## Decision

Default engine is now `say` (interactive latency, ~60× faster). Chatterbox
remains fully working and selectable via `speech.tts_engine: "chatterbox"`;
nothing about its path changed. Voice identity is NOT preserved across the
switch (Samantha vs the clone) — explicit, documented trade-off. No model
was downloaded; no new dependency added.

## Before vs. after

| Metric | Before (Chatterbox default) | After (`say` default) |
|---|---|---|
| Warm short reply | ~54–67 s median | ~1.06 s median |
| Cold first audio | ~38 s (load + first gen) | ~1.06 s (no model) |
| Time to first audio | tens of seconds | ~1 s (file render then play; marginally over the 1 s target — reported, not rounded down) |
| Repeated calls | model reused (Phase 6) | stateless subprocess, no reload possible |
| Cloned voice | yes | no (opt-in via config) |

## Implementation notes

- `speech._synthesize_say` (temp AIFF → 16 kHz float32, always unlinked),
  `_speak_say`, `synthesize_for_engine` dispatcher; `SpeechPlayer` default
  resolves the engine at call time. Legacy non-`say` engines keep the
  Chatterbox array path (today's effective voice-mode behavior).
- Arg-list subprocess, timeouts, `TTSEngineError` on failure → console-text
  fallback (text is always printed first). Tracing spans kept.
- Chatterbox singleton, locks, warm-up, conds cache untouched.

## Weather fix (separate)

Symptom `Unable to parse exact temperature for delhi` was NOT a parser
field change: Google returns HTTP 429 (bot throttling, 3 KB page, zero
`BNeawe` divs). Fix: wttr.in primary (verified live: Delhi `Sunny +36°C`),
Google scrape retained as fallback, honest empty + message when both fail.
No invented values. Covered by `tests/test_weather.py` (mocked).

## Manual verification

1. `printf 'what time is it\n' | python -m jarvis --text` → spoken + printed
   reply in ~1 s instead of minutes (speaker-dependent for audibility).
2. Set `speech.tts_engine: "chatterbox"` → cloned voice path unchanged.
3. `python scripts/tts_latency.py` reproduces the table above.
4. `python -m pytest tests/test_tts_say.py tests/test_weather.py` green.
