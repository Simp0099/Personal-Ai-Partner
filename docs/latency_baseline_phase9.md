# Latency Baseline — Phase 9

Turn tracing (`jarvis/trace.py`): one `TurnTrace` per assistant turn,
monotonic `perf_counter` spans, statuses ok/fail/cancelled/skipped, DEBUG-only
compact summaries. Spans carry names + model/tool identifiers only — never
prompts, bodies, audio, or secrets. Overhead is two clock reads per span
(~100 ns); tracing degrades to no-ops rather than raising.

Historical reports (`tts_baseline.md`, `tts_optimization_phase6.md`,
`wake_phrases.md`) are preserved; this file adds turn-level tracing data.

## Environment

- Date: 2026-10-09. Python 3.14.7 (suite), project venv 3.12.15 for live TTS.
- Live provider quota exhausted during this phase: provider-backed turns are
  MOCKED (instant scripted replies); local turns are REAL (real intent
  matching + real tools, stubbed only where noted).

## Methodology

- `printf 'what time is it' | python -m jarvis --text --debug` → live local trace.
- Mocked turns: `JarvisBrain` + `tests/mock_providers.py` scripted replies,
  5 repetitions, real `perf_counter` (mocks return ~instantly, so absolute
  values measure harness overhead, not providers — structure is the point).
- No p95 (n too small): min/median/max only.
- Wake/STT/TSS device numbers cited from prior reports, not re-measured.

## Stage names

| Stage | Includes |
|---|---|
| `intent_route` | Phase 2 local-intent match + dispatch (tool runtime inside) |
| `classify` | Deterministic `layer.classify` call |
| `provider_request` | One `send_message` network call; attrs `model`, `provider`, `attempt`/`followup` |
| `tool_execute` | One `_execute_tool_call`; attr `tool` (name only, never args) |
| `stt` / `think` / `speak` | Voice-loop transcribe / brain-ask / synth+playback |
| `tts_ready` | `ensure_chatterbox_ready` (load+prepare+warm on first call, ~0 ms warm) |
| `tts_generate` | `model.generate` call only |
| `wake_frame_ms` | Firing-frame inference time (note, not span) |
| Skipped stages | Absent from the trace (e.g. no `provider_request` on local route) |

## Measured turns

Live local (`what time is it`, real tool, text mode):
```text
trace turn_ed733fe6 route=local total=117.4ms
  stages=[intent_route=117.3ms:ok tts_ready=0.1ms:fail]
```
The `tts_ready:fail` is honest: the turn's spoken confirmation attempted
Chatterbox (absent in system python) and fell back to console text.

Mocked provider-backed (`Write a story about Mars`, n=5):
total min 0.2 / median 0.2 / max 4.8 ms; `provider_request` median 0.0 ms
(mock latency, not provider latency).

Mocked tool-loop (`check my inbox`):
`intent_route → classify → provider_request → tool_execute(get_system_time)
→ provider_request(followup)` — fallback attempts each get their own span;
no double counting.

## Cited device numbers (prior phases, unchanged code paths)

- Wake inference: ~2 ms/chunk (`wake_eval.py`); VAD/STT live numbers unavailable.
- TTS cold ~38 s first-audio / warm medians ~54–67 s (Phase 5/6 reports).
- `tts_ready` warm ≈ 0.1 ms; no `tts_load` span exists by construction.

## Limitations

- No live provider/STT/microphone numbers (quota/hardware); voice `stt`/`think`/
  `speak` spans are test-covered, not live-measured.
- Mocked absolute durations reflect harness speed, not production latency.
- First-turn `tts_ready` cost is measured in the TTS reports, not here.
- Trace covers one turn; cross-turn aggregation is future work.
