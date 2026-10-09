# TTS Optimization — Phase 6 Report

Phase 5 (`docs/tts_baseline.md`, preserved untouched) measured the old path:
reference preprocessed inside every `generate()` via `audio_prompt_path`.
Phase 6 prepares conditionals once and reuses them. Same phrases, same
environment (venv Python 3.12.15, torch 2.6.0, Chatterbox 0.1.7, MPS),
same 5-rep methodology.

## Environment and Device

- Date: 2026-10-09 11:44:06 IST
- Python / PyTorch / Chatterbox: 3.12.15 / 2.6.0 / 0.1.7 (identical to Phase 5)
- Device: MPS, model reports `mps` (no silent fallback, both runs)
- Generation settings, unchanged: exaggeration 0.5, cfg_weight 0.5,
  temperature 0.8, no `max_new_tokens` in this path (hardcoded 1000 inside
  the library's `t3.inference`, not exposed)

## Before vs. After

| Metric | Phase 5 baseline | Phase 6 result |
|---|---:|---:|
| Model initialization | 20,017 ms | 34,428 ms (one-time; delta is run-to-run load variance, not code) |
| Reference resolve | 0.02 ms | 0.03 ms (path validation only, both runs) |
| Reference prepare | n/a (inseparable, paid inside every generation) | 4,557 ms once, 0.0 ms on reuse |
| Warm-up | 18,455 ms (same-text generation) | 19,892 ms (controlled, once per process) |
| Cold P1 generation | 18,188 ms (first generation ever) | 23,894 ms (first generation after warm-up — not directly comparable) |
| Warm P2 median | 59,851 ms | 88,643 ms |
| Warm P3 median | 66,860 ms | 96,906 ms |
| Warm P4 median | 54,428 ms | 73,996 ms |
| Warm P5 median | 64,202 ms | 63,291 ms |
| Tensor-to-numpy | 0.5 ms | 2.9 ms (negligible both runs) |

## Implementation Changes

- `jarvis/speech.py`: module RLock (double-checked init, serialized
  inference); `prepare_chatterbox_reference()` (conditionals prepared once
  per reference file via the API-sanctioned `prepare_conditionals`, reused
  after); `warm_up_chatterbox()` (one controlled pass, best-effort);
  `ensure_chatterbox_ready()` orchestrator; `generate()` called without
  `audio_prompt_path` under an explicit `torch.inference_mode()` boundary.
- Justification: Phase 5 showed per-call reference preprocessing with no
  reuse, no init lock, no warm-up lifecycle. The library already runs its
  own inference under `inference_mode`; the outer boundary is conformance,
  not a claimed speedup. No sampling/device/voice parameter was touched.

## Results

- Reference preparation (~4.6 s) moved from per-call to once-per-process:
  the only guaranteed, measured structural saving.
- Warm-generation medians overlap within measured run variance (Phase 5
  stdevs 5–54 s; Phase 6 stdevs 5–23 s): no significant generation-speed
  change in either direction. The dominant cost (autoregressive sampling,
  tens of seconds) is untouched by design, so this is the expected outcome,
  not a regression signal.
- Model-load delta (20 s vs 34 s) is machine-state noise on a one-time cost.

## Voice and Stability

- Reference voice and all generation parameters unchanged (locked by tests).
- Audio pipeline untouched; synthesis output shape/dtype contract unchanged.
- Focused TTS tests: 12 new service tests pass; existing
  `test_tts_chatterbox.py` updated only where it encoded the old per-call
  contract (documented in commit).
- Full suite: 918 passed, 1 skipped (pre-existing native-wake opt-in).

## Remaining Bottlenecks

- Autoregressive generation throughput (tens of seconds per utterance) —
  requires sampling/model-level work, out of scope here.
- One-time ~20–35 s model load on first use (lazy init preserved; no
  startup delay introduced).
- `max_new_tokens=1000` is hardcoded inside the installed library and not
  exposed; changing it means forking library behavior — not attempted.
- Single global inference lock serializes concurrent synthesis; matches the
  single-consumer voice loop, revisit only if concurrent TTS is ever needed.
