# TTS Baseline

## Environment

- Date: 2026-10-08 21:34:19 IST
- Python: 3.12.15
- PyTorch: 2.6.0 (MPS available: True)
- Chatterbox: 0.1.7
- Device: mps (model reports: mps)

## Configuration

- max_new_tokens: n/a (not a parameter of this `generate()` path)
- exaggeration: 0.5
- cfg_weight: 0.5
- temperature: 0.8
- Reference audio: chatterbox_emotion_test.wav (resolved per call; preprocessed inside every `generate()` -- no caching)

## Initialization

- Model load: 20017 ms
- Reference resolve: 0.02 ms (path validation only)
- Warm-up (discarded): 18455 ms

## Cold Generation

- Phrase: "Yes, Boss?"
- Generation: 18188 ms
- Tensor-to-numpy: 0.5 ms
- Samples: 30720
- Cold total (load + resolve + generation): 38205 ms

## Warm Generation (reps per phrase)

| Phrase | Text |
|---|---|
| P2 Short (~10 words) | Boss, your system is ready and waiting for your next command. |
| P3 Medium (~30 words) | Boss, I have checked your schedule, the weather in Delhi, and your latest messages. Everything looks completely normal, and there is nothing urgent that needs your attention right now. |
| P4 Repeated short (== P2) | Boss, your system is ready and waiting for your next command. |
| P5 Representative response | The current time is nine twenty PM. Your meeting tomorrow is at ten, and traffic looks light. |

| Phrase | Min ms | Median ms | Mean ms | Max ms | Stdev ms |
|---|---:|---:|---:|---:|---:|
| P2 | 32097 | 59851 | 53616 | 62674 | 12772 |
| P3 | 63348 | 66860 | 90634 | 186959 | 53974 |
| P4 | 52768 | 54428 | 56163 | 65568 | 5309 |
| P5 | 55176 | 64202 | 64406 | 75894 | 7790 |

## Observations

- Model load (~20 s, weights cached) and first generation (~18 s) put time-to-first-audio at ~38 s cold.
- Warm-up shows no benefit: same-text warm-up (18.5 s) ≈ cold (18.2 s).
- Warm generation is slow everywhere: medians P2 ~60 s, P3 ~67 s, P4 ~54 s, P5 ~64 s.
- Repeating identical text (P4 == P2) gives no meaningful speedup (54 s vs 60 s median).
- Latency grows with text length (2 words ~18 s → 11 words ~55-60 s → 29 words ~67 s) but noisily.
- P3 max 187 s is an outlier (3× its median); single-run instability, not a conclusion.
- Tensor-to-numpy conversion (~0.5 ms) is negligible.
- Reference-resolve is path validation only (0.02 ms); actual reference preprocessing happens inside every `generate()` call and cannot be separated without rewriting the call.
- Device is genuinely MPS (model reports `mps`); no silent CPU fallback. torch 2.6.0, Chatterbox 0.1.7.
- The `x/1000` progress denominator is an autoregressive step budget display, not audio samples (probe generation stopped at step ~25); per the Phase 5 warning, no conclusion is drawn from it.
- No `max_new_tokens` or equivalent exists anywhere in this path, so token-limit blame is not applicable.

## Bottleneck Assessment

- Model initialization: one-time ~20 s. Significant for cold start, irrelevant per-turn.
- Reference preprocessing: contribution unknown (inseparable from generation).
- Generation: dominates everything (tens of seconds per utterance). The measured bottleneck.
- Post-processing: negligible (~0.5 ms).
- Device: as expected (MPS). Not the problem.

## Optimization Decision

Yes, with a scoped mandate for Phase 6: warm-generation latency (~1 minute medians) is far above interactive needs, and generation itself is the measured dominant cost, so generation-speed work is evidence-backed. One-time costs (model load, warm-up behavior) are secondary targets. Reference-audio caching has NO measured contribution yet -- Phase 6 must measure before/after if attempted, not assume it. No parameter looks "wrong"; change nothing without a measured win.

Baseline provenance: Phase 0 recorded no numeric TTS latency. This file establishes the first measured TTS baseline.
