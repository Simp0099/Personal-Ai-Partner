# Wake Phrases (Phase 8)

openWakeWord 0.6.0 built-ins (both interpreters agree):
`alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`, `timer`, `weather`.

## Active

| Phrase | Model | Verified | Threshold | Why |
|---|---|---|---|---|
| hey jarvis | `hey_jarvis` | built-in, loads and scores live | 0.5 | Pre-existing default; synthetic negatives peak ≤ 0.008 (`scripts/wake_eval.py`) |

## Disabled and why

| Phrase | Reason |
|---|---|
| jarvis (bare) | No verified built-in model. Only `hey_jarvis` exists; assigning it to bare "jarvis" would claim untested coverage. Needs a custom trained model. Entry present in `config.yaml`, `enabled: false`. |
| you there | No built-in model. Needs custom model + evaluation. Not configured. |
| wake up | No built-in model. Needs custom model + evaluation. Not configured. |
| hello | No built-in model AND high false-activation risk (common word). Evaluated last, only with a compatible model. Not configured. |

## Evaluation (2026-10-09, `scripts/wake_eval.py`)

Real `hey_jarvis` ONNX model, 80 ms chunks, per-chunk latency median ~2 ms.
Negatives (60 chunks silence/white-noise, 30 tones, 42 TTS-speech chunks):
peak scores 0.000–0.008 vs threshold 0.5 → PASS, 0 firings.
Limitation: synthetic negatives only. TTS positives (`--with-tts`) check the
detector chain, not human speech. Real-world false-positive claims need a
recorded corpus: ≥20 positive utterances (≥3 speakers, quiet + noisy rooms)
and ≥5 min of near-miss/confusable speech. Never commit private recordings.

## Architecture notes

- All active models load in ONE `Model(wakeword_models=[...])` session over
  the existing shared microphone stream. No second stream per phrase.
- Config order = fire priority; the wake latch allows one turn per utterance.
- `LISTENING` transition on wake is unchanged (Phase 7 intact).
