"""Reproducible wake-word evaluation (Phase 8).

Scores the REAL configured WakeWordEngine (no mic) on synthetic negatives
and -- optionally -- Chatterbox-spoken positives, all through the same
80 ms / 16 kHz int16 chunks the shared stream feeds it.

    python3 scripts/wake_eval.py                 # fast: negatives + latency
    .venv/bin/python scripts/wake_eval.py --with-tts   # + TTS positives (~5 gens)

Negative set (seeded, deterministic): digital silence, white noise, pure
tones, and the repo's own TTS sample (real speech, no wake phrase).
Positive set (only with --with-tts): Chatterbox saying the active phrases
plus near-miss confusables. TTS voice != human voice: positives measure the
detector chain, not real-world accuracy. A recorded human corpus is still
needed for real-world false-positive claims -- see docs/wake_phrases.md.

Exit 0: no negative scored at/above its phrase threshold.
Exit 2: engine could not load (deps/model), or a negative fired.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHUNK = 1280  # 80 ms at 16 kHz, same as the shared stream
RATE = 16000


def _to_chunks(mono_float, rate=RATE):
    import numpy as np
    pcm = np.asarray(mono_float, dtype=np.float64).reshape(-1)
    if rate != RATE:  # crude linear resample; negatives only need energy
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * RATE / rate))
        pcm = np.interp(idx, np.arange(len(pcm)), pcm)
    pcm = np.clip(pcm, -1.0, 1.0)
    raw = (pcm * 32767).astype(np.int16).tobytes()
    return [raw[i:i + CHUNK * 2] for i in range(0, len(raw) - CHUNK * 2 + 1, CHUNK * 2)]


def _negatives():
    import numpy as np
    rng = np.random.default_rng(8)
    n = CHUNK * 60
    t = np.arange(n) / RATE
    out = {
        "silence": np.zeros(n),
        "white_noise": rng.standard_normal(n) * 0.3,
        "tone_440": 0.4 * np.sin(2 * np.pi * 440 * t),
        "tone_880": 0.4 * np.sin(2 * np.pi * 880 * t),
    }
    try:
        import soundfile as sf
        for name in ("chatterbox_test.wav",):
            data, rate = sf.read(str(PROJECT_ROOT / name), dtype="float64")
            out[f"tts_speech_{name}"] = np.asarray(data).reshape(-1)
    except Exception as e:
        print(f"note: speech negative unavailable ({e})")
    return out


def _score_chunks(engine, chunks):
    """Max per-phrase score + median chunk latency over the sample."""
    lat, best = [], {}
    for pcm in chunks:
        start = time.perf_counter_ns()
        scores = engine.scores(pcm)
        lat.append((time.perf_counter_ns() - start) / 1_000_000.0)
        for phrase, score in scores.items():
            best[phrase] = max(best.get(phrase, 0.0), score)
    lat.sort()
    return best, lat[len(lat) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-tts", action="store_true",
                        help="synthesize positives with Chatterbox (slow)")
    args = parser.parse_args()

    from jarvis.config import WAKE_WORD_PHRASES
    from jarvis.wake_word import WakeWordEngine

    active = [p for p in WAKE_WORD_PHRASES if p.get("enabled")]
    print(f"active phrases: {[(p['phrase'], p['model'], p['threshold']) for p in active]}")
    engine = WakeWordEngine(phrases=WAKE_WORD_PHRASES)
    if not engine.load():
        print("FAIL: wake engine did not load (see log). No measurements taken.")
        return 2

    failures = 0
    print("\n-- negatives (must all stay below threshold) --")
    for name, audio in _negatives().items():
        chunks = _to_chunks(audio)
        best, med_ms = _score_chunks(engine, chunks)
        for phrase, peak in sorted(best.items()):
            threshold = engine.thresholds[phrase]
            flag = "FIRE" if peak >= threshold else "ok"
            if flag == "FIRE":
                failures += 1
            print(f"[{flag}] {name:28s} {phrase:12s} peak={peak:.3f} "
                  f"threshold={threshold} chunks={len(chunks)} lat_med={med_ms:.1f}ms")

    if args.with_tts:
        print("\n-- TTS positives (synthetic speech, chain check only) --")
        from jarvis import speech
        model = speech._get_chatterbox_model()
        ref = str(speech._resolve_chatterbox_reference())
        texts = (["Hey Jarvis"] * 2 + ["Hey Jarvis, what time is it",
                                       "Hey service", "Jarvis"])
        for text in texts:
            wav = model.generate(text, audio_prompt_path=ref)
            import numpy as np
            arr = wav.detach().cpu().numpy().reshape(-1) if hasattr(wav, "detach") else wav
            best, _ = _score_chunks(engine, _to_chunks(np.asarray(arr, dtype=np.float64)))
            print(f"tts {text!r:38s} " +
                  " ".join(f"{p}={s:.3f}" for p, s in sorted(best.items())))

    print(f"\nresult: {'FAIL' if failures else 'PASS'} ({failures} negative firing(s))")
    engine.close()
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
