"""Chatterbox TTS performance baseline (Phase 5: measurement only).

Exercises the REAL Jarvis TTS path -- same model loader, same reference
voice, same generation kwargs as `jarvis.speech._synthesize_chatterbox` --
and reports cold/warm latency, device, and config. No optimization, no
network, no providers, no tools, no playback, no disk I/O in the hot loop.

Run with the project environment (Chatterbox lives there, not in system python):
    .venv/bin/python scripts/tts_benchmark.py [--reps 5] [--md docs/tts_baseline.md]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PHRASES = {
    # id: (label, text) -- P4 deliberately repeats P2 verbatim.
    "P1": ("Very short (cold)", "Yes, Boss?"),
    "P2": ("Short (~10 words)",
           "Boss, your system is ready and waiting for your next command."),
    "P3": ("Medium (~30 words)",
           "Boss, I have checked your schedule, the weather in Delhi, and "
           "your latest messages. Everything looks completely normal, and "
           "there is nothing urgent that needs your attention right now."),
    "P4": ("Repeated short (== P2)",
           "Boss, your system is ready and waiting for your next command."),
    "P5": ("Representative response",
           "The current time is nine twenty PM. Your meeting tomorrow is "
           "at ten, and traffic looks light."),
}


def _ms(nanoseconds: int) -> float:
    return nanoseconds / 1_000_000.0


def _stats(samples):
    return {
        "n": len(samples),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5,
                        help="warm repetitions per phrase (default: 5)")
    parser.add_argument("--md", default=None,
                        help="write measured tables to this markdown file")
    args = parser.parse_args()

    import torch
    from jarvis import speech
    from jarvis.config import (CHATTERBOX_REFERENCE_AUDIO, CHATTERBOX_EXAGGERATION,
                               CHATTERBOX_CFG_WEIGHT, CHATTERBOX_TEMPERATURE)

    header = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "chatterbox": getattr(__import__("chatterbox"), "__version__", "unknown"),
        "device": speech._chatterbox_device(),
        "exaggeration": CHATTERBOX_EXAGGERATION,
        "cfg_weight": CHATTERBOX_CFG_WEIGHT,
        "temperature": CHATTERBOX_TEMPERATURE,
        "max_new_tokens": None,  # not a parameter of this generate() path
        "reference": Path(CHATTERBOX_REFERENCE_AUDIO).name,
    }
    print("TTS benchmark header:")
    for key, value in header.items():
        print(f"  {key}: {value}")

    # -- reference resolution (path validation only; the WAV itself is
    #    preprocessed inside every generate() call -- inseparable) ----------
    start = time.perf_counter_ns()
    reference = speech._resolve_chatterbox_reference()
    ref_ms = _ms(time.perf_counter_ns() - start)
    print(f"reference resolve: {ref_ms:.2f} ms ({reference.name})")

    # -- model initialization (singleton, as in production) ------------------
    start = time.perf_counter_ns()
    model = speech._get_chatterbox_model()
    load_ms = _ms(time.perf_counter_ns() - start)
    model_device = str(getattr(model, "device", getattr(model, "_device", "?")))
    print(f"model load: {load_ms:.0f} ms (model device: {model_device})")

    # -- reference preparation (cached conditionals; timed once) -------------
    prepare_ms = speech.prepare_chatterbox_reference(model)
    print(f"reference prepare: {prepare_ms:.0f} ms (reused on later calls)")

    gen_kwargs = dict(
        exaggeration=CHATTERBOX_EXAGGERATION,
        cfg_weight=CHATTERBOX_CFG_WEIGHT,
        temperature=CHATTERBOX_TEMPERATURE,
    )

    def _generate(text: str):
        """One timed generation; returns (generate_ms, convert_ms, samples)."""
        start = time.perf_counter_ns()
        wav = model.generate(text, **gen_kwargs)
        gen_ms = _ms(time.perf_counter_ns() - start)
        import numpy as np
        start = time.perf_counter_ns()
        array = wav.detach().cpu().numpy() if hasattr(wav, "detach") else wav
        array = __import__("numpy").asarray(array, dtype="float32").reshape(-1)
        conv_ms = _ms(time.perf_counter_ns() - start)
        assert array.size > 0 and abs(array).max() > 0, "silent/empty output"
        return gen_ms, conv_ms, int(array.size)

    # -- cold: first generation after init -----------------------------------
    label, text = PHRASES["P1"]
    cold_gen_ms, cold_conv_ms, cold_samples = _generate(text)
    print(f"cold P1 {label!r}: generate={cold_gen_ms:.0f} ms "
          f"convert={cold_conv_ms:.1f} ms samples={cold_samples}")

    # -- warm-up (discarded; one controlled pass like production) -------------
    warm_ms = speech.warm_up_chatterbox(model)
    print(f"warm-up (discarded): {warm_ms:.0f} ms")

    # -- warm repetitions -----------------------------------------------------
    results = {}
    for pid in ("P2", "P3", "P4", "P5"):
        label, text = PHRASES[pid]
        gens, convs = [], []
        for _ in range(args.reps):
            gen_ms, conv_ms, _ = _generate(text)
            gens.append(gen_ms)
            convs.append(conv_ms)
        results[pid] = (_stats(gens), _stats(convs))
        s = results[pid][0]
        print(f"warm {pid} {label!r} x{args.reps}: min={s['min']:.0f} "
              f"median={s['median']:.0f} mean={s['mean']:.0f} "
              f"max={s['max']:.0f} stdev={s['stdev']:.0f} ms")

    if args.md:
        _write_md(args.md, header, ref_ms, load_ms, prepare_ms, model_device,
                  (cold_gen_ms, cold_conv_ms, cold_samples),
                  (warm_ms,), results)
        print(f"wrote {args.md}")
    return 0


def _write_md(path, header, ref_ms, load_ms, prepare_ms, model_device,
              cold, warmup, results):
    pid_rows = "\n".join(
        f"| {pid} {PHRASES[pid][0]} | {PHRASES[pid][1]} |" for pid in results)
    warm_rows = "\n".join(
        f"| {pid} | {g['min']:.0f} | {g['median']:.0f} | {g['mean']:.0f} | "
        f"{g['max']:.0f} | {g['stdev']:.0f} |"
        for pid, (g, _) in results.items())
    Path(path).write_text(f"""# TTS Baseline

## Environment

- Date: {header['timestamp']}
- Python: {header['python']}
- PyTorch: {header['torch']} (MPS available: {header['mps_available']})
- Chatterbox: {header['chatterbox']}
- Device: {header['device']} (model reports: {model_device})

## Configuration

- max_new_tokens: n/a (not a parameter of this `generate()` path)
- exaggeration: {header['exaggeration']}
- cfg_weight: {header['cfg_weight']}
- temperature: {header['temperature']}
- Reference audio: {header['reference']} (prepared once into cached conditionals; per-call generation reuses them)

## Initialization

- Model load: {load_ms:.0f} ms
- Reference resolve: {ref_ms:.2f} ms (path validation only)
- Reference prepare (cached conditionals): {prepare_ms:.0f} ms
- Warm-up (discarded): {warmup[0]:.0f} ms

## Cold Generation

- Phrase: "Yes, Boss?"
- Generation: {cold[0]:.0f} ms
- Tensor-to-numpy: {cold[1]:.1f} ms
- Samples: {cold[2]}
- Cold total (load + resolve + prepare + warm-up + generation): {load_ms + ref_ms + prepare_ms + warmup[0] + cold[0]:.0f} ms

## Warm Generation (reps per phrase)

| Phrase | Text |
|---|---|
{pid_rows}

| Phrase | Min ms | Median ms | Mean ms | Max ms | Stdev ms |
|---|---:|---:|---:|---:|---:|
{warm_rows}

## Observations

<!-- Added by hand after reviewing the numbers. -->

## Bottleneck Assessment

<!-- Added by hand after reviewing the numbers. -->

## Optimization Decision

<!-- Added by hand after reviewing the numbers. -->
""", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
