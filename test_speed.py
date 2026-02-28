#!/usr/bin/env python3
"""Fair speed comparison — interleaved runs to cancel thermal/load bias.

Runs baseline and test configs in alternating order on the same audio,
multiple times, and reports median timings.

Usage:
    python test_speed.py
"""
import os
import time
import statistics

import numpy as np
import mlx.core as mx
import mlx.nn as nn


def load_model():
    import nemotron_asr_mlx as nm
    local_path = os.path.expanduser("~/Models/nemotron-asr-mlx")
    if os.path.isdir(local_path):
        return nm.from_pretrained(local_path)
    return nm.from_pretrained("dboris/nemotron-asr-mlx")


def time_transcribe(model, audio, n_runs=5):
    """Time transcription over n_runs, return list of times."""
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        model.transcribe(audio)
        elapsed = time.time() - t0
        times.append(elapsed)
    return times


def main():
    print("=" * 60)
    print("SPEED BENCHMARK — INTERLEAVED RUNS")
    print("=" * 60)

    # Generate test audio of various lengths
    np.random.seed(42)
    audio_60s = np.random.randn(16000 * 60).astype(np.float32) * 0.01  # 60s noise
    audio_10s = np.random.randn(16000 * 10).astype(np.float32) * 0.01  # 10s noise

    print("Loading model...")
    model = load_model()

    # Warmup
    print("Warming up (3 runs)...")
    for _ in range(3):
        model.transcribe(np.zeros(16000, dtype=np.float32))

    configs = {}
    n_runs = 5

    # ── Baseline (fp32) ──
    print("\n[baseline] float32...")
    configs["baseline"] = time_transcribe(model, audio_60s, n_runs)

    # ── bfloat16 ──
    from mlx.utils import tree_flatten, tree_unflatten
    orig_weights = [(k, mx.array(v)) for k, v in tree_flatten(model.parameters())]
    new_w = [(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(new_w))
    mx.eval(model.parameters())
    model.transcribe(np.zeros(16000, dtype=np.float32))  # warmup

    print("[bf16] bfloat16 weights...")
    configs["bf16"] = time_transcribe(model, audio_60s, n_runs)

    # Revert to fp32
    model.update(tree_unflatten(orig_weights))
    mx.eval(model.parameters())

    # ── Baseline again (check thermal drift) ──
    print("[baseline_2] float32 again...")
    configs["baseline_2"] = time_transcribe(model, audio_60s, n_runs)

    # ── mx.compile on joint ──
    orig_joint_call = model.joint_net.__call__
    model.joint_net.__call__ = mx.compile(orig_joint_call)
    model.transcribe(np.zeros(16000, dtype=np.float32))

    print("[compile_joint] mx.compile joint net...")
    configs["compile_joint"] = time_transcribe(model, audio_60s, n_runs)
    model.joint_net.__call__ = orig_joint_call

    # ── mx.compile on encoder ──
    orig_enc_call = model.encoder.__call__
    model.encoder.__call__ = mx.compile(orig_enc_call)
    model.transcribe(np.zeros(16000, dtype=np.float32))

    print("[compile_enc] mx.compile encoder...")
    configs["compile_enc"] = time_transcribe(model, audio_60s, n_runs)
    model.encoder.__call__ = orig_enc_call

    # ── Both compiled ──
    model.joint_net.__call__ = mx.compile(orig_joint_call)
    model.encoder.__call__ = mx.compile(orig_enc_call)
    model.transcribe(np.zeros(16000, dtype=np.float32))

    print("[compile_both] mx.compile encoder + joint...")
    configs["compile_both"] = time_transcribe(model, audio_60s, n_runs)
    model.joint_net.__call__ = orig_joint_call
    model.encoder.__call__ = orig_enc_call

    # ── bf16 + compile both ──
    new_w = [(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(new_w))
    mx.eval(model.parameters())
    model.joint_net.__call__ = mx.compile(model.joint_net.__call__.__func__ if hasattr(model.joint_net.__call__, '__func__') else type(model.joint_net).__call__)
    model.encoder.__call__ = mx.compile(model.encoder.__call__.__func__ if hasattr(model.encoder.__call__, '__func__') else type(model.encoder).__call__)
    model.transcribe(np.zeros(16000, dtype=np.float32))

    print("[bf16+compile] bf16 + compile both...")
    configs["bf16+compile"] = time_transcribe(model, audio_60s, n_runs)

    # Revert
    model.update(tree_unflatten(orig_weights))
    mx.eval(model.parameters())

    # ── INT8 quantization ──
    print("[int8] Reloading for INT8...")
    model = load_model()
    model.transcribe(np.zeros(16000, dtype=np.float32))
    nn.quantize(model, bits=8, group_size=64)
    mx.eval(model.parameters())
    model.transcribe(np.zeros(16000, dtype=np.float32))

    print("[int8] INT8 quantization...")
    configs["int8"] = time_transcribe(model, audio_60s, n_runs)

    # ── INT4 quantization ──
    print("[int4] Reloading for INT4...")
    model = load_model()
    model.transcribe(np.zeros(16000, dtype=np.float32))
    nn.quantize(model, bits=4, group_size=64)
    mx.eval(model.parameters())
    model.transcribe(np.zeros(16000, dtype=np.float32))

    print("[int4] INT4 quantization...")
    configs["int4"] = time_transcribe(model, audio_60s, n_runs)

    # ── Final baseline ──
    print("[baseline_final] Reloading for final baseline...")
    model = load_model()
    model.transcribe(np.zeros(16000, dtype=np.float32))
    configs["baseline_final"] = time_transcribe(model, audio_60s, n_runs)

    # ── Results ──
    print(f"\n{'=' * 60}")
    print("RESULTS (60s audio, median of 5 runs)")
    print(f"{'=' * 60}")
    print(f"{'Config':<20} {'Median':>8} {'Min':>8} {'Max':>8} {'RTFx':>8}")
    print(f"{'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for name, times in configs.items():
        med = statistics.median(times)
        mn = min(times)
        mx_t = max(times)
        rtfx = 60.0 / med
        print(f"{name:<20} {med:>7.3f}s {mn:>7.3f}s {mx_t:>7.3f}s {rtfx:>7.1f}x")

    # Compute drift
    b1 = statistics.median(configs.get("baseline", [1]))
    b2 = statistics.median(configs.get("baseline_final", [1]))
    drift = (b2 - b1) / b1 * 100
    print(f"\nBaseline drift: {drift:+.1f}% (first→last)")
    if abs(drift) > 5:
        print("  WARNING: significant thermal/load drift detected")


if __name__ == "__main__":
    main()
