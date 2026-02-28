#!/usr/bin/env python3
"""A/B test framework for nemotron-asr-mlx optimizations.

Tests each optimization individually and in combinations on a subset of
LibriSpeech test-clean, measuring WER and RTFx.

Usage:
    python test_optimizations.py                    # run all tests
    python test_optimizations.py --samples 50       # quick run (50 samples)
    python test_optimizations.py --test reflect     # run single test
    python test_optimizations.py --list             # list available tests
"""

import os
import sys
import time
import json
import copy
import types
import functools
from dataclasses import dataclass

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ── Result type ──────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    wer: float
    rtfx: float
    inference_s: float
    audio_s: float
    samples: int
    notes: str = ""


# ── Text normalization (same as eval_wer.py) ─────────────────────────

def normalize_text(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Load test data ───────────────────────────────────────────────────

def load_test_samples(n_samples: int = 200):
    """Load first N samples from LibriSpeech test-clean."""
    from datasets import load_dataset
    print(f"Loading {n_samples} samples from LibriSpeech test-clean...")
    ds = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        "librispeech",
        split="test.clean",
    )
    samples = []
    for i, sample in enumerate(ds):
        if i >= n_samples:
            break
        ref = sample.get("text", "").strip()
        if not ref:
            continue
        audio_data = sample["audio"]
        audio = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]
        if sr != 16000:
            from scipy.signal import resample
            n = int(len(audio) * 16000 / sr)
            audio = resample(audio, n).astype(np.float32)
        samples.append((audio, ref))
    print(f"Loaded {len(samples)} samples")
    return samples


# ── Evaluation runner ────────────────────────────────────────────────

def evaluate(model, samples, label="test") -> TestResult:
    """Run transcription on samples and return WER + RTFx."""
    from jiwer import wer as compute_wer

    refs, hyps = [], []
    total_audio_s = 0.0
    total_infer_s = 0.0

    for i, (audio, ref) in enumerate(samples):
        audio_s = len(audio) / 16000
        total_audio_s += audio_s

        t0 = time.time()
        result = model.transcribe(audio)
        elapsed = time.time() - t0
        total_infer_s += elapsed

        refs.append(normalize_text(ref))
        hyps.append(normalize_text(result.text))

        if (i + 1) % 100 == 0:
            partial_wer = compute_wer(refs, hyps) * 100
            rtfx = total_audio_s / total_infer_s if total_infer_s > 0 else 0
            print(f"  [{i+1}/{len(samples)}] WER: {partial_wer:.2f}% | RTFx: {rtfx:.1f}x")

    final_wer = compute_wer(refs, hyps) * 100
    rtfx = total_audio_s / total_infer_s if total_infer_s > 0 else 0

    return TestResult(
        name=label,
        wer=round(final_wer, 2),
        rtfx=round(rtfx, 1),
        inference_s=round(total_infer_s, 2),
        audio_s=round(total_audio_s, 1),
        samples=len(samples),
    )


# ══════════════════════════════════════════════════════════════════════
# OPTIMIZATION PATCHES
# Each returns a cleanup function to revert the change.
# ══════════════════════════════════════════════════════════════════════

def apply_reflect_padding(model):
    """Switch STFT padding from 'constant' to 'reflect'."""
    import nemotron_asr_mlx.audio as audio_mod

    orig_stft = audio_mod._stft_np

    def _stft_reflect(x, n_fft, hop_length, win_length, window):
        pad_len = n_fft // 2
        # Reflect padding instead of zero padding
        x = np.pad(x, (pad_len, pad_len), mode="reflect")
        if win_length < n_fft:
            window = np.pad(window, (0, n_fft - win_length))
        elif win_length > n_fft:
            window = window[:n_fft]
        n_frames = 1 + (len(x) - n_fft) // hop_length
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(n_frames, n_fft),
            strides=(x.strides[0] * hop_length, x.strides[0]),
        )
        windowed = frames * window
        return np.fft.rfft(windowed, n=n_fft)

    audio_mod._stft_np = _stft_reflect
    return lambda: setattr(audio_mod, '_stft_np', orig_stft)


def apply_periodic_hann(model):
    """Use periodic Hann window (torch-style) instead of symmetric (numpy-style)."""
    import nemotron_asr_mlx.audio as audio_mod

    orig_window = audio_mod._window

    @functools.lru_cache(maxsize=4)
    def _periodic_window(name, length):
        if name in ("hann", "hanning"):
            # Periodic: compute for length+1, drop last sample (matches torch.hann_window)
            return np.hanning(length + 1)[:-1].astype(np.float32)
        # Fall back to standard windows
        funcs = {"hamming": np.hamming, "blackman": np.blackman, "bartlett": np.bartlett}
        fn = funcs.get(name)
        if fn is None:
            raise ValueError(f"Unknown window type: {name}")
        return fn(length).astype(np.float32)

    audio_mod._window = _periodic_window
    return lambda: setattr(audio_mod, '_window', orig_window)


def apply_bfloat16(model):
    """Cast all model weights to bfloat16."""
    from mlx.utils import tree_flatten, tree_unflatten

    # Save original weights for revert
    orig_weights = [(k, mx.array(v)) for k, v in tree_flatten(model.parameters())]

    new_weights = [(k, v.astype(mx.bfloat16)) for k, v in orig_weights]
    model.update(tree_unflatten(new_weights))
    mx.eval(model.parameters())

    def revert():
        model.update(tree_unflatten(orig_weights))
        mx.eval(model.parameters())

    return revert


def apply_float16(model):
    """Cast all model weights to float16."""
    from mlx.utils import tree_flatten, tree_unflatten

    orig_weights = [(k, mx.array(v)) for k, v in tree_flatten(model.parameters())]

    new_weights = [(k, v.astype(mx.float16)) for k, v in orig_weights]
    model.update(tree_unflatten(new_weights))
    mx.eval(model.parameters())

    def revert():
        model.update(tree_unflatten(orig_weights))
        mx.eval(model.parameters())

    return revert


def apply_compile_joint(model):
    """Wrap joint network __call__ with mx.compile."""
    orig_call = model.joint_net.__call__

    compiled = mx.compile(orig_call)
    model.joint_net.__call__ = compiled

    return lambda: setattr(model.joint_net, '__call__', orig_call)


def apply_compile_encoder(model):
    """Wrap encoder __call__ with mx.compile."""
    orig_call = model.encoder.__call__

    compiled = mx.compile(orig_call)
    model.encoder.__call__ = compiled

    return lambda: setattr(model.encoder, '__call__', orig_call)


def apply_remove_eval(model):
    """Remove explicit mx.eval(logits) from greedy decode loop.

    The int(mx.argmax(...)) already forces evaluation implicitly,
    so the explicit eval is redundant work.
    """
    import nemotron_asr_mlx.decoder as dec_mod

    orig_decode = dec_mod.greedy_decode

    def greedy_decode_no_eval(
        encoder_output, predict_net, joint_net,
        hidden=None, last_token=dec_mod.BLANK_ID, max_symbols=10,
    ):
        T = encoder_output.shape[1]
        if T == 0:
            return [], hidden, last_token
        tokens = []
        for t in range(T):
            frame = encoder_output[:, t:t+1, :]
            symbols_emitted = 0
            while symbols_emitted < max_symbols:
                token_input = mx.array([[last_token]])
                pred_out, new_h = predict_net(token_input, hidden)
                logits = joint_net(frame, pred_out)
                # No explicit mx.eval — let int() trigger it
                pred_token = int(mx.argmax(logits[0, 0, 0]))
                if pred_token == dec_mod.BLANK_ID:
                    break
                tokens.append(pred_token)
                hidden = new_h
                last_token = pred_token
                symbols_emitted += 1
        return tokens, hidden, last_token

    # Patch both the module and the import in model.py
    dec_mod.greedy_decode = greedy_decode_no_eval
    import nemotron_asr_mlx.model as model_mod
    model_mod.greedy_decode = greedy_decode_no_eval

    def revert():
        dec_mod.greedy_decode = orig_decode
        model_mod.greedy_decode = orig_decode

    return revert


def apply_quantize_int8(model):
    """Quantize linear layers to INT8 (8-bit, group_size=64)."""
    from mlx.utils import tree_flatten

    # Save original state
    orig_weights = [(k, mx.array(v)) for k, v in tree_flatten(model.parameters())]

    nn.quantize(model, bits=8, group_size=64)
    mx.eval(model.parameters())

    def revert():
        from mlx.utils import tree_unflatten
        # Reload model fresh — quantization changes module structure
        # so we need to rebuild
        pass  # Will be handled by reloading

    return revert


def apply_quantize_int4(model):
    """Quantize linear layers to INT4 (4-bit, group_size=64)."""
    from mlx.utils import tree_flatten

    orig_weights = [(k, mx.array(v)) for k, v in tree_flatten(model.parameters())]

    nn.quantize(model, bits=4, group_size=64)
    mx.eval(model.parameters())

    def revert():
        pass  # Will be handled by reloading

    return revert


# ── Registry ─────────────────────────────────────────────────────────

OPTIMIZATIONS = {
    # WER-affecting
    "reflect":        ("STFT reflect padding (vs constant)", apply_reflect_padding, False),
    "periodic_hann":  ("Periodic Hann window (torch-style)", apply_periodic_hann, False),
    "bf16":           ("bfloat16 weights", apply_bfloat16, False),
    "fp16":           ("float16 weights", apply_float16, False),

    # Speed-affecting
    "compile_joint":  ("mx.compile on joint network", apply_compile_joint, False),
    "compile_enc":    ("mx.compile on encoder", apply_compile_encoder, False),
    "no_eval":        ("Remove explicit mx.eval in decode", apply_remove_eval, False),

    # Quantization (destructive — need model reload)
    "int8":           ("INT8 quantization (8-bit, g=64)", apply_quantize_int8, True),
    "int4":           ("INT4 quantization (4-bit, g=64)", apply_quantize_int4, True),
}

# Combination tests
COMBOS = {
    "reflect+periodic":        ["reflect", "periodic_hann"],
    "bf16+compile":            ["bf16", "compile_joint", "compile_enc"],
    "bf16+compile+no_eval":    ["bf16", "compile_joint", "compile_enc", "no_eval"],
    "reflect+bf16+compile":    ["reflect", "bf16", "compile_joint", "compile_enc", "no_eval"],
    "int8+compile":            ["int8", "compile_joint", "compile_enc"],
    "bf16+no_eval":            ["bf16", "no_eval"],
}


# ── Load model ───────────────────────────────────────────────────────

def load_model():
    """Load fresh model from local or HF."""
    import nemotron_asr_mlx as nm
    local_path = os.path.expanduser("~/Models/nemotron-asr-mlx")
    if os.path.isdir(local_path):
        return nm.from_pretrained(local_path)
    return nm.from_pretrained("dboris/nemotron-asr-mlx")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if "--list" in sys.argv:
        print("Available individual tests:")
        for key, (desc, _, destructive) in OPTIMIZATIONS.items():
            flag = " [DESTRUCTIVE]" if destructive else ""
            print(f"  {key:20s} {desc}{flag}")
        print("\nAvailable combination tests:")
        for key, opts in COMBOS.items():
            print(f"  {key:30s} = {' + '.join(opts)}")
        return

    n_samples = 200
    specific_test = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--samples" and i + 2 < len(sys.argv):
            n_samples = int(sys.argv[i + 2])
        if arg == "--test" and i + 2 < len(sys.argv):
            specific_test = sys.argv[i + 2]

    print("=" * 70)
    print("NEMOTRON ASR MLX — OPTIMIZATION A/B TEST FRAMEWORK")
    print("=" * 70)
    print(f"Samples: {n_samples} (from LibriSpeech test-clean)")
    print()

    # Load test data
    samples = load_test_samples(n_samples)

    # Warmup + baseline
    print("\nLoading model...")
    model = load_model()
    print("Warming up...")
    model.transcribe(np.zeros(16000, dtype=np.float32))

    results = []

    # ── Baseline ──
    print("\n" + "─" * 70)
    print("BASELINE (current code, float32, greedy)")
    print("─" * 70)
    baseline = evaluate(model, samples, "baseline")
    results.append(baseline)
    print(f"  WER: {baseline.wer}% | RTFx: {baseline.rtfx}x | Time: {baseline.inference_s}s")

    # Determine which tests to run
    if specific_test:
        if specific_test in OPTIMIZATIONS:
            tests_to_run = {specific_test: OPTIMIZATIONS[specific_test]}
            combos_to_run = {}
        elif specific_test in COMBOS:
            tests_to_run = {}
            combos_to_run = {specific_test: COMBOS[specific_test]}
        else:
            print(f"Unknown test: {specific_test}")
            sys.exit(1)
    else:
        tests_to_run = OPTIMIZATIONS
        combos_to_run = COMBOS

    # ── Individual tests ──
    for key, (desc, apply_fn, destructive) in tests_to_run.items():
        print(f"\n{'─' * 70}")
        print(f"TEST: {key} — {desc}")
        print("─" * 70)

        if destructive:
            # Reload model fresh for destructive tests
            print("  (Reloading model for destructive test...)")
            model = load_model()
            model.transcribe(np.zeros(16000, dtype=np.float32))

        try:
            cleanup = apply_fn(model)
            # Warmup with optimization applied
            model.transcribe(np.zeros(16000, dtype=np.float32))
            result = evaluate(model, samples, key)
            results.append(result)

            wer_delta = result.wer - baseline.wer
            speed_delta = result.rtfx - baseline.rtfx
            print(f"  WER: {result.wer}% (Δ {wer_delta:+.2f}%) | "
                  f"RTFx: {result.rtfx}x (Δ {speed_delta:+.1f}x) | "
                  f"Time: {result.inference_s}s")

            if cleanup and not destructive:
                cleanup()
            elif destructive:
                # Reload for next test
                model = load_model()
                model.transcribe(np.zeros(16000, dtype=np.float32))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(TestResult(key, -1, -1, 0, 0, 0, notes=str(e)))
            if destructive:
                model = load_model()
                model.transcribe(np.zeros(16000, dtype=np.float32))

    # ── Combination tests ──
    for combo_name, opt_keys in combos_to_run.items():
        print(f"\n{'─' * 70}")
        print(f"COMBO: {combo_name} — {' + '.join(opt_keys)}")
        print("─" * 70)

        has_destructive = any(OPTIMIZATIONS[k][2] for k in opt_keys if k in OPTIMIZATIONS)
        if has_destructive:
            print("  (Reloading model for destructive combo...)")
            model = load_model()
            model.transcribe(np.zeros(16000, dtype=np.float32))

        cleanups = []
        try:
            for k in opt_keys:
                if k in OPTIMIZATIONS:
                    _, apply_fn, _ = OPTIMIZATIONS[k]
                    c = apply_fn(model)
                    cleanups.append((c, k))

            model.transcribe(np.zeros(16000, dtype=np.float32))
            result = evaluate(model, samples, combo_name)
            results.append(result)

            wer_delta = result.wer - baseline.wer
            speed_delta = result.rtfx - baseline.rtfx
            print(f"  WER: {result.wer}% (Δ {wer_delta:+.2f}%) | "
                  f"RTFx: {result.rtfx}x (Δ {speed_delta:+.1f}x) | "
                  f"Time: {result.inference_s}s")

            # Revert non-destructive
            for cleanup, k in reversed(cleanups):
                if cleanup and not OPTIMIZATIONS.get(k, (None, None, False))[2]:
                    cleanup()

            if has_destructive:
                model = load_model()
                model.transcribe(np.zeros(16000, dtype=np.float32))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(TestResult(combo_name, -1, -1, 0, 0, 0, notes=str(e)))
            model = load_model()
            model.transcribe(np.zeros(16000, dtype=np.float32))

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Config':<30} {'WER':>8} {'Δ WER':>8} {'RTFx':>8} {'Δ RTFx':>8} {'Time':>8}")
    print(f"{'─' * 30} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for r in results:
        if r.wer < 0:
            print(f"{r.name:<30} {'FAILED':>8}")
            continue
        wer_d = r.wer - baseline.wer if r.name != "baseline" else 0
        rtfx_d = r.rtfx - baseline.rtfx if r.name != "baseline" else 0
        marker = " ★" if wer_d < -0.05 or rtfx_d > 5 else ""
        print(f"{r.name:<30} {r.wer:>7.2f}% {wer_d:>+7.2f}% {r.rtfx:>7.1f}x {rtfx_d:>+7.1f}x {r.inference_s:>7.1f}s{marker}")

    # Best WER
    valid = [r for r in results if r.wer >= 0]
    if valid:
        best_wer = min(valid, key=lambda r: r.wer)
        best_speed = max(valid, key=lambda r: r.rtfx)
        print(f"\n  Best WER:   {best_wer.name} ({best_wer.wer}%)")
        print(f"  Best Speed: {best_speed.name} ({best_speed.rtfx}x)")

    # Save results
    output = {
        "machine": "Apple M4 Max, 16-core, 64GB",
        "samples": n_samples,
        "dataset": "librispeech test-clean (subset)",
        "results": [
            {
                "name": r.name,
                "wer": r.wer,
                "rtfx": r.rtfx,
                "inference_s": r.inference_s,
                "audio_s": r.audio_s,
                "notes": r.notes,
            }
            for r in results
        ],
    }
    with open("optimization_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to optimization_results.json")


if __name__ == "__main__":
    main()
