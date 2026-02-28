#!/usr/bin/env python3
"""Evaluate beam search WER improvement on LibriSpeech test-clean."""
import os, sys, time, re
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1,
                          d[i-1][j-1] + (0 if r[i-1] == h[j-1] else 1))
    return d[len(r)][len(h)], len(r)

def eval_config(model, ds, beam_size, label):
    errs, total = 0, 0
    t0 = time.time()
    for i, s in enumerate(ds):
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        ref = normalize_text(s["text"])
        hyp = normalize_text(model.transcribe(audio, beam_size=beam_size).text)
        e, r = wer(ref, hyp)
        errs += e; total += r
        if (i + 1) % 25 == 0:
            print(f"  [{label}] {i+1}/{len(ds)} — running WER: {errs/total*100:.2f}%")
    elapsed = time.time() - t0
    w = errs / total * 100
    print(f"  {label}: {w:.2f}% WER ({elapsed:.1f}s)")
    return w, elapsed

if __name__ == "__main__":
    from nemotron_asr_mlx.model import from_pretrained
    from datasets import load_dataset

    n = 200
    print(f"Loading model...")
    model = from_pretrained(os.path.expanduser("~/Models/nemotron-asr-mlx"))
    model.transcribe(np.zeros(16000, dtype=np.float32))  # warmup

    print(f"Loading {n} samples from LibriSpeech test-clean...")
    ds = load_dataset("hf-audio/esb-datasets-test-only-sorted", "librispeech", split="test.clean")
    ds = list(ds.select(range(n)))

    results = {}
    for bs, label in [(1, "greedy"), (4, "beam-4"), (8, "beam-8")]:
        print(f"\n--- {label} (beam_size={bs}) ---")
        w, t = eval_config(model, ds, bs, label)
        results[label] = (w, t)

    print(f"\n{'='*50}")
    print(f"RESULTS ({n} samples)")
    print(f"{'='*50}")
    for label, (w, t) in results.items():
        print(f"  {label:>8}: {w:.2f}% WER ({t:.1f}s)")
    gw = results["greedy"][0]
    for label in ["beam-4", "beam-8"]:
        delta = results[label][0] - gw
        print(f"  {label} delta: {delta:+.2f}%")
