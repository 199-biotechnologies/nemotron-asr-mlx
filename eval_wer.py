#!/usr/bin/env python3
"""Official WER benchmark for nemotron-asr-mlx.

Evaluates on the standard Open ASR Leaderboard datasets using the
pre-packaged hf-audio/esb-datasets-test-only-sorted collection.

Usage:
    python eval_wer.py                          # run all datasets
    python eval_wer.py librispeech tedlium      # run specific datasets
    python eval_wer.py --list                   # list available datasets

Machine: Apple M4 Max, 16-core, 64 GB
"""
import os
import sys
import time
import json
import tempfile

import numpy as np
from jiwer import wer as compute_wer

DATASETS = {
    "librispeech-clean": ("librispeech", "test.clean", "LibriSpeech test-clean"),
    "librispeech-other": ("librispeech", "test.other", "LibriSpeech test-other"),
    "tedlium": ("tedlium", "test", "TED-LIUM v3"),
    "voxpopuli": ("voxpopuli", "test", "VoxPopuli"),
    "common_voice": ("common_voice", "test", "Common Voice"),
    "gigaspeech": ("gigaspeech", "test", "GigaSpeech"),
    "spgispeech": ("spgispeech", "test", "SPGISpeech"),
    "ami": ("ami", "test", "AMI"),
    "earnings22": ("earnings22", "test", "Earnings22"),
}


def normalize_text(text: str) -> str:
    """Basic text normalization for WER: lowercase, strip punctuation."""
    import re
    text = text.lower().strip()
    # Remove punctuation except apostrophes in contractions
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def eval_dataset(model, name: str, config: str, split: str, display_name: str):
    """Evaluate WER on a single dataset."""
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"  {display_name}")
    print(f"  Dataset: hf-audio/esb-datasets-test-only-sorted/{config}")
    print(f"  Split: {split}")
    print(f"{'='*60}")

    print("  Loading dataset...")
    ds = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        config,
        split=split,
    )
    print(f"  Loaded {len(ds)} samples")

    references = []
    hypotheses = []
    total_audio_s = 0.0
    total_inference_s = 0.0
    errors = 0

    for i, sample in enumerate(ds):
        ref_text = sample.get("text", "").strip()
        if not ref_text:
            continue

        # Get audio
        audio_data = sample["audio"]
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy.signal import resample
            n_samples = int(len(audio_array) * 16000 / sr)
            audio_array = resample(audio_array, n_samples).astype(np.float32)

        audio_duration = len(audio_array) / 16000
        total_audio_s += audio_duration

        try:
            t0 = time.time()
            result = model.transcribe(audio_array)
            elapsed = time.time() - t0
            total_inference_s += elapsed

            hyp_text = result.text.strip()
        except Exception as e:
            errors += 1
            hyp_text = ""

        references.append(normalize_text(ref_text))
        hypotheses.append(normalize_text(hyp_text))

        if (i + 1) % 200 == 0:
            partial_wer = compute_wer(references, hypotheses) * 100
            rtfx = total_audio_s / total_inference_s if total_inference_s > 0 else 0
            print(f"  [{i+1}/{len(ds)}] WER: {partial_wer:.2f}% | RTFx: {rtfx:.1f}x | Audio: {total_audio_s/60:.1f}min")

    # Final WER
    final_wer = compute_wer(references, hypotheses) * 100
    rtfx = total_audio_s / total_inference_s if total_inference_s > 0 else 0

    print(f"\n  Results for {display_name}:")
    print(f"  {'─'*40}")
    print(f"  WER:              {final_wer:.2f}%")
    print(f"  Samples:          {len(references)}")
    print(f"  Total audio:      {total_audio_s/60:.1f} min ({total_audio_s/3600:.2f}h)")
    print(f"  Total inference:  {total_inference_s:.1f}s")
    print(f"  RTFx:             {rtfx:.1f}x realtime")
    if errors:
        print(f"  Errors:           {errors}")

    return {
        "dataset": display_name,
        "wer": round(final_wer, 2),
        "samples": len(references),
        "audio_hours": round(total_audio_s / 3600, 2),
        "inference_s": round(total_inference_s, 1),
        "rtfx": round(rtfx, 1),
    }


def main():
    if "--list" in sys.argv:
        print("Available datasets:")
        for key, (_, _, name) in DATASETS.items():
            print(f"  {key:20s} {name}")
        return

    # Which datasets to run
    requested = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not requested:
        # Default: the most impactful ones first
        requested = ["librispeech-clean", "librispeech-other", "tedlium",
                      "voxpopuli", "common_voice", "gigaspeech",
                      "spgispeech", "ami", "earnings22"]

    # Resolve dataset keys (allow partial matches)
    to_run = []
    for r in requested:
        matches = [k for k in DATASETS if r in k]
        if matches:
            to_run.extend(matches)
        else:
            print(f"Unknown dataset: {r}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)
    # Deduplicate preserving order
    seen = set()
    to_run = [x for x in to_run if not (x in seen or seen.add(x))]

    print("=" * 60)
    print("NEMOTRON ASR MLX — OFFICIAL WER BENCHMARK")
    print("=" * 60)
    print(f"Machine: Apple M4 Max, 16-core CPU, 64 GB RAM")
    print(f"Datasets: {', '.join(to_run)}")
    print()

    # Load model
    import nemotron_asr_mlx as nm
    print("Loading model...")
    t0 = time.time()
    local_path = os.path.expanduser("~/Models/nemotron-asr-mlx")
    if os.path.isdir(local_path):
        model = nm.from_pretrained(local_path)
    else:
        model = nm.from_pretrained("dboris/nemotron-asr-mlx")
    print(f"Model loaded in {time.time() - t0:.2f}s")

    # Warmup
    print("Warming up...")
    model.transcribe(np.zeros(16000, dtype=np.float32))

    results = []
    for key in to_run:
        config, split, display_name = DATASETS[key]
        r = eval_dataset(model, key, config, split, display_name)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<25} {'WER':>8} {'Audio':>8} {'Infer':>8} {'RTFx':>8}")
    print(f"{'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    total_audio_h = 0
    total_infer_s = 0
    wers = []
    for r in results:
        print(f"{r['dataset']:<25} {r['wer']:>7.2f}% {r['audio_hours']:>7.2f}h {r['inference_s']:>7.1f}s {r['rtfx']:>7.1f}x")
        total_audio_h += r["audio_hours"]
        total_infer_s += r["inference_s"]
        wers.append(r["wer"])

    avg_wer = sum(wers) / len(wers) if wers else 0
    avg_rtfx = (total_audio_h * 3600) / total_infer_s if total_infer_s > 0 else 0
    print(f"{'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    print(f"{'Average':<25} {avg_wer:>7.2f}% {total_audio_h:>7.2f}h {total_infer_s:>7.1f}s {avg_rtfx:>7.1f}x")

    # Save results
    output = {
        "machine": "Apple M4 Max, 16-core, 64GB",
        "model": "dboris/nemotron-asr-mlx",
        "model_version": "0.1.0",
        "results": results,
        "average_wer": round(avg_wer, 2),
    }
    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to eval_results.json")


if __name__ == "__main__":
    main()
