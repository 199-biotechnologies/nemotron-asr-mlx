#!/usr/bin/env python3
"""Benchmark for nemotron-asr-mlx.

Point this at a directory of audio files (wav, mp3, flac, etc.)
and it will transcribe each one, measuring speed and quality.

Usage:
    python benchmark.py /path/to/audio/files
    python benchmark.py file1.wav file2.mp3
    python benchmark.py  # uses current directory
"""
import os
import sys
import time
import subprocess
import tempfile

import numpy as np
import mlx.core as mx


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}


def get_audio_duration(path: str) -> float:
    """Get duration in seconds using ffprobe."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            stderr=subprocess.DEVNULL, text=True,
        )
        return float(out.strip())
    except Exception:
        return 0.0


def convert_to_wav(path: str) -> str:
    """Convert any audio/video to 16kHz mono WAV, return temp path."""
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp],
        capture_output=True,
    )
    return tmp


def discover_audio_files(paths: list[str]) -> list[str]:
    """Discover audio files from paths (files or directories)."""
    files = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                ext = os.path.splitext(name)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    files.append(os.path.join(p, name))
        elif os.path.isfile(p):
            files.append(p)
    return files


def anonymize_filename(path: str) -> str:
    """Strip extension and truncate for display."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name[:48]


def main():
    import nemotron_asr_mlx as nm
    from nemotron_asr_mlx.audio import load_audio

    # --- Discover test files ---
    args = sys.argv[1:] if len(sys.argv) > 1 else ["."]
    test_files = discover_audio_files(args)

    if not test_files:
        print("No audio files found. Usage: python benchmark.py /path/to/audio/files")
        sys.exit(1)

    print("=" * 80)
    print("NEMOTRON ASR MLX — BENCHMARK")
    print("=" * 80)
    print()

    # --- Model loading ---
    print("[1] Model Loading")
    print("-" * 40)
    t0 = time.time()
    model = nm.from_pretrained("dboris/nemotron-asr-mlx")
    load_time = time.time() - t0
    print(f"  Model load time:    {load_time:.2f}s")

    def count_params(obj):
        if isinstance(obj, mx.array):
            return obj.size
        elif isinstance(obj, dict):
            return sum(count_params(v) for v in obj.values())
        elif isinstance(obj, list):
            return sum(count_params(v) for v in obj)
        return 0
    n_params = count_params(model.parameters())
    print(f"  Parameters:         {n_params / 1e6:.1f}M")
    print()

    # --- Warmup ---
    print("  Warming up...")
    model.transcribe(test_files[0])
    print()

    # --- Helper: transcribe with chunking for long files ---
    MAX_SEGMENT_S = 30

    def transcribe_chunked(model, path):
        audio = load_audio(path)
        duration = len(audio) / 16000
        if duration <= MAX_SEGMENT_S + 5:
            result = model.transcribe(audio)
            return result.text, result.tokens, duration

        sr = 16000
        seg_samples = MAX_SEGMENT_S * sr
        all_tokens = []
        all_text = []
        for start in range(0, len(audio), seg_samples):
            segment = audio[start : start + seg_samples]
            result = model.transcribe(segment)
            all_tokens.extend(result.tokens)
            all_text.append(result.text)
        return " ".join(all_text), all_tokens, duration

    # --- Batch transcription benchmark ---
    print("[2] Batch Transcription")
    print("-" * 40)
    print(f"  {'File':<50} {'Duration':>8} {'Time':>8} {'RTFx':>8} {'Tokens':>7}")
    print(f"  {'─'*50} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")

    results = []
    for path in test_files:
        fname = anonymize_filename(path)
        duration = get_audio_duration(path)

        if not path.lower().endswith(".wav"):
            wav_path = convert_to_wav(path)
        else:
            wav_path = path

        try:
            t0 = time.time()
            text, tokens, dur = transcribe_chunked(model, wav_path)
            elapsed = time.time() - t0
            rtf = duration / elapsed if elapsed > 0 else 0

            results.append({
                "file": fname,
                "duration_s": round(duration, 1),
                "time_s": round(elapsed, 2),
                "rtfx": round(rtf, 1),
                "tokens": len(tokens),
                "text": text,
            })
            print(f"  {fname:<50} {duration:>7.1f}s {elapsed:>7.2f}s {rtf:>7.1f}x {len(tokens):>7}")

            if wav_path != path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as e:
            print(f"  {fname:<50} ERROR: {e}")
            if wav_path != path and os.path.exists(wav_path):
                os.unlink(wav_path)

    # Summary
    if results:
        total_audio = sum(r["duration_s"] for r in results)
        total_time = sum(r["time_s"] for r in results)
        avg_rtf = total_audio / total_time if total_time > 0 else 0
        print(f"\n  Total audio:        {total_audio:.1f}s ({total_audio/60:.1f} min)")
        print(f"  Total inference:    {total_time:.2f}s")
        print(f"  Average RTFx:       {avg_rtf:.1f}x realtime")

    # --- Transcription samples ---
    print()
    print("[3] Transcription Samples")
    print("-" * 40)
    for r in results[:5]:
        print(f"\n  [{r['file']}] ({r['duration_s']}s)")
        text = r["text"]
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"  > {text}")

    # --- Memory usage ---
    print()
    print("[4] Memory")
    print("-" * 40)
    try:
        peak = mx.get_peak_memory() / 1024 / 1024
        active = mx.get_active_memory() / 1024 / 1024
        print(f"  Peak GPU memory:    {peak:.0f} MB")
        print(f"  Active GPU memory:  {active:.0f} MB")
    except Exception:
        print("  (GPU memory stats not available)")

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
