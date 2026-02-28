# Streaming ASR on Apple Silicon — Benchmark Comparison (Feb 2026)

## Comparison Table

| Model | Params | Framework | Hardware | Latency (10s audio) | RTFx | True Streaming? | WER (LS clean) |
|---|---|---|---|---|---|---|---|
| **FluidAudio Parakeet-TDT v3** | 600M | CoreML/ANE | M4 Pro | ~0.19s (batch) | **155.6x** | Yes (320ms chunks, RTFx 12.5) | 2.5% |
| **parakeet.cpp TDT-600M** | 600M | C++/Metal | M3 16GB | 520ms | ~19x | Yes (EOU modes) | N/A |
| **parakeet.cpp RNNT-600M** | 600M | C++/Metal | M3 16GB | 1,468ms | ~7x | Yes | N/A |
| **Qwen3-ASR 0.6B (MLX fp16)** | 600M | MLX | M4 Pro | 830ms | ~12x | Experimental rolling | 2.29% |
| **Qwen3-ASR 0.6B (MLX 4-bit)** | 600M | MLX | M4 Pro | 180ms | **55x** | Experimental rolling | 2.72% |
| **Qwen3-ASR 0.6B (MLX 8-bit)** | 600M | MLX | M4 Pro | 270ms | ~37x | Experimental rolling | 2.33% |
| **Moonshine v2 Medium** | 245M | ONNX/CoreML | M3 | 258ms/chunk | N/A | Yes (sliding window) | 2.08% |
| **Voxtral Realtime 4B** | 4.4B | C/Metal | M3 Max 128GB | 284ms enc + 23.5ms/step | ~2.5x RT | Yes (native) | ~8.5% |
| **Whisper Large v3 Turbo** | 809M | MLX | M4 Pro | 1.02s (batch) | ~10x | No (batch only) | 7.75% |

## Nemotron Speech ASR 0.6B (GPU-only benchmarks, no Apple Silicon port exists)

| Metric | Value |
|---|---|
| Time-to-final transcription | 24ms median (H100 GPU) |
| WER (1.12s chunk, LS clean) | 2.31% |
| WER (160ms chunk, LS clean) | 2.43% |
| WER (80ms chunk, LS clean) | 2.55% |
| Concurrent streams (H100) | 560 at 320ms chunks |

## Projected Nemotron on Apple Silicon (based on parakeet.cpp RNNT-600M)

- RNNT-600M on M3 Metal: ~1,468ms for 10s audio (RTFx ~7x)
- MLX fp16 estimate: ~800-1000ms for 10s (RTFx ~10-12x)
- MLX 4-bit estimate: ~200-400ms for 10s (RTFx ~25-50x)
- **RNNT decoder is the bottleneck** — autoregressive (sequential token generation)

## Verdict: "Fastest Streaming ASR on Mac"?

**INCORRECT as stated.** Corrections:

- **Fastest raw batch**: FluidAudio CoreML Parakeet-TDT (RTFx 155.6x)
- **Fastest MLX**: Qwen3-ASR 0.6B 4-bit (RTFx 55x)
- **Fastest true streaming**: FluidAudio Parakeet EOU at 320ms chunks (RTFx 12.5x)

**What Nemotron DOES uniquely offer:**
1. **Lowest configurable chunk size**: 80ms (no other model goes this low)
2. **Cache-aware architecture**: Zero recomputation — each frame processed exactly once
3. **Best quality at ultra-low latency**: 2.55% WER at 80ms chunks
4. **No other MLX model has native cache-aware streaming**

The value proposition is "best streaming architecture with configurable latency-accuracy tradeoff" — NOT raw speed.

## Sources
- FluidAudio Benchmarks: https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md
- parakeet.cpp: https://github.com/Frikallo/parakeet.cpp
- mlx-qwen3-asr: https://github.com/moona3k/mlx-qwen3-asr
- mac-whisper-speedtest: https://github.com/anvanvan/mac-whisper-speedtest
- Moonshine v2 paper: https://arxiv.org/abs/2602.12241
- Voicci Whisper Apple Silicon: https://www.voicci.com/blog/apple-silicon-whisper-performance.html
