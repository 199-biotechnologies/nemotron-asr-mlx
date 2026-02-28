# nemotron-asr-mlx — Design Document
**Date**: 2026-02-28
**Author**: 199 Biotechnologies

## Goal

Port NVIDIA Nemotron Speech Streaming ASR (0.6B) to Apple Silicon via MLX. Streaming-first: the cache-aware streaming inference is the primary differentiator.

## Value Proposition

Not the fastest ASR on Mac (Qwen3-ASR 4-bit is ~55x RTFx). The unique value is:
1. **Lowest chunk size**: 80ms (nobody else goes this low)
2. **Cache-aware streaming**: Each frame processed exactly once, zero recomputation
3. **Best quality at ultra-low latency**: 2.55% WER at 80ms, 2.43% at 160ms
4. **Runtime-configurable**: Switch 80/160/560/1120ms without retraining
5. **No MLX model has native cache-aware streaming** — this is the gap

## Architecture

Clean-room implementation referencing parakeet-mlx (Apache 2.0). Functional/session-based API.

### Core Components

1. **FastConformer Encoder (24 layers)**
   - Cache-aware with 8x depthwise separable conv subsampling
   - Convolution cache: fixed-size ring buffer, K-1=8 activations per layer
   - Self-attention cache: fixed-size ring buffer up to L_c=70 frames (NOT dynamic — avoids shape churn/recompilation in MLX)
   - Layer norm throughout (matches checkpoint exactly)
   - Causal convolutions: left-pad with K-1 cached values
   - d_model=1024, 128 mel features, normalize="NA"

2. **RNNT Decoder**
   - PredictNetwork: 2-layer LSTM, pred_hidden=640
   - JointNetwork: joint_hidden=640, vocab_size=1025 (1024 BPE + blank)
   - Hidden state persistence between streaming chunks
   - Greedy decoding with max_symbols and blank handling
   - mx.compile boundary includes decoder+joint to minimize dispatch overhead

3. **Streaming Inference (Session-based API)**
   - Stateless model, state passed via NemotronCache dataclass
   - Configurable chunk sizes via att_context_size=[70, M]
   - Warm-up call during from_pretrained() to pre-compile Metal kernels
   - Fixed-size ring buffers for all caches (static shapes for MLX compile)

4. **Audio Preprocessing**
   - MLX-native STFT + mel filterbank (from mlx-audio utils, avoids librosa CPU bottleneck)
   - 128 mel bins, 25ms window, 10ms stride, 16kHz, normalize="NA"
   - Stateful STFT cache for streaming (no full-window recomputation)

5. **Tokenizer**
   - SentencePiece BPE (1024 tokens + blank)
   - Extract from .nemo checkpoint or embed in config
   - Simple decode: index → vocab, replace ▁ with space

6. **Weight Conversion**
   - .nemo (NeMo/PyTorch) → config.json + model.safetensors
   - Tensor reordering (PyTorch channel-first → MLX)
   - LSTM weight renaming (weight_ih_l0 → .Wx, etc.)
   - Conv2d transposition
   - Reference: parakeet-mlx gist + sherpa-onnx export script

### API Design (Incorporating Codex + Gemini Feedback)

```python
from nemotron_asr_mlx import from_pretrained

model = from_pretrained("199-biotechnologies/nemotron-asr-mlx")

# Streaming (primary) — session-based, structured events
session = model.create_stream(chunk_ms=160)
event = session.push(pcm16_chunk)     # StreamEvent(text_delta, text, is_final)
final = session.flush()
session.reset()

# Convenience: mic streaming
with model.listen(chunk_ms=160) as stream:
    for event in stream:
        print(event.text_delta, end="", flush=True)

# Batch (secondary)
result = model.transcribe("audio.wav")
print(result.text)
```

### CLI

```bash
nemotron-asr listen --chunk-ms 160          # stream from mic
nemotron-asr transcribe file.wav            # batch transcribe
nemotron-asr transcribe file.wav --chunk-ms 560  # streaming on file
```

### Project Structure

```
nemotron_asr_mlx/
├── __init__.py          # Public API: from_pretrained, StreamEvent
├── model.py             # NemotronASR class, create_stream, transcribe, listen
├── encoder.py           # Cache-aware FastConformer (24 layers)
├── decoder.py           # RNNT: PredictNetwork + JointNetwork + greedy decode
├── cache.py             # NemotronCache dataclass (conv + attention + decoder state)
├── attention.py         # RelPositionMultiHeadAttention with cache support
├── audio.py             # MLX-native mel spectrogram, streaming STFT
├── tokenizer.py         # SentencePiece BPE decode
├── convert.py           # .nemo → safetensors weight conversion
└── cli.py               # Typer CLI
```

### Dependencies

```
mlx >= 0.22.1
huggingface-hub
numpy
sounddevice           # mic streaming
sentencepiece         # tokenizer
typer                 # CLI
```

### Implementation Phases (Codex's recommendation)

1. **Phase 1: Offline parity** — Batch transcription matching NeMo output on fixed files
2. **Phase 2: Streaming encoder** — Cache-aware FastConformer with fixed-size ring buffers
3. **Phase 3: Streaming decoder** — RNNT incremental decode with hypothesis state
4. **Phase 4: Real-time** — Mic wrapper, CLI, UX polish

### Testing Strategy

- Golden tests: compare output at chunk boundaries (80/160/560/1120ms) vs NeMo reference
- Step-level parity: encoder logits match NeMo (not just final WER)
- Long-stream soak test: 10+ minute streams for memory stability and numerical drift
- Quantization validation: 4-bit and 8-bit paths

### Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LSTM numerical drift (float16) over long streams | Default to float32 for LSTM state; benchmark both |
| MLX compile shape churn | Fixed-size ring buffers; static shapes everywhere |
| CPU preprocessing bottleneck at 80ms chunks | MLX-native mel; avoid librosa |
| 8x subsampling edge artifacts between chunks | Cache trailing audio samples at input level |
| Context warm-up (first chunks have no history) | Handle pre-fill state in cache; test specifically |
| NVIDIA license for weight redistribution | Verify NVIDIA Open Model License allows it |

### Success Criteria

- Streaming transcription from mic with perceived latency matching chunk size
- WER within 1% of NeMo reference on LibriSpeech test-clean
- pip-installable package on PyPI
- Pre-converted weights on HuggingFace
- README matching nanaban style
