# nemotron-asr-mlx Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port NVIDIA Nemotron Speech Streaming ASR (0.6B) to Apple Silicon MLX with cache-aware streaming inference.

**Architecture:** Clean-room Python package referencing parakeet-mlx (Apache 2.0) for MLX patterns. Session-based streaming API with fixed-size ring buffer caches. FastConformer encoder (24 layers, d_model=1024) + RNNT decoder (2-layer LSTM).

**Tech Stack:** Python 3.10+, MLX >= 0.22.1, huggingface-hub, numpy, sounddevice, sentencepiece, typer

---

## Team Assignment (4 workers, exclusive file ownership)

### Worker: foundation
- `pyproject.toml`
- `nemotron_asr_mlx/__init__.py`
- `nemotron_asr_mlx/cache.py`
- `nemotron_asr_mlx/audio.py`
- `nemotron_asr_mlx/tokenizer.py`

### Worker: encoder
- `nemotron_asr_mlx/attention.py`
- `nemotron_asr_mlx/encoder.py`

### Worker: decoder
- `nemotron_asr_mlx/decoder.py`
- `nemotron_asr_mlx/convert.py`

### Worker: integration
- `nemotron_asr_mlx/model.py`
- `nemotron_asr_mlx/cli.py`
- `README.md`
- `LICENSE`
- `tests/`

---

## Phase 1: Foundation (Workers: foundation, encoder, decoder — PARALLEL)

### Task 1: Project scaffold (foundation)

**Files:**
- Create: `pyproject.toml`
- Create: `nemotron_asr_mlx/__init__.py`

pyproject.toml with:
- name: nemotron-asr-mlx
- version: 0.1.0
- python >= 3.10
- dependencies: mlx>=0.22.1, huggingface-hub, numpy, sounddevice, sentencepiece, typer
- cli entry point: nemotron-asr = nemotron_asr_mlx.cli:app
- author: 199 Biotechnologies

__init__.py with placeholder exports: from_pretrained, StreamEvent, NemotronASR

### Task 2: Cache dataclass (foundation)

**Files:**
- Create: `nemotron_asr_mlx/cache.py`

NemotronCache dataclass holding:
- cache_last_channel: mx.array shape [n_layers, cache_size, d_model] — attention KV cache per layer
- cache_last_time: mx.array shape [n_layers, d_model, conv_context] — conv cache per layer
- cache_last_channel_len: int — valid frames in attention cache
- decoder_hidden: tuple of mx.array — LSTM (h, c) states for 2 layers
- decoder_last_token: int — last emitted token ID

Factory method: NemotronCache.initial(n_layers=24, d_model=1024, cache_size=70, conv_context=8, pred_hidden=640, pred_rnn_layers=2)

Methods: update_attention_cache(), update_conv_cache() — fixed-size ring buffer operations

### Task 3: Audio preprocessing (foundation)

**Files:**
- Create: `nemotron_asr_mlx/audio.py`

Reference: parakeet-mlx audio.py but adapted for Nemotron config:
- 128 mel bins (not 80)
- window_size=0.025s (400 samples at 16kHz)
- window_stride=0.01s (160 samples)
- n_fft=512
- normalize="NA" (no normalization)
- preemphasis=0.97

Functions:
- load_audio(path, sr=16000) -> mx.array — load via ffmpeg subprocess
- get_logmel(audio, config) -> mx.array — STFT + mel filterbank + log
- StreamingMelProcessor class — maintains overlap buffer for chunked processing

### Task 4: Tokenizer (foundation)

**Files:**
- Create: `nemotron_asr_mlx/tokenizer.py`

Simple BPE tokenizer:
- Load vocab list from config (1024 tokens + blank at index 1024)
- decode(token_ids: list[int]) -> str — join tokens, replace ▁ with space, strip
- BLANK_ID = 1024

### Task 5: Attention with cache support (encoder)

**Files:**
- Create: `nemotron_asr_mlx/attention.py`

Reference: parakeet-mlx attention.py (RelPositionMultiHeadAttention)

Classes:
- RelPositionalEncoding — sinusoidal positional encoding
- MultiHeadAttention — standard MHA with cache_last_channel input/output
  - forward(x, pos_emb, cache=None) -> (output, new_cache)
  - When cache provided: prepend cached KV, compute attention over extended sequence, return updated cache (trim to max_cache_size)

### Task 6: FastConformer encoder (encoder) — THE HARD PART

**Files:**
- Create: `nemotron_asr_mlx/encoder.py`

Reference: parakeet-mlx conformer.py + NeMo conformer_encoder.py

Classes:
- DwStridingSubsampling — 8x depthwise separable conv subsampling
  - 3 conv layers: stride 2 each → 8x total
  - Must handle streaming: cache trailing samples between chunks
- CausalConvModule — conformer convolution with left-only padding
  - Kernel size 9, causal padding [8, 0]
  - Accepts conv_cache input, returns updated conv_cache
- ConformerBlock — single conformer layer
  - ff1 → mha (with attn cache) → conv (with conv cache) → ff2 → layernorm
  - forward(x, pos_emb, attn_cache, conv_cache) -> (x, new_attn_cache, new_conv_cache)
- FastConformerEncoder — 24 ConformerBlocks
  - forward(audio_signal, length) -> (encoded, encoded_len) — batch mode
  - stream_step(chunk, cache: NemotronCache) -> (encoded, new_cache) — streaming mode
  - Streaming: subsampling → per-layer process with cache → post-process

### Task 7: RNNT decoder (decoder)

**Files:**
- Create: `nemotron_asr_mlx/decoder.py`

Reference: parakeet-mlx rnnt.py (nearly identical)

Classes:
- PredictNetwork — Embedding + 2-layer LSTM (pred_hidden=640)
  - forward(targets, hidden=None) -> (output, new_hidden)
- JointNetwork — Linear projections + ReLU + output linear
  - forward(encoder_out, decoder_out) -> logits [vocab_size + 1]
- GreedyRNNTDecoder
  - decode(encoder_output, decoder_hidden, last_token) -> (tokens, new_hidden, new_last_token)
  - Implements max_symbols_per_step, blank detection, hypothesis management

### Task 8: Weight conversion (decoder)

**Files:**
- Create: `nemotron_asr_mlx/convert.py`

Reference: parakeet-mlx gist + sherpa-onnx export script

Function: convert_nemo_to_mlx(nemo_path, output_dir)
1. Extract .nemo tarball
2. Load PyTorch checkpoint (model_weights.ckpt)
3. Key transformations:
   - Remove preprocessor.* keys
   - Transpose Conv2d weights (PyTorch OIHW → MLX)
   - Rename LSTM: weight_ih_l{n} → layers.{n}.Wx, weight_hh_l{n} → layers.{n}.Wh
   - Rename biases similarly
4. Extract tokenizer vocab from model config
5. Save: config.json + model.safetensors + vocab.json

CLI entry: convert subcommand

---

## Phase 2: Integration (Worker: integration, AFTER Phase 1)

### Task 9: Model class and streaming session

**Files:**
- Create: `nemotron_asr_mlx/model.py`

Classes:
- StreamEvent(text_delta: str, text: str, is_final: bool, tokens: list[int])
- StreamSession — holds NemotronCache, processes chunks
  - push(pcm_chunk: mx.array) -> StreamEvent
  - flush() -> StreamEvent (is_final=True)
  - reset() -> None
- NemotronASR(nn.Module)
  - __init__(config, encoder, decoder, joint, tokenizer)
  - create_stream(chunk_ms=160) -> StreamSession
  - transcribe(path_or_audio) -> TranscribeResult
  - listen(chunk_ms=160) -> context manager yielding StreamEvents via sounddevice

Function: from_pretrained(model_id_or_path) -> NemotronASR
- Downloads from HuggingFace hub
- Loads config.json, model.safetensors, vocab
- Warm-up call to pre-compile Metal kernels

### Task 10: CLI

**Files:**
- Create: `nemotron_asr_mlx/cli.py`

Typer app with:
- `transcribe <file> [--chunk-ms 560]` — batch or chunked transcription
- `listen [--chunk-ms 160]` — live mic transcription
- `convert <nemo_path> <output_dir>` — weight conversion

### Task 11: Update __init__.py exports

**Files:**
- Modify: `nemotron_asr_mlx/__init__.py`

Export: from_pretrained, NemotronASR, StreamEvent, StreamSession, NemotronCache

---

## Phase 3: Testing & Polish

### Task 12: Basic smoke tests

**Files:**
- Create: `tests/test_audio.py` — mel spectrogram shape/value tests
- Create: `tests/test_cache.py` — cache init, update, ring buffer behavior
- Create: `tests/test_encoder.py` — encoder forward pass shape tests
- Create: `tests/test_decoder.py` — RNNT decoder shape tests

### Task 13: README (nanaban style)

**Files:**
- Create: `README.md` — matching nanaban format with banner, badges, examples
- Create: `LICENSE` — Apache 2.0

### Task 14: End-to-end test with converted weights

Download/convert Nemotron weights, run transcription on a test audio file, verify output text is reasonable English.
