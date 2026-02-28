# Reusable Components for Nemotron ASR MLX Port

## Reuse Matrix

| Component | Best Source | Reuse Level | Adaptation Needed |
|-----------|-----------|-------------|-------------------|
| FastConformer encoder | parakeet-mlx `conformer.py` | **95%** | Add causal conv mode, NeMo-style cache I/O |
| RNNT decoder (LSTM) | parakeet-mlx `rnnt.py` | **100%** | None — identical architecture |
| Joint network | parakeet-mlx `rnnt.py` | **100%** | None |
| Attention layers | parakeet-mlx `attention.py` | **90%** | Add cache_last_channel support to MHA |
| Audio preprocessing | parakeet-mlx `audio.py` | **90%** | Change normalize="NA", features=128 |
| Streaming cache | parakeet-mlx `cache.py` | **30%** | Need NeMo-style cache_last_channel/time |
| Streaming inference | NeMo `cache_aware_stream_step` | Reference only | Reimplement in MLX from scratch |
| Weight conversion | sherpa-onnx export script | Reference | Need NeMo → safetensors converter |
| Tokenizer | parakeet-mlx `tokenizer.py` | **100%** | None |
| Model loading | parakeet-mlx `utils.py` | **95%** | Add Nemotron config support |
| Greedy RNNT decode | parakeet-mlx `parakeet.py` | **90%** | Adapt for streaming state |

## Key Source Repos

### 1. senstella/parakeet-mlx (PRIMARY — fork/reference this)
- **URL**: https://github.com/senstella/parakeet-mlx
- **License**: Apache 2.0
- **Key files**:
  - `conformer.py` (~250 lines) — FastConformer with DwStridingSubsampling, ConformerBlock
  - `rnnt.py` (~180 lines) — LSTM, PredictNetwork, JointNetwork
  - `cache.py` (~130 lines) — ConformerCache, RotatingConformerCache
  - `attention.py` (~350 lines) — RelPositionMHA + local attention + custom Metal kernel
  - `audio.py` (~130 lines) — PreprocessArgs, load_audio (ffmpeg), get_logmel (STFT/mel)
  - `parakeet.py` (~1107 lines) — BaseParakeet, ParakeetRNNT, StreamingParakeet
  - `tokenizer.py` (~3 lines) — BPE decode (index → vocab, replace ▁ with space)
  - `utils.py` (~80 lines) — from_config(), from_pretrained()
- **NeMo weight conversion gist**: https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed
- **What's missing**: NeMo-native cache-aware streaming (uses local attention instead)

### 2. NVIDIA/NeMo (REFERENCE — streaming implementation)
- **URL**: https://github.com/NVIDIA-NeMo/NeMo
- **Key files**:
  - `nemo/collections/asr/modules/conformer_encoder.py` — `cache_aware_stream_step()`
  - `nemo/collections/asr/parts/submodules/conformer_modules.py` — ConformerLayer with cache I/O
  - `nemo/collections/asr/parts/submodules/causal_convs.py` — CausalConv1D
  - `nemo/collections/asr/inference/model_wrappers/cache_aware_rnnt_inference_wrapper.py` — full pipeline
  - `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`
  - `tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb`
- **Cache tensor shapes**:
  - `cache_last_channel`: `[batch, n_layers, cache_size, d_model]`
  - `cache_last_time`: `[batch, n_layers, d_model, conv_context_size[0]]`
  - `cache_last_channel_len`: `[batch]`

### 3. sherpa-onnx (REFERENCE — working non-NVIDIA Nemotron port)
- **URL**: https://github.com/k2-fsa/sherpa-onnx
- **Nemotron ONNX model**: https://huggingface.co/csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-2026-01-14
- **Export script**: `scripts/nemo/nemotron-speech-streaming-en-0.6b/export_onnx.py`
- **Key learnings**:
  - Model splits into: encoder.onnx, decoder.onnx, joint.onnx, tokens.txt
  - Streaming config: `chunk_size`, `pre_encode_cache_size` from `encoder.streaming_cfg`
  - Vocabulary in `model.joint.vocabulary` (1024 BPE + blank)
  - `pad_and_drop_preencoded` flag can degrade WER (1.79% → 3.57%)

### 4. Blaizzy/mlx-audio (REFERENCE — MLX-native preprocessing)
- **URL**: https://github.com/Blaizzy/mlx-audio
- **Key**: `mlx_audio.utils` has pure-MLX STFT and mel filterbank (avoids librosa)
- **Parakeet in mlx-audio**: Simplified fork of parakeet-mlx (no caching, no streaming)

### 5. moona3k/mlx-qwen3-asr (REFERENCE — ground-up MLX ASR)
- **URL**: https://github.com/moona3k/mlx-qwen3-asr
- **Key**: Full reimplementation showing encoder-decoder ASR on MLX from scratch
- **Performance**: 4.19x faster than PyTorch, useful MLX patterns

## Nemotron Model Specifics

| Parameter | Value |
|-----------|-------|
| Architecture | EncDecHybridRNNTCTCBPEModel |
| Encoder | Cache-Aware FastConformer, 24 layers, d_model=1024 |
| Attention | att_context_size=[70, M] configurable |
| Subsampling | dw_striding, factor 8 |
| Conv kernel | 9, causal mode [8, 0] |
| Decoder | RNNT: pred_hidden=640, 2 LSTM layers, joint_hidden=640 |
| Vocab | 1024 BPE tokens + 1 blank |
| Mel features | 128 bins (not 80!) |
| Window | 25ms, stride 10ms |
| Normalization | "NA" (none) |
| Total params | 600M |

## Existing Non-NVIDIA Ports

| Port | Platform | Status | Streaming? |
|------|----------|--------|------------|
| sherpa-onnx | ONNX (CPU, Android, iOS, RPi) | **Working** | Yes (cache-aware) |
| onnx-asr | ONNX Runtime (CPU/GPU/CoreML) | **Working** | Partial |
| parakeet-mlx | MLX (Parakeet only, not Nemotron) | **Working** | Offline only |
| **Nemotron on MLX** | MLX (Apple Silicon) | **NOT STARTED** | — |
