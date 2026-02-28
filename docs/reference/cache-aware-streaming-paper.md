# Cache-Aware Streaming: Key Implementation Details
## Source: arXiv:2312.17279 — "Stateful Conformer with Cache-based Inference for Streaming ASR"

## Core Idea
Convert FastConformer's non-autoregressive encoder into autoregressive mode during inference via activation caching. Model trains non-autoregressively (efficient), infers autoregressively (streaming).

## Two Cache Types

### 1. Convolution Cache (Fixed Size)
- **Purpose**: Store trailing activations for causal 1D depthwise convolutions
- **Size per layer**: K-1 activations (K = kernel size = 9, so cache = 8 activations)
- **Total shape**: `[n_layers, batch, d_model, K-1]`
- **Behavior**: Rolling buffer — new activations overwrite oldest each step
- **Implementation**: Left-pad convolutions with cached values from previous chunk

### 2. Self-Attention Cache (Dynamic, bounded)
- **Purpose**: Provide left context for attention computation
- **Max size**: L_c (configurable left context, e.g., 70 frames)
- **Shape**: `[n_layers, batch, 0..L_c, d_model]`
- **Behavior**: Grows from 0 to L_c, then drops oldest values (FIFO)
- **Implementation**: Concatenate cache with current chunk before attention computation

## Streaming Inference Algorithm

```
1. Initialize: conv_cache = zeros, attn_cache = empty
2. For each audio chunk (C frames):
   a. Compute mel spectrogram of chunk
   b. Pre-encode (subsampling): reduce to C/8 frames
   c. Drop overlap frames from subsampling
   d. For each encoder layer:
      - Prepend attn_cache to input → compute attention
      - Left-pad conv_cache → compute causal convolution
      - Update attn_cache (append new, drop oldest if > L_c)
      - Update conv_cache (store last K-1 activations)
   e. Run RNNT decoder on encoder output:
      - Use cached decoder LSTM hidden states
      - Greedy decode: emit tokens or blanks
      - Update LSTM hidden state cache
   f. Return partial transcription
```

## Chunk-Aware Lookahead (Key Innovation)

**Problem**: Regular lookahead of M frames per layer → M*N effective lookahead over N layers → redundant recomputation.

**Solution**: Split input into chunks of size C = M+1. Within each chunk, all tokens can attend to each other. Between chunks, only cached context is used. This keeps effective lookahead constant regardless of depth.

## Latency Configurations

| Mode | att_context_size | Right context (M) | Chunk frames | Duration |
|------|-----------------|-------------------|-------------|----------|
| Ultra-low | [70, 0] | 0 | 1 | 80ms |
| Low | [70, 1] | 1 | 2 | 160ms |
| Balanced | [70, 6] | 6 | 7 | 560ms |
| High accuracy | [70, 13] | 13 | 14 | 1.12s |

All configurable at runtime without retraining.

## Critical Implementation Notes

1. **Causal convolutions**: Left-pad with K-1 zeros (or cached values). No future context.
2. **Layer norm, NOT batch norm**: Batch norm requires global statistics → breaks streaming.
3. **No mel normalization**: NeMo uses normalize="NA" for streaming models.
4. **Hybrid CTC/RNNT loss**: `l_total = 0.3 * l_ctc + l_rnnt` (but we only use RNNT decoder at inference).
5. **Context warm-up**: First few chunks have limited/no left context. Cache must handle this gracefully.
6. **LSTM state persistence**: RNNT decoder LSTM hidden states must carry over between chunks.
