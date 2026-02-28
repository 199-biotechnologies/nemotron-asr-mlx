"""Cache-aware FastConformer encoder for streaming ASR.

Implements the core encoder from NVIDIA Nemotron Speech Streaming (0.6B):
24 conformer layers with 8x depthwise striding subsampling, causal
convolutions, and relative-positional self-attention.

The streaming path (stream_step) implements the cache-aware algorithm from
arXiv:2312.17279 — each audio frame is processed exactly once with fixed-size
caches for both convolution (K-1=8 activations) and self-attention (L_c=70
frames).

Reference: parakeet-mlx conformer.py (Apache 2.0), NeMo conformer_encoder.py.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from nemotron_asr_mlx.attention import MultiHeadAttention, RelPositionalEncoding
from nemotron_asr_mlx.cache import NemotronCache


# ──────────────────────────────────────────────────────────────────────
# Feed-forward module
# ──────────────────────────────────────────────────────────────────────


class FeedForward(nn.Module):
    """Standard conformer feed-forward: Linear -> SiLU -> Linear.

    Parameters
    ----------
    d_model : int
        Input/output dimension.
    d_ff : int
        Hidden dimension (typically 4 * d_model).
    use_bias : bool
        Whether linear layers use bias.
    """

    def __init__(self, d_model: int, d_ff: int, use_bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))


# ──────────────────────────────────────────────────────────────────────
# Causal convolution module
# ──────────────────────────────────────────────────────────────────────


class CausalConvModule(nn.Module):
    """Conformer convolution module with causal (left-only) padding.

    Architecture: pointwise_conv1 -> GLU -> depthwise_conv (causal) ->
    batch_norm -> SiLU -> pointwise_conv2.

    For streaming: left-pad with K-1=8 cached activations (no right padding).
    For batch: symmetric padding (K-1)/2 on each side.

    Parameters
    ----------
    d_model : int
        Model dimension.
    kernel_size : int
        Depthwise convolution kernel size (default 9).
    use_bias : bool
        Whether convolutions use bias.
    """

    def __init__(
        self,
        d_model: int = 1024,
        kernel_size: int = 9,
        use_bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal_padding = kernel_size - 1  # 8 for K=9

        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 2, kernel_size=1, stride=1, padding=0, bias=use_bias
        )
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=d_model,
            bias=use_bias,
        )
        self.batch_norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=1, stride=1, padding=0, bias=use_bias
        )

    def __call__(
        self,
        x: mx.array,
        conv_cache: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """Forward pass.

        Parameters
        ----------
        x : [batch, time, d_model]
        conv_cache : [d_model, K-1] — cached activations from previous chunk.
                     None for batch mode.

        Returns
        -------
        output     : [batch, time, d_model]
        new_cache  : [d_model, K-1] or None
        """
        # Pointwise expansion + GLU gate
        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=2)
        # x: [batch, time, d_model]

        if conv_cache is not None:
            # Streaming: causal left-pad with cached activations
            # conv_cache: [d_model, K-1] -> [1, K-1, d_model]
            pad = mx.expand_dims(conv_cache.T, 0)  # [1, K-1, d_model]
            if x.shape[0] > 1:
                pad = mx.broadcast_to(pad, (x.shape[0],) + pad.shape[1:])
            x_padded = mx.concatenate([pad, x], axis=1)

            # Save new cache: last K-1 activations from the concatenated
            # sequence (pad + x). This ensures we always get exactly K-1
            # frames even when the current chunk is shorter than K-1.
            combined = x_padded  # [batch, K-1 + T, d_model]
            new_cache = combined[0, -self.causal_padding :, :].T  # [d_model, K-1]
        else:
            # Batch: causal padding (left-only, same as streaming)
            x_padded = mx.pad(x, ((0, 0), (self.causal_padding, 0), (0, 0)))
            new_cache = None

        x = self.depthwise_conv(x_padded)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        return x, new_cache


# ──────────────────────────────────────────────────────────────────────
# Conformer block
# ──────────────────────────────────────────────────────────────────────


class ConformerBlock(nn.Module):
    """Single conformer layer: ff1 -> mha -> conv -> ff2 -> layernorm.

    In streaming mode, takes per-layer attention and convolution caches and
    returns updated caches.

    Parameters
    ----------
    d_model : int
    n_heads : int
    ff_dim : int
        Feed-forward hidden dimension.
    conv_kernel : int
        Convolution kernel size.
    use_bias : bool
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 8,
        ff_dim: int = 4096,
        conv_kernel: int = 9,
        use_bias: bool = True,
    ):
        super().__init__()

        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1 = FeedForward(d_model, ff_dim, use_bias)

        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, use_bias)

        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = CausalConvModule(d_model, conv_kernel, use_bias)

        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2 = FeedForward(d_model, ff_dim, use_bias)

        self.norm_out = nn.LayerNorm(d_model)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        attn_cache: tuple[mx.array, mx.array] | None = None,
        conv_cache: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None, mx.array | None]:
        """Forward pass.

        Parameters
        ----------
        x          : [batch, time, d_model]
        pos_emb    : positional embeddings
        mask       : attention mask
        attn_cache : tuple(K, V) — cached PROJECTED KV activations.
                     K, V each [valid_len, d_model].
        conv_cache : [d_model, K-1] — cached activations for convolution

        Returns
        -------
        output         : [batch, time, d_model]
        new_attn_cache : tuple(K, V) — updated projected KV.
        new_conv_cache : [d_model, K-1] or None
        """
        # FF1 (half-step residual)
        x = x + 0.5 * self.ff1(self.norm_ff1(x))

        # Self-attention with cache
        x_norm = self.norm_attn(x)
        attn_out, new_attn_cache = self.self_attn(
            x_norm, x_norm, x_norm,
            pos_emb=pos_emb,
            mask=mask,
            cache=attn_cache,
        )
        x = x + attn_out

        # Convolution with cache
        conv_out, new_conv_cache = self.conv(self.norm_conv(x), conv_cache=conv_cache)
        x = x + conv_out

        # FF2 (half-step residual)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))

        x = self.norm_out(x)

        return x, new_attn_cache, new_conv_cache


# ──────────────────────────────────────────────────────────────────────
# 8x depthwise striding subsampling
# ──────────────────────────────────────────────────────────────────────


class DwStridingSubsampling(nn.Module):
    """8x depthwise separable conv subsampling (3 stages, each stride 2).

    Input:  [batch, time, feat_in]  (e.g. 128 mel bins)
    Output: [batch, time//8, d_model]

    For streaming: caches trailing audio frames between chunks to avoid
    boundary artifacts at stride boundaries.

    Parameters
    ----------
    feat_in : int
        Input feature dimension (128 mel bins).
    d_model : int
        Output model dimension.
    conv_channels : int
        Intermediate convolution channels.
    """

    def __init__(
        self,
        feat_in: int = 128,
        d_model: int = 1024,
        conv_channels: int = 256,
    ):
        super().__init__()
        self._sampling_num = 3  # log2(8) = 3 stages
        self._stride = 2
        self._kernel_size = 3
        self._conv_channels = conv_channels

        # NeMo causal subsampling: asymmetric padding (left=k-1, right=s-1)
        self._left_padding = self._kernel_size - 1   # 2
        self._right_padding = self._stride - 1        # 1
        all_paddings = self._left_padding + self._right_padding  # 3

        # Compute final frequency dimension after 3 stages of stride-2
        # Formula: floor((input + all_paddings - kernel) / stride) + 1
        final_freq = feat_in
        for _ in range(self._sampling_num):
            final_freq = (
                math.floor(
                    (final_freq + all_paddings - self._kernel_size)
                    / self._stride
                )
                + 1
            )

        # Stage 1: regular Conv2d (no built-in padding — applied manually)
        self.conv = [
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=0,
            ),
            nn.ReLU(),
        ]

        # Stages 2-3: depthwise separable Conv2d
        for _ in range(self._sampling_num - 1):
            self.conv.append(
                nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=0,
                    groups=conv_channels,
                )
            )
            self.conv.append(
                nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.conv.append(nn.ReLU())

        self.out = nn.Linear(conv_channels * final_freq, d_model)

    def _causal_pad(self, x: mx.array) -> mx.array:
        """Apply NeMo causal padding to [B, T, F, C] tensor."""
        # Pad both time and freq with (left=k-1, right=s-1)
        return mx.pad(
            x,
            pad_width=[
                (0, 0),                                          # batch
                (self._left_padding, self._right_padding),       # time
                (self._left_padding, self._right_padding),       # freq
                (0, 0),                                          # channels
            ],
        )

    def _conv_forward(self, x: mx.array) -> mx.array:
        """Run conv layers. x: [B, C, T, F] (channels-first, NeMo layout)."""
        # MLX Conv2d expects [B, H, W, C] (channels-last)
        x = x.transpose(0, 2, 3, 1)  # -> [B, T, F, C]
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d) and layer.weight.shape[1] > 1:
                # Striding conv — apply causal padding
                x = self._causal_pad(x)
            x = layer(x)
        return x.transpose(0, 3, 1, 2)  # -> [B, C, T', F']

    def __call__(
        self,
        x: mx.array,
        lengths: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Batch forward pass.

        Parameters
        ----------
        x       : [batch, time, feat_in]
        lengths : [batch] — valid lengths per sample.

        Returns
        -------
        output  : [batch, time//8, d_model]
        lengths : [batch] — output lengths.
        """
        all_pad = self._left_padding + self._right_padding
        for _ in range(self._sampling_num):
            lengths = (
                mx.floor(
                    (lengths + all_pad - self._kernel_size) / self._stride
                )
                + 1.0
            )
        lengths = lengths.astype(mx.int32)

        # NeMo processes mel as [B, 1, T, F] — unsqueeze channel dim
        x = mx.expand_dims(x, axis=1)     # [B, T, F] -> [B, 1, T, F]
        x = self._conv_forward(x)
        # x: [B, C, T', F'] -> [B, T', C*F'] (NeMo reshape: transpose(1,2) then flatten)
        x = x.transpose(0, 2, 1, 3)       # [B, C, T', F'] -> [B, T', C, F']
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, T', C*F']
        x = self.out(x)

        return x, lengths

    def stream_forward(
        self,
        x: mx.array,
        pre_encode_cache: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Streaming subsampling with cache for trailing samples.

        Parameters
        ----------
        x : [1, time, feat_in] — a single streaming chunk.
        pre_encode_cache : [1, cache_len, feat_in] — trailing samples from
                           previous chunk that couldn't complete a stride.
                           None on first chunk.

        Returns
        -------
        output           : [1, time', d_model]
        new_cache        : [1, remainder, feat_in] — samples to carry over.
        """
        if pre_encode_cache is not None and pre_encode_cache.shape[1] > 0:
            x = mx.concatenate([pre_encode_cache, x], axis=1)

        T = x.shape[1]
        all_pad = self._left_padding + self._right_padding
        # The total stride is 8. We need T to be processable by 3 stride-2
        # stages. Compute how many output frames we'll get:
        out_t = T
        for _ in range(self._sampling_num):
            out_t = (out_t + all_pad - self._kernel_size) // self._stride + 1

        if out_t <= 0:
            # Not enough frames — cache everything
            d_model = self.out.weight.shape[0]
            return mx.zeros((1, 0, d_model)), x

        # How many input frames produce out_t output frames?
        # Reverse the length calculation to find the minimum input needed
        needed = out_t
        for _ in range(self._sampling_num):
            needed = (needed - 1) * self._stride + self._kernel_size - all_pad

        # Cache the remainder
        remainder = T - needed
        if remainder > 0:
            new_cache = x[:, needed:, :]
            x = x[:, :needed, :]
        else:
            new_cache = mx.zeros((1, 0, x.shape[2]))

        # Run through conv layers
        lengths = mx.array([x.shape[1]])
        output, _ = self.__call__(x, lengths)

        return output, new_cache


# ──────────────────────────────────────────────────────────────────────
# FastConformer encoder
# ──────────────────────────────────────────────────────────────────────


class FastConformerEncoder(nn.Module):
    """Cache-aware FastConformer encoder (24 layers).

    Batch mode: standard conformer forward.
    Streaming mode: cache-aware stream_step processes one chunk at a time,
    maintaining fixed-size caches for attention (L_c=70) and convolution (K-1=8).

    Parameters
    ----------
    n_layers : int
    d_model : int
    n_heads : int
    ff_dim : int
    conv_kernel : int
    subsampling_factor : int
    feat_in : int
        Input feature dimension (mel bins).
    subsampling_conv_channels : int
        Channels in subsampling convolutions.
    att_context_size : list[int]
        [left_context, right_context] for attention. left_context=70 (L_c).
    pos_emb_max_len : int
        Maximum positional encoding length.
    use_bias : bool
    """

    def __init__(
        self,
        n_layers: int = 24,
        d_model: int = 1024,
        n_heads: int = 8,
        ff_dim: int = 4096,
        conv_kernel: int = 9,
        subsampling_factor: int = 8,
        feat_in: int = 128,
        subsampling_conv_channels: int = 256,
        att_context_size: Optional[list[int]] = None,
        pos_emb_max_len: int = 5000,
        use_bias: bool = True,
    ):
        super().__init__()
        if att_context_size is None:
            att_context_size = [70, 1]

        self.n_layers = n_layers
        self.d_model = d_model

        # att_context_size can be a flat [left, right] or nested
        # [[left, right], ...] per layer group. Expand to per-layer list.
        if isinstance(att_context_size[0], (list, tuple)):
            # Nested: divide layers evenly among groups
            n_groups = len(att_context_size)
            layers_per_group = n_layers // n_groups
            self._layer_context = []
            for g, (left, right) in enumerate(att_context_size):
                count = layers_per_group if g < n_groups - 1 else n_layers - g * layers_per_group
                self._layer_context.extend([(left, right)] * count)
        else:
            self._layer_context = [tuple(att_context_size)] * n_layers

        self.cache_size = self._layer_context[0][0]  # L_c = left context of first layer

        self.pre_encode = DwStridingSubsampling(
            feat_in=feat_in,
            d_model=d_model,
            conv_channels=subsampling_conv_channels,
        )

        self.pos_enc = RelPositionalEncoding(
            d_model=d_model,
            max_len=pos_emb_max_len,
            scale_input=False,  # NeMo xscaling=False
        )

        self.layers = [
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                conv_kernel=conv_kernel,
                use_bias=use_bias,
            )
            for _ in range(n_layers)
        ]

    def __call__(
        self,
        audio_signal: mx.array,
        length: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Batch forward pass (non-streaming).

        Parameters
        ----------
        audio_signal : [batch, time, feat_in]
        length       : [batch] valid lengths.

        Returns
        -------
        encoded     : [batch, time', d_model]
        encoded_len : [batch]
        """
        x, out_len = self.pre_encode(audio_signal, length)
        x, pos_emb = self.pos_enc(x)

        T = x.shape[1]
        # Build attention masks per layer (True = blocked)
        positions = mx.arange(T)
        for i, layer in enumerate(self.layers):
            left, right = self._layer_context[i]
            # Mask: query at row q, key at col k
            # Allow if k in [q - left, q + right]
            diff = mx.expand_dims(positions, 1) - mx.expand_dims(positions, 0)
            mask = (diff < -right) | (diff > left)  # [T, T] boolean
            x, _, _ = layer(x, pos_emb=pos_emb, mask=mask)

        return x, out_len

    def stream_step(
        self,
        chunk: mx.array,
        cache: NemotronCache,
    ) -> tuple[mx.array, NemotronCache]:
        """Process one streaming chunk with cache-aware inference.

        Implements the algorithm from arXiv:2312.17279:
        1. Subsample with pre-encode cache (trailing audio samples).
        2. For each layer: prepend attention cache, left-pad conv cache.
        3. Update caches (fixed size, drop oldest when full).

        Parameters
        ----------
        chunk : [1, time, feat_in] — raw mel features for this chunk.
        cache : NemotronCache — all caches from previous step.

        Returns
        -------
        encoded   : [1, time', d_model] — encoder output for this chunk.
        new_cache : NemotronCache — updated caches.
        """
        # ── Step 1: Subsampling with pre-encode cache ──
        # For now, we process the chunk directly. The caller (StreamSession)
        # handles audio-level caching. The subsampling handles its own
        # stride boundary caching.
        lengths = mx.array([chunk.shape[1]])
        x, out_len = self.pre_encode(chunk, lengths)

        if x.shape[1] == 0:
            return x, cache

        # ── Step 2: Positional encoding ──
        # Offset = number of frames already in cache (for correct positions)
        offset = cache.cache_last_channel_len
        x, pos_emb = self.pos_enc(x, offset=offset)

        # ── Step 3: Process each layer with caches ──
        # Define the layer-by-layer sequential processing.
        # We must keep this sequential because each layer depends on the output
        # of the previous layer.
        
        if not hasattr(self, "_compiled_step"):
            def _step(x_in, pos_emb_in, attn_cache_in, conv_cache_in, orig_len_in):
                cache_size = attn_cache_in.shape[2]
                n_new = x_in.shape[1]
                
                curr_x = x_in
                new_attns = []
                new_convs = []

                for i, layer in enumerate(self.layers):
                    # Slice per-layer caches (K and V)
                    l_k_cache = attn_cache_in[i, 0, :orig_len_in, :]
                    l_v_cache = attn_cache_in[i, 1, :orig_len_in, :]
                    l_attn_cache = (l_k_cache, l_v_cache)
                    
                    l_conv_cache = conv_cache_in[i]

                    # WE MUST CAPTURE THE INPUT TO THIS LAYER FOR THE CACHE
                    # Wait! MultiHeadAttention projects the input and returns
                    # the full projected KV (including past).
                    # So we just pass the updated KV back to the cache.
                    
                    curr_x, new_l_kv, new_l_conv = layer(
                        curr_x,
                        pos_emb=pos_emb_in,
                        attn_cache=l_attn_cache,
                        conv_cache=l_conv_cache,
                    )

                    # Update per-layer attention cache (K and V)
                    new_k, new_v = new_l_kv # [T_k_full, d_model]
                    
                    # Compute new ring buffer state for K and V
                    def update_single_cache(old_buf, new_full):
                        if orig_len_in + n_new <= cache_size:
                            indices = mx.arange(orig_len_in, orig_len_in + n_new)
                            # new_full contains [prev_k, current_k]
                            current_data = new_full[-n_new:]
                            return old_buf.at[indices].add(current_data - old_buf[indices])
                        else:
                            return new_full[-cache_size:]

                    updated_k = update_single_cache(attn_cache_in[i, 0], new_k)
                    updated_v = update_single_cache(attn_cache_in[i, 1], new_v)
                    
                    new_attns.append(mx.stack([updated_k, updated_v], axis=0))
                    new_convs.append(new_l_conv)

                return curr_x, mx.stack(new_attns, axis=0), mx.stack(new_convs, axis=0)
            
            self._compiled_step = mx.compile(_step)

        x, updated_attn, updated_conv = self._compiled_step(
            x,
            pos_emb,
            cache.cache_last_channel,
            cache.cache_last_time,
            cache.cache_last_channel_len,
        )

        new_len = min(cache.cache_last_channel_len + x.shape[1], cache.cache_last_channel.shape[1])
        new_cache = NemotronCache(
            cache_last_channel=updated_attn,
            cache_last_time=updated_conv,
            cache_last_channel_len=new_len,
            decoder_hidden=cache.decoder_hidden,
            decoder_last_token=cache.decoder_last_token,
        )

        return x, new_cache
