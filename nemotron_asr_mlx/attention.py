"""Relative-positional multi-head attention with cache-aware streaming support.

Implements the two attention components needed by the FastConformer encoder:

1. RelPositionalEncoding — sinusoidal positional encoding that produces
   relative position embeddings spanning [-(T-1), +(T-1)].
2. MultiHeadAttention — multi-head self-attention with relative positional
   bias (content + position decomposition) and fixed-size KV cache support
   for cache-aware streaming inference.

Reference: parakeet-mlx attention.py (Apache 2.0), adapted for NeMo-style
cache_last_channel I/O.
"""

import math

import mlx.core as mx
import mlx.nn as nn


# ──────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────


class RelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding.

    Produces position embeddings of shape [1, 2*(T+offset)-1, d_model] that
    encode relative distances from -(T+offset-1) to +(T+offset-1).

    Parameters
    ----------
    d_model : int
        Model dimension (must be even).
    max_len : int
        Initial maximum sequence length. Automatically grows if exceeded.
    scale_input : bool
        If True, scale input by sqrt(d_model) (xscaling).
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        scale_input: bool = True,
    ):
        assert d_model % 2 == 0 and max_len > 0
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.scale = math.sqrt(self.d_model) if scale_input else 1.0
        self._build_pe()

    def _build_pe(self):
        """Pre-compute the sinusoidal PE table for positions [max_len-1 .. -(max_len-1)]."""
        positions = mx.arange(
            self.max_len - 1, -self.max_len, -1, dtype=mx.int32
        )
        positions = mx.expand_dims(positions, axis=1).astype(mx.float32)

        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = mx.zeros((2 * self.max_len - 1, self.d_model), dtype=mx.float32)
        pe[:, 0::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)

        self._pe = mx.expand_dims(pe, axis=0)  # [1, 2*max_len-1, d_model]
        mx.eval(self._pe)

    def __call__(self, x: mx.array, offset: int = 0) -> tuple[mx.array, mx.array]:
        """Return (scaled_x, pos_emb).

        Parameters
        ----------
        x : mx.array [batch, time, d_model]
        offset : int
            Cumulative frame offset for streaming (grows each chunk).

        Returns
        -------
        scaled_x : mx.array [batch, time, d_model]
        pos_emb  : mx.array [1, 2*(time+offset)-1, d_model]
        """
        input_len = x.shape[1] + offset

        if input_len > self.max_len:
            self.max_len = input_len + 1
            self._build_pe()

        x = x * self.scale

        buf_len = self._pe.shape[1]
        start = buf_len // 2 - (input_len - 1)
        end = buf_len // 2 + (input_len - 1) + 1
        pos_emb = self._pe[:, start:end].astype(x.dtype)

        return x, pos_emb


# ──────────────────────────────────────────────────────────────────────
# Multi-Head Attention with relative positional bias + KV cache
# ──────────────────────────────────────────────────────────────────────


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with relative positional bias and KV cache.

    Decomposes attention scores into content-content (AC) and content-position
    (BD) terms following Dai et al. (Transformer-XL):

        A = (q + pos_bias_u) @ k^T           (content)
        B = (q + pos_bias_v) @ pos_emb^T      (position, rel-shifted)
        score = (A + B) / sqrt(head_dim)

    For cache-aware streaming, cached K and V from previous chunks are
    prepended to the current K and V before attention computation.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    use_bias : bool
        Whether linear projections use bias.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 8,
        use_bias: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.linear_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_out = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)

        # Learnable biases for content vs position (Transformer-XL style)
        self.pos_bias_u = mx.zeros((n_heads, self.head_dim))
        self.pos_bias_v = mx.zeros((n_heads, self.head_dim))

    @staticmethod
    def _rel_shift(x: mx.array) -> mx.array:
        """Perform relative shift on position scores.

        Input : [B, H, Tq, pos_len]
        Output: [B, H, Tq, pos_len]

        Shifts each row so that position index 0 aligns with key index 0.
        """
        B, H, Tq, pos_len = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])
        x = x.reshape(B, H, pos_len + 1, Tq)
        x = x[:, :, 1:, :]
        x = x.reshape(B, H, Tq, pos_len)
        return x

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass with optional KV cache for streaming.

        Parameters
        ----------
        query : [batch, T_q, d_model]
        key   : [batch, T_k, d_model]
        value : [batch, T_k, d_model]
        pos_emb : [1, pos_len, d_model] — relative positional embeddings.
        mask : optional attention mask.
        cache : tuple(K, V) — cached PROJECTED KV activations.
                K, V each [valid_len, d_model].
                None on first chunk or in batch mode.

        Returns
        -------
        output    : [batch, T_q, d_model]
        new_cache : tuple(K, V) — updated projected KV.
        """
        batch = query.shape[0]
        T_q = query.shape[1]

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        if cache is not None:
            prev_k, prev_v = cache
            # prev_k: [valid_len, d_model]
            # k: [batch, T_q, d_model]
            if batch > 1:
                # We assume single-stream or same valid_len for batch
                pk = mx.broadcast_to(mx.expand_dims(prev_k, 0), (batch,) + prev_k.shape)
                pv = mx.broadcast_to(mx.expand_dims(prev_v, 0), (batch,) + prev_v.shape)
                k = mx.concatenate([pk, k], axis=1)
                v = mx.concatenate([pv, v], axis=1)
            else:
                k = mx.concatenate([mx.expand_dims(prev_k, 0), k], axis=1)
                v = mx.concatenate([mx.expand_dims(prev_v, 0), v], axis=1)

        T_k = k.shape[1]
        
        # Capture full projected KV to return
        new_cache = (k[0], v[0]) if batch == 1 else (k[0], v[0]) # simplified for single-stream

        # Reshape to [B, H, T, head_dim]
        q = q.reshape(batch, T_q, self.n_heads, self.head_dim)
        k_heads = k.reshape(batch, T_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v_heads = v.reshape(batch, T_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if pos_emb is not None:
            # Relative positional attention (Transformer-XL decomposition)
            p = self.linear_pos(pos_emb)  # [1, pos_len, d_model]
            p_batch = p.shape[0]
            pos_len = p.shape[1]
            if p_batch == 1 and batch > 1:
                p = mx.broadcast_to(p, (batch, pos_len, p.shape[-1]))

            p = p.reshape(batch, pos_len, self.n_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )

            # q + bias terms: [B, T_q, H, head_dim]
            q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)  # content
            q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)  # position

            # BD: position scores with relative shift
            matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
            matrix_bd = self._rel_shift(matrix_bd)
            matrix_bd = matrix_bd[:, :, :, :T_k] * self.scale

            if mask is not None:
                expanded_mask = mx.expand_dims(mask, 0)
                matrix_bd = mx.where(expanded_mask, mx.array(-1e9), matrix_bd)

            # AC + BD via scaled_dot_product_attention
            output = mx.fast.scaled_dot_product_attention(
                q_u, k_heads, v_heads, scale=self.scale, mask=matrix_bd
            )
        else:
            # Standard attention without positional bias
            q = q.transpose(0, 2, 1, 3)
            output = mx.fast.scaled_dot_product_attention(
                q, k_heads, v_heads, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(batch, T_q, -1)
        output = self.linear_out(output)

        return output, new_cache
