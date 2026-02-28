"""NemotronCache — fixed-size ring-buffer caches for cache-aware streaming.

All cache arrays have static shapes so that mx.compile never sees shape churn.
Attention cache uses a ring-buffer with a write pointer; convolution cache is a
simple FIFO shift register.
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class NemotronCache:
    """Holds all persistent state for streaming inference.

    Shapes (batch dimension omitted — single-stream only):
        cache_last_channel : [n_layers, 2, cache_size, d_model]
            Ring buffer of projected KV activations. [:, 0, ...] is K, [:, 1, ...] is V.
        cache_last_time    : [n_layers, d_model, conv_context]
            FIFO buffer of trailing conv activations per layer.
        cache_last_channel_len : int
            How many valid frames are in the attention cache (0 .. cache_size).
        decoder_hidden : tuple[tuple[mx.array, mx.array], ...]
            LSTM (h, c) pairs for each predictor RNN layer.
            Each h and c has shape [1, pred_hidden].
        decoder_last_token : int
            Last emitted non-blank token id (initialised to blank=1024).
    """

    cache_last_channel: mx.array
    cache_last_time: mx.array
    cache_last_channel_len: int
    decoder_hidden: tuple[tuple[mx.array, mx.array], ...]
    decoder_last_token: int

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def initial(
        n_layers: int = 24,
        d_model: int = 1024,
        cache_size: int = 70,
        conv_context: int = 8,
        pred_hidden: int = 640,
        pred_rnn_layers: int = 2,
    ) -> "NemotronCache":
        """Create a zero-initialised cache for the start of a stream."""
        cache_last_channel = mx.zeros((n_layers, 2, cache_size, d_model))
        cache_last_time = mx.zeros((n_layers, d_model, conv_context))
        decoder_hidden = tuple(
            (mx.zeros((1, pred_hidden)), mx.zeros((1, pred_hidden)))
            for _ in range(pred_rnn_layers)
        )
        return NemotronCache(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=0,
            decoder_hidden=decoder_hidden,
            decoder_last_token=1024,  # blank id
        )

    # ------------------------------------------------------------------
    # Attention cache — fixed-size ring buffer
    # ------------------------------------------------------------------

    def update_attention_cache(
        self, layer_idx: int, new_kv: mx.array
    ) -> "NemotronCache":
        """Append *new_kv* ([T, d_model]) to the attention cache for *layer_idx*.

        Uses a fixed-size ring buffer so the array shape never changes.
        Returns a **new** NemotronCache (functional style for mx.compile).
        """
        cache_size = self.cache_last_channel.shape[1]
        n_new = new_kv.shape[0]

        # Current valid length (capped at cache_size)
        cur_len = self.cache_last_channel_len
        new_len = min(cur_len + n_new, cache_size)

        layer_cache = self.cache_last_channel[layer_idx]  # [cache_size, d_model]

        if cur_len + n_new <= cache_size:
            # Cache not full yet — simple append at cur_len
            layer_cache = _write_at(layer_cache, cur_len, new_kv)
        else:
            # Cache full — shift left, write at end
            keep = cache_size - n_new
            if keep > 0:
                layer_cache = mx.concatenate(
                    [layer_cache[cur_len - keep : cur_len], new_kv], axis=0
                )
            else:
                # n_new >= cache_size: just take the last cache_size frames
                layer_cache = new_kv[-cache_size:]

        new_channel = self.cache_last_channel.at[layer_idx].add(
            layer_cache - self.cache_last_channel[layer_idx]
        )

        return NemotronCache(
            cache_last_channel=new_channel,
            cache_last_time=self.cache_last_time,
            cache_last_channel_len=new_len,
            decoder_hidden=self.decoder_hidden,
            decoder_last_token=self.decoder_last_token,
        )

    def get_attention_cache(self, layer_idx: int) -> mx.array:
        """Return the valid portion of the attention cache for *layer_idx*.

        Returns shape [cache_last_channel_len, d_model].
        """
        length = self.cache_last_channel_len
        return self.cache_last_channel[layer_idx, :length, :]

    # ------------------------------------------------------------------
    # Convolution cache — FIFO shift register
    # ------------------------------------------------------------------

    def update_conv_cache(
        self, layer_idx: int, new_conv: mx.array
    ) -> "NemotronCache":
        """Replace the conv cache for *layer_idx* with *new_conv*.

        *new_conv* must have shape [d_model, conv_context].
        Returns a **new** NemotronCache.
        """
        new_time = self.cache_last_time.at[layer_idx].add(
            new_conv - self.cache_last_time[layer_idx]
        )
        return NemotronCache(
            cache_last_channel=self.cache_last_channel,
            cache_last_time=new_time,
            cache_last_channel_len=self.cache_last_channel_len,
            decoder_hidden=self.decoder_hidden,
            decoder_last_token=self.decoder_last_token,
        )

    def get_conv_cache(self, layer_idx: int) -> mx.array:
        """Return the conv cache for *layer_idx* — shape [d_model, conv_context]."""
        return self.cache_last_time[layer_idx]

    # ------------------------------------------------------------------
    # Decoder state helpers
    # ------------------------------------------------------------------

    def with_decoder_state(
        self,
        hidden: tuple[tuple[mx.array, mx.array], ...],
        last_token: int,
    ) -> "NemotronCache":
        """Return a new cache with updated decoder LSTM state."""
        return NemotronCache(
            cache_last_channel=self.cache_last_channel,
            cache_last_time=self.cache_last_time,
            cache_last_channel_len=self.cache_last_channel_len,
            decoder_hidden=hidden,
            decoder_last_token=last_token,
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _write_at(buf: mx.array, start: int, data: mx.array) -> mx.array:
    """Write *data* into *buf* starting at row *start* (no shape change)."""
    n = data.shape[0]
    indices = mx.arange(start, start + n)
    return buf.at[indices].add(data - buf[indices])
