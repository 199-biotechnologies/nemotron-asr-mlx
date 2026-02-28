"""Tests for NemotronCache — shapes, updates, ring buffer overflow."""

import mlx.core as mx

from nemotron_asr_mlx.cache import NemotronCache


def test_initial_shapes():
    cache = NemotronCache.initial()
    assert cache.cache_last_channel.shape == (24, 70, 1024)
    assert cache.cache_last_time.shape == (24, 1024, 8)
    assert cache.cache_last_channel_len == 0
    assert cache.decoder_last_token == 1024
    assert len(cache.decoder_hidden) == 2
    for h, c in cache.decoder_hidden:
        assert h.shape == (1, 640)
        assert c.shape == (1, 640)


def test_initial_custom_shapes():
    cache = NemotronCache.initial(
        n_layers=4, d_model=64, cache_size=10,
        conv_context=4, pred_hidden=32, pred_rnn_layers=1,
    )
    assert cache.cache_last_channel.shape == (4, 10, 64)
    assert cache.cache_last_time.shape == (4, 64, 4)
    assert len(cache.decoder_hidden) == 1


def test_attention_cache_append():
    cache = NemotronCache.initial(n_layers=2, d_model=8, cache_size=5,
                                  conv_context=2, pred_hidden=4, pred_rnn_layers=1)
    new_kv = mx.ones((2, 8))
    updated = cache.update_attention_cache(0, new_kv)
    assert updated.cache_last_channel_len == 2
    # First 2 rows of layer 0 should be ones
    layer0 = updated.cache_last_channel[0]
    mx.eval(layer0)
    assert float(layer0[0, 0]) == 1.0
    assert float(layer0[1, 0]) == 1.0
    # Remaining rows should be zeros
    assert float(layer0[2, 0]) == 0.0


def test_attention_cache_overflow():
    cache = NemotronCache.initial(n_layers=1, d_model=4, cache_size=3,
                                  conv_context=2, pred_hidden=4, pred_rnn_layers=1)
    # Fill to capacity
    data1 = mx.ones((3, 4))
    cache = cache.update_attention_cache(0, data1)
    assert cache.cache_last_channel_len == 3

    # Overflow: add 2 more, should keep last 3
    data2 = mx.ones((2, 4)) * 2.0
    cache2 = NemotronCache(
        cache_last_channel=cache.cache_last_channel,
        cache_last_time=cache.cache_last_time,
        cache_last_channel_len=cache.cache_last_channel_len,
        decoder_hidden=cache.decoder_hidden,
        decoder_last_token=cache.decoder_last_token,
    )
    updated = cache2.update_attention_cache(0, data2)
    assert updated.cache_last_channel_len == 3
    layer0 = updated.cache_last_channel[0]
    mx.eval(layer0)
    # Last 2 should be 2.0, first 1 should be 1.0
    assert float(layer0[0, 0]) == 1.0
    assert float(layer0[1, 0]) == 2.0
    assert float(layer0[2, 0]) == 2.0


def test_conv_cache_update():
    cache = NemotronCache.initial(n_layers=2, d_model=4, cache_size=5,
                                  conv_context=3, pred_hidden=4, pred_rnn_layers=1)
    new_conv = mx.ones((4, 3))
    updated = cache.update_conv_cache(1, new_conv)
    mx.eval(updated.cache_last_time)
    # Layer 1 should be ones
    assert float(updated.cache_last_time[1, 0, 0]) == 1.0
    # Layer 0 should still be zeros
    assert float(updated.cache_last_time[0, 0, 0]) == 0.0


def test_with_decoder_state():
    cache = NemotronCache.initial(n_layers=1, d_model=4, cache_size=3,
                                  conv_context=2, pred_hidden=4, pred_rnn_layers=1)
    new_hidden = ((mx.ones((1, 4)), mx.ones((1, 4))),)
    updated = cache.with_decoder_state(new_hidden, 42)
    assert updated.decoder_last_token == 42
    mx.eval(updated.decoder_hidden[0][0])
    assert float(updated.decoder_hidden[0][0][0, 0]) == 1.0
    # Original fields unchanged
    assert updated.cache_last_channel_len == 0
