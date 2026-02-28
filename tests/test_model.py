"""Tests for NemotronASR model — instantiation and StreamSession creation."""

import mlx.core as mx
import mlx.nn as nn

from nemotron_asr_mlx.decoder import JointNetwork, PredictNetwork
from nemotron_asr_mlx.encoder import FastConformerEncoder
from nemotron_asr_mlx.model import NemotronASR, StreamEvent, StreamSession
from nemotron_asr_mlx.tokenizer import NemotronTokenizer


def _make_small_model():
    """Build a tiny NemotronASR for testing (no real weights).

    Uses feat_in=128 (real mel bins) to keep subsampling shapes valid,
    but reduces d_model, layers, and hidden dims to keep it fast.
    """
    d_model = 64
    n_layers = 2
    n_heads = 2
    vocab_size = 32
    pred_hidden = 16
    pred_rnn_layers = 1
    joint_hidden = 16
    feat_in = 128
    subsampling_conv_channels = 32

    config = {
        "encoder": {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "kernel_size": 9,
            "subsampling_factor": 8,
            "feat_in": feat_in,
            "att_context_size": [10, 1],
        },
        "decoder": {
            "vocab_size": vocab_size,
            "pred_hidden": pred_hidden,
            "pred_rnn_layers": pred_rnn_layers,
        },
        "joint": {
            "joint_hidden": joint_hidden,
        },
    }

    encoder = FastConformerEncoder(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ff_dim=d_model * 4,
        conv_kernel=9,
        feat_in=feat_in,
        subsampling_conv_channels=subsampling_conv_channels,
        att_context_size=[10, 1],
    )

    predict_net = PredictNetwork(
        vocab_size=vocab_size,
        embed_dim=pred_hidden,
        num_layers=pred_rnn_layers,
    )

    joint_net = JointNetwork(
        encoder_dim=d_model,
        decoder_dim=pred_hidden,
        joint_dim=joint_hidden,
        vocab_size=vocab_size,
    )

    tokenizer = NemotronTokenizer(
        vocab=[f"tok{i}" for i in range(vocab_size)],
        blank_id=vocab_size,
    )

    return NemotronASR(
        encoder=encoder,
        predict_net=predict_net,
        joint_net=joint_net,
        tokenizer=tokenizer,
        config=config,
    )


def test_model_instantiation():
    model = _make_small_model()
    assert isinstance(model, NemotronASR)
    assert isinstance(model.encoder, FastConformerEncoder)
    assert isinstance(model.predict_net, PredictNetwork)
    assert isinstance(model.joint_net, JointNetwork)


def test_create_stream():
    model = _make_small_model()
    session = model.create_stream(chunk_ms=160)
    assert isinstance(session, StreamSession)


def test_stream_session_push():
    model = _make_small_model()
    session = model.create_stream(chunk_ms=160)
    # Push a chunk of audio (2560 samples = 160ms at 16kHz)
    chunk = mx.zeros((2560,))
    event = session.push(chunk)
    assert isinstance(event, StreamEvent)
    assert isinstance(event.text, str)
    assert isinstance(event.text_delta, str)
    assert event.is_final is False
    assert isinstance(event.tokens, list)


def test_stream_session_flush():
    model = _make_small_model()
    session = model.create_stream(chunk_ms=160)
    chunk = mx.zeros((2560,))
    session.push(chunk)
    final = session.flush()
    assert final.is_final is True


def test_stream_session_reset():
    model = _make_small_model()
    session = model.create_stream(chunk_ms=160)
    chunk = mx.zeros((2560,))
    session.push(chunk)
    session.reset()
    # After reset, tokens should be empty
    assert session._all_tokens == []
    assert session._prev_text == ""


def test_stream_event_fields():
    event = StreamEvent(
        text_delta="hello",
        text="hello world",
        is_final=False,
        tokens=[1, 2, 3],
    )
    assert event.text_delta == "hello"
    assert event.text == "hello world"
    assert event.is_final is False
    assert event.tokens == [1, 2, 3]


def test_batch_transcribe_with_array():
    model = _make_small_model()
    # 1 second of silence at 16kHz
    import numpy as np
    audio = np.zeros(16000, dtype=np.float32)
    result = model.transcribe(audio)
    assert isinstance(result, StreamEvent)
    assert result.is_final is True
    assert isinstance(result.text, str)
