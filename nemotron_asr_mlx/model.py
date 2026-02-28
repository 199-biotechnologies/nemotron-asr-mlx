"""NemotronASR — top-level model class with streaming and batch transcription.

Ties together the FastConformer encoder, RNNT decoder, tokenizer, and cache
into a single high-level API.  Streaming is the primary path; batch
transcription is a convenience wrapper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from nemotron_asr_mlx.audio import MelConfig, get_logmel, load_audio
from nemotron_asr_mlx.cache import NemotronCache
from nemotron_asr_mlx.decoder import (
    BLANK_ID,
    JointNetwork,
    PredictNetwork,
    greedy_decode,
)
from nemotron_asr_mlx.encoder import FastConformerEncoder
from nemotron_asr_mlx.tokenizer import NemotronTokenizer


# ------------------------------------------------------------------
# StreamEvent — structured result from push / flush
# ------------------------------------------------------------------


@dataclass
class StreamEvent:
    """Result from a single streaming step.

    Attributes
    ----------
    text_delta : str
        New text produced by this step (relative to previous event).
    text : str
        Full accumulated text so far.
    is_final : bool
        True only on the event returned by ``flush()``.
    tokens : list[int]
        All non-blank token IDs accumulated so far.
    """

    text_delta: str
    text: str
    is_final: bool
    tokens: list[int]


# ------------------------------------------------------------------
# StreamSession — holds cache state, processes audio chunks
# ------------------------------------------------------------------


class StreamSession:
    """Stateful streaming session wrapping encoder + decoder + cache.

    Created via ``NemotronASR.create_stream()``.  Feed PCM chunks via
    ``push()``, call ``flush()`` when the utterance ends.

    Parameters
    ----------
    model : NemotronASR
        Parent model (encoder, predict_net, joint_net, tokenizer).
    chunk_ms : int
        Chunk duration in milliseconds (80, 160, 560, 1120).
    """

    def __init__(self, model: NemotronASR, chunk_ms: int = 160):
        self._model = model
        self._chunk_ms = chunk_ms
        self._mel_config = MelConfig()

        # Encoder config
        cfg = model._config
        n_layers = cfg.get("encoder", {}).get("n_layers", 24)
        d_model = cfg.get("encoder", {}).get("d_model", 1024)
        att_ctx = cfg.get("encoder", {}).get("att_context_size", [70, 1])
        cache_size = att_ctx[0][0] if isinstance(att_ctx[0], (list, tuple)) else att_ctx[0]
        conv_context = cfg.get("encoder", {}).get("kernel_size", 9) - 1
        pred_hidden = cfg.get("decoder", {}).get("pred_hidden", 640)
        pred_rnn_layers = cfg.get("decoder", {}).get("pred_rnn_layers", 2)

        self._cache = NemotronCache.initial(
            n_layers=n_layers,
            d_model=d_model,
            cache_size=cache_size,
            conv_context=conv_context,
            pred_hidden=pred_hidden,
            pred_rnn_layers=pred_rnn_layers,
        )

        self._all_tokens: list[int] = []
        self._prev_text = ""

        # Streaming mel: accumulate PCM and compute mel incrementally
        # Keep overlap of (n_fft - hop_length) samples for correct STFT
        self._pcm_buffer = np.zeros(0, dtype=np.float32)
        self._mel_frames_processed = 0

    def push(self, pcm_chunk: mx.array) -> StreamEvent:
        """Process one PCM audio chunk and return a StreamEvent.

        Parameters
        ----------
        pcm_chunk : mx.array
            Raw PCM samples (float32, mono, 16 kHz).  Shape ``[N]`` or
            ``[1, N]``.

        Returns
        -------
        StreamEvent
        """
        import numpy as np

        # Ensure 1-D numpy for mel computation
        if pcm_chunk.ndim == 2:
            pcm_chunk = pcm_chunk.squeeze(0)
        pcm_np = np.array(pcm_chunk, dtype=np.float32)

        # Accumulate PCM and compute mel over full buffer for correct STFT
        self._pcm_buffer = np.concatenate([self._pcm_buffer, pcm_np])
        mel_full = get_logmel(self._pcm_buffer, self._mel_config)  # [1, T_total, 128]

        # Extract only new mel frames (not yet processed)
        new_start = self._mel_frames_processed
        if mel_full.shape[1] <= new_start:
            return StreamEvent(
                text_delta="",
                text=self._prev_text,
                is_final=False,
                tokens=list(self._all_tokens),
            )
        mel = mel_full[:, new_start:, :]
        self._mel_frames_processed = mel_full.shape[1]

        # Encode
        encoded, self._cache = self._model.encoder.stream_step(mel, self._cache)

        if encoded.shape[1] == 0:
            return StreamEvent(
                text_delta="",
                text=self._prev_text,
                is_final=False,
                tokens=list(self._all_tokens),
            )

        # Decode
        # Convert cache format (per-layer tuples) to stacked format for greedy_decode
        cache_hidden = self._cache.decoder_hidden
        # cache_hidden: tuple of (h, c) per layer, each [1, pred_hidden]
        # greedy_decode expects: (h_stacked, c_stacked) each [n_layers, 1, H]
        h_stacked = mx.stack([h for h, c in cache_hidden], axis=0)  # [n_layers, 1, H]
        c_stacked = mx.stack([c for h, c in cache_hidden], axis=0)  # [n_layers, 1, H]
        hidden = (h_stacked, c_stacked)
        last_token = self._cache.decoder_last_token

        new_tokens, new_hidden, new_last_token = greedy_decode(
            encoded,
            self._model.predict_net,
            self._model.joint_net,
            hidden=hidden,
            last_token=last_token,
        )

        # Update cache with decoder state
        # Convert stacked format back to per-layer tuples for cache
        if new_hidden is not None:
            h_all, c_all = new_hidden
            n_layers = h_all.shape[0]
            per_layer = tuple(
                (h_all[i], c_all[i]) for i in range(n_layers)
            )
            self._cache = self._cache.with_decoder_state(per_layer, new_last_token)

        # Accumulate tokens and compute text
        self._all_tokens.extend(new_tokens)
        full_text = self._model.tokenizer.decode(self._all_tokens)
        delta = full_text[len(self._prev_text):]
        self._prev_text = full_text

        return StreamEvent(
            text_delta=delta,
            text=full_text,
            is_final=False,
            tokens=list(self._all_tokens),
        )

    def flush(self) -> StreamEvent:
        """Signal end of utterance and return the final StreamEvent."""
        return StreamEvent(
            text_delta="",
            text=self._prev_text,
            is_final=True,
            tokens=list(self._all_tokens),
        )

    def reset(self):
        """Reset session state for a new utterance."""
        cfg = self._model._config
        n_layers = cfg.get("encoder", {}).get("n_layers", 24)
        d_model = cfg.get("encoder", {}).get("d_model", 1024)
        att_ctx = cfg.get("encoder", {}).get("att_context_size", [70, 1])
        cache_size = att_ctx[0][0] if isinstance(att_ctx[0], (list, tuple)) else att_ctx[0]
        conv_context = cfg.get("encoder", {}).get("kernel_size", 9) - 1
        pred_hidden = cfg.get("decoder", {}).get("pred_hidden", 640)
        pred_rnn_layers = cfg.get("decoder", {}).get("pred_rnn_layers", 2)

        self._cache = NemotronCache.initial(
            n_layers=n_layers,
            d_model=d_model,
            cache_size=cache_size,
            conv_context=conv_context,
            pred_hidden=pred_hidden,
            pred_rnn_layers=pred_rnn_layers,
        )
        self._all_tokens = []
        self._prev_text = ""
        self._pcm_buffer = np.zeros(0, dtype=np.float32)
        self._mel_frames_processed = 0


# ------------------------------------------------------------------
# NemotronASR — main model class
# ------------------------------------------------------------------


class NemotronASR(nn.Module):
    """Top-level Nemotron ASR model.

    Wraps the FastConformer encoder, RNNT prediction network, joint network,
    and tokenizer.  Provides ``create_stream()`` for streaming and
    ``transcribe()`` for batch mode.
    """

    def __init__(
        self,
        encoder: FastConformerEncoder,
        predict_net: PredictNetwork,
        joint_net: JointNetwork,
        tokenizer: NemotronTokenizer,
        config: dict,
    ):
        super().__init__()
        self.encoder = encoder
        self.predict_net = predict_net
        self.joint_net = joint_net
        self.tokenizer = tokenizer
        self._config = config

    def create_stream(self, chunk_ms: int = 160) -> StreamSession:
        """Create a new streaming session.

        Parameters
        ----------
        chunk_ms : int
            Chunk duration in milliseconds: 80, 160, 560, or 1120.

        Returns
        -------
        StreamSession
        """
        return StreamSession(self, chunk_ms=chunk_ms)

    def transcribe(self, path_or_audio) -> StreamEvent:
        """Batch-transcribe an audio file or array.

        Parameters
        ----------
        path_or_audio : str | Path | numpy.ndarray | mx.array
            Path to an audio file, or raw PCM samples (float32, 16 kHz).

        Returns
        -------
        StreamEvent
            The final transcription result.
        """
        import numpy as np

        if isinstance(path_or_audio, (str, Path)):
            audio_np = load_audio(str(path_or_audio))
        elif isinstance(path_or_audio, mx.array):
            audio_np = np.array(path_or_audio, dtype=np.float32)
        else:
            audio_np = np.asarray(path_or_audio, dtype=np.float32)

        mel = get_logmel(audio_np)  # [1, T, 128]
        length = mx.array([mel.shape[1]])

        encoded, enc_len = self.encoder(mel, length)

        tokens, _, _ = greedy_decode(
            encoded,
            self.predict_net,
            self.joint_net,
        )

        text = self.tokenizer.decode(tokens)

        return StreamEvent(
            text_delta=text,
            text=text,
            is_final=True,
            tokens=tokens,
        )

    def listen(self, chunk_ms: int = 160) -> _MicStream:
        """Stream from the microphone.

        Usage::

            with model.listen(chunk_ms=160) as stream:
                for event in stream:
                    print(event.text_delta, end="", flush=True)
        """
        return _MicStream(self, chunk_ms=chunk_ms)


# ------------------------------------------------------------------
# Mic streaming context manager
# ------------------------------------------------------------------


class _MicStream:
    """Context manager for live microphone streaming."""

    def __init__(self, model: NemotronASR, chunk_ms: int = 160):
        self._model = model
        self._chunk_ms = chunk_ms
        self._session: StreamSession | None = None
        self._stream = None

    def __enter__(self) -> Iterator[StreamEvent]:
        import queue
        import sounddevice as sd

        self._queue: queue.Queue = queue.Queue()
        sample_rate = 16000
        chunk_samples = int(sample_rate * self._chunk_ms / 1000)
        self._session = self._model.create_stream(self._chunk_ms)

        def callback(indata, frames, time, status):
            self._queue.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=callback,
        )
        self._stream.start()
        return self._iter_events()

    def _iter_events(self) -> Iterator[StreamEvent]:
        while True:
            try:
                pcm = self._queue.get(timeout=0.1)
                chunk = mx.array(pcm)
                event = self._session.push(chunk)
                yield event
            except Exception:
                continue

    def __exit__(self, *args):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._session is not None:
            self._session.flush()


# ------------------------------------------------------------------
# from_pretrained — download + build model
# ------------------------------------------------------------------


def from_pretrained(model_id_or_path: str) -> NemotronASR:
    """Load a pretrained NemotronASR model.

    Parameters
    ----------
    model_id_or_path : str
        Either a HuggingFace model ID (e.g. ``"dboris/nemotron-asr-mlx"``)
        or a local directory containing ``config.json`` and ``model.safetensors``.

    Returns
    -------
    NemotronASR
    """
    model_path = Path(model_id_or_path)

    if not model_path.is_dir():
        from huggingface_hub import snapshot_download

        model_path = Path(
            snapshot_download(
                model_id_or_path,
                allow_patterns=["config.json", "model.safetensors"],
            )
        )

    # Load config
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Build encoder
    enc_cfg = config.get("encoder", {})
    encoder = FastConformerEncoder(
        n_layers=enc_cfg.get("n_layers", 24),
        d_model=enc_cfg.get("d_model", 1024),
        n_heads=enc_cfg.get("n_heads", 8),
        ff_dim=enc_cfg.get("d_model", 1024) * 4,
        conv_kernel=enc_cfg.get("kernel_size", 9),
        subsampling_factor=enc_cfg.get("subsampling_factor", 8),
        feat_in=enc_cfg.get("feat_in", 128),
        att_context_size=enc_cfg.get("att_context_size", [70, 1]),
    )

    # Build predict network
    dec_cfg = config.get("decoder", {})
    predict_net = PredictNetwork(
        vocab_size=dec_cfg.get("vocab_size", 1024),
        embed_dim=dec_cfg.get("pred_hidden", 640),
        num_layers=dec_cfg.get("pred_rnn_layers", 2),
    )

    # Build joint network
    jnt_cfg = config.get("joint", {})
    joint_net = JointNetwork(
        encoder_dim=enc_cfg.get("d_model", 1024),
        decoder_dim=dec_cfg.get("pred_hidden", 640),
        joint_dim=jnt_cfg.get("joint_hidden", 640),
        vocab_size=dec_cfg.get("vocab_size", 1024),
    )

    # Assemble model so nn.Module tree is complete before loading weights
    model = NemotronASR(
        encoder=encoder,
        predict_net=predict_net,
        joint_net=joint_net,
        tokenizer=NemotronTokenizer(vocab=[]),  # placeholder
        config=config,
    )

    # Load weights
    weights_path = model_path / "model.safetensors"
    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Build tokenizer — normalize key name from converter output
    tok_config = dict(config)
    if "vocabulary" in tok_config and "vocab" not in tok_config:
        tok_config["vocab"] = tok_config["vocabulary"]
    tokenizer = NemotronTokenizer.from_config(tok_config)
    model.tokenizer = tokenizer

    return model
