# nemotron-asr-mlx

<p align="center">
  <img src="banner.png" alt="nemotron-asr-mlx" width="600">
</p>

<p align="center">
  <strong>NVIDIA Nemotron Speech ASR on Apple Silicon. Cache-aware streaming. One install, one command.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/nemotron-asr-mlx/"><img src="https://img.shields.io/pypi/v/nemotron-asr-mlx.svg" alt="PyPI version"></a>
  <a href="https://github.com/199-biotechnologies/nemotron-asr-mlx/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/nemotron-asr-mlx.svg" alt="license"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/pypi/pyversions/nemotron-asr-mlx.svg" alt="python version"></a>
</p>

An MLX port of [NVIDIA's Nemotron-ASR 0.6B](https://huggingface.co/nvidia/nemotron-asr-speech-streaming-en-0.6b) for Apple Silicon. Each audio frame gets processed exactly once: no recomputation, no sliding windows, no rewinding. The cache-aware conformer encoder holds state in fixed-size ring buffers, so latency stays flat regardless of how long you talk.

```bash
pip install nemotron-asr-mlx
nemotron-asr listen
```

Model downloads on first run. Start talking.

## What it looks like

```python
from nemotron_asr_mlx import from_pretrained

model = from_pretrained("199-biotechnologies/nemotron-asr-mlx")

session = model.create_stream(chunk_ms=160)
event = session.push(pcm_chunk)
print(event.text_delta, end="", flush=True)
# "The quick brown fox "
event = session.push(next_chunk)
print(event.text_delta, end="", flush=True)
# "jumps over "
final = session.flush()
print(final.text)
# "The quick brown fox jumps over the lazy dog"
```

Batch mode works too:

```python
result = model.transcribe("meeting.wav")
print(result.text)
```

## Why nemotron-asr-mlx

Most streaming ASR systems use overlapping sliding windows, reprocessing seconds of audio on every step. Nemotron's cache-aware conformer doesn't. It processes each frame once and carries forward just enough state in ring buffers. That makes it fundamentally different from whisper-style or attention-recompute approaches.

- **Cache-aware streaming** — each frame processed once. State lives in fixed-size ring buffers. No sliding window, no recomputation.
- **80ms minimum chunk** — the smallest chunk any streaming ASR supports on Mac. 160ms default balances latency and throughput.
- **Runtime-configurable** — switch between 80, 160, 560, 1120ms chunks without reloading the model.
- **2.43% WER** on LibriSpeech test-clean at 160ms chunks. Quality holds steady over long streams.
- **Constant memory** — ring buffers are pre-allocated. No growing KV caches, no memory spikes.
- **Native MLX** — no PyTorch, no ONNX, no bridge layers. Runs directly on Metal.

## Install

```bash
pip install nemotron-asr-mlx
```

Python 3.10+ and an Apple Silicon Mac. That's it.

## Setup

No setup. The model downloads from HuggingFace on first use (~1.2 GB) and caches locally.

To use a local model directory instead:

```python
model = from_pretrained("/path/to/local/model")
```

## Usage

### CLI

```bash
nemotron-asr listen                               # stream from mic
nemotron-asr listen --chunk-ms 80                  # lowest latency
nemotron-asr transcribe meeting.wav                # batch transcribe
nemotron-asr transcribe call.mp3 --chunk-ms 560    # streaming on file
```

### Python API

```python
from nemotron_asr_mlx import from_pretrained

model = from_pretrained("199-biotechnologies/nemotron-asr-mlx")

# Batch
result = model.transcribe("audio.wav")
print(result.text)

# Streaming from mic
with model.listen(chunk_ms=160) as stream:
    for event in stream:
        print(event.text_delta, end="", flush=True)
```

## Streaming API

Session-based. Create a session, push audio chunks, flush when done.

```python
session = model.create_stream(chunk_ms=160)

# Push PCM chunks (float32, mono, 16kHz)
event = session.push(chunk_1)  # StreamEvent(text_delta="Hello ", text="Hello ", ...)
event = session.push(chunk_2)  # StreamEvent(text_delta="world", text="Hello world", ...)

# End of utterance
final = session.flush()        # StreamEvent(is_final=True, text="Hello world")

# Reuse for next utterance
session.reset()
```

`StreamEvent` fields:

| Field | Type | Description |
|-------|------|-------------|
| `text_delta` | `str` | New text since last event |
| `text` | `str` | Full accumulated text |
| `is_final` | `bool` | True only from `flush()` |
| `tokens` | `list[int]` | All accumulated token IDs |

## Architecture

FastConformer encoder (24 layers, 1024-dim) with 8x depthwise striding subsampling. RNNT decoder with 2-layer LSTM prediction network and joint network. Cache-aware streaming uses fixed-size ring buffers for attention context (70 frames) and causal convolution context (8 activations per layer). Greedy decoding with blank suppression.

Based on [Cache-aware Streaming Conformer](https://arxiv.org/abs/2312.17279) and the [NeMo](https://github.com/NVIDIA/NeMo) implementation. Weight conversion from `.nemo` checkpoint format, with MLX-native mel spectrogram computation (no librosa).

## Weight conversion

If you have a `.nemo` checkpoint:

```bash
nemotron-asr convert model.nemo ./output_dir
```

Produces `config.json` and `model.safetensors`. Requires PyTorch and safetensors (not needed for inference).

## Dependencies

Deliberately small:

- `mlx` — Apple's ML framework
- `huggingface-hub` — model download
- `numpy` — mel spectrogram
- `sounddevice` — mic access
- `typer` — CLI

## License

Apache 2.0

## Author

[Boris Djordjevic](https://github.com/199-biotechnologies) / [199 Biotechnologies](https://github.com/199-biotechnologies)
