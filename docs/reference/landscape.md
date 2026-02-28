# Nemotron ASR Porting Landscape (Feb 2026)

## Who's Working on What

| Effort | Platform | Status | Who |
|--------|----------|--------|-----|
| **sherpa-onnx** Nemotron | ONNX (CPU, Android, iOS, RPi) | **Working** | csukuangfj (k2-fsa) |
| **onnx-asr** package | ONNX Runtime (CPU/GPU/CoreML) | **Working** | istupakov |
| **parakeet-mlx** | MLX (Apple Silicon) | Working (Parakeet only) | senstella |
| **mlx-audio** STT | MLX (Apple Silicon) | Working (no Nemotron) | Prince Canuma |
| **parakeet.cpp** | C++/Metal | Working (Parakeet only) | Frikallo |
| **FluidAudio** | CoreML/ANE | Working (Parakeet only) | FluidInference |
| **Nemotron on MLX** | MLX (Apple Silicon) | **NOBODY** | — |
| **Nemotron on CoreML** | CoreML (Apple Silicon) | **NOBODY** | — |

## Key People

- **Prince Canuma** (@Prince_Canuma): mlx-audio maintainer. No Nemotron plans. Focused on Qwen3-ASR, Voxtral, mlx-audio-swift.
- **Awni Hannun** (@awnihannun): MLX co-creator at Apple. Left Apple on Feb 27, 2026. Previously said "I'd love to see someone build [NeMo on MLX]!"
- **csukuangfj** (Fangjun Kuang): sherpa-onnx/k2-fsa. Did the ONNX Nemotron port. Key reference for cache-aware export.
- **senstella**: parakeet-mlx author. Apache 2.0 licensed. Best MLX reference for FastConformer.

## The Gap We're Filling

Nobody is building Nemotron Speech Streaming with native MLX GPU acceleration + cache-aware streaming.

sherpa-onnx runs on CPU only (no Metal). FluidAudio uses CoreML/ANE but only for Parakeet.

## License

NVIDIA Open Model License — allows redistribution of converted weights.
Need to verify: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

## Key Reference URLs

### Model & Weights
- HuggingFace model: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- sherpa-onnx conversion: https://huggingface.co/csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-2026-01-14

### Source Code References
- parakeet-mlx: https://github.com/senstella/parakeet-mlx
- NeMo: https://github.com/NVIDIA-NeMo/NeMo
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- mlx-audio: https://github.com/Blaizzy/mlx-audio
- mlx-qwen3-asr: https://github.com/moona3k/mlx-qwen3-asr
- parakeet.cpp: https://github.com/Frikallo/parakeet.cpp
- FluidAudio: https://github.com/FluidInference/FluidAudio

### Papers
- Cache-aware streaming: https://arxiv.org/abs/2312.17279
- FastConformer: https://arxiv.org/abs/2305.05084

### Blogs & Tutorials
- NVIDIA streaming blog: https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents
- NeMo streaming tutorial: https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb
- Daily.co voice agents: https://www.daily.co/blog/building-voice-agents-with-nvidia-open-models/
- Pipecat demo: https://github.com/pipecat-ai/nemotron-january-2026

### Benchmarks
- FluidAudio benchmarks: https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md
- mac-whisper-speedtest: https://github.com/anvanvan/mac-whisper-speedtest
- MLX benchmark paper: https://arxiv.org/abs/2510.18921

### MLX Documentation
- MLX nn layers: https://ml-explore.github.io/mlx/build/html/python/nn/layers.html
- MLX compile: https://ml-explore.github.io/mlx/build/html/usage/compile.html
- MLX LSTM: available in nn.LSTM
- MLX scaled_dot_product_attention: available in mx.fast
