"""Audio preprocessing for Nemotron ASR.

128 mel bins, 25 ms window, 10 ms stride, 16 kHz, no normalisation (normalize="NA"),
preemphasis=0.97.  References parakeet-mlx audio.py (Apache 2.0) but adapted for
Nemotron's configuration.
"""

import functools
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import CalledProcessError, run

import mlx.core as mx
import numpy as np


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class MelConfig:
    """Mel spectrogram parameters matching Nemotron's preprocessor."""

    sample_rate: int = 16000
    window_size: float = 0.025       # 25 ms -> 400 samples
    window_stride: float = 0.01      # 10 ms -> 160 samples
    n_fft: int = 512
    features: int = 128              # mel bins
    preemphasis: float = 0.97
    normalize: str = "NA"            # no normalisation
    window: str = "hann"
    dither: float = 0.0
    mag_power: float = 2.0

    # Derived properties
    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)

    # Lazily-built mel filterbank (numpy, converted to mx at call site)
    _filterbank: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def filterbank(self) -> np.ndarray:
        if self._filterbank is None:
            self._filterbank = _mel_filterbank(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.features,
            )
        return self._filterbank


# ------------------------------------------------------------------
# Audio loading (ffmpeg, no librosa dependency)
# ------------------------------------------------------------------

def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Load an audio file to mono float32 numpy array at *sr* Hz via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH.")

    cmd = [
        "ffmpeg", "-nostdin", "-i", str(path),
        "-threads", "0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-",
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


# ------------------------------------------------------------------
# Mel spectrogram (numpy STFT, converted to mx.array at the end)
# ------------------------------------------------------------------

def get_logmel(
    audio: np.ndarray | mx.array,
    config: MelConfig | None = None,
) -> mx.array:
    """Compute log-mel spectrogram.

    Parameters
    ----------
    audio : 1-D float array (numpy or mx)
    config : MelConfig (default Nemotron config)

    Returns
    -------
    mx.array of shape [1, T, n_mels]   (batch=1)
    """
    if config is None:
        config = MelConfig()

    # Work in numpy for the STFT (simple, proven, avoids mx fft edge-cases)
    if isinstance(audio, mx.array):
        x = np.array(audio, dtype=np.float32)
    else:
        x = np.asarray(audio, dtype=np.float32)

    # Pre-emphasis
    if config.preemphasis > 0:
        x = np.concatenate([x[:1], x[1:] - config.preemphasis * x[:-1]])

    # STFT
    win = _window(config.window, config.win_length)
    S = _stft_np(x, config.n_fft, config.hop_length, config.win_length, win)

    # Power spectrum
    power = np.abs(S) ** config.mag_power  # [T, n_fft//2 + 1]

    # Mel filterbank
    fb = config.filterbank  # [n_mels, n_fft//2 + 1]
    mel = fb @ power.T  # [n_mels, T]

    # Log
    log_mel = np.log(mel + 1e-5)  # [n_mels, T]

    # Nemotron uses normalize="NA" — no normalisation applied.

    # Transpose to [T, n_mels] and add batch dim
    log_mel = log_mel.T[np.newaxis, :, :]  # [1, T, n_mels]

    return mx.array(log_mel, dtype=mx.float32)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _window(name: str, length: int) -> np.ndarray:
    """Return a window function as a numpy array."""
    funcs = {
        "hann": np.hanning,
        "hanning": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
        "bartlett": np.bartlett,
    }
    fn = funcs.get(name)
    if fn is None:
        raise ValueError(f"Unknown window type: {name}")
    return fn(length).astype(np.float32)


def _stft_np(
    x: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """Real-valued STFT using numpy. Returns complex array [T, n_fft//2+1]."""
    # Reflect-pad so first frame is centred
    pad_len = n_fft // 2
    x = np.pad(x, (pad_len, pad_len), mode="reflect")

    # Zero-pad window to n_fft if needed
    if win_length < n_fft:
        window = np.pad(window, (0, n_fft - win_length))
    elif win_length > n_fft:
        window = window[:n_fft]

    # Frame the signal
    n_frames = 1 + (len(x) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(x.strides[0] * hop_length, x.strides[0]),
    )

    # Apply window and FFT
    windowed = frames * window
    return np.fft.rfft(windowed, n=n_fft)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Compute a Slaney-normalised mel filterbank (no librosa dependency).

    Returns shape [n_mels, n_fft//2 + 1].
    """
    fmin = 0.0
    fmax = sr / 2.0

    # Mel scale (HTK formula)
    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = np.array([mel_to_hz(m) for m in mels])

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower = freqs[i]
        center = freqs[i + 1]
        upper = freqs[i + 2]

        # Rising slope: lower -> center
        up_slope = np.where(
            (fft_freqs >= lower) & (fft_freqs <= center) & (center > lower),
            (fft_freqs - lower) / (center - lower),
            0.0,
        )
        # Falling slope: center -> upper
        down_slope = np.where(
            (fft_freqs >= center) & (fft_freqs <= upper) & (upper > center),
            (upper - fft_freqs) / (upper - center),
            0.0,
        )
        fb[i] = np.maximum(up_slope, down_slope)

        # Slaney normalisation
        enorm = 2.0 / (upper - lower) if upper > lower else 1.0
        fb[i] *= enorm

    return fb
