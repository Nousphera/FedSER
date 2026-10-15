"""Two-second, 16 kHz log-Mel spectrograms for speech emotion recognition."""

from functools import lru_cache

import numpy as np
import soundfile as sf
import torch


def read_wave(path):
    """Read a finite, nonempty mono waveform at IEMOCAP's 16 kHz rate."""
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 16000:
        raise ValueError(f"Expected 16000 Hz audio: {path}; got {rate}")
    if wave.shape[1] != 1:
        raise ValueError(f"Expected mono audio: {path}")
    wave = wave[:, 0]
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    return wave


@lru_cache(maxsize=1)
def _mel_weights():
    # TensorFlow's linear_to_mel_weight_matrix uses triangles in Mel space,
    # no area normalization and a zero DC bin. Generic HTK filters whose
    # triangles are linear in Hertz do not give the same transform.
    frequencies = torch.linspace(0.0, 8000.0, 513, dtype=torch.float32, device="cpu")
    mel = 1127.0 * torch.log(1.0 + frequencies / 700.0)
    limits = 1127.0 * torch.log(
        1.0 + torch.tensor([60.0, 7800.0], dtype=torch.float32, device="cpu") / 700.0
    )
    edges = torch.linspace(limits[0], limits[1], 66, dtype=torch.float32, device="cpu")
    lower = (mel[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2])
    upper = (edges[2:] - mel[:, None]) / (edges[2:] - edges[1:-1])
    weights = torch.minimum(lower, upper).clamp_min(0)
    weights[0] = 0
    return weights


def features(waveform, training, rng=None):
    """Return [1,198,64] in training or [segments,1,198,64] in evaluation.

    Normalize the whole utterance before a random two-second training crop.
    At evaluation, normalize each complete non-overlapping segment separately
    and discard an incomplete tail. Short utterances are center padded first.
    The 1e-9 epsilon bounds the squared L2 norm, matching the source transform.
    """
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Expected a nonempty finite mono waveform")
    if len(wave) < 32000:
        padding = 32000 - len(wave)
        wave = np.pad(wave, (padding // 2, padding - padding // 2))
    if training:
        rng = rng if rng is not None else np.random.default_rng()
        wave = wave / np.sqrt(max(float(np.dot(wave, wave)), 1e-9))
        start = int(rng.integers(len(wave) - 32000 + 1))
        segments = wave[start : start + 32000][None]
    else:
        segments = wave[: len(wave) // 32000 * 32000].reshape(-1, 32000)
        norms = np.sqrt(np.maximum(np.sum(segments * segments, axis=1, keepdims=True), 1e-9))
        segments = segments / norms
    samples = torch.from_numpy(np.ascontiguousarray(segments, dtype=np.float32))
    # Explicit 400-sample framing preserves TF STFT's 198 frames. torch.stft
    # otherwise frames according to n_fft=1024, changing this time dimension.
    frames = samples.unfold(-1, 400, 160)
    window = torch.hann_window(400, periodic=True, dtype=torch.float32, device="cpu")
    magnitude = torch.fft.rfft(frames * window, n=1024).abs()
    result = (magnitude @ _mel_weights()).clamp(1e-5, 1e8).log().unsqueeze(1)
    return result[0] if training else result
