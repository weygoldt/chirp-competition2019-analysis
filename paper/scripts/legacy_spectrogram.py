"""NumPy reimplementation of the two spectrogram helpers of old gridtools (v1).

Fig. 1 (``plot_what_is_a_chirp_and_why_use_spetrograms.py``) imported
``compute_spectrogram`` and ``to_decibel`` from
``gridtools.utils.spectrograms`` (``~/wrk/tools/gridtools``), which wrap
torchaudio. The paper environment has no torch, so this module reproduces
them in NumPy:

``compute_spectrogram(data, samplingrate, nfft, hop_length)``
    torchaudio ``Spectrogram(n_fft=nfft, hop_length=hop_length, power=2,
    normalized=False, window_fn=torch.hann_window, pad=nfft)``: constant zero
    padding of ``nfft`` samples on both sides, then ``torch.stft`` with
    ``win_length=n_fft``, ``center=True`` (reflect padding of ``n_fft // 2``),
    periodic Hann window, onesided; power spectrum. gridtools then dropped
    ``ceil(pad / hop_length)`` frames on each side and returned
    ``(spec[freq, time], time, freq)``.
``to_decibel(spec)``
    torchaudio ``AmplitudeToDB(stype="power", top_db=60)``:
    ``10 * log10(max(spec, 1e-10))``, floored at its maximum minus 60 dB.

Both return NumPy arrays (the old ones returned torch tensors; the caller's
``.cpu().numpy()`` is dropped). ``paper/tests/test_legacy_spectrogram.py``
checks them against ``torch.stft``.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window, as ``torch.hann_window(n)`` (``periodic=True``)."""
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)


def compute_spectrogram(
    data: np.ndarray,
    samplingrate: float,
    nfft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power spectrogram ``(freq, time)`` and its time and frequency axes."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("data must be one-dimensional")
    pad = nfft
    x = np.pad(x, (pad, pad), mode="constant")  # Spectrogram(pad=nfft)
    x = np.pad(x, (nfft // 2, nfft // 2), mode="reflect")  # stft(center=True)
    n_frames = 1 + (len(x) - nfft) // hop_length
    frames = sliding_window_view(x, nfft)[::hop_length][:n_frames]
    spec = np.abs(np.fft.rfft(frames * hann_window(nfft), axis=-1)) ** 2
    spec = spec.T  # (freq, time) as torch
    pad_samples = int(np.ceil(pad / hop_length))
    spec = spec[:, pad_samples:-pad_samples]

    time = np.arange(0, spec.shape[1]) * hop_length / samplingrate
    freq = np.arange(0, spec.shape[0]) * samplingrate / nfft
    return spec, time, freq


def to_decibel(spec: np.ndarray, top_db: float = 60.0) -> np.ndarray:
    """Power to dB, clipped at ``top_db`` below the maximum (AmplitudeToDB)."""
    db = 10.0 * np.log10(np.maximum(np.asarray(spec, dtype=np.float64), 1e-10))
    return np.maximum(db, db.max() - top_db)
