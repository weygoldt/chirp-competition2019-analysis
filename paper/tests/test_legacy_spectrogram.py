"""legacy_spectrogram reproduces old gridtools' torchaudio spectrogram (Fig. 1).

The reference is torchaudio's ``Spectrogram(pad=nfft, power=2,
normalized=False, window_fn=torch.hann_window)`` + ``AmplitudeToDB("power",
top_db=60)`` + gridtools' frame trimming, written out with plain torch
(``torch.nn.functional.pad`` + ``torch.stft``, as torchaudio.functional does),
so torchaudio itself is not needed. The paper environment has no torch: run
with an environment that has it, e.g.

    ~/wrk/analyses/chirp_competition_upgrade/.venv/bin/python -m pytest \
        -p no:cacheprovider --rootdir paper paper/tests/test_legacy_spectrogram.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from legacy_spectrogram import compute_spectrogram, to_decibel

torch = pytest.importorskip("torch")


def _torch_reference(data: np.ndarray, samplingrate: float, nfft: int, hop: int):
    x = torch.from_numpy(data)  # float64, as in the old code
    window = torch.hann_window(nfft)  # float32 buffer, as in torchaudio
    pad = nfft
    x = torch.nn.functional.pad(x[None], (pad, pad), "constant")
    spec = torch.stft(
        x,
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=window,  # float32 window on float64 input, as torchaudio
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )[0]
    spec = spec.abs().pow(2.0)
    k = int(np.ceil(pad / hop))
    spec = spec[:, k:-k]
    # AmplitudeToDB(stype="power", top_db=60)
    db = 10.0 * torch.log10(torch.clamp(spec, min=1e-10))
    db = torch.max(db, db.amax() - 60.0)
    time = np.arange(spec.shape[1]) * hop / samplingrate
    freq = np.arange(spec.shape[0]) * samplingrate / nfft
    return spec.numpy(), db.numpy(), time, freq


def _fig1_like_signal(seed: int = 0) -> np.ndarray:
    """Two wave-type EODs plus a frequency-modulated one and noise (2 s, 20 kHz)."""
    rng = np.random.default_rng(seed)
    rate, n = 20000, 40000
    t = np.arange(n) / rate
    f = 700 + 300 * np.exp(-0.5 * ((t - 1.0) / 0.02) ** 2)
    phase = 2 * np.pi * np.cumsum(f) / rate
    x = np.sin(phase) + 0.3 * np.sin(2 * phase + 0.5)
    x += 1.2 * np.linspace(0, 1, n) * np.sin(2 * np.pi * 650 * t)
    x += 0.3 * np.linspace(1, 0, n) * np.sin(2 * np.pi * 924 * t)
    x += 0.1 * rng.standard_normal(n)
    return x - x.mean()


@pytest.mark.parametrize(
    ("nfft", "hop", "n"),
    [(4096, 4096 // 10, 40000), (512, 64, 10001), (1024, 333, 7777)],
)
def test_matches_torch(nfft, hop, n):
    data = _fig1_like_signal()[:n]
    ref_spec, ref_db, ref_time, ref_freq = _torch_reference(data, 20000, nfft, hop)
    spec, time, freq = compute_spectrogram(data, 20000, nfft=nfft, hop_length=hop)
    assert spec.shape == ref_spec.shape
    np.testing.assert_array_equal(time, ref_time)
    np.testing.assert_array_equal(freq, ref_freq)
    np.testing.assert_allclose(spec, ref_spec, rtol=1e-4, atol=1e-6 * ref_spec.max())
    db = to_decibel(spec)
    assert np.max(np.abs(db - ref_db)) < 1e-3


def test_fig1_shape():
    """Fig. 1's call: 2 s at 20 kHz, nfft 4096, hop 409."""
    nfft, hop = 4096, 409
    spec, time, freq = compute_spectrogram(np.zeros(40000), 20000, nfft, hop)
    n_frames = 1 + (40000 + 2 * nfft + 2 * (nfft // 2) - nfft) // hop
    assert spec.shape == (nfft // 2 + 1, n_frames - 2 * math.ceil(nfft / hop))
