"""Each fish's EOD amplitude on the 11 electrodes, every 0.08 s, from the raw traces.

The wavetracker's ``sign_v`` gives power per electrode on spectrogram windows of
1.64 s (``nfft`` 2^15 at 20 kHz). A chasing fish swims ~0.5 m in that time, so
here the raw traces are demodulated at each fish's own EOD frequency and the
complex amplitude is taken on short windows instead:

1. Reference frequency of the fish: its gap-filled wavetracker track
   (``tracks_gapfill.npz``), linearly interpolated to every sample.
2. Mix every channel down by that frequency (cumulative phase), low-pass and
   decimate to 200 Hz (block mean of 10 samples, then ``resample_poly`` by 10).
3. On Hann windows of ``win_s`` (0.16 s) every ``step_s`` (0.08 s): DFT of the
   baseband on a grid of frequency offsets within +-``band_hz``; the offset with
   the largest power summed over channels wins; its complex value per channel is
   the fish's amplitude there (scaled so that a sine of amplitude 1 gives 1).

The EOD field is quasi-static: all electrodes see the same oscillation, scaled
by a real, signed factor. :func:`signed_pattern` turns the complex amplitudes of
one window into that real pattern (up to one global sign) and reports how much
of the power is off that real axis (``imag_frac``; near 0 for a clean window).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

FS = 20000.0
N_CH = 11


@dataclass(frozen=True)
class DemodParams:
    win_s: float = 0.16
    step_s: float = 0.08
    band_hz: float = 4.0
    df_hz: float = 0.25
    fs_base: float = 200.0
    pad_s: float = 1.0  # extra data read on each side of a chunk (filter edges)
    max_track_gap_s: float = 1.0  # windows farther than this from a track point have no amplitude


DEFAULT = DemodParams()


def near_track(t: np.ndarray, t_track: np.ndarray, max_gap: float) -> np.ndarray:
    """True where ``t`` lies within ``max_gap`` of a point of the (sorted) track times."""
    j = np.clip(np.searchsorted(t_track, t), 1, len(t_track) - 1)
    d = np.minimum(np.abs(t - t_track[j - 1]), np.abs(t_track[j] - t))
    return d <= max_gap


def read_traces(path: Path, t0: float, t1: float, n_ch: int = N_CH, fs: float = FS) -> np.ndarray:
    """Samples [t0, t1) s of a fishgrid ``traces-grid1.raw`` (float32, channels interleaved) as float64."""
    mm = np.memmap(path, dtype=np.float32, mode="r")
    n_total = mm.size // n_ch
    i0, i1 = max(0, int(round(t0 * fs))), min(n_total, int(round(t1 * fs)))
    return np.asarray(mm[i0 * n_ch:i1 * n_ch], dtype=np.float64).reshape(-1, n_ch)


def baseband(x: np.ndarray, t0: float, t_ref: np.ndarray, f_ref: np.ndarray, fs: float = FS,
             fs_base: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """Mix ``x`` (n, n_ch; first sample at time ``t0``) down by the reference frequency and decimate.

    Returns the baseband (m, n_ch) complex at ``fs_base`` and its sample times. A component at
    ``f_ref + d`` ends up at frequency ``d``.
    """
    down = int(round(fs / fs_base))
    if down % 10:
        raise ValueError("fs / fs_base must be a multiple of 10")
    n = (len(x) // down) * down
    t = t0 + np.arange(n) / fs
    f = np.interp(t, t_ref, f_ref)
    phase = 2 * np.pi * np.cumsum(f) / fs
    z = x[:n] * np.exp(-1j * phase)[:, None]
    z = z.reshape(-1, 10, x.shape[1]).mean(axis=1)  # block mean: 2 kHz for fs = 20 kHz
    z = resample_poly(z, 1, down // 10, axis=0)
    tb = t0 + (np.arange(len(z)) * down + (down - 1) / 2) / fs
    return z, tb


def window_amplitudes(z: np.ndarray, tb: np.ndarray, centers: np.ndarray,
                      p: DemodParams = DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Complex amplitude per channel on Hann windows centred at ``centers`` (s).

    Returns ``A`` (len(centers), n_ch) complex (NaN where the window does not fit) and the frequency
    offset ``df`` (Hz) of the peak within +-``band_hz``.
    """
    fsb = 1.0 / np.median(np.diff(tb))
    win = int(round(p.win_s * fsb))
    w = np.hanning(win + 2)[1:-1]
    offs = np.arange(-p.band_hz, p.band_hz + p.df_hz / 2, p.df_hz)
    tw = (np.arange(win) - (win - 1) / 2) / fsb
    E = np.exp(-2j * np.pi * offs[None, :] * tw[:, None]) * w[:, None] * (2.0 / w.sum())  # (win, n_off)
    start = np.round((centers - tb[0]) * fsb - (win - 1) / 2).astype(int)
    A = np.full((len(centers), z.shape[1]), np.nan + 0j)
    df = np.full(len(centers), np.nan)
    ok = (start >= 0) & (start + win <= len(z))
    if not ok.any():
        return A, df
    idx = start[ok][:, None] + np.arange(win)[None, :]
    seg = z[idx]  # (k, win, n_ch)
    S = np.einsum("kwc,wf->kfc", seg, E)  # (k, n_off, n_ch)
    best = np.argmax((np.abs(S) ** 2).sum(axis=2), axis=1)
    A[ok] = S[np.arange(len(best)), best]
    df[ok] = offs[best]
    return A, df


def signed_pattern(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real signed field pattern (up to a global sign) and the share of power off the real axis.

    ``A`` (..., n_ch) complex. The pattern is ``Re(A * exp(-i theta))`` with ``theta`` the axis that
    holds the most power (``theta = angle(sum A^2) / 2``).
    """
    theta = np.angle(np.sum(A ** 2, axis=-1)) / 2
    rot = A * np.exp(-1j * theta)[..., None]
    total = np.sum(np.abs(A) ** 2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        imag_frac = np.sum(rot.imag ** 2, axis=-1) / total
    return rot.real, imag_frac


def fish_references(gapfill: Path) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Per trial fish role (winner / loser): (wavetracker id, times, EODf) of its gap-filled track."""
    z = np.load(gapfill)
    out = {}
    for role, fid in zip(z["fish_role"], z["fish_id"], strict=True):
        m = z["ident_v"] == fid
        t = z["times"][z["idx_v"][m]]
        order = np.argsort(t)
        out[str(role)] = (float(fid), t[order], z["fund_v"][m][order])
    return out


def recording_amplitudes(raw: Path, refs: dict[str, tuple[float, np.ndarray, np.ndarray]], t_end: float,
                         chunk_s: float = 60.0, p: DemodParams = DEFAULT,
                         progress=None) -> dict[str, np.ndarray]:
    """Amplitudes of every fish in ``refs`` on the window grid ``step_s * k`` from 0 to ``t_end``.

    A window is NaN outside the fish's track (no extrapolation of its frequency). Returns
    ``t`` (centres), and per role ``A_<role>`` (n, n_ch) complex64 and ``df_<role>``.
    """
    centers = np.arange(int(np.ceil(p.win_s / 2 / p.step_s)), int(t_end / p.step_s)) * p.step_s
    out: dict[str, np.ndarray] = {"t": centers}
    for role in refs:
        out[f"A_{role}"] = np.full((len(centers), N_CH), np.nan + 0j, dtype=np.complex64)
        out[f"df_{role}"] = np.full(len(centers), np.nan, dtype=np.float32)
    for c0 in np.arange(0.0, t_end, chunk_s):
        c1 = min(c0 + chunk_s, t_end)
        sel = np.flatnonzero((centers >= c0) & (centers < c1))
        if not len(sel):
            continue
        r0 = max(0.0, c0 - p.pad_s)
        x = read_traces(raw, r0, c1 + p.pad_s)
        x -= x.mean(axis=0)
        for role, (_, tr, fr) in refs.items():
            inside = sel[near_track(centers[sel], tr, p.max_track_gap_s)]
            if not len(inside):
                continue
            z, tb = baseband(x, r0, tr, fr, fs_base=p.fs_base)
            A, df = window_amplitudes(z, tb, centers[inside], p)
            out[f"A_{role}"][inside] = A
            out[f"df_{role}"][inside] = df
        if progress:
            progress(c1, t_end)
    return out
