"""Loser-chirp kernels: locked model_v2 vs model_v3_filled (filled trajectories), as is and as episodes (fast estimator)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_kernel_blur as dk
import matplotlib
import merge_gap_test as mg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = '/mnt/data1/chirp_v2/video2022/chases/model/'
ts = pd.read_csv(mg.DATASET / 'trial_summary.csv')
chirps = {r.recording: None for r in ts.itertuples()}
for r in ts.itertuples():
    t = np.load(mg.DATASET / r.recording / 'chirp_times_rtdetr_t080.npy')
    ids = np.load(mg.DATASET / r.recording / 'chirp_ids_rtdetr_t080.npy')
    chirps[r.recording] = t[ids == r.lose_ID]
variants = [('model_v2', 0.0, 0.0, '#fdae6b', 1.2), ('model_v3_filled', 0.0, 0.0, '#9ecae1', 1.2),
            ('model_v2', 0.9, 5.0, '#e6550d', 1.8), ('model_v3_filled', 0.9, 5.0, '#08519c', 1.8)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for col, center in enumerate(('agon', 'agoff')):
    ax = axes[col]
    t19, k19 = dk.ratio(dk.RATES_2019 / f'lchirps@{center}.json')
    ax.plot(t19, k19, color='0.25', lw=2, label='2019 (BORIS chases)')
    for name, min_p, gap, color, lw in variants:
        ev = pd.read_csv(M + name + '/events.csv')
        trials, n = [], 0
        for r in ts.itertuples():
            c = ev[(ev['recording'] == r.recording) & (ev['p_mean'] >= min_p)]
            mon, moff = mg.merge(c['t_start'].to_numpy(), c['t_end'].to_numpy(), gap)
            n += len(mon)
            trials.append((mon if center == 'agon' else moff, chirps[r.recording]))
        k = mg.kernel(trials, np.random.default_rng(1))
        T = mg.T
        label = f"{name}{', confident, gaps < 5 s merged' if gap else ', as is'} ({n})"
        ax.plot(T, k, color=color, lw=lw, label=label)
        print(f"{center} {label}: peak {k.max():.2f} at {T[np.argmax(k)]:+.1f} s, -2..0 {k[(T > -2) & (T < 0)].mean():.2f},"
              f" 0..2 {k[(T > 0) & (T < 2)].mean():.2f}")
    ax.axhline(1, color='0.7', lw=0.8)
    ax.axvline(0, color='k', ls=':', lw=0.8)
    ax.set_xlim(-30, 30)
    ax.set_xlabel(f"time from the chase {'onset' if center == 'agon' else 'offset'} (s)")
    ax.legend(fontsize=7, frameon=False, loc='upper left')
axes[0].set_ylabel('rate / jittered baseline')
fig.suptitle("Loser's chirps: locked model_v2 vs model_v3 on filled trajectories (orange vs blue)", fontsize=10)
fig.savefig('/mnt/data1/chirp_v2/paper2022/output/chases_model_v2/figures/kernels_v2_vs_v3.png', dpi=150)
print('ok')
