"""Per labelled chase: chaser / chasee speed, distance, cue; author's, model's and refined offsets."""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.claude/worktrees/data2022/experiments/video2022')
import c8_chase_model as c8  # noqa: E402
import offset_refine as orf  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

clips = c8.load_clips(["chase_v1", "chase_v2"], "poses_r3", "identity")
d = pd.read_csv('/mnt/data1/chirp_v2/video2022/chases/model/offset_refine/errors.csv')
d = pd.concat([d.sort_values('err_model', key=np.abs, ascending=False).head(8),
               d.sort_values('err_model', key=np.abs).head(8)])
r = orf.Refine(delta_s=0.6, w_chasee=1.0, w_open=2.0, before_s=2.0, after_s=2.0)
fig, axes = plt.subplots(4, 4, figsize=(15, 11), constrained_layout=True)
for ax, row in zip(axes.flat, d.itertuples(), strict=False):
    s = clips[row.clip]['s']
    t = s['t'].to_numpy()
    ch, ce = row.chaser, 'B' if row.chaser == 'A' else 'A'
    w = (t > row.label_off - 6) & (t < row.label_off + 6)
    ax.plot(t[w], s[f'speed_{ch}'].to_numpy()[w], color='#b30000', lw=1.4, label='chaser speed')
    ax.plot(t[w], s[f'speed_{ce}'].to_numpy()[w], color='0.4', lw=1.2, label='chasee speed')
    ax2 = ax.twinx()
    ax2.plot(t[w], s['dist'].to_numpy()[w], color='#2b8cbe', lw=1, ls='--')
    ax2.set_ylim(0, 4)
    c = orf.cue(s, ch, ce, r)
    ax.plot(t[w], np.clip(c[w], -3, 6), color='#31a354', lw=0.8, alpha=0.8, label='cue')
    ax.axvline(row.label_off, color='k', lw=2, label='author')
    ax.axvline(row.model_off, color='#fd8d3c', lw=1.5, ls='--', label='model')
    ax.axvline(row.label_off + row.err_refined, color='#31a354', lw=1.5, ls=':', label='refined')
    ax.set_title(f"{row.clip_id} model err {row.err_model:+.1f} s{' contact' if row.contact else ''}", fontsize=8)
    ax.set_ylim(-3, 6)
axes[0, 0].legend(fontsize=6, frameon=False)
fig.suptitle('Labelled chase offsets: top 8 = largest model errors, bottom 8 = smallest (dashed blue: distance, '
             'right axis)', fontsize=10)
fig.savefig('/mnt/data1/chirp_v2/video2022/chases/model/offset_refine/cases.png', dpi=110)
print('ok')
