"""Offset errors vs tracking gaps (both fish not seen) within 1 s of the author's offset."""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.claude/worktrees/data2022/experiments/video2022')
import c8_chase_model as c8  # noqa: E402

clips = c8.load_clips(["chase_v1", "chase_v2"], "poses_r3", "identity")
d = pd.read_csv('/mnt/data1/chirp_v2/video2022/chases/model/offset_refine/errors.csv')
rows = []
for r in d.itertuples():
    s = clips[r.clip]['s']
    t = s['t'].to_numpy()
    seen = np.isfinite(s['dist'].to_numpy())
    w = np.abs(t - r.label_off) <= 1.0
    rows.append({'abs_err': abs(r.err_model), 'seen': seen[w].mean() if w.any() else np.nan, 'contact': r.contact})
x = pd.DataFrame(rows)
x['group'] = pd.cut(x['abs_err'], [-0.01, 0.5, 1.0, 100], labels=['<= 0.5 s', '0.5-1 s', '> 1 s'])
print(x.groupby('group', observed=True).agg(chases=('abs_err', 'size'), both_seen_share=('seen', 'mean'),
                                            share_with_gap=('seen', lambda v: float((v < 0.8).mean())),
                                            contact=('contact', 'mean')).round(2))
