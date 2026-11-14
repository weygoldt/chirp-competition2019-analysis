# Front-end ablation: three STFT windows vs the long window only

2026-11-14. Branch `v2-ablation-frontend` of `chirp_competition_upgrade`
(scripts in `experiments/ablation/`). Tables: `compare/tables.md`; per-stratum
CSVs and `differences.png` in `compare/`; validation curves `val_curves.png`.

## Question and design

Plan decision 6: do the two short STFT windows (0.0512 s, 0.0128 s) in image
channels 1–2 help, compared with the long 0.2048 s window alone?

- **Arm A**: the released dual-head RT-DETRv2-r18 (`dualhead_r18_sim_v4_r1`,
  weights 9fcdef3c), three windows on one frequency grid. Not retrained.
- **Arm B**: the same config, seed, schedule and early stopping on the same
  sim_v4 scenes and tiling, with the 0.2048 s window in all three channels
  (`dualhead_r18_sim_v4_longonly_r1`). `check_identity.py`: channel 0 and all
  boxes of all 81,000 tiles identical to sim_v4; `tile_comparison.png` shows
  one tile in both front ends.
- Both arms scored with the identical current evaluation code.
- **One training run per arm**: seed-to-seed variance is not measured.

## Training

| | epochs run | best epoch | best val fitness | val F1 / AP / assignment at best |
|---|---|---|---|---|
| A | 22 | 16 | 0.883 | 0.968 / 0.970 / 0.910 |
| B | 10 (early stop, patience 6) | 4 | 0.893 | 0.963 / 0.981 / 0.910 |

B converged faster and reached a higher validation fitness; the same
stopping rule therefore stopped it after 10 epochs.

## Results

**Simulated test** (300 scenes of unseen fish, 8,100 tiles, 9,401 chirps;
after post-processing, score ≥ 0.5; B − A with a paired scene bootstrap, pp):

| metric | A | B | B − A [95% CI] |
|---|---|---|---|
| precision | 97.6 | 96.2 | −1.38 [−1.86, −0.93] |
| recall | 95.7 | 95.3 | −0.40 [−0.79, −0.01] |
| F1 | 96.6 | 95.8 | −0.89 [−1.17, −0.61] |
| average precision | 97.5 | 98.1 | +0.57 [+0.24, +0.95] |
| correct emitter (of TPs) | 92.1 | 91.3 | −0.84 [−1.61, −0.04] |
| end-to-end recall (found + right emitter) | 88.2 | 87.0 | −1.17 [−2.00, −0.32] |

Strata: the gap is largest for big chirps shorter than 0.3 s (end-to-end
−2.0 pp [−3.1, −0.9]) and mid-SNR chirps (−1.8 to −2.1 pp); long chirps
(≥ 0.3 s, n = 219) are identical (94.1% recall in both); ΔF 0–10 Hz
assignment 88.1 vs 86.9% (−1.2 pp [−2.5, +0.05]). Emitter-head error of the
TPs, median: 1.27 vs 1.48 Hz.

**Real hand labels** (4 held-out recordings, 6 regions, score ≥ 0.5):

| labels | metric | A | B | B − A [95% CI] |
|---|---|---|---|---|
| 365 (335 + 30 confirmed) | precision | 93.8 | 93.8 | +0.0 [−3.5, +3.5] |
| | recall | 99.2 | 99.5 | +0.3 [−1.3, +1.9] |
| 335 original | precision / recall | 86.8 / 100 | 86.6 / 100 | |

Labels found by one arm only: 1 (A) vs 2 (B), exact McNemar p = 1. Of B's
24 false positives on the 365 labels, 5 lie where the author never reviewed
a box (A's were all reviewed), which can only favour A.

**Blind benchmark** (1,010 simulated items: 536 chirps, 409 negatives):

| metric | A | B | B − A [95% CI] |
|---|---|---|---|
| recall | 96.3 | 97.0 | +0.75 [−0.75, +2.43] (McNemar p 0.48) |
| chirp with the right emitter | 86.8 | 87.7 | +0.93 [−1.68, +3.54] (p 0.57) |
| false-alarm rate | 9.0 | 4.9 | −4.2 [−6.8, −1.7] (p 0.003) |

The false-alarm difference is a selection artefact: it comes from the
"model false alarm" negatives, which were drawn from arm A's own false
alarms when the benchmark was built (A 16.0%, B 7.8%, human 7.3%); on every
other negative type both arms are at ~0%.

## Reading

The two extra short windows bring **no practically relevant benefit**. On
simulated test scenes the three-window model is about 1 pp better in
precision, F1, assignment and end-to-end recall (paired-bootstrap CIs
exclude zero), while the long-only model has the higher average precision
and the higher validation fitness. On real recordings and on the blind
benchmark the arms are indistinguishable. With one training run per arm and
different stopping epochs, differences of ~1 pp are within what seed and
schedule could produce, so the direction is not established.

Consequences: no reason to retrain or replace the released model; the
manuscript can describe the multi-window front end as a design choice and
report this ablation ("comparable; ≤ 1 pp differences on simulated data,
none on real data"), but must not claim that the short windows improve
detection or assignment.

## Operational note

Arm B's evaluation hard-reset the workstation three times (2026-11-13
~23:20, ~23:33; 2026-11-14 00:22:33). A 1-Hz fsync'd telemetry log
(`/mnt/data1/chirp_v2/logs/telemetry_ablation_20261113-235239.csv`) shows the
last reset ~1 s after a core dropped from 28 min of sustained ~5.05 GHz
single-core boost to idle; RAM, GPU and temperatures were normal. With CPU
boost disabled (`/sys/devices/system/cpu/cpufreq/boost = 0`; cores ≤ 3.39 GHz,
telemetry `telemetry_ablation_boostoff_20261114-003836.csv`) the same
evaluation and all remaining steps completed (00:38–01:48). Partial outputs
of the crashed runs are kept as `*_crash*` / `*_partial_crash*`.
