# 2022 competitions: chirp detection (milestone A1 of `docs/spec_2022_dataset.md`)

Diagnosis of why the production detector's 2022 detections agree so much
less with the expert than the 2019/2020 ones (real_v1 session: 0.37 at 0.5,
0.70 at 0.8, 0.92 at 0.9; 2019/2020 0.83 / 0.90 / 0.94), and what to do about
it. Nothing in the detector or the paper analyses was changed. Outputs are
under `/mnt/data1/chirp_v2/data2022/a1/` (written 2026-11-19).

"Agreement" is the share of detections the expert judged chirps (real_v1),
not precision; the simulator is the only ground truth.

## In short

1. **The 2022 agreement of the 0.5-0.7, 0.7-0.9 and 0.9-1 bins is decided by
   single items.** real_v1 spread its 2022 sample evenly over recordings, but
   the 2022 detections are concentrated in pair 3 (2022-06-20/22/24: 67 % of
   the dark-phase detections at 0.7-0.9, 61 % at >= 0.9; 2022-06-20 alone 47 %
   and 42 %). One "not a chirp" from 2022-06-20 carries 66 % of the 0.7-0.9
   bin's weight (reported 0.048; 0.129 without that recording); one "chirp"
   from the same recording carries 61 % of the >= 0.9 bin (0.923; 0.809
   without it). Both items are the same kind of detection: small loser chirps
   of 40-60 Hz.
2. **In the exhaustive recall windows, that recording's detections are
   chirps.** The expert marked 16 chirps in the 25 windows of 2022-06-20. The
   detector found all 16 at >= 0.8, and 16 of its 18 detections >= 0.8 in
   those zones match a mark. In the 38 windows from the 14 recordings outside
   pair 3 the expert marked no chirp, and the 4 detections >= 0.8 there match
   nothing. Recall in 2022 windows: 19 of 19 marks at 0.8, 7 of 19 at 0.9.
3. **The false positives are the same kinds as in 2019/2020, at the same rate
   per fish.** They are short symmetric brightenings of a fish's baseline (an
   amplitude change, no upward frequency excursion) and tall boxes over
   interference lines or noise. Winners chirp little in both datasets, so
   their mid-score detections estimate the false-positive floor: 0.5-0.9
   detections per trial-hour, median 10.3 (2019/2020) vs 14.2 (2022).
4. **What differs is how much the fish chirp.** Outside pair 3, 2022 losers
   have 10 detections >= 0.9 per hour (median over 15 trials) against 203 in
   2019/2020, about 20x fewer. So a similar floor of false positives is a much
   larger share of the 2022 detections. Pair 3's loser is the exception (about
   970 detections >= 0.8 per hour in 2022-06-20).
5. **On simulated scenes the detector is only slightly worse on 2022
   backgrounds.** The test set has 125 scenes on 2022 backgrounds (the whole
   1x11 line; training had 690). At equal recall (0.96 at 0.5) it makes 1.5x
   the false positives per scene-minute of 2019/2020 backgrounds: 0.83 vs 0.53
   at 0.5, 0.18 vs 0.12 at 0.8.
6. **For the replication, the 2022 winner-loser contrast is near the
   false-positive floor outside pair 3**: per hour, 19 (loser) vs 15 (winner)
   detections at 0.5-0.9, and 10 vs 5.7 at >= 0.9.

## What to decide

Fixes, ordered by cost. My recommendation is 1, then 2; no detector change.

1. **A small 2022 labelling session (about 25 min of your time) to replace
   the fragile estimate.** Candidates would be sampled in proportion to the
   detections, not evenly per recording: 2022 dark phase, kept, scores >= 0.5,
   stratified by score bin (0.5-0.7, 0.7-0.8, 0.8-0.9, >= 0.9) x assigned fish
   (winner, loser, other) x pair 3 vs the other trials. That is about 300
   candidates at up to 15 per cell (real_v1 mean 1.7 s each). Add 150 recall
   windows drawn uniformly over the dark phase, not by activity (4.1 s each),
   to estimate the 2022 chirp rate and recall at 0.8 vs 0.9. Same blind
   labeller and view as real_v1, about 20-25 min in total. It gives
   per-threshold agreement for winners and losers that a 2022 analysis can
   rely on.
2. **A 2022 operating threshold chosen from those labels.** The current
   evidence favours 0.8 over 0.9: 0.9 kept only 7 of the 19 window chirps,
   and real_v1's >= 0.8 agreement for 2022 is 0.70 [0.37, 0.86]. Use the same
   threshold for both fish.
3. **A per-fish false-positive allowance in the analysis** (author's
   decision, analysis not detector): subtract or model a floor of about
   10-15 mid-score detections per fish-hour, estimated from the winners'
   labelled detections. Needed only if the 2022 analysis uses low counts.
4. **A post-processing rule for the swelling false positives: not
   recommended.** A detector-independent check (the fish's instantaneous
   frequency in the box) separates the expert's chirps from not-chirps only
   weakly at 0.5-0.9 (AUC 0.64 in 2019/2020, 0.72 in 2022). Any rule on it
   would drop many real chirps. The tall boxes over interference lines are
   mostly unassigned ("other") and do not enter fish counts.
5. **Fine-tuning on more 2022-like scenes: not recommended now.** Training
   already covers the 1x11 line and 2022 noise, and the simulated
   false-positive excess on 2022 backgrounds is small. A retrain would change
   the 2019/2020 detections and would need the full 2019/2020 re-validation.

## Evidence

### 1. The 2022 estimates rest on single items

`a1_influence.py` → `influence/by_bin.csv`, `heaviest_items.csv`, `influence.png`.

| 2022 bin | items | recordings | agreement | heaviest item's weight share | without the heaviest recording | one recording left out: range |
|---|---|---|---|---|---|---|
| 0.5-0.7 | 56 | 16 | 0.299 | 0.234 (06-20, chirp) | 0.053 | 0.053-0.377 |
| 0.7-0.9 | 56 | 15 | 0.048 | 0.660 (06-20, not a chirp) | 0.129 | 0.031-0.129 |
| 0.9-1 | 46 | 14 | 0.923 | 0.606 (06-20, chirp) | 0.809 | 0.809-0.943 |

For comparison, 2019/2020 0.7-0.9: 0.669, heaviest item 0.240, range
0.583-0.706. Even without 2022-06-20, the 2022 0.7-0.9 bin is mostly
non-chirps (0.13).

![influence](/mnt/data1/chirp_v2/data2022/a1/influence/influence.png)

### 2. Recall windows (exhaustive labels)

From `labels/real_v1/report/windows.csv`. 2022: 75 windows, 19 marks; 16 in
2022-06-20 (25 windows), 2 in 2022-06-22 (4), 1 in 2022-06-24 (8), none in the
38 windows of the other 14 recordings. Detections >= 0.8 inside the zones:
2022-06-20 18 (16 matched), all 2022 26 (19 matched). The windows were drawn
80 % by minute-level activity, so they over-represent active minutes.

### 3. What the false positives look like

`a1_examples.py`, `cutouts.py` → `examples/<group>.png` (upper panel: the
labeller's 5-s view; lower: 0.8 s with a 51.2 ms window). Visual reading (mine,
not the expert's):

- `2022_winner_not.png`, `2022_loser_not.png`: mostly short, symmetric
  brightenings of the assigned fish's baseline. The box is 25-45 Hz high and
  the zoom shows no thin upward excursion. The rest are tall boxes (300-1300 Hz)
  over interference lines or broadband noise.
- `2022_other_not.png`: tall boxes, and detections on faint, wavy, untracked
  lines in 2022-05-10/11/12 that are not fish tracks.
- `2019_not.png`: the same two kinds in 2019/2020.
- `2022_chirps.png`, `2019_chirps.png`: judged chirps; small ones show a thin
  upward excursion from the baseline in the zoom.
- `2022-06-20_loser_unlabelled.png`: 12 random unlabelled detections of the
  pair-3 loser; most show such an upward tick.

### 4. What the detector sees: 2019/2020 vs 2022

`a1_signal.py` (24 random 4-s segments per recording from the dark phase) and
`a1_figures.py` → `signal/recordings.csv`, `signal.png`. Medians over trial
recordings:

| | 2019/2020 | 2022 |
|---|---|---|
| electrodes | 9 (15 recordings) or 15 (34), grids | 11 (1x11 line) |
| fish SNR, winner / loser (dB, peak over 300-2000 Hz floor) | 40.6 / 37.3 | 52.3 / 35.7 |
| electrodes within 10 dB of the fish's best, winner / loser | 3.5 / 3.5 | 1 / 2 |
| spectral lines no fish explains (per segment) | 8 | 15 |
| strongest such line vs the weaker fish (dB) | -12.7 | -12.2 |
| broadband transients (> 3 kHz, per s) | 0.0 | 0.0 |
| samples repaired by click removal | ~0 | ~0 |
| share of dark-phase seconds with a track detection, winner / loser | 0.99 / 0.97 | 0.98 / 0.98 |
| winner-loser frequency difference (Hz; 10th-90th percentile) | 113 (22-188) | 86 (48-134) |

The 2022 winner is strong on only one electrode in every trial. It seems to
stay at one place on the line, probably the tube; B1's video work can check
this. The fish are not weaker in 2022, and clicks are not more common.
Interference lines are more numerous. In pair 3 and 2022-06-27/29 they come
within 0-5 dB of the weaker fish. In the 2022-06-20 minute I looked at, they
form a comb at about 95 Hz spacing (285, 380, 475, 570, 760, 855, 950 ... Hz).
Their source is unknown (recording equipment?).

![signal](/mnt/data1/chirp_v2/data2022/a1/signal/signal.png)

### 5. Simulated test scenes by background (ground truth)

`a1_sim_by_background.py` → `sim_by_background/by_threshold.csv`,
`by_score_bin.csv`. Re-reads the existing evaluation of the production model
after post-processing (`runs/dualhead_r18_sim_v4_r1/eval_postprocess/sim_test_after/items.csv`).

| backgrounds | scenes (recordings) | recall at 0.5 / 0.8 / 0.9 | false positives per scene-minute at 0.5 / 0.8 | precision 0.7-0.8 / 0.8-0.9 |
|---|---|---|---|---|
| 2019/2020 | 139 (2) | 0.957 / 0.931 / 0.822 | 0.53 / 0.12 | 0.78 / 0.97 |
| 2022 (1x11 line) | 125 (2) | 0.958 / 0.929 / 0.830 | 0.83 / 0.18 | 0.64 / 0.95 |
| field | 36 (7) | 0.957 / 0.933 / 0.831 | 1.31 / 0.25 | 0.54 / 0.96 |

The simulated scenes have about 30 chirps per minute, far more than real
trials, so simulated precision is high everywhere. False positives per minute
are what carries over to real data. Only 2 background recordings per
competition dataset are in the test split, so these are rough numbers.

### 6. Chirp rates and the false-positive floor

`a1_figures.py` → `signal/rates.csv`, `rates_summary.csv`, `rates.png`. These
are kept detections per hour of dark phase, by the assigned fish's trial
status. 2019/2020 uses the 33 analysed trials with the gap-filled assignment;
2022 uses the 18 trials of `meta.csv`. Medians over trials:

| | 2019/2020 winner | 2019/2020 loser | 2022 winner | 2022 loser | 2022 loser without pair 3 |
|---|---|---|---|---|---|
| 0.5-0.9 | 10.3 | 68.3 | 14.2 | 23.3 | 19.3 |
| >= 0.9 | 3.0 | 202.7 | 4.7 | 11.5 | 10.0 |

![rates](/mnt/data1/chirp_v2/data2022/a1/signal/rates.png)

### 7. A detector-independent check that did not separate well

`a1_excursion.py` → `excursion/items.csv`, `auc.csv`, `excursion.png`. For
each labelled candidate (score >= 0.2), it measures the assigned fish's
instantaneous frequency on its least-contaminated electrode. The signal is
band-passed around its baseline, clear of the other fish, and samples where
the envelope collapses are ignored. The largest upward excursion in the box
separates the expert's chirps from not-chirps with AUC 0.64 (2019/2020) and
0.72 (2022) at 0.5-0.9, and 0.89 / 0.78 at >= 0.9.

## Open questions of the spec, as far as A1 answers them

- **Which 2022 recordings were in real_v1?** All 19 provided candidates
  (6-37 each); 17 provided recall windows. 2022-07-07-10_00 is not in
  `meta.csv`, so its detections count as "other".
- **Does the simulator cover a 1x11 line and 2022 backgrounds?** Yes: 690
  training, 84 validation and 125 test scenes use 2022 backgrounds with their
  own 11-electrode line. The test scenes come from recordings 2022-05-10 and
  2022-06-29.

## Scripts

| script | what | output (`/mnt/data1/chirp_v2/data2022/a1/`) |
|---|---|---|
| `a1_influence.py` | weighted agreement by bin, weight shares, leave-one-recording-out | `influence/` |
| `a1_examples.py`, `cutouts.py` | contact sheets of labelled and unlabelled detections | `examples/` |
| `a1_signal.py` | per-recording signal measures and detection rates (about 5 min) | `signal/*.csv` |
| `a1_figures.py` | rates and signal figures, `rates_summary.csv` | `signal/*.png` |
| `a1_sim_by_background.py` | simulated test evaluation split by background dataset | `sim_by_background/` |
| `a1_excursion.py` | instantaneous-frequency excursion of labelled candidates | `excursion/` |
| `test_a1.py` | tests of the pure functions | |

Run them with the root environment, e.g.
`/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python experiments/data2022/a1_signal.py`.
