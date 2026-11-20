# 2019 paper analyses on the 2022 data (only the chase-triggered chirp rates so far)

Author (2026-11-20): of the 2019 pipeline only the chase onset / offset-triggered chirp
rates (the kernel-convolution plots) are of interest now; no contacts. The 2019 code in
`paper/` (main checkout) is run **unchanged**, on a 2022 dataset folder in the 2019
layout; nothing in `paper/` or `manuscript/` is changed and no output goes there.

1. `build_dataset.py` → `/mnt/data1/chirp_v2/paper2022/dataset/` (README in its
   docstring): the 18 trials, the **locked** model chases
   (`video2022/chases/model/model_v2`) as BORIS onset / offset rows in video time
   (overlapping chases merged: 5 in pair 3, 2,042 chases), the chirps at 0.8
   (unweighted, as in 2019), LED files, an empty rise array (the script loads rises),
   `trial_summary.csv` (`group` = 5 for all, the 2019 filter for annotated trials).
   `--check` runs the 2019 loader on every trial: chase times come back exactly
   (0.0 ms) through its LED map.
2. The unchanged 2019 analysis, in the paper's environment, outputs to
   `/mnt/data1/chirp_v2/paper2022/output/chases_model_v2/`:

       O=/mnt/data1/chirp_v2/paper2022/output/chases_model_v2; P=<main checkout>/paper
       cd $O && CHIRP_REC_2019=/mnt/data1/chirp_v2/paper2022/dataset PAPER_DETECTIONS=rtdetr_t080 \
         PAPER_OUT=$O PAPER_FIGURE_DIR=$O/figures PAPER_PRIMARY=chases_model_v2 MPLBACKEND=Agg \
         $P/.venv/bin/python $P/scripts/analyze_event_time_rates_trial.py --ag-events

   (run detached with `systemd-run` + `scripts/guarded --mem 12G`). The two contact
   centrings have no centres in 2022.
3. `plot_chase_triggered_rates.py`: the chase columns of the 2019 figure for 2022
   (the 2019 plot script expects contact results and cannot run on 2022), and 2019 vs
   2022 from the same code.

Handoff for running more of the 2019 pipeline later: `docs/handoff_paper_2022.md`.

## Result (2026-11-20)

The unchanged script wrote the four chase centrings (`{l,w}chirps@{agon,agoff}.json`)
and then stopped at the contact centrings (no contacts in 2022: `ValueError: a must be
greater than 0`), as expected. Peak / jittered baseline (`plot_chase_triggered_rates.py`):

| | 2019 (15 trials) | 2022 (18 trials) |
|---|---|---|
| loser, chase onset | 2.25x at +7.6 s; dip to ~0.5x just before | 2.16x at +7.1 s; already rising from ~-8 s |
| loser, chase offset | 3.2x at +0.4 s | ~2.2x from 0 to +3 s, broader |
| winner, chase onset | <= 1.3x | ~4x right at the onset |
| winner, chase offset | drop to ~0.3x | raised during the chase, drop at the offset |

The 2022 bands are wide (pair 3 dominates the chirps); the model splits chase bouts
into single chases, which blunts the offset peak and fills the pre-onset dip.
