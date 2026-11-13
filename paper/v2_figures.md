# Pipeline-v2 figures (methods/supplement)

Three figures replace the v1 pipeline figures of the manuscript. Outputs
(PDF + 150-dpi PNG) in `/mnt/data1/chirp_v2/paper_rerun/rtdetr/figures/`;
all are 17.4 cm wide (the width of most manuscript figures; the template
scales them to the 16-cm text width).

| new figure | replaces | script | size |
|---|---|---|---|
| `plot_v2_pipeline_explainer` | `plot_assignment_feature_extraction.pdf` | `scripts/plot_v2_pipeline_explainer.py` | 17.4 x 13.0 cm |
| `plot_v2_performance` | `plot_pipeline_performances_manuscript_supplement_supplement.pdf` | `scripts/plot_v2_performance.py` | 17.4 x 12.0 cm |
| `plot_v2_failure_modes` | `plot_pipeline_performances_manuscript_supplement.pdf` | `scripts/plot_v2_failure_modes.py` | 17.4 x 12.6 cm |

## How to run

1. Cache the raw-data inputs (upgrade repo env: chirpdetector/gridtools;
   CPU; ~15 s for `snippets traces`, ~1 min for `assign-baseline`, which
   loads the tracks of the 300 simulated test scenes and should go through
   `guarded`):

   ```sh
   cd paper
   /home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python scripts/v2_fig_extract.py snippets traces
   ~/wrk/analyses/chirp_competition_upgrade/scripts/guarded --name v2fig-assign --mem 16G --vram 0.2 -- \
     /home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python scripts/v2_fig_extract.py assign-baseline
   ```

   Writes `$PAPER_OUT/data/processed/v2_figures/` (default
   `/mnt/data1/chirp_v2/paper_rerun/rtdetr/data/processed/v2_figures/`).
2. Plot (paper env; ~20 s each on a cold LaTeX cache, ~3 s warm):

   ```sh
   cd paper && unset VIRTUAL_ENV
   uv run python scripts/plot_v2_pipeline_explainer.py
   uv run python scripts/plot_v2_performance.py
   uv run python scripts/plot_v2_failure_modes.py
   ```

   Each prints the numbers it shows (JSON or tables).

`scripts/v2_fig_common.py` holds paths, colours and drawing helpers (paper
env). Spectrograms are the production front end (`span_spectrogram` with the
released bundle: click removal on >= 9 electrodes, 205/51/13-ms Hann windows
on one 4.88-Hz grid, 5-ms hop, 500-1500 Hz, linear power summed over
electrodes), normalised with the front end's rule (60-dB range ending 10 dB
above the median per-frame peak of the 205-ms channel) over each snippet.
Display levels 0.3-0.95 of that range; Fig. 1A and the 51-ms crop of
Fig. 3C use per-crop levels (40th-99.8th percentile). Fish colours: EOD$f_1$
= higher-frequency fish of the snippet (orange), EOD$f_2$ (reddish purple),
as in the old figures (EOD$f_1$ above EOD$f_2$).

## Numbers and their sources

### Fig. 1 (explainer)

Source: `detections/competition2019_v2_20261113/2020-05-29-10_00/chirps_rtdetr.parquet`
(rows printed by the script), tracks of that recording, traces from
`v2_fig_extract.py traces` (gridtools harvest extraction).

- Front end (A): `2020-06-05-10_00`, 89.9-91.1 s; fish tracks 860.0 Hz
  (track 14, EOD$f_1$) and 830.9 Hz (track 10, EOD$f_2$), ΔF 29.1 Hz;
  chirp 133 (EOD$f_2$, 90.28 s, score 0.96, height 63.5 Hz, FWHM 10 ms) and
  chirp 134 (EOD$f_1$, 90.67 s, score 0.95, height 387.7 Hz, FWHM 5 ms).
  No snippet in the 33 trials had a visible long chirp and a small chirp
  between fish < 35 Hz apart (the candidates in 2019 recordings with
  long chirps showed them only faintly); the long chirp is in C and E2.
  Panels A1-A3 use per-panel display contrast; A4 is the detector's input
  on its own levels (R, G, B = 205, 51, 13 ms).
- Tile 10850.4-10853.6 s (tile 4932), fish tracks 764.0 Hz (track 10,
  EOD$f_1$) and 650.9-651.2 Hz (track 12, EOD$f_2$).
- Chirp 2085 (A, E1): score 0.95, emitter 651.9 Hz, track 651.2 Hz;
  decoder height 69.4 Hz, FWHM 10.6 ms, 10 % width 17.5 ms, undershoot
  1.9 Hz (not drawn, < 5 % of the height). Trace peak 61.7 Hz above the
  trace baseline 651.2 Hz.
- Chirp 2086 (C, E2): score 0.95, box 675-1167 Hz, emitter 760.6 Hz, track
  764.0 Hz (3.4 Hz; other fish 650.9 Hz); height 409.3 Hz, FWHM 79.1 ms,
  10 % width 86.5 ms, undershoot 42.3 Hz. Trace peak 378.9 Hz above the
  trace baseline 764.1 Hz.
- Chirp 2087 (C): score 0.91, box 611-766 Hz (contains both baselines),
  emitter 650.5 Hz, track 650.9 Hz; onset 61 ms after 2086's.
- Harmonic copy 2084: score 0.84, box 1261-1501 Hz, emitter 1307.9 Hz
  (2 x 651.2 = 1302.5 Hz), `pp_harmonic_of_track` 12, dropped.
- Chirp 2088: score 0.65, fish 12.
- Duplicate (D2): `2020-03-18-10_34` (a draw, not among the 33 trials),
  chirp 49386 kept (score 0.86, box 629-1386 Hz), 49385 dropped (score
  0.72, box 637-836 Hz).

### Fig. 2 (performance)

| panel | number | source |
|---|---|---|
| A real, n = 365 labels | dual-head P 0.938, R 0.992 (TP 362, FP 24, FN 3); YOLO26s 0.947 / 0.984 (359 / 20 / 6); v1 1.000 / 0.701 (256 / 0 / 109) | `eval/real_labels_365/<model>/items.csv` (counted); curves `pr_curve.csv` level "all" |
| B sim test after post-processing, 9,401 chirps | dual-head 0.976 / 0.957 (8,999 / 225 / 402); YOLO26s 0.989 / 0.952 (8,948 / 101 / 453) | `runs/<model>/eval_postprocess/sim_test_after/items.csv`; curves `pr_curve.csv` |
| C correct emitter by ΔF (TP at >= 0.5) | emitter head: <10 Hz 0.881 (n 5,605), 10-25 0.982 (2,218), 25-50 0.994 (659), 50-100 1.000 (299), >=100 1.000 (120); box edge + 41.15 Hz: 0.578, 0.716, 0.836, 0.866, 0.983. All 8,999 TPs: head 0.921, box edge + offset 0.651, raw box edge 0.402 | `data/processed/v2_figures/assign_baseline.csv` (recomputed: kept predictions of `sim_test_predictions_flagged.parquet`, scene tracks, `assign_by_frequency`, items of `sim_test_after`) |
| D benchmark, 536 true chirps (1,010 items) | recall expert 0.924, dual-head 0.963; by SNR (<-20, -20..-10, -10..0, 0..10, >=10 dB; n 85, 123, 130, 100, 98): expert 0.671, 0.959, 0.946, 0.990, 1.000; dual-head 0.835, 0.992, 0.962, 1.000, 1.000. Right emitter among true chirps: expert 0.812, dual-head 0.868 (not plotted) | `labels/benchmark/report_v2/items.csv` (truth == chirp & scored; unweighted) |
| E, F decoder | sim (n 7,678): height median abs. error 9.54 Hz (6.7 %), FWHM 1.16 ms (9.8 %); real long chirps (n 513): median abs. difference to trace reference height 17.7 Hz (4.1 %), FWHM 24.5 ms (18.7 %) | `params/runs/v4-dualhead-sim-v4-r2/bundle/eval/predictions.parquet` (cal_* vs true_*), `real_agreement.csv` (detected) |

### Fig. 3 (failure modes)

- A: `2019-12-09-10_00`, 4312-4328 s (hand-labelled region 2,
  4260-4423 s; fish 747.7 and 727.3 Hz). 20 hand-labelled chirps, 21
  detections kept at >= 0.5; matched one-to-one by centre (<= 0.1 s): the
  label at 4320.31 s has no detection (detection 4080 scored 0.33), and two
  detections have no label: 4069 (4316.49 s, score 0.80, assigned to the
  727-Hz fish) and 4084 (4324.98 s, score 0.72, a second box 93 ms before
  the detection 4085 of a labelled chirp; emitters 3.3 Hz apart, not
  flagged as duplicate). Neither can be judged by eye in any of the three
  windows; the evaluation in `eval/real_labels_365` counts 4084 as a
  false positive. Labels:
  `eval/real_labels_365/labels/2019_tube_competition/2019-12-09-10_00/traces-grid1-editable-labels.csv`
  (the missed one is one of the 30 labels the author added in the v2 FP
  review, note `confirmed ...#v4fp-yolo-119`).
- B: `2019-12-04-10_00`, 7468-7488 s. Track 8 (~700 Hz) has its last
  detection at 7477.81 s and resumes at 7618.05 s (140-s gap); 8 chirps
  assigned in the window, 3 left unassigned (`beyond_tolerance`; emitter
  696-700 Hz, nearest live track 830 Hz). The recording has 97
  beyond-tolerance chirps of 1,100.
- C1: `2019-12-09-10_00` chirp 14783 (15388.57 s): fish 699.2 and 701.9 Hz
  (ΔF 2.7 Hz), emitter 700.5 Hz, confidence 0.53.
- C2: `2020-03-13-10_00` chirps 2974/2975 (9394.45/9394.52 s, 77 ms apart):
  fish 674.0 and 680.4 Hz (ΔF 6.4 Hz), emitters 676.5 and 677.4 Hz,
  assigned to different fish with confidence 0.72 and 0.56; shown in the
  51-ms channel, where the two chirps are separate.
- C3: `2019-12-09-10_00` detection 453 (1208.53 s), score 0.29, on a
  hand-labelled chirp (label added in the v2 FP review, note
  `#v4fp-yolo-18`).
- Prevalence in the 33 analysed trials (all `chirp` rows): 49,433 chirps;
  47,809 assigned, 1,579 beyond tolerance, 20 no live track, 25
  re-attributed harmonics; unassigned 3.2 %. Assigned chirps between fish
  < 10 Hz apart: 1,100 (2.3 %); assignment confidence < 0.9: 920 (1.9 %).

## Disagreements with the README / reports

1. **Assignment accuracy.** README, `design.md`, `limitations.md`,
   `pipeline.md`: 0.914 (0.875 below 10 Hz) and box-edge baseline 0.648.
   These come from `runs/dualhead_r18_sim_v4_r1/eval/sim_v4_test`
   (`emitter_by_df_bin.csv`; 8,802 TPs), an evaluation made before
   post-processing and before the matching rule compared labels by box
   midpoint (its label times are chirp centres; `contract.py` now uses box
   midpoints). Fig. 2C uses the post-processed evaluation that also gives
   the README's detection numbers (0.976 / 0.957): 8,999 TPs, emitter head
   0.921 (0.881 below 10 Hz), box edge + offset 0.651 (0.578 below 10 Hz).
   The README mixes the two evaluations. Small difference, same
   conclusion; the manuscript should quote one of them consistently.
2. **Benchmark strata.** `report_v2/report.md` gives recall per SNR bin
   reweighted to the population of visible test chirps (e.g. < -20 dB:
   expert 0.609, model 0.811). Fig. 2D plots unweighted proportions with
   Wilson 95 % intervals (0.671, 0.835). Overall unweighted recall (0.924,
   0.963) and right-emitter rate (0.812, 0.868) agree with the report.
3. **Decoder report caveat.** `bundle/eval/report.md` says under Caveats
   "Test chirps reuse the training templates ... the error on chirps of new
   fish is not measured", while its Summary says 0.0 % template reuse
   (fish-disjoint splits). The caveat looks stale; check before citing.
4. **Real long chirps.** On the 513 real long chirps the decoder's FWHM is
   a median 24.5 ms (18.7 %) longer than the trace reference (visible in
   Fig. 2F as orange points above the diagonal), against 13.6 ms simulated
   error in the same range (`report.md`, real check). In Fig. 1E the
   decoded heights lie above the extracted traces' peaks (69 vs 62 Hz, 409
   vs 379 Hz). Either the decoder overestimates on real chirps or the trace
   reference sits low (smoothing; the report notes the detector box top is
   35 Hz above the trace top on real chirps). Not resolved here.
5. **Real precision** (0.938) counts unlabelled faint chirps as false
   positives (labels prioritised visible chirps), and recall is measured on
   these labels; both stated in the caption draft.

## Choices worth a second look

- Fig. 1D duplicate comes from `2020-03-18-10_34` (a draw, not among the
  33 trials); the duplicates in the 33 trials I looked at were ambiguous
  (tall box vs small box with unclear truth).
- Fig. 1A: the snippet has a big short chirp, not a long one (none found
  with a small chirp between close fish and clearly visible); in the 13-ms
  window the beat of the two fish (29 Hz) and of the noise lines shows as
  vertical stripes.
- Fig. 3C1/C2 are ambiguous assignments, not verified errors (no emitter
  ground truth on real data); worded as such.
- Click-related detections (e.g. `2020-05-13-10_00` 8673.9 s, 67 % of the
  box repaired) could not be judged by eye and are not shown.
- The spectrograms were recomputed with the bundle's click removal; the
  production run's own chunking can differ in the last float digits (CUDA vs
  CPU), not visibly.

## Caption drafts

**Pipeline v2 (replaces the assignment figure).** Chirp detection, emitter
assignment and parameter estimation in pipeline v2, on recordings of
_Apteronotus leptorhynchus_ dyads. *A*, the spectrogram front end: the same
1.2-s snippet (two fish 29 Hz apart; a small chirp of the lower fish
followed by a large, short chirp of the upper fish) computed with three
analysis windows (205, 51 and 13 ms) on one frequency grid (4.88-Hz bins,
500-1500 Hz; shorter windows zero-padded, frames centred on the same
samples; power summed over 15 electrodes; display contrast per panel), and
the three stacked as the red, green and blue channels of the image the
detector receives (right). The long window separates the two fish's
baselines but smears each chirp over about 200 ms; the short windows merge
the baselines but resolve the chirps in time. The detector receives the
recording as such images in tiles of 3.2 s. *B*, one tile of another
recording (baseline EODfs 764 and 651 Hz) with the detector's output at the
operating threshold (score >= 0.5): each box is coloured by the fish it was
assigned to, circles mark the emitter baseline EODf the detector predicts
with each box, lines are the wavetracker tracks. A copy of the first chirp
on the second EOD harmonic (dashed) is removed. *C*, two chirps 61 ms apart
whose boxes both overlap the other fish's baseline: each box is assigned to
the live track nearest its predicted emitter EODf (3.4 and 0.4 Hz away),
provided it lies within 20 Hz (shaded). *D*, post-processing removes boxes
on the second harmonic of a tracked fish (left) and duplicate boxes on one
chirp (right; another recording). *E*, chirp parameters estimated by the
decoder for the first small chirp of B (height 69 Hz, FWHM 10.6 ms, 10 %
width 17.5 ms, undershoot 1.9 Hz) and the long chirp of C (409 Hz, 79.1 ms,
86.5 ms, 42.3 Hz), drawn on each chirp's instantaneous frequency relative to
its baseline, extracted from the raw recording (line: spatially filtered,
1-ms smoothed estimate; grey dots: frequency of single EOD periods). Width
bars are centred on the trace's own crossings.

(Note for the text: the multi-window front end is a design choice; no
ablation compared it with a single-window front end (docs/status.md,
"Front-end ablation"), so neither caption nor text should claim that the
extra channels improve detection.)

**Pipeline v2 performance (replaces the YOLOv8 cross-validation figure).**
*A*, precision against recall over the detection score threshold on 365
hand-labelled chirps in six regions of four held-out recordings (2019, 2020,
2022 tube competitions; no labelled recording was used for training or
model selection); circles, operating threshold 0.5. v2 dual-head (the
production detector): precision 0.938, recall 0.992; YOLO26s (box-only
reference): 0.947, 0.984; v1 (detector of the original analysis): 1.000,
0.701. The labels prioritised clearly visible chirps, so faint unlabelled
chirps count as false positives (precision is underestimated) and recall is
measured on the more visible chirps. A detection matched a label if its
centre lay within 100 ms and the boxes overlapped in frequency (one-to-one).
*B*, the same on simulated test scenes (300 one-minute hybrid scenes:
real background recordings and chirp templates of fish never used in
training; 9,401 visible chirps) after post-processing: dual-head 0.976,
0.957; YOLO26s 0.989, 0.952. *C*, fraction of detected simulated chirps
(true positives at 0.5) assigned to the correct fish, by the difference
between the emitter's baseline EODf and that of the closest other fish
(numbers: chirps per bin), for the dual-head's predicted emitter EODf and
for a box-only rule (track nearest to the box's lower edge + 41 Hz, the
median offset of the training chirps). Overall 0.921 vs 0.651 (n = 8,999);
below 10 Hz 0.881 vs 0.578. *D*, blind benchmark on 1,010 simulated items
(536 true chirps): recall of one expert labeller and of the dual-head
detector by chirp SNR (n per bin below). Overall recall 0.924 vs 0.963;
chirp found with the correct emitter 0.812 vs 0.868. *E*,*F*, decoded
(calibrated) chirp height and FWHM against the true values of 7,678
simulated test chirps of unseen fish (median absolute error 9.5 Hz, 6.7 %,
and 1.16 ms, 9.8 %), and against an independent frequency-trace estimate for
513 long chirps of the competition recordings (orange; median absolute
difference 17.7 Hz and 24.5 ms). Error bars: Wilson 95 % intervals.

**Pipeline v2 failure modes (replaces the v1 failure-mode figure).**
Spectrograms (205-ms window, power summed over electrodes) of two-fish
competitions with the v2 detections at score >= 0.5, coloured by the fish
they were assigned to (lines: wavetracker tracks; circles: predicted emitter
EODf; white: unassigned; dashed grey: removed by post-processing; dotted:
detection below the operating threshold). EOD$f_1$ is the higher-frequency
fish of each panel. *A*, a hand-labelled test region (2019-12-09; fish 21 Hz apart).
Arrowheads mark hand-labelled chirps. Of the 20 labelled chirps, 19 were
detected; the missed chirp was faint and scored 0.33 (dotted box). Two of
the 21 detections have no label (one assigned to EOD$f_2$ at 4.5 s, one a
second box on the chirp at 13.1 s); whether they are unlabelled chirps
cannot be decided from the spectrogram. *B*, the wavetracker track of
EOD$f_2$ ends at 7477.8 s (it resumes 140 s later) while the fish keeps
chirping: the later chirps are detected but stay unassigned, because no
live track lies within 20 Hz of their predicted emitter EODf. In the 33
analysed trials, 1,599 of 49,433 detected chirps (3.2 %) were unassigned.
*C*, examples of uncertain or failed cases. Left, a chirp between two fish
2.7 Hz apart, assigned with confidence 0.53: the emitter cannot be
identified by eye. Middle, two chirps 77 ms apart between fish 6.4 Hz apart
(51-ms window, in which they are separate), assigned to different fish with
confidence 0.72 and 0.56. In the 33 trials, 1,100 assigned chirps (2.3 %)
occurred between fish less than 10 Hz apart, where assignment was correct
for 0.881 of simulated chirps (Fig. 2C). Right, a faint hand-labelled chirp
scored 0.29 and missed. Scale bars: 1 s (A, B), 100 ms (C).
