# Pipeline-v2 figures (methods/supplement)

Three figures replace the v1 pipeline figures of the manuscript. Outputs
(PDF + 300-dpi PNG) in `paper/output/rtdetr/figures/`;
all are 17.4 cm wide (the width of most manuscript figures; the template
scales them to the 16-cm text width). Earlier versions for comparison:
`figures/_draft1/` (first drafts, 2026-11-13) and `figures/_draft2/` (first
redesign, 2026-11-14, before the author's second design round).

| new figure | replaces | script | size |
|---|---|---|---|
| `plot_v2_pipeline_explainer` | `plot_assignment_feature_extraction.pdf` | `scripts/plot_v2_pipeline_explainer.py` | 17.4 x 15.3 cm |
| `plot_v2_performance` | `plot_pipeline_performances_manuscript_supplement_supplement.pdf` | `scripts/plot_v2_performance.py` | 17.4 x 15.7 cm |
| `plot_v2_failure_modes` | `plot_pipeline_performances_manuscript_supplement.pdf` | `scripts/plot_v2_failure_modes.py` | 17.4 x 12.05 cm |

## Detections and output roots

The figures follow `PAPER_DETECTIONS` (`paper_config`), as the rest of the
paper rerun:

| `PAPER_DETECTIONS` | detections (per recording) | tracks | output root (`PAPER_OUT`) |
|---|---|---|---|
| `rtdetr_gapfill_t080` (the dataset the paper uses, 2026-11-16) | `detections/competition2019_v2_20261113_gapfill_t080/<rec>/chirps_rtdetr_gapfill_t080.parquet` (the gap-filled run's rows; `operating`/`chirp` recomputed at 0.8; `meta.json` `operating_threshold` 0.8) | as `rtdetr_gapfill` (the gap-filled run's `tracks_gapfill.npz`) | `paper/output/rtdetr_gapfill_t080` |
| `rtdetr_gapfill` (operating threshold 0.5) | `detections/competition2019_v2_20261113_gapfill/<rec>/chirps_rtdetr_gapfill.parquet` | wavetracker tracks, with the two trial fish's tracks replaced by `<rec>/tracks_gapfill.npz` (interior gaps 10-300 s with an EOD$f$ jump <= 15 Hz filled by linear interpolation; `filled` marks the interpolated samples) | `paper/output/rtdetr_gapfill` |
| `rtdetr` (production run, unpatched) | `detections/competition2019_v2_20261113/<rec>/chirps_rtdetr.parquet` | wavetracker tracks | `paper/output/rtdetr` |

`v2_fig_extract.py` caches the tracks of the selected run (field
`track_filled` in each snippet; `meta.detections`/`meta.run`), and the plot
scripts read `C.RUN`/`C.PARQUET`; interpolated track samples are drawn
dashed. The patch changes, among the examples of the three figures, only the
chirps of Fig. 3B (2019-12-04, chirps 4303-4305: unassigned -> fish 8); all
other shown rows have `gapfill_changed` = False, and Figs. 1 and 2 render
pixel-identical from both runs (Fig. 2 does not depend on the run at all:
simulations and the hand-label evaluation; `assign_baseline.csv` is
byte-identical in both roots). With `PAPER_DETECTIONS=rtdetr` the scripts
reproduce the kept `rtdetr` renders pixel for pixel (checked 2026-11-14).

**Operating threshold.** `C.OPERATING` is read from the run's `meta.json`
(`operating_threshold`; one value over all recordings, else an error),
never hard-coded. Everything that encodes the operating point follows it:
which boxes are drawn as detections (the run's `operating` column) and which
as "below threshold", the operating-point markers and their label in
Fig. 2C, the model's recall and emitter in Fig. 2D, E (benchmark: kept and
score >= threshold), the true positives of Fig. 2F (score >= threshold) and
the boxes of Fig. 2G, H (score >= threshold), and the counts printed for the
captions.

**What changes at 0.8 (`rtdetr_gapfill_t080`, 2026-11-16).**

- Fig. 1B: chirp 2088 (score 0.65) is no longer a detection; 2084-2087
  (0.84-0.95) are unchanged. Fig. 1D, duplicate: the earlier example's
  removed box (49385, 0.72) is below 0.8 and would not be in the output
  anyway, so the panel now shows `2019-12-02-10_00` chirps 244 (kept, 0.91)
  / 245 (removed, 0.87), snippet `f1_dup_c2` (proposed and accepted
  2026-11-16). The two other candidates checked (both >= 0.8, analysed
  trials) were fainter
: `2019-12-09-10_00`
  10851/10852 (0.83/0.80; `f1_dup_c1`) and `2019-11-27-10_00` 1461/1456
  (0.88/0.80; `f1_dup_c3`). Preview of all four:
  `/home/weygoldt/.claude/jobs/f8fa9695/tmp/figredesign/dup_candidates_t080.png`.
  Of the duplicate pairs with both boxes >= 0.8 there are 46 in the run,
  9 in the 33 analysed trials.
- Fig. 3A: the two detections without a hand label (4069, score 0.797;
  4084, 0.717) are below 0.8 and no longer drawn, so the "?" marks and their
  key entry disappear; 19 detections, all matched; 19 of the 20 labelled
  chirps still found (the faint one scored 0.33). B and C unchanged (all
  shown chirps >= 0.85). Prevalence (33 trials, `chirp` rows): 44,571
  chirps; 44,360 assigned, 199 beyond tolerance, 7 no live track, 5
  re-attributed; unassigned 206 (0.46 %); assigned between fish < 10 Hz
  apart 1,127 (2.5 %); confidence < 0.9 736 (1.7 %). 2019-12-04 at 0.8:
  1,040 chirps, 1 beyond tolerance.
- Fig. 2: see the Fig. 2 table (all simulated operating points at 0.8).

## How to run

1. Cache the raw-data inputs (upgrade repo env: chirpdetector/gridtools;
   CPU; ~15 s for `snippets traces raw`, ~1 min for `assign-baseline`,
   which loads the tracks of the 300 simulated test scenes and should go
   through `guarded`):

   ```sh
   cd paper
   /home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python scripts/v2_fig_extract.py snippets traces raw
   ~/wrk/analyses/chirp_competition_upgrade/scripts/guarded --name v2fig-assign --mem 16G --vram 0.2 -- \
     /home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python scripts/v2_fig_extract.py assign-baseline
   ```

   Writes `$PAPER_OUT/data/processed/v2_figures/` (default
   `paper/output/rtdetr/data/processed/v2_figures/`).
   `raw` (new) caches 1.2 s of the raw voltage of one electrode for the
   "Recording" thumbnail of Fig. 1. The snippet `g_missed_1208` is no
   longer plotted.
   For the gap-filled dataset prefix both commands with
   `PAPER_DETECTIONS=rtdetr_gapfill PAPER_OUT=paper/output/rtdetr_gapfill`
   (`run_all.sh` exports both; its `v2figs` chain runs the same steps).
2. Plot (paper env; ~20 s each on a cold LaTeX cache, ~3 s warm):

   ```sh
   cd paper && unset VIRTUAL_ENV
   # export PAPER_DETECTIONS=rtdetr_gapfill PAPER_OUT=paper/output/rtdetr_gapfill
   uv run python scripts/plot_v2_pipeline_explainer.py
   uv run python scripts/plot_v2_performance.py
   uv run python scripts/plot_v2_failure_modes.py
   ```

   Each prints the numbers it shows (JSON or tables), including the display
   levels used.

`scripts/v2_fig_common.py` holds paths, the palette, the symbol conventions
and drawing helpers (arrows, keys, thumbnails, zoom connectors, display
levels) of all three figures (paper env). Spectrograms are the production
front end (`span_spectrogram` with the released bundle: click removal on
>= 9 electrodes, 205/51/13-ms Hann windows on one 4.88-Hz grid, 5-ms hop,
500-1500 Hz, linear power summed over electrodes), normalised with the
front end's rule (60-dB range ending 10 dB above the median per-frame peak
of the 205-ms channel) over each snippet. Display levels: Fig. 1B-D and
Fig. 3A, B and C left use `chirp_levels()`: from the 85th percentile of the
crop (mostly background) to the 99th percentile of the pixels inside the
crop's chirp boxes, excluding +-15 Hz around the tracks, so that the chirps
fill the colour range and the baselines saturate (Fig. 1B and C share one
set, 0.27-0.63 of the 0-1 range; the draft's fixed 0.30-0.95 left the
chirps, at 0.26-0.62, in the dark half). Fig. 3A (main and zoom) uses
`background_levels()` instead (floor from the background pixels only; see
Fig. 3 below). Fig. 1A (single windows) and the 51-ms panel of Fig. 3C use
per-crop percentiles (40th-99.8th); the RGB image of Fig. 1A is on the front
end's own levels.

## Figure style

Written down from the paper's figures (`manuscript/figures/`:
`plot_competition_trial_overview`, `plot_what_is_a_chirp_and_why_use_spetrograms`,
`plot_chirp_coupling`, `plot_event_time_rates_winner_loser_chirp`,
`plot_obliterate_receiver_effect`, `plot_hurdle_decomposition`, and the v1
pipeline figures) and `scripts/style.py`; the v2 figures follow it, with
the author's second-round rules on top (below).

- **Size.** Full width 17.4 cm (printed at 16 cm, x0.92), heights 8-13 cm;
  half width 8.5 cm. 16-cm figures exist (what-is-a-chirp, event-time rates).
- **Type.** Noto Sans through LaTeX (`set_light_style`), 8-pt base: tick
  labels, axis labels and the few titles 8 pt; annotations, legends, keys
  and per-bin n "small" = 6.7 pt (6.1 pt printed); nothing smaller (checked
  programmatically: the v2 figures use only 6.7, 8 and 9.6 pt). Panel
  letters bold 9.6 pt ("large") from `letter_subplots`, at the upper left
  outside the axes.
- **Axes.** Left and bottom spines, `sns.despine(offset=3-5, trim=True)`.
  Spectrograms: only a frequency axis, offset from the image; no time axis
  but white scale bars inside the image ("200 ms"). Labels in sentence case
  with units in square brackets ("Frequency [Hz]", "Time [ms]"); EOD$f$ in
  math; fish labelled at their bands (EOD$f_1$, EOD$f_2$).
- **Lines and markers.** Data lines 1-1.6 pt, reference lines 0.5-0.6 pt
  dashed or dotted black, axes 0.8 pt, ticks 3 pt. Points: circles of
  3.5-4.5 pt filled with the series colour and a thin black edge; an
  estimate with its interval as a white circle on a black line (forest
  plots); intervals without caps.
- **Colour.** Spectrograms in seaborn `mako`; series colours from the
  mako/crest/flare families (`style.py`); grey for null or reference.
- **Explaining with little text.** Direct labels at the data instead of
  legends; zoom rectangles joined to the enlarged panel by thin connector
  lines; frameless legends, at the figure's top or bottom when shared by
  several panels.

Rules of the v2 figures (`v2_fig_common.py`; second design round):

- **One palette** from seaborn `mako` (cool) and its companion `flare`
  (warm), the families of `style.py`: blue `#3e4d93` (mako 0.35), teal
  `#3eb4ad` (mako 0.70), peach `#eca479` (flare 0.05), coral `#e2655c`
  (flare 0.30), rose `#c2456a` (flare ~0.5), grey `#a0a0a0`. Roles:
  spectrograms in mako capped at 0.93 (nothing in the image is white, so
  white marks stand out); densities in mako reversed (light mint -> blue);
  the two fish warm on the cool spectrogram (EOD$f_1$ peach, EOD$f_2$ rose;
  dataviz validator on the spectrogram's black: CVD separation 21 dE,
  contrast >= 3:1); models on white: RT-DETR dual-head blue (thicker line),
  YOLO26s teal, human expert coral (validator: all checks
  pass; teal has 2.45:1 contrast on white and is always in the key);
  box-position-only assignment grey open circles; real decoder chirps black.
- **Spectrogram symbols** (Figs. 1 and 3; one key in Fig. 3): box in the
  colour of the fish it was assigned to (EOD$f_1$ = higher-frequency fish of
  the panel); circle = predicted emitter EOD$f$ in the box colour, drawn
  only if its centre is inside the image and never clipped; white box =
  unassigned; dashed grey = removed by post-processing; dotted white =
  below the operating threshold; black triangles above the image = hand
  labels; "?" = detection without a hand label; numbers above boxes =
  assignment confidence, in the box colour; tracks as thin lines in the
  fish colour.
- **Arrows**: one gentle curvature (`arc3`, rad +-0.2), small filled head,
  0.7 pt, a 2-pt gap at the text and a 2.5-4-pt gap at the target
  (`C.annotate`); connectors between thumbnails thin grey arcs.
- **Text**: no panel titles except where the panel is ambiguous without
  them (Fig. 1: window lengths, "detector input", "harmonic"/"duplicate";
  Fig. 2: "Real recordings"/"Simulated"); grouping by spacing, no brackets
  or stage headings; no project terms (v1/v2, production, sim_v4, decoder,
  labeller): models are "RT-DETR dual-head", "YOLO26s (boxes only)",
  "human expert". White labels that would sit on a
  faint noise line get a dark backing (`DARK_BACK`). Checked for every
  figure: no text overlapping another text (programmatic), no text touching
  a line, box or marker (300-dpi PNGs inspected region by region).
- **Markers** are never cut at a border (`clip_on=False` with axis margins).

## Redesign history

- **Round 1 (2026-11-14, `_draft2/`).** From the first drafts
  (`_draft1/`): a roadmap of stage boxes on top of Fig. 1, R/G/B chips and
  the window drawn to scale in A, direct labels instead of a floating
  legend, decoder estimates as markers +- SD next to the trace; Fig. 2
  grouped under stage brackets with one colour per model, bin-edge axes,
  a density for the decoder test chirps; Fig. 3 with one key, titles
  naming each failure and a zoom on the missed chirp.
- **Round 2 (author's design round, current).** The checklist and how each
  item was addressed are in the report of 2026-11-14; in short: one
  mako+flare palette across the three figures (fish peach/rose instead of
  Okabe-Ito orange/pink); the roadmap of grey boxes replaced by a rail of
  data thumbnails of the panels (raw voltage, RGB input, a detection, the
  assignment, the removed harmonic copy, the frequency trace) with thin
  grey arc connectors and one-word stage names; Fig. 1B and C cropped to
  the fundamentals (590-1190 Hz; the harmonic copy only in D) with
  data-derived display levels so that the chirps are clearly visible;
  titles, brackets and redundant notes removed ("in between", "stack",
  "removed" labels on the boxes, median-deviation text, Fig. 3 titles and
  the explanatory key line); Fig. 2 groups by spacing (A and B share the
  precision axis; D set apart from E, F), jargon-free names, the E/F key
  below the panels; Fig. 3 hand labels as black triangles above the image,
  labels next to the tracks they name, confidences in the box colour above
  the highest box.

## Numbers and their sources

### Fig. 1 (explainer)

Source: `detections/competition2019_v2_20261113/2020-05-29-10_00/chirps_rtdetr.parquet`
(rows printed by the script), tracks of that recording, traces from
`v2_fig_extract.py traces` (gridtools harvest extraction).

- Recording thumbnail: `raw.npz` (`v2_fig_extract.py raw`): the same
  snippet as A, the electrode with the largest RMS (channel 1 of 15),
  60 ms from 0.74 s into the snippet.
- Front end (A): `2020-06-05-10_00`, 89.9-91.1 s; fish tracks 860.1 Hz
  (track 14, EOD$f_1$) and 830.7 Hz (track 10, EOD$f_2$; medians over the
  snippet), ΔF 29 Hz; chirp 133 (EOD$f_2$, 90.28 s, score 0.96, height
  63.5 Hz, FWHM 10 ms) and chirp 134 (EOD$f_1$, 90.67 s, score 0.95, height
  387.7 Hz, FWHM 5 ms). No snippet in the 33 trials had a visible long
  chirp and a small chirp between fish < 35 Hz apart (the candidates in 2019
  recordings with long chirps showed them only faintly); the long chirp is
  in C and E. A1-A3 use per-panel display contrast; the RGB image is the
  detector's input on its own levels (R, G, B = 205, 51, 13 ms).
- Tile 10850.4-10853.6 s (tile 4932), fish tracks 764.0 Hz (track 10,
  EOD$f_1$) and 650.9-651.2 Hz (track 12, EOD$f_2$). B and C show
  590-1190 Hz (the fundamentals; the harmonic copy is shown only in D),
  display levels 0.266-0.629 (`chirp_levels` of the tile); D left
  0.234-0.530, D right 0.289-0.684.
- Chirp 2085 (B, E left): score 0.95, emitter 651.9 Hz, track 651.2 Hz;
  decoder height 69.4 +- 4.3 Hz, FWHM 10.6 +- 0.8 ms, 10 % width 17.5 +-
  1.0 ms, undershoot 1.9 +- 2.4 Hz (not drawn, < 5 % of the height); +-
  values are the calibrated SDs of the run (`*_sd_*`). Trace peak 61.7 Hz
  above the trace baseline 651.2 Hz (at -11.5 ms from the box centre).
- Chirp 2086 (C, E right): score 0.95, box 675-1167 Hz, emitter 760.6 Hz,
  track 764.0 Hz (3.4 Hz; other fish 650.9 Hz); height 409.3 +- 34.2 Hz,
  FWHM 79.1 +- 13.2 ms, 10 % width 86.5 +- 7.9 ms, undershoot 42.3 +-
  9.0 Hz. Trace peak 378.9 Hz above the trace baseline 764.1 Hz. The trace
  crosses 50 % of the decoded height at -47.2 and +28.3 ms (75.5 ms apart)
  and 10 % at -54.2 and +31.8 ms (86.0 ms apart).
- Chirp 2087 (C): score 0.91, box 611-766 Hz (contains both baselines),
  emitter 650.5 Hz, track 650.9 Hz (0.4 Hz); onset 61 ms after 2086's.
- Harmonic copy 2084 (D left): score 0.84, box 1261-1501 Hz, emitter
  1307.9 Hz (2 x 651.2 = 1302.5 Hz), `pp_harmonic_of_track` 12, dropped.
- Chirp 2088 (B, right): score 0.65, fish 12.
- Duplicate (D right; author's choice 2026-11-16): `2019-12-02-10_00`
  (an analysed trial), chirp 244 kept (score 0.91, box 698-802 Hz), 245
  removed as its duplicate (score 0.87, box 688-896 Hz), 41 ms apart; both
  above the operating threshold of 0.8; snippet `f1_dup_c2`, display levels
  0.33-0.70. Earlier example (until 2026-11-16, valid only at 0.5):
  `2020-03-18-10_34` chirps 49386 (0.86) / 49385 (0.72).
- E draws each decoded width at 50 % / 10 % of the decoded height, centred
  on the trace's own crossings of that level (the decoder estimates no
  timing), and the height marker at the trace's peak time. Key: "trace",
  "estimate +- SD".

### Fig. 2 (performance)

Framing (author, 2026-11-16, binding): on real recordings a human expert is
not ground truth, so A and B show **agreement with the expert** (used to
choose the operating threshold) and never precision/recall or accuracy;
only the simulated panels (C-H) give performance against truth. Names:
"RT-DETR dual-head" = `dualhead_r18_sim_v4_r1` (the detector of all
analyses); "YOLO26s (boxes only)" = `yolo26s_sim_v4_r1`; "human expert" =
the author labelling blind; "box position only" = nearest live track to the
box's lower edge + 41.15 Hz. Numbers at the operating threshold 0.8
(`rtdetr_gapfill_t080`); the old real-data panel (precision-recall on the
365 hand labels of `eval/real_labels_365`) is removed.

Layout: three rows grouped by spacing (real recordings A, B; simulated C-F
with the model key above them, E and F close together on shared ΔF and
y axes with one x label; chirp parameters G, H with their key at the
right); A: the two agreement measures over the score threshold (dashed line:
the operating threshold; key in the empty lower middle, left of the line);
B: winner / loser in the paper's status colours (`style.winner_color`,
`style.loser_color`). 17.4 x 15.7 cm.

| panel | number | source |
|---|---|---|
| A real (2019/2020 competitions), agreement over the score threshold (0.01, 0.05, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95) | detections the expert judged chirps / expert's chirps found: 0.5: 0.834 [0.729, 0.902] / 0.935 [0.852, 0.996]; 0.8: 0.899 [0.799, 0.961] / 0.863 [0.739, 0.963]; 0.9: 0.942 [0.828, 0.997] / 0.728 [0.598, 0.847]; candidates labelled at those thresholds 286, 146, 82 (451 in all); 125 windows with 130 expert chirps | `labels/real_v1/report/precision_at_threshold.csv` and `recall.csv`, rows `dataset == 2019`; design-weighted (inclusion probabilities of the stratified sample; windows: 1/pi), 95 % percentile bootstrap (2,000 resamples) within strata; `labels/real_v1/README.md` |
| B real, by assigned fish (winner / loser) and score bin 0.5-0.7, 0.7-0.9, >= 0.9 | judged chirps, winner: 0.066 [0.011, 0.145], 0.427 [0.157, 0.630], 0.972 [0.804, 0.995] (n 60, 60, 40); loser: 0.184 [0.016, 0.386], 0.698 [0.328, 0.885], 0.920 [0.731, 1.000] (n 30, 30, 30) | `labels/real_v1/report/precision_by_status.csv`, rows `dataset == 2019`, scores `0.5-0.7`, `0.7-0.9`, `0.9-1` (status from the gap-filled assignment) |
| C sim test after post-processing, 9,401 chirps | at 0.8: RT-DETR dual-head P 0.995, R 0.931 (TP 8,757, FP 44, FN 644); YOLO26s 0.997 / 0.919 (8,639 / 30 / 762) | `runs/<model>/eval_postprocess/sim_test_after/pr_curve.csv` (level "all"; the row at the first threshold >= 0.8; the evaluation re-matches at each threshold) |
| D benchmark, 536 true chirps (1,010 items) | recall expert 0.924, RT-DETR dual-head at 0.8 0.935; by SNR (<-20, -20..-10, -10..0, 0..10, >=10 dB; n 85, 123, 130, 100, 98): expert 0.671, 0.959, 0.946, 0.990, 1.000; detector 0.741, 0.968, 0.931, 1.000, 1.000. Right emitter among true chirps: expert 0.812, detector 0.851 (not plotted) | `labels/benchmark/report_v2/items.csv` (truth == chirp & scored; model: `dualhead_chirp` (kept, score >= 0.5) and `dualhead_score >= 0.8`; unweighted, Wilson 95 % intervals) |
| E benchmark, emitter among own detections, by the emitter's ΔF to the nearest other live fish (536 true chirps; 2 without ΔF left out) | expert (items called chirp, n 493) / detector (kept, score >= 0.8, n 499): 0-10 Hz 0.608 [0.520, 0.689] / 0.728 [0.648, 0.796] (n 125 / 136); 10-25 0.924 / 0.943; 25-50 0.986 / 0.986; 50-100 0.992 / 0.991; >= 100 0.986 / 1.000; overall 0.878 / 0.910. Chirps per bin 143, 114, 78, 124, 75 | `labels/benchmark/report_v2/items.csv` (`df_nearest_hz`, `human_emitter_ok`, `dualhead_emitter_ok`; unweighted, Wilson 95 % intervals) |
| F correct emitter by the emitter's ΔF to the nearest other live track (true positives with score >= 0.8 and a known ΔF, n = 8,652) | predicted emitter EODf: 0-10 Hz 0.780 [0.763, 0.795] (n 2,571), 10-25 0.963 (2,154), 25-50 0.992 (1,713), 50-100 0.996 (1,107), >= 100 0.997 (1,107); box position only: 0.380, 0.552, 0.806, 0.880, 0.968. Overall 0.923 vs 0.646 (all 8,748 TPs >= 0.8: 0.924 / 0.650). Until 2026-11-16 this panel was binned by the evaluation's ΔF (closest pair of any two live tracks; most scenes have 3-7 fish), which put easy chirps into the 0-10 Hz bin (then 0.884 vs 0.577) | `data/processed/v2_figures/assign_baseline.csv` column `emitter_df_bin` (TPs of the evaluation at 0.5, filtered to score >= 0.8; regenerated 2026-11-16 in `rtdetr_gapfill_t080`) |
| G, H decoder (boxes >= 0.8) | sim (n 7,395): height median abs. error 9.17 Hz (6.5 %), FWHM 1.12 ms (9.4 %); real long chirps (n 512): median abs. difference to the trace reference 17.7 Hz (4.1 %), 24.5 ms (18.7 %); median signed difference +8.2 Hz, +23.9 ms | `params/runs/v4-dualhead-sim-v4-r2/bundle/eval/predictions.parquet` (score >= 0.8), `real_agreement.csv` (detected, score >= 0.8) |

### Fig. 3 (failure modes)

Display levels: A (main and zoom) 0.324-0.579 from `background_levels`
(60th percentile of the background pixels only, i.e. outside every stored
box with score >= 0.2 and > 15 Hz from the tracks -> 97th percentile of the
pixels inside the chirp boxes away from the tracks). The earlier
`chirp_levels` floor (85th percentile of the whole crop, 0.519) lay above
the median chirp pixel (0.35) because the glow of the two baselines 20 Hz
apart fills > 15 % of this 310-Hz crop, so most chirps were nearly
invisible (`_draft3/`). The zoom uses the main panel's levels on purpose:
the missed chirp's pixels do not exceed the background of the same rows, so
it stays as faint relative to the detected chirps as it is (not
stretched). B 0.471-0.620 and C left 0.426-0.615 (`chirp_levels`), C right
per-crop percentiles (0.254-0.894): unchanged, their chirps read clearly.
No titles; the panels carry the minimal cues: "?", "score 0.33", "track
ends", "$\Delta f$ = 2.7 Hz", "$\Delta f$ = 6.4 Hz, 51-ms window",
"confidence 0.53", "0.72", "0.56".

- A: `2019-12-09-10_00`, 4312-4328 s (hand-labelled region 2,
  4260-4423 s; fish 747.7 and 727.3 Hz, 20 Hz apart). 20 hand-labelled
  chirps, 21 detections kept at >= 0.5; matched one-to-one by centre
  (<= 0.1 s): the label at 4320.31 s has no detection (detection 4080
  scored 0.33; zoom 4319.9-4320.7 s, 685-845 Hz, per-crop contrast), and
  two detections have no label ("?"): 4069 (4316.49 s, score 0.80, assigned
  to the 727-Hz fish) and 4084 (4324.98 s, score 0.72, a second box 93 ms
  before the detection 4085 of a labelled chirp; emitters 3.3 Hz apart, not
  flagged as duplicate). Neither can be judged by eye in any of the three
  windows; the evaluation in `eval/real_labels_365` counts 4084 as a false
  positive. Labels:
  `eval/real_labels_365/labels/2019_tube_competition/2019-12-09-10_00/traces-grid1-editable-labels.csv`
  (the missed one is one of the 30 labels the author added in the v2 FP
  review, note `confirmed ...#v4fp-yolo-119`).
- B, gap-filled run (`rtdetr_gapfill`, the paper's dataset): the same
  window; the gap of track 8 (7477.81-7618.05 s, 140.2 s; EOD$f$ 700.7 Hz
  before, 699.5 Hz after; action "filled") from
  `competition2019_v2_20261113_gapfill/2019-12-04-10_00/tracks_gapfill.npz`
  (gap table) is bridged by linear interpolation (dashed in the figure);
  the three chirps of the window inside the gap (4303, 4304, 4305) are now
  assigned to fish 8 (confidence 0.999, 0.999, 0.994); the window has 11
  assigned and no unassigned chirps; the recording has 12 beyond-tolerance
  chirps of 1,099 (source run: 97 of 1,100). The key of Fig. 3 lists only
  symbols drawn, so "unassigned" is absent in this version. Gap numbers
  (33 analysed trials): 1,328 interior gaps > 10 s in the trial fish's
  tracks, in 56 of 66 fish (`competition2019_v2_20261113/track_gaps/README.md`);
  1,275 filled (8.87 h), 53 left because the EOD$f$ jump exceeds 15 Hz,
  none longer than 300 s; unassigned chirps 1,599 -> 924 (whole
  recordings), 1,260 -> 761 (first 3 h)
  (`competition2019_v2_20261113_gapfill/README.md`).
- B, source run (`rtdetr`): `2019-12-04-10_00`, 7468-7488 s. Track 8 (the loser, ~700 Hz) has an
  interior gap 7477.81-7618.05 s (140.2 s; 18 chirps in the gap, all
  unassigned, all 18 assignable after linear interpolation of the track):
  `detections/competition2019_v2_20261113/track_gaps/gaps.csv` (the snippet
  cache holds the tracks only +- 15 s around the snippet). In the window: 8
  chirps assigned, 3 unassigned (`beyond_tolerance`; emitter 696-700 Hz,
  nearest live track 830 Hz). The recording has 97 beyond-tolerance chirps
  of 1,100. Gap numbers for the 33 trials (README of that directory,
  2026-11-14): 1,328 interior gaps > 10 s in 56 of 66 trial fish (median
  17.7 s, max 283 s); 733 chirps with onset in an interior gap left
  unassigned, 639 of which linear interpolation would assign. Whether
  tracks are interpolated is now decided (a dataset-only patch fills
  interior gaps <= 300 s with an EOD$f$ jump <= 15 Hz and re-runs the
  assignment, run `competition2019_v2_20261113_gapfill`); B keeps the
  current content until that run exists, then gets redrawn (and its counts
  updated) from it.
- C left: `2019-12-09-10_00` chirp 14783 (15388.57 s): fish 701.9 and
  699.2 Hz; ΔF at the chirp onset 2.67 Hz (`track_df_other_hz`; 2.64 Hz
  between the track medians over +- 5 s), emitter 700.5 Hz, confidence
  0.53. A detection below threshold (14782, score 0.19) at the same time is
  not drawn.
- C right: `2020-03-13-10_00` chirps 2974/2975 (9394.45/9394.52 s, 77 ms
  apart): fish 674.0 and 680.4 Hz (ΔF at onset 6.37 and 6.40 Hz), emitters
  676.5 and 677.4 Hz, assigned to different fish with confidence 0.72 and
  0.56; shown in the 51-ms channel, where the two chirps are separate.
- Assignment confidence: posterior probability that the chosen track is the
  emitter under a Student-t error model of the predicted emitter EOD$f$
  fitted on simulated test scenes (chirpdetector `docs/pipeline.md`); an
  upper bound on real data (tracks add the tracker's error).
- Prevalence in the 33 analysed trials (all `chirp` rows; printed by the
  script). Gap-filled run: 49,375 chirps; 48,425 assigned, 904 beyond
  tolerance, 20 no live track, 26 re-attributed harmonics; unassigned 924
  (1.9 %). Assigned chirps between fish < 10 Hz apart: 1,192 (2.5 %);
  assignment confidence < 0.9: 1,032 (2.1 %). Source run: 49,433 chirps;
  47,809 assigned, 1,579 beyond tolerance, 20 no live track, 25
  re-attributed; unassigned 1,599 (3.2 %); < 10 Hz apart 1,100 (2.3 %);
  confidence < 0.9 920 (1.9 %). (The gap-filled run has 58 more rows flagged
  by post-processing, hence 49,375 chirps.)

## Disagreements with the README / reports

1. **Assignment accuracy.** README, `design.md`, `limitations.md`,
   `pipeline.md`: 0.914 (0.875 below 10 Hz) and box-edge baseline 0.648.
   These come from `runs/dualhead_r18_sim_v4_r1/eval/sim_v4_test`
   (`emitter_by_df_bin.csv`; 8,802 TPs), an evaluation made before
   post-processing and before the matching rule compared labels by box
   midpoint (its label times are chirp centres; `contract.py` now uses box
   midpoints). Fig. 2F uses the post-processed evaluation that also gives
   the README's detection numbers (0.976 / 0.957): 8,999 TPs, emitter head
   0.921, box edge + offset 0.651. Below 10 Hz by the evaluation's ΔF
   (closest pair of any two live tracks): 0.881 / 0.578; by the emitter's
   ΔF to the nearest other live track (what Fig. 2F plots since
   2026-11-16): 0.774 / 0.382.
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
   a median 24.5 ms (18.7 %) away from the trace reference, and longer in
   most chirps (median signed difference +23.9 ms; black points above the
   diagonal in Fig. 2G), against 13.6 ms simulated error in the same range
   (`report.md`, real check). In Fig. 1E the decoded heights lie above the
   extracted traces' peaks (69 +- 4 vs 62 Hz; 409 +- 34 vs 379 Hz, within
   1 SD for the big chirp, 1.8 SD for the small one). Either the decoder
   overestimates on real chirps or the trace reference sits low (smoothing;
   the report notes the detector box top is 35 Hz above the trace top on
   real chirps). Not resolved here; Fig. 1E shows both estimates side by
   side and does not adjust either.
5. **Real data.** The former real-data panel (precision 0.938 / recall
   0.992 on the 365 hand labels) is replaced by the expert-agreement panels
   (author's framing of 2026-11-16: an expert is not ground truth on real
   data); the old numbers should not be quoted as precision or recall.

## Choices worth a second look

- Fig. 1D duplicate comes from `2020-03-18-10_34` (a draw, not among the
  33 trials); the duplicates in the 33 trials I looked at were ambiguous
  (tall box vs small box with unclear truth).
- Fig. 1A: the snippet has a big short chirp, not a long one (none found
  with a small chirp between close fish and clearly visible); in the 13-ms
  window the beat of the two fish (29 Hz) and of the noise lines shows as
  vertical stripes.
- Fig. 1 rail: the decoder runs on every box together with the detector
  in production (step 2 of `chirpdetector run`); the rail draws it last
  because it is explained last. The caption says so.
- Fig. 1E: the legend's "trace" line is drawn in the small chirp's fish
  colour (the big chirp's trace is peach, its fish's colour).
- Fig. 2D plots only the human expert vs RT-DETR dual-head; the benchmark
  also scored YOLO26s (recall 0.974) (`report_v2/report.md`), not plotted.
- Fig. 2 no longer shows the detector of the previous analysis (author's
  decision, 2026-11-14; its evaluation stays in `eval/real_labels_365/v1`).
- Fig. 1 B/C display levels make the chirps visible by saturating the
  baselines; the relative brightness of baseline and chirp is therefore not
  shown (A shows it).
- Fig. 3B (gap-filled run) shows a filled gap: how gaps are handled. A
  remaining unfilled gap (53 in the 33 trials, EOD$f$ jump > 15 Hz) would
  instead show a residual failure; not drawn (no example inspected).
- Fig. 3C left/right are ambiguous assignments, not verified errors (no
  emitter ground truth on real data); worded as such.
- Click-related detections (e.g. `2020-05-13-10_00` 8673.9 s, 67 % of the
  box repaired) could not be judged by eye and are not shown.
- The spectrograms were recomputed with the bundle's click removal; the
  production run's own chunking can differ in the last float digits (CUDA vs
  CPU), not visibly.

## Caption drafts

Note for the text: the multi-window front end is a design choice. The
front-end ablation of 2026-11-14 (`$CHIRP_DATA/runs/ablation_frontend/report.md`)
found no practically relevant benefit of the extra channels (<= 1 pp on
simulated data, none on real labels; one run per arm), so neither caption
nor text should claim that they improve detection.

**Fig. 1, pipeline v2 (replaces the assignment figure).** *Chirp
detection, emitter assignment and parameter estimation in recordings of
_Apteronotus leptorhynchus_ dyads.* Top, the stages of the pipeline,
previewed by thumbnails of the panels with the same letters: the grid
recording (raw voltage of one electrode, 60 ms), spectrograms (A),
detection (B), assignment to a fish (C), post-processing (D) and chirp
parameters (E). *A*, one 1.2-s snippet (two fish 29 Hz apart; a small chirp
of the lower fish, EOD$f_2$, then a big, short chirp of the upper fish,
EOD$f_1$; power summed over 15 electrodes) computed with three Hann windows
of 205, 51 and 13 ms (white curves, drawn to scale) on one frequency grid
(4.88-Hz bins, 5-ms steps; display contrast per panel). The long window
resolves the two fish's baseline EOD$f$s but blurs each chirp over about
200 ms; the short windows merge the baselines but resolve the chirps in
time. The detector receives the three spectrograms as the red, green and
blue channels of one image (right), in tiles of 3.2 s. *B*, detector output
on one tile of another recording (band of the fundamentals; baseline
EOD$f$s 764 and 651 Hz) at the operating threshold (score >= 0.8): for each
chirp a box and a predicted baseline EOD$f$ of the emitting fish (circle).
Boxes are coloured by the fish they were assigned to in C; lines, frequency
tracks of the two fish (wavetracker); dashed rectangle, the time range
shown in C. *C*, assignment of two chirps 61 ms apart: each box goes to the
track nearest its predicted emitter EOD$f$ (3.4 and 0.4 Hz away), provided
that the track lies within 20 Hz (shaded). The lower box spans both fish's
baselines, so the box alone would not identify the emitter. *D*,
post-processing removes boxes on the second harmonic of a tracked fish
(left: a copy of the first chirp of B, box 1,261-1,501 Hz, predicted
emitter 1,308 Hz, twice the 651-Hz baseline) and the weaker of two boxes on
one chirp (right, another recording; scores 0.91 and 0.87, 41 ms
apart). *E*, chirp
parameters estimated by a second network from each box (white circles and
black bars; +-1 calibrated SD as vertical lines and as grey patches at the
bar ends), next to the chirp's instantaneous frequency extracted
independently from the raw recording (coloured line: spatially filtered,
1-ms smoothed trace; grey dots: single EOD periods), relative to the
baseline before the chirp. Left, the first chirp of B: height 69 +- 4 Hz,
full width at half maximum (FWHM) 10.6 +- 0.8 ms, width at 10 % of the
height 17.5 +- 1.0 ms (undershoot 1.9 Hz, not drawn); right, the big chirp
of C: 409 +- 34 Hz, 79 +- 13 ms, 86 +- 8 ms, undershoot below the baseline
after the chirp 42 +- 9 Hz. Widths are drawn at 50 % and 10 % of the
estimated height and centred on the trace's own crossings of that level
(the network estimates no timing). The estimated heights lie above the
traces' peaks (62 and 379 Hz). The parameter network runs on every detected
box. Display levels in B-D: from the 85th percentile of the crop to the
99th percentile of the pixels inside the chirp boxes away from the
baselines, so the baselines saturate.

**Fig. 2, pipeline v2 performance (replaces the YOLOv8 cross-validation
figure).** *Agreement with a human expert on real recordings (A, B) and
performance on simulated recordings (C-H).* On real recordings the
expert's judgement is not ground truth, so A and B show agreement with the
expert, which we used to choose the operating threshold; the simulator's
ground truth gives absolute performance (C-H). Models as in the key:
RT-DETR dual-head, the detector used for all analyses (an RT-DETR detector
with a second head that predicts the emitter's baseline EOD$f$ for each
box); YOLO26s, a detector trained on the same data that predicts boxes only
(reference); one human expert. *A*, agreement over the detector's score
threshold on recordings of the 2019/2020 competitions analysed here (blind
labelling: candidates and windows shown without box, score or model output,
in the display of the simulated benchmark): share of detections the expert
judged chirps (451 detections sampled across score bins, weighted to the
recordings' detections) and share of the chirps the expert marked in 125
random 2-s windows (130 chirps) that the detector found. Error bars, 95 %
bootstrap intervals at 0.5, 0.8 (dashed, the operating threshold) and 0.9:
at 0.5, 0.83 [0.73, 0.90] of the detections judged chirps and 0.94 [0.85,
1.00] of the expert's chirps found; at 0.8, 0.90 [0.80, 0.96] and 0.86
[0.74, 0.96]; at 0.9, 0.94 [0.83, 1.00] and 0.73 [0.60, 0.85]. *B*, share of
detections the expert judged chirps by score bin, for detections assigned
to the trial's winner (n = 60, 60, 40) or loser (30, 30, 30), as in the
key: below a score
of 0.9 the expert judged winner-assigned detections chirps less often
(0.5-0.7: 0.07 [0.01, 0.15]; 0.7-0.9: 0.43 [0.16, 0.63]) than loser-assigned
ones (0.18 [0.02, 0.39]; 0.70 [0.33, 0.89]); at >= 0.9, 0.97 and 0.92. *C*,
precision against recall over the score threshold on simulated test scenes
after post-processing (300 one-minute hybrid scenes: real background
recordings with chirp templates of fish never used in training; 9,401
visible chirps); circles, operating threshold (score >= 0.8): RT-DETR
dual-head precision 0.995, recall 0.931; YOLO26s 0.997, 0.919. *D*, blind
benchmark on 1,010 simulated items (536 true chirps and 409 negatives
scored): recall of the human expert and of the RT-DETR dual-head detector
(score >= 0.8) by chirp signal-to-noise ratio (SNR; outer bins open-ended;
numbers, chirps per bin). Overall recall 0.924 (expert) and 0.935
(detector); chirp found and assigned to the correct fish, 0.812 and 0.851.
*E*, the same benchmark: fraction of each one's own detections (expert:
chirps called, n = 493; detector: score >= 0.8, n = 499) assigned to the
emitting fish, by the difference between the emitter's baseline EOD$f$ and
that of the nearest other fish ($\Delta f$; numbers, true chirps per bin;
two chirps without $\Delta f$ left out). Below 10 Hz the expert assigned
0.61 [0.52, 0.69] and the detector 0.73 [0.65, 0.80] correctly; above
25 Hz both $\geq$ 0.98. The expert gave the same emitter on repeated items
in 0.80-0.81 of cases (16 and 56 repeated pairs). *F*, fraction of detected
simulated test chirps (true positives with score >= 0.8 and a known
$\Delta f$; n = 8,652; 2,571, 2,154, 1,713, 1,107 and 1,107 per bin) assigned to
the emitting fish, by $\Delta f$ as in E (to the nearest other tracked fish),
with the predicted emitter EOD$f$ (nearest track, as in the pipeline) and
with the box position only (track nearest to the box's lower edge + 41 Hz,
the median offset of the training chirps). Overall 0.923 vs 0.646;
$\Delta f$ < 10 Hz, 0.780 [0.763, 0.795] vs 0.380 [0.362, 0.399]. *G*, *H*, estimated (calibrated) chirp height and
FWHM for boxes with score >= 0.8 against a reference value: the true values
of 7,395 simulated test chirps of unseen fish (density; median absolute
error 9.2 Hz, 6.5 %, and 1.12 ms, 9.4 %; dashed, identity) and an
independent estimate from the frequency trace for 512 long chirps of the
competition recordings (black; median absolute difference 17.7 Hz, 4.1 %,
and 24.5 ms, 18.7 %; the estimated FWHM is longer by a median 23.9 ms).
Error bars in D-F: Wilson 95 % confidence intervals. FWHM, full width at
half maximum.

**Fig. 3, pipeline v2 failure modes (replaces the v1 failure-mode
figure).** *Where the pipeline fails on real recordings.* Spectrograms
(205-ms window unless stated; power summed over electrodes) of
_A. leptorhynchus_ competition recordings with the detections at score >=
0.8 (the operating threshold), drawn as in Fig. 1 (key at top): boxes coloured by the fish they were
assigned to (EOD$f_1$, the higher-frequency fish of each panel), white
boxes unassigned, circles the predicted emitter EOD$f$, dotted boxes
detections below the operating threshold; lines, frequency tracks. *A*, a
hand-labelled test region (fish 20 Hz apart; triangles above the image,
hand labels): 19 of the 20 chirps the expert labelled were detected; the
remaining one was faint and scored 0.33 (right, enlarged with the display
levels of the main panel). *B*, a gap in the
frequency track: the tracker lost EOD$f_2$ for 140 s while the fish kept
chirping. Such interior gaps of the two competitors' tracks (10-300 s, EOD$f$
change across the gap <= 15 Hz) were bridged by linear interpolation
(dashed) before assignment, so the chirps in the gap are assigned to the
fish. In the 33 analysed trials, the competitors' tracks had 1,328 interior
gaps longer than 10 s (56 of 66 fish); 1,275 were filled and 53 left
because the EOD$f$ changed by more than 15 Hz across the gap. *C*,
assignments between fish less
than 7 Hz apart; numbers above the boxes, assignment confidence (posterior
probability of the chosen track under an error model fitted on simulated
data; an upper bound on real data). Left, one chirp between fish 2.7 Hz
apart, assigned with confidence 0.53: the emitter cannot be identified by
eye. Right, two chirps 77 ms apart between fish 6.4 Hz apart, shown in the
51-ms window, in which they are separate; assigned to different fish with
confidence 0.72 and 0.56. Real recordings have no emitter ground truth, so
these are uncertain assignments, not verified errors. In the 33 trials,
1,127 assigned chirps (2.5 %) occurred between fish less than 10 Hz apart,
where the predicted emitter EOD$f$ assigned 0.78 of simulated chirps
correctly (Fig. 2F), and 736 (1.7 %) had a confidence below 0.9; 206 of
44,571 detected chirps (0.5 %) remained unassigned.
Display levels: from a background percentile to a percentile of the
pixels inside the chirp boxes, so the baselines saturate; the enlargement
in A uses the levels of the main panel (C right: per-crop percentiles).
Scale bars: 1 s (A, B), 200 ms (A, right), 100 ms (C).
