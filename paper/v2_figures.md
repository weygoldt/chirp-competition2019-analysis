# Pipeline-v2 figures (methods/supplement)

Three figures replace the v1 pipeline figures of the manuscript. Outputs
(PDF + 300-dpi PNG) in `/mnt/data1/chirp_v2/paper_rerun/rtdetr/figures/`;
all are 17.4 cm wide (the width of most manuscript figures; the template
scales them to the 16-cm text width). Earlier versions for comparison:
`figures/_draft1/` (first drafts, 2026-11-13) and `figures/_draft2/` (first
redesign, 2026-11-14, before the author's second design round).

| new figure | replaces | script | size |
|---|---|---|---|
| `plot_v2_pipeline_explainer` | `plot_assignment_feature_extraction.pdf` | `scripts/plot_v2_pipeline_explainer.py` | 17.4 x 15.3 cm |
| `plot_v2_performance` | `plot_pipeline_performances_manuscript_supplement_supplement.pdf` | `scripts/plot_v2_performance.py` | 17.4 x 12.4 cm |
| `plot_v2_failure_modes` | `plot_pipeline_performances_manuscript_supplement.pdf` | `scripts/plot_v2_failure_modes.py` | 17.4 x 12.05 cm |

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
   `/mnt/data1/chirp_v2/paper_rerun/rtdetr/data/processed/v2_figures/`).
   `raw` (new) caches 1.2 s of the raw voltage of one electrode for the
   "Recording" thumbnail of Fig. 1. The snippet `g_missed_1208` is no
   longer plotted.
2. Plot (paper env; ~20 s each on a cold LaTeX cache, ~3 s warm):

   ```sh
   cd paper && unset VIRTUAL_ENV
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

Written down from the paper's figures (`~/wrk/analyses/chirp_competition_paper/figures/`:
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
  YOLO26s teal, YOLOv8l grey, human expert coral (validator: all checks
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
  "YOLOv8l" (the YOLOv8-large detector of the previous analysis, "v1" in
  the evaluation files), "human expert". White labels that would sit on a
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
- Duplicate (D right): `2020-03-18-10_34` (a draw, not among the 33
  trials), chirp 49386 kept (score 0.86, box 629-1386 Hz), 49385 dropped
  (score 0.72, box 637-836 Hz).
- E draws each decoded width at 50 % / 10 % of the decoded height, centred
  on the trace's own crossings of that level (the decoder estimates no
  timing), and the height marker at the trace's peak time. Key: "trace",
  "estimate +- SD".

### Fig. 2 (performance)

Names in the figure: "RT-DETR dual-head" = `dualhead_r18_sim_v4_r1` (the
production detector); "YOLO26s (boxes only)" = `yolo26s_sim_v4_r1`;
"YOLOv8l" = `v1` (the detector of the previous analysis); "human expert" =
the blind labeller of the benchmark; "box position only" = nearest live
track to the box's lower edge + 41.15 Hz.

| panel | number | source |
|---|---|---|
| A real, n = 365 labels | dual-head P 0.938, R 0.992 (TP 362, FP 24, FN 3); YOLO26s 0.947 / 0.984 (359 / 20 / 6); v1 1.000 / 0.701 (256 / 0 / 109). Curves end at the lowest stored score (v1 0.27, YOLO26s 0.06, dual-head 0.05) | `eval/real_labels_365/<model>/items.csv` (counted); curves `pr_curve.csv` level "all" |
| B sim test after post-processing, 9,401 chirps | dual-head 0.976 / 0.957 (8,999 / 225 / 402); YOLO26s 0.989 / 0.952 (8,948 / 101 / 453) | `runs/<model>/eval_postprocess/sim_test_after/items.csv`; curves `pr_curve.csv` |
| C benchmark, 536 true chirps (1,010 items) | recall expert 0.924, dual-head 0.963; by SNR (<-20, -20..-10, -10..0, 0..10, >=10 dB; n 85, 123, 130, 100, 98): expert 0.671, 0.959, 0.946, 0.990, 1.000; dual-head 0.835, 0.992, 0.962, 1.000, 1.000. Right emitter among true chirps: expert 0.812, dual-head 0.868 (not plotted) | `labels/benchmark/report_v2/items.csv` (truth == chirp & scored; unweighted) |
| D correct emitter by ΔF (TP at >= 0.5, scenes with >= 2 live fish, n = 8,901) | emitter head: <10 Hz 0.881 (n 5,605), 10-25 0.982 (2,218), 25-50 0.994 (659), 50-100 1.000 (299), >=100 1.000 (120); box edge + 41.15 Hz: 0.578, 0.716, 0.836, 0.866, 0.983. Plotted population: head 0.920, box edge + offset 0.647. All 8,999 TPs (incl. 98 in single-fish scenes): head 0.921, box edge + offset 0.651, raw box edge 0.402 | `data/processed/v2_figures/assign_baseline.csv` (recomputed: kept predictions of `sim_test_predictions_flagged.parquet`, scene tracks, `assign_by_frequency`, items of `sim_test_after`) |
| E, F decoder | sim (n 7,678): height median abs. error 9.54 Hz (6.7 %), FWHM 1.16 ms (9.8 %); real long chirps (n 513): median abs. difference to trace reference height 17.7 Hz (4.1 %), FWHM 24.5 ms (18.7 %); median signed difference (decoder - trace) +8.2 Hz and +23.9 ms | `params/runs/v4-dualhead-sim-v4-r2/bundle/eval/predictions.parquet` (cal_* vs true_*), `real_agreement.csv` (detected) |

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
- B: `2019-12-04-10_00`, 7468-7488 s. Track 8 (the loser, ~700 Hz) has an
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
- Prevalence in the 33 analysed trials (all `chirp` rows): 49,433 chirps;
  47,809 assigned, 1,579 beyond tolerance, 20 no live track, 25
  re-attributed harmonics; unassigned 1,599 (3.2 %). Assigned chirps between
  fish < 10 Hz apart: 1,100 (2.3 %); assignment confidence < 0.9: 920
  (1.9 %).

## Disagreements with the README / reports

1. **Assignment accuracy.** README, `design.md`, `limitations.md`,
   `pipeline.md`: 0.914 (0.875 below 10 Hz) and box-edge baseline 0.648.
   These come from `runs/dualhead_r18_sim_v4_r1/eval/sim_v4_test`
   (`emitter_by_df_bin.csv`; 8,802 TPs), an evaluation made before
   post-processing and before the matching rule compared labels by box
   midpoint (its label times are chirp centres; `contract.py` now uses box
   midpoints). Fig. 2D uses the post-processed evaluation that also gives
   the README's detection numbers (0.976 / 0.957): 8,999 TPs, emitter head
   0.921 (0.881 below 10 Hz), box edge + offset 0.651 (0.578 below 10 Hz);
   it plots the 8,901 TPs of scenes with >= 2 live fish (0.920 / 0.647).
   The README mixes the two evaluations. Small difference, same
   conclusion; the manuscript should quote one of them consistently.
2. **Benchmark strata.** `report_v2/report.md` gives recall per SNR bin
   reweighted to the population of visible test chirps (e.g. < -20 dB:
   expert 0.609, model 0.811). Fig. 2C plots unweighted proportions with
   Wilson 95 % intervals (0.671, 0.835). Overall unweighted recall (0.924,
   0.963) and right-emitter rate (0.812, 0.868) agree with the report.
3. **Decoder report caveat.** `bundle/eval/report.md` says under Caveats
   "Test chirps reuse the training templates ... the error on chirps of new
   fish is not measured", while its Summary says 0.0 % template reuse
   (fish-disjoint splits). The caveat looks stale; check before citing.
4. **Real long chirps.** On the 513 real long chirps the decoder's FWHM is
   a median 24.5 ms (18.7 %) away from the trace reference, and longer in
   most chirps (median signed difference +23.9 ms; black points above the
   diagonal in Fig. 2F), against 13.6 ms simulated error in the same range
   (`report.md`, real check). In Fig. 1E the decoded heights lie above the
   extracted traces' peaks (69 +- 4 vs 62 Hz; 409 +- 34 vs 379 Hz, within
   1 SD for the big chirp, 1.8 SD for the small one). Either the decoder
   overestimates on real chirps or the trace reference sits low (smoothing;
   the report notes the detector box top is 35 Hz above the trace top on
   real chirps). Not resolved here; Fig. 1E shows both estimates side by
   side and does not adjust either.
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
- Fig. 1 rail: the decoder runs on every box together with the detector
  in production (step 2 of `chirpdetector run`); the rail draws it last
  because it is explained last. The caption says so.
- Fig. 1E: the legend's "trace" line is drawn in the small chirp's fish
  colour (the big chirp's trace is peach, its fish's colour).
- Fig. 2C plots only the human expert vs RT-DETR dual-head; the benchmark
  also scored YOLO26s (recall 0.974) and YOLOv8l (0.438)
  (`report_v2/report.md`), not plotted.
- Fig. 1 B/C display levels make the chirps visible by saturating the
  baselines; the relative brightness of baseline and chirp is therefore not
  shown (A shows it).
- Fig. 3B depends on the gap-filled run (see Fig. 3 above).
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
EOD$f$s 764 and 651 Hz) at the operating threshold (score >= 0.5): for each
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
one chirp (right, another recording; scores 0.86 and 0.72). *E*, chirp
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
figure).** *Performance of chirp detection (A-C), emitter assignment (D)
and parameter estimation (E, F).* Models as in the key: RT-DETR dual-head,
the detector used for all analyses (an RT-DETR detector with a second head
that predicts the emitter's baseline EOD$f$ for each box); YOLO26s, a
detector trained on the same data that predicts boxes only (reference);
YOLOv8l, the detector of the previous analysis; one human expert. Circles
in A and B, operating threshold (score >= 0.5). *A*, precision against
recall over the detection score threshold on 365 hand-labelled chirps in
six regions of four held-out recordings of _A. leptorhynchus_ dyads (2019,
2020 and 2022 competitions; no labelled recording was used for training or
model selection). At the operating threshold: RT-DETR dual-head precision
0.938, recall 0.992 (362 of 365 labels found); YOLO26s 0.947, 0.984;
YOLOv8l 1.000, 0.701. The labels prioritised clearly visible chirps, so
faint unlabelled chirps count as false positives (precision is
underestimated) and recall refers to the more visible chirps. A detection
matched a label if its centre lay within 100 ms and the boxes overlapped in
frequency (one-to-one). Curves end at the lowest stored score. *B*, the
same on simulated test scenes after post-processing (300 one-minute hybrid
scenes: real background recordings with chirp templates of fish never used
in training; 9,401 visible chirps): RT-DETR dual-head 0.976, 0.957;
YOLO26s 0.989, 0.952. *C*, blind benchmark on 1,010 simulated items (536
true chirps and 409 negatives scored): recall of the human expert and of
the RT-DETR dual-head detector by chirp signal-to-noise ratio (SNR; outer
bins open-ended; numbers, chirps per bin). Overall recall 0.924 (expert)
and 0.963 (detector); chirp found and assigned to the correct fish, 0.812
and 0.868. *D*, fraction of detected simulated chirps (true positives at
score >= 0.5 in scenes with at least two fish; n = 8,901) assigned to the
emitting fish, by the difference between the emitter's baseline EOD$f$ and
that of the closest other fish ($\Delta f$; numbers, chirps per bin), with
the predicted emitter EOD$f$ (nearest track, as in the pipeline) and with
the box position only (track nearest to the box's lower edge + 41 Hz, the
median offset of the training chirps). Overall 0.920 vs 0.647; $\Delta f$ <
10 Hz, 0.881 vs 0.578. *E*, *F*, estimated (calibrated) chirp height and
FWHM against a reference value: the true values of 7,678 simulated test
chirps of unseen fish (density; median absolute error 9.5 Hz, 6.7 %, and
1.16 ms, 9.8 %; dashed, identity), and an independent estimate from the
frequency trace for 513 long chirps of the competition recordings (black;
median absolute difference 17.7 Hz, 4.1 %, and 24.5 ms, 18.7 %; the
estimated FWHM is longer by a median 23.9 ms). Vertical lines: Wilson 95 %
confidence intervals. FWHM, full width at half maximum.

**Fig. 3, pipeline v2 failure modes (replaces the v1 failure-mode
figure).** *Where the pipeline fails on real recordings.* Spectrograms
(205-ms window unless stated; power summed over electrodes) of
_A. leptorhynchus_ competition recordings with the detections at score >=
0.5, drawn as in Fig. 1 (key at top): boxes coloured by the fish they were
assigned to (EOD$f_1$, the higher-frequency fish of each panel), white
boxes unassigned, circles the predicted emitter EOD$f$, dotted boxes
detections below the operating threshold; lines, frequency tracks. *A*, a
hand-labelled test region (fish 20 Hz apart; triangles above the image,
hand labels): 19 of the 20 labelled chirps were detected; the missed chirp
was faint and scored 0.33 (right, enlarged with its own display contrast).
Two of the 21 detections have no label (?; one assigned to EOD$f_2$, one a
second box 93 ms before the box of a labelled chirp); whether they are
unlabelled chirps cannot be decided from the spectrogram, and the
evaluation of Fig. 2A counts them as false positives. *B*, the frequency
track of EOD$f_2$ is interrupted for 140 s while the fish keeps chirping:
its chirps in the gap are detected but stay unassigned, because no track
lies within 20 Hz of their predicted emitter EOD$f$. [To be redrawn from
the gap-filled run; counts to follow.] *C*, assignments between fish less
than 7 Hz apart; numbers above the boxes, assignment confidence (posterior
probability of the chosen track under an error model fitted on simulated
data; an upper bound on real data). Left, one chirp between fish 2.7 Hz
apart, assigned with confidence 0.53: the emitter cannot be identified by
eye. Right, two chirps 77 ms apart between fish 6.4 Hz apart, shown in the
51-ms window, in which they are separate; assigned to different fish with
confidence 0.72 and 0.56. Real recordings have no emitter ground truth, so
these are uncertain assignments, not verified errors. In the 33 trials,
1,100 assigned chirps (2.3 %) occurred between fish less than 10 Hz apart,
where the predicted emitter EOD$f$ assigned 0.881 of simulated chirps
correctly (Fig. 2D), and 920 (1.9 %) had a confidence below 0.9; 1,599 of
49,433 detected chirps (3.2 %) were unassigned [before the gap fill].
Display levels: from a background percentile to a percentile of the
pixels inside the chirp boxes, so the baselines saturate; the enlargement
in A uses the levels of the main panel (C right: per-crop percentiles).
Scale bars: 1 s (A, B), 200 ms (A, right), 100 ms (C).
