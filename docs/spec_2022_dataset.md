# Spec: making the 2022 competition dataset usable

Written 2026-11-19 for a new session. Author: Patrick Weygoldt (behavioural
neuroscience, first author of the manuscript in `manuscript/`). Read this whole
file, then `README.md`, `docs/plan.md`, `paper/README.md` and
`manuscript/CLAUDE.md` (style and claims rules of the manuscript) before you start.

## Why

The manuscript analyses chirps of *Apteronotus leptorhynchus* in staged two-fish
shelter ("tube") competitions recorded in 2019/2020 (33 analysed trials, 15 of
them with hand-annotated chases and contacts from video, BORIS). Its main
results: losers chirp far more than winners, chirping is coupled between
opponents, loser chirping peaks at the end of chases, and there is no detectable
effect of loser chirps on the winner's chasing (receiver effect). A second
dataset, the 2022 competitions, has everything the 2019 one has except the hand
annotations of chases and contacts. If it can be used, it gives a replication in
a different design (repeated bouts of the same pairs). Two things block it:

1. **Chirp detection on 2022 is much worse than on 2019** (numbers below).
2. **The 2022 video tracks are too fragmented** to derive chasing events from.

Extensive hand labelling (chases, contacts) is off the table. Small, targeted
labelling sessions by the author are fine (state the time they need up front).

## Ground rules (binding, from the author)

- Work in your own git worktree of this repository; commit there; push your
  branch as a backup; ask the author before merging into `main`. Never
  force-push. Commit messages end with
  `Co-Authored-By: Claude <model> <noreply@anthropic.com>`.
- Raw data are read-only. Never move or delete data. New files inside
  `/mnt/data2/2022_tube_competition/` only with the author's explicit OK; put
  derived data and models under `/mnt/data1/chirp_v2/` (detector material) or a
  new folder there (e.g. `/mnt/data1/chirp_v2/video2022/`), and analysis outputs
  under `paper/output/` (git-ignored). If a disk fills up, stop and tell the
  author; do not free space yourself.
- Never fabricate results, numbers or citations. Every number that may enter
  the manuscript must be written by the module that computes it
  (`paper/scripts/manuscript_stats.py: write_stats`, see `paper/README.md`,
  "Numbers into the manuscript").
- **The simulator is the ground truth for chirp detection; the author's labels on
  real recordings are not.** Real-recording labels measure *agreement* with the
  expert, used to choose thresholds. Never call them precision/recall on real
  data.
- Long jobs: detach with `systemd-run --user --unit=<name>` (unit names must
  not start with `chirpv2-`); jobs started from a Claude shell die with the
  session. Use `scripts/guarded` (memory/VRAM caps) for heavy steps. A watchdog
  (`oomwatch-chirp`) kills runaway jobs. CPU boost stays off
  (`/sys/devices/system/cpu/cpufreq/boost` = 0); the machine (Ryzen 5950X) has
  crashed at single-core boost transitions before.
- Hardware: RTX 4080 (16 GB; part is in use by the desktop), 62 GB RAM, 32 cores.
  Free disk: `/mnt/data1` ~13 TB, `/mnt/data2` ~2.8 TB.
- Shell aliases are odd: use `/usr/bin/ls`, `/usr/bin/rm -f`, `/usr/bin/grep -e`;
  never `pkill` a pattern that appears on the same command line.
- Do not touch `manuscript/`, the paper analysis chains in `paper/scripts/` or
  their outputs. Another session is finishing a rerun of the paper analyses
  (after an LED-alignment fix) from the main checkout; do not pull into or edit
  the main checkout while `systemctl --user is-active paper-ledfix-095331`
  says `active`.
- Simple, readable code; reusable tools with tests; human-readable diagnostic
  figures (one message per figure, plain-language title, labelled axes with
  units). Always look at your own figures before showing them.
- Report to the author at each milestone below and wait for decisions where
  marked.

## What exists

### The 2022 dataset: `/mnt/data2/2022_tube_competition/` (645 GB)

- `README.md`: `raw/` (voltage traces, videos), `interrim/` (frequency tracks,
  video tracks, old chirp detections), `processed/` (empty), `sleap_tracking/`.
- `raw/<rec>/` for 19 recordings (`2022-05-10-10_00` … `2022-07-07-10_00`):
  `traces-grid1.raw` (electrode recording), `fishgrid.cfg`, the video
  `<date>_0_converted.mp4` (2048×1536, 25 fps, ~4.05 h, i.e. shorter than the
  electric recording), `led_idxs.csv` (LED flashes as sample indices in the
  electric recording), `led_times.csv`, `traces-grid1-wavetracker/`. A
  recording-level `README.md` says "CHANGED THE CONFIG FILE TO 1x11 grid!!!":
  **the electrodes are a 1×11 line, a different geometry from 2019/2020.**
- `raw/meta.csv` (18 trials; `2022-07-07-10_00` is not in it): `Pairing`
  (`1a, 1b, 1c, …`), `Fish1/2`, pitches, `Winner_pitch`, `Win_ID`, `Lose_ID`
  (track ids in the wavetracker output), `Win_EODf`, `Lose_EODf`, rise counts,
  `light_sec` (10800: 3 h dark, then light, as in 2019). **Design: 6 pairs, each
  fought on 3 consecutive days, and the same fish won all three bouts in every
  pair** (6 independent dyads). Also `meta2.csv`, `meta.ods`, `weight_size.ods`,
  `Versuchszettel.odt`.
- `interrim/<rec>/` (19 recordings): wavetracker tracks `fund_v.npy`,
  `ident_v.npy`, `idx_v.npy`, `sign_v.npy`, `times.npy`; spectrograms
  (`spec.npy`, `fill_*`); `LED_frames.npy`, `LED_val.npy` (LED detected in the
  video); `meta.npy`; and the SLEAP export
  `competition.005_<date>_0.analysis.nix` (HDF5/NIX). In `2022-05-10-10_00`:
  465,404 instances with nodes head/centre/tail (`position` (n, 2, 3)),
  `centroid`, per-instance `track` id and scores; **9,479 track ids for 2 fish**;
  frames 1462–364739; both fish detected in 39.6 % of frames, one in 49.0 %,
  none in 11.4 % (data under `data/<video>/data_arrays/`, the frame index of
  each instance in `centroid/dimensions/1/link/<uuid>/data`).
- `sleap_tracking/`: `labels.v000.slp` (hand labels), `models/` (top-down
  centroid + centred-instance models trained on n = 6, 31 and 100 labelled
  frames), `predictions/` (12 `.slp`), `predicted_videos/`, `videos/` (also
  DeepLabCut pickles from an earlier attempt).
- `bsc_leonie_reetz.zip`: a BSc thesis on rises; `code/read_sleap_export.py`,
  `positions_nix.py` (how the NIX export was read), and
  `analysis_data/<rec>/` with fish-to-fish `distance.npy`, `min_distance.npy`,
  `distance_frame.npy`, `frame_times.npy` for 19 recordings.

### The 2019/2020 dataset: `/mnt/data2/2019_tube_competition/<rec>/`

Video `<date>_0.avi` (752×580, 25 fps, ~6.25 h; no video tracks), BORIS
annotations of chases and contacts for 15 analysed trials (onset/offset of
chases, contact points; the chasing fish is not recorded), `LED_frames.npy`,
`led_idxs.csv`, v2 chirp detections `chirps_rtdetr*.parquet`, frequency tracks.
This is the only place with hand-annotated chasing, so it is the validation set
for anything that detects chases automatically.

### Pipeline v2 (chirp detection), in this repository

- Packages (submodules): `packages/chirpdetector` (RT-DETRv2 detector with an
  emitter-EODf head, three-window spectrogram front end, post-processing,
  assignment to frequency tracks, chirp-parameter decoder; docs in
  `packages/chirpdetector/docs/`), `packages/gridtools` (simulation of hybrid
  scenes: real chirp templates on simulated EODs mixed into real recordings).
  Training backgrounds included one-minute segments of 2022 recordings.
- Production runs: `/mnt/data1/chirp_v2/detections/`:
  `competition2019_v2_20261113*` (2019/2020; `_gapfill`, `_gapfill_t080`,
  `_gapfill_t090` are the gap-filled tracks and threshold variants used by the
  paper) and **`competition2022_v2_t001`** (all 19 recordings of 2022, not
  gap-filled; per recording `chirps_rtdetr.parquet`, `chirp_times_rtdetr.npy`,
  `chunks_rtdetr.parquet`, `meta.json`, `qc/`, `windows/`; run-level
  `run.json`, `run.log`, `config.toml`, `candidates*.csv`, `columns.md`), plus a
  dry run.
- Real-recording labelling session `real_v1`: `/mnt/data1/chirp_v2/labels/real_v1/`
  (report `report/report.md`; scoring `experiments/real_labels/score_session.py`).
  The author judged candidate detections by score bin and marked chirps in
  random windows, blind, in a 5 s × 600 Hz spectrogram display. **The 2022 items
  are labelled too.** Agreement (share of detections the expert judged chirps,
  weighted, 95 % bootstrap CI):

  | threshold | 2019/2020 | 2022 |
  |---|---|---|
  | 0.5 | 0.83 | 0.37 [0.12, 0.72] |
  | 0.8 | 0.90 | 0.70 [0.37, 0.86] |
  | 0.9 | 0.94 | 0.92 [0.66, 0.99] |

  By score bin in 2022: 0.7–0.9 → 0.048 (3,017 detections in the session's
  population); ≥ 0.9 → 0.923 (1,831 detections; 2019/2020: 20,186). So in 2022
  almost everything below 0.9 is judged not to be a chirp.

### Video ↔ recording time

`paper/scripts/led_alignment.py` (`fit_led_alignment`, `check_led_alignment`)
maps video frames to recording samples from the LED pairs, robust to missed or
extra flashes and lost video frames (it replaced a first/last-LED straight line
that misaligned two 2019 trials by 1.4 s and 7.6 s). Use it for 2022 and check it
with the same diagnostics as `paper/scripts/plot_led_alignment.py` (LED
residual over the recording per trial).

## Goals

### A. Find and fix the 2022 chirp-detection problem

1. **Characterise the false positives.** Use the expert's 2022 decisions in
   `real_v1` (items judged "not a chirp" at scores 0.5–0.9, and missed chirps
   in the marked windows). Show example spectrogram cut-outs by score bin, and
   compare 2022 with 2019 on what the detector sees: electrode geometry (1×11
   line), signal-to-noise, clicks/artefacts, EOD harmonics, the number and
   spacing of fish EODfs (2022 pairs: ~630–660 Hz vs ~740–760 Hz), track quality
   and gaps (wavetracker on a line array), and the assignment (emitter EODf vs
   tracks). Find out which cause(s) explain the 0.7–0.9 bin.
2. **Check the simulator.** Does the training/simulation cover a 1×11 line and
   2022-like backgrounds? Measure detector performance on simulated scenes built
   on 2022 backgrounds with a 1×11 geometry (the benchmark code in
   `chirpdetector`/`experiments`), against the simulator ground truth.
3. **Propose fixes, ordered by cost**, and wait for the author's decision before
   implementing anything that changes the detector:
   - a 2022-specific post-processing rule or QC flag (if the false positives
     have a recognisable signature);
   - a 2022-specific operating threshold (0.9) as the fallback;
   - retraining/fine-tuning with 2022-like scenes. This would change the
     2019/2020 detections too unless kept as a separate model; any change must
     show no degradation on the 2019/2020 simulated benchmark and the
     2019/2020 agreement.
4. Validate the chosen fix on simulations (ground truth) and report agreement
   with the existing 2022 labels; if more labels are needed, prepare a small
   blind session in the existing labelling tool and state its size and time.

### B. Clean video tracklets for 2022 (and for 2019, for validation)

1. **Assess the SLEAP export**: per recording, frames with 0/1/2 fish,
   instance/node score distributions, tracklet lengths, gaps, where in the
   arena detections drop out (shelter tube, walls), and whether the low
   2-fish coverage is occlusion (fish in the tube) or missed detections.
2. **Mine training data from confident SLEAP detections** (high instance and
   node scores, two well-separated fish, temporally consistent), spread over
   recordings, times of day (dark/light) and arena positions. Add **targeted
   hand annotation** only for what mining cannot provide (hard frames: fish
   close together, crossing, in/at the tube, motion blur). Build or reuse a
   simple labelling tool; budget a few hundred frames and say how long the
   author needs.
3. **Train a recent YOLO pose model** (Ultralytics; check the current release
   and pick a size that fits the GPU), keypoints head/centre/tail, one fish
   class. Keep a held-out set of hand-labelled frames from recordings not used
   in training; report detection rate and keypoint error on it, against SLEAP.
4. **Link detections into tracklets.** Identity across the whole recording is
   not needed (the analyses do not rely on video identity; the 2019 annotations
   do not record the chaser either), but short, clean tracklets with few
   switches are. Use a standard tracker (e.g. ByteTrack/BoT-SORT on the pose
   detections) or simple motion-based linking; report tracklet lengths and a
   switch-rate proxy (e.g. implausible jumps). If useful later, identity can be
   recovered from the electrode line (EOD power along the 11 electrodes +
   EODf) rather than from the video.
5. **2019 videos** (752×580, different camera and resolution): check whether
   the model transfers; if not, add a small set of 2019 frames (hand-labelled or
   mined from a first pass) so that the same tracker runs on 2019. This is
   needed for C.
6. Outputs per recording: frames → instances (keypoints, scores, tracklet id),
   in a documented file format with a reader and tests, plus diagnostics
   (coverage over the recording, tracklet lengths, example frames with
   overlays).

### C. Later, only after A and B and the author's go: chasing from video

An identity-free chase detector from the two fish's tracks (distance, speeds,
heading, one fish following the other, the other fleeing), validated on the
2019 trials against the BORIS chases. The paper's analyses are locked to chase
onsets/offsets at the scale of ~0.5 s (loser chirping peaks at the chase
offset; analysis windows run from −3 to +13 s), so the detector is usable only
if its onsets/offsets match the annotations at that precision; report hit rate,
false chases and onset/offset timing errors per trial. Do not start C without
the author's decision.

## Milestones (report to the author at each; ⏸ = wait for a decision)

1. **A1** Diagnosis of the 2022 false positives, with example cut-outs and the
   2019 vs 2022 comparison. ⏸ choice of fix.
2. **B1** Assessment of the SLEAP export (coverage, failure modes, examples).
3. **B2** Mined training set and a prepared labelling session with its size and
   time estimate. ⏸ the author labels.
4. **B3** YOLO pose model trained and evaluated against SLEAP on held-out frames;
   tracklets for all 2022 recordings (and a first 2019 transfer check).
5. **A2** Fix implemented and validated (if chosen).
6. ⏸ Proposal for C, with what it would take.

## Open questions to settle early

- Why `2022-07-07-10_00` is not in `meta.csv` (control? extra bout?), and what
  `meta2.csv` adds.
- The video covers ~4.05 h of each recording: from which recording time (check
  with the LED alignment), and does it cover the whole 3-h dark phase?
- Which 2022 recordings were in the `real_v1` session, and whether its 2022
  counts cover all recordings.
- Whether the electric localisation on a 1×11 line is good enough to attach
  EOD identities to video tracklets if that is needed later.
