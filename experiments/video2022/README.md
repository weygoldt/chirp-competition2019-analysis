# 2022 competitions: video tracks (milestones B1, B2 of `docs/spec_2022_dataset.md`)

Outputs under `/mnt/data1/chirp_v2/video2022/`. Use the root environment
(`/home/weygoldt/wrk/analyses/chirp_competition_upgrade/.venv/bin/python`).

## B1: the SLEAP export (2026-11-19)

Report: `/mnt/data1/chirp_v2/video2022/b1_sleap/report.md`, with its per-recording
table, 17 figures and example frames. Code: `sleap_nix.py` (reader of the NIX/HDF5
exports, tested), `recordings2022.py`, `b1_assess_sleap.py` (stages: tables,
background, locations, examples, figures, report). In short:

- **Video and recording time.** The LED map (`paper/scripts/led_alignment.py`)
  fits all 19 recordings to about one frame. Every video covers the whole
  electric recording.
- **Coverage.** SLEAP finds both fish in 46 % of the dark-phase and 20 % of
  the light-phase frames; some of these are one fish counted twice.
- **Tracks.** The track ids are fragments (median 1 s).
- **Missing fish.** About half of them are hidden (tube, blocks), the rest
  in view and missed.

## B2: training data for a pose model (2026-11-19)

### Frames and SLEAP's frame numbers

`frames.py` reads frames by index, either sequentially (exact by
construction) or by seeking (fast). `b2_frame_check.py` decoded three whole
videos (2022-05-10, 2022-06-20, 2022-07-06; output
`/mnt/data1/chirp_v2/video2022/b2/frame_check/`) and found seeking exact
(12 of 12 frames identical in each).

SLEAP's frame numbers agree with `*converted.mp4` only to within one frame.
For confident, fast-moving fish, the keypoints of SLEAP frame f are darkest
(i.e. on the fish) in video frame f or in f + 1:

| video | 0 | +1 | other offsets |
|---|---|---|---|
| 2022-05-10 | 61 | 37 | 10 |
| 2022-06-20 | 76 | 38 | 30 |
| 2022-07-06 | 13 | 11 | 2 |

In 2022-05-10 the mix runs through the whole video with no drift. I did not
find the cause; frames barely changing from one to the next occur, but in
runs that look like still fish, not clear duplicates. Consequences:

- **Mining** keeps only frames where both fish move at most 0.05 body
  lengths (~10 px) per frame. A one-frame offset then moves a mined label by
  less than the fish's body width.
- **Hand labels** are made on the converted frames themselves, so they are
  exact.
- **B3's comparison with SLEAP** must allow for it: score SLEAP's keypoints
  at offset 0 and at the better of 0 and +1, or on slow fish only.

### Mined frames (`b2_mine.py`)

`/mnt/data1/chirp_v2/video2022/datasets/mined_v1/` holds the Ultralytics YOLO
pose layout, `manifest.csv`, `summary.csv`, `dataset.yaml`, and the check
figures `check_crops.png` and `check_coverage.png` (`b2_check_dataset.py`).

A frame is mined when SLEAP found exactly the two fish:

- each fish with instance, head, centre and tail scores >= 0.5 and head-tail
  >= 0.5 body lengths;
- the two more than 2 body lengths apart, with no shared keypoint;
- both slow (see above);
- inside the electric recording.

Selection: at most 150 frames per recording, at least 2 s apart, spread
round-robin over light/dark x 15-min block x the arena cells of the two
fish. Labels: SLEAP's three keypoints (visible), and a box made of the
keypoints padded by 0.1 body lengths. The same box rule applies to
hand-labelled frames.

Splits by recording (`b2_mine.SPLITS`):

| split | recordings | use |
|---|---|---|
| test | 2022-05-11 (pair 1b), 2022-06-14 (6c), 2022-07-01 (4c) | never mined; hand-labelled evaluation (blind) |
| val | 2022-06-13 (5c), 2022-06-22 (3b) | mined + hand-labelled hard frames, for model selection |
| train | the other 14 | mined + hand-labelled hard frames |

The test recordings share their fish with other bouts of the same pairs, so
the test measures new recordings and days, not new individuals. That is the
use case: the model will run on these 19 recordings.

### The 2022 SLEAP hand labels (`b2_convert_sleap_labels.py`, `sleap_slp.py`)

`/mnt/data1/chirp_v2/video2022/datasets/sleap_v000/` is the YOLO version of
`sleap_tracking/labels.v000.slp`: 100 hand-labelled frames from May 2022
(project `/home/efish/leonie`) with an 8-node skeleton (snout, spine1-3, tail,
left/right fin, LED_on). It has the same layout as `mined_v1`, plus
`manifest.csv`, `summary.json` and the check figures.

The 3-node project behind the exports (`competition.slp`) and its labels are
not on this machine. I searched `/mnt/data1`, `/mnt/data2`, `/home/weygoldt`
and `/home/efish`.

- **Mapping:** snout -> head, spine1 -> centre, tail -> tail. spine1 lies 18 %
  of the way from snout to tail (median of the 149 fully labelled fish),
  exactly where the exports' "centre" lies.
- **Cleaning:** only user instances are kept. 32 duplicate instances (a
  second, near-identical instance on the same fish, shared body points within
  20 px) were dropped, leaving exactly 2 fish in every frame (198 fish; 148
  with a centre).
- **Video frame:** the labels index the original videos. Each frame is the
  one of f - 1, f, f + 1 in which the labelled body points are darkest: 0 in
  57 frames, +1 in 25, -1 in 17. The -1 choices gain a median of 2 grey
  levels (still fish), the +1 choices up to 37 (moving fish).
- **Splits:** 54 train frames (2022-05-10, 2022-05-12) and 45 **test**
  frames (2022-05-11, a held-out recording: never train on them; they are a
  second, independent test set from another labeller). One frame of 2022-05-03
  has no video on disk and was skipped. 11 frames are from the light phase.
- **Harder than the mined frames:** crossing and touching fish, fish under
  a block labelled by head and tail only, tails sticking out
  (`check_crops.png`).

### The labelling session (`b2_session.py`, `pose_labeller/`)

Session: `/mnt/data1/chirp_v2/video2022/labels/b2_v1/`. It holds
`items.json` (the queue), `images/`, and `truth.csv` (recording, frame,
stratum and SLEAP's instances per item; never served). Your labels go to
`labels.jsonl`.

- **Test items, blind (no points shown):** per test recording, 20 frames
  drawn uniformly over the electric recording (an unbiased sample, about a
  quarter of them in the light) and up to 18 hard frames.
- **Hard items, with SLEAP's points shown to correct:** per train / val
  recording, 2 frames from each hard stratum. The strata are: a dropout of
  >= 1 s, 0.5 s in, by where the lost fish was last seen (tube, block, edge,
  open water); two instances less than 1 body length apart; a random
  light-phase frame.

**Rule (yours, 2026-11-19):** label every visible fish. A partly hidden fish
is labelled by its visible part (e.g. only the head out of the tube; skip
the hidden points with `s`). A fish you cannot see is not labelled.

**The "centre" point is SLEAP's, not the middle of the body.** In the mined
frames it lies 18 % of the way from snout to tail (median of 3,098 fish;
10th-90th percentile 15-21 %): just behind the head, at the pectoral fins.
The hand labels use the same point, so mined and hand labels agree
(`check_crops.png` shows it).

Start, label, stop:

```sh
experiments/video2022/serve_b2.sh                  # user service pose-labeller-b2, port 8766
ssh -N -L 8766:localhost:8766 polarbear            # on the laptop, then open http://localhost:8766/
systemctl --user stop pose-labeller-b2             # stop (labels are saved at every Enter)
```

Keys are in the page's help panel (`h`):

- click places the active fish's next point (head, then centre, then tail);
- `s` skips a point that is not visible, `n` starts a new fish;
- drag moves a point, right click deletes one, `u` undoes;
- Enter marks the frame done, `x` marks it unusable;
- the arrows move to the previous / next frame, the wheel zooms;
- `c` turns the contrast on or off.

Tests: `test_pose_labels.py`, `test_labeller_server.py`, `test_labeller_page.py`
(runs `pose_labeller/test_labeller.mjs` under node: clicks, skips, drags,
deletes, undo and saves of the real page against a fake DOM), `test_frames.py`,
`test_sleap_nix.py`.

## B3: the 7-point pose model, trained in rounds (2026-11-19)

**Skeleton (your decision):** the 2022 SLEAP hand-label skeleton, `snout, spine1,
spine2, spine3, tail, left_fin, right_fin` (`pose_labels.SKELETON`). The flip map
swaps the fins (a mirror image swaps the fish's left and right; Ultralytics
applies it to horizontal and vertical flips). Older 3-point labels (mined
frames, session b2_v1) are read as snout / spine1 / tail; their other points
are "not labelled", which YOLO's keypoint loss ignores.

**The loop:**

1. `b3_dataset.py <name> mined_v1 sleap_v000 labels/<sessions>...` assembles a
   YOLO dataset. Every source keeps its labels in `labels.jsonl` + `index.csv`;
   images are symlinked; splits are by recording; blind test items go to test.
2. `b3_train.py <name>` trains YOLO26 pose with flips, any rotation, mosaic and
   brightness, and early stopping.
3. `b3_eval.py <name> --model <best.pt>` scores the model and SLEAP on the
   test frames: detection rate, false detections per frame, point error and
   PCK@0.1 body lengths. SLEAP is scored at its frame f and at the better of
   f / f - 1.
4. `b3_session.py <round> --model <best.pt> [--with-test]` builds the next
   correction session:
   - the blind test frames, never pre-filled by a model;
   - hard frames, pre-filled with your earlier labels or the model;
   - the model's most doubtful frames (not exactly two confident fish,
     ambiguous detections, weak points), pre-filled with the model.
   `b3_session_sheet.py` draws a session's pre-fill.
5. The labeller (`pose_labeller/`) takes the skeleton from the session's
   `session.json`. A click far from the active fish starts a new one.

Model (round 1, `runs/pose7_r1_yolo26l-pose_1280`): YOLO26l-pose at 1280 px,
batch 4 (11 GB), about 77 s per epoch including validation, up to 100 epochs,
patience 20. The first attempt used the x model (`runs/pose7_r1_yolo26x-pose_1280`,
stopped after 6 epochs and kept on disk). At 1280 px with 16 GB only batch 2
fits for x (9.2 GB). AutoBatch's probe pushed the GPU past the watchdog's
15 GB and was stopped, and at batch 2 an epoch takes about 2.6 min. The
switch to l was your call: with ~1,400 training frames capacity is not the
limit. Batch 4 gives steadier normalisation statistics. Inference over all
videos (~7 M frames) is faster.

**Caveat for the comparison with SLEAP:** the 3-node SLEAP model behind the
exports was most likely trained on `labels.v000.slp` (its "center" is exactly
spine1), probably including the 45 frames of 2022-05-11 we use as test. Its
scores on those frames are therefore optimistic. Your blind test frames
(b2_v1) are the fair comparison.

### Round 2 (2026-11-19)

- **Data:** `pose7_r2`: mined frames with the round-1 model's spine2/3
  (`b3_pseudo_inner.py`; 2,960 of 3,098 fish), old SLEAP labels, `b2_v1`, and
  the 210 frames of round 1. One label per frame (the latest session wins).
  Points inside the tube and the blocks are removed (`shelters.py`): your
  round-1 labels mixed both ways, deleting such points at first and
  estimating them later.
- **Training:** fine-tuned from round 1, YOLO26l-pose at 1280 px, batch 4,
  stopped at epoch 24 (your call). Model: `weights/pose7_r2_l.pt`.
- **Test:** `b3_eval.py pose7_r2`, your 112 blind test frames (134 visible
  fish) from the three held-out recordings:

| | round 1 | round 2 | SLEAP (best of f / f-1) |
|---|---|---|---|
| fish found | 0.59 | 0.87 | 0.81 |
| ... dark (105 fish) | | 0.87 | 0.76 |
| ... light (29 fish) | | 0.90 | 0.97 |
| false detections per frame | 0.07 | 0.00 | 0.33 |
| median error snout / spine1 / tail (body lengths) | 0.026 / 0.029 / 0.022 | 0.023 / 0.021 / 0.019 | 0.016 / 0.031 / 0.018 |

Round 2's other points: spine2 0.057, spine3 0.075, fins 0.08 body lengths
(share within 0.1 body lengths: 0.73, 0.69, 0.59). The fins rest on only
34 test labels.
