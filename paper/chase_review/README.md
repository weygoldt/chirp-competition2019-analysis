# Chase review: the BORIS chase onsets and offsets of the 2019/2020 competitions

The losers' chirp rate in Fig. 6 C1 (`fig:event_correlations`) peaks right at the BORIS
chase offset. Whether that peak is before, at or after the end of a chase depends on how
the annotator placed the offsets, which we did not do ourselves. This browser tool shows
every annotated chase in the video, with the BORIS onset and offset and the winner's and
loser's chirps on a timeline, and lets you mark where *you* see each chase start and end.
The differences tell how the annotator defined onset and offset, and how precise they are.

Nothing in the dataset is changed: the BORIS files, videos and LED files are only read; your
marks go to a separate `revisions.jsonl`.

## Run

```sh
# once: the session (all chases of the 15 trials behind Fig. 6), ~10 s
paper/.venv/bin/python paper/chase_review/build_session.py offsets_v1
# start the server as a user service, open http://localhost:8769/
paper/chase_review/serve.sh offsets_v1
#   from a laptop: ssh -N -L 8769:localhost:8769 polarbear
# any time: the table of your reviews
paper/.venv/bin/python paper/chase_review/export_revisions.py offsets_v1
systemctl --user stop chase-review-offsets_v1   # stop the server
```

| file | what |
|---|---|
| `build_session.py` | the chases, LED maps, chirps and contacts → `session.json` (paper environment) |
| `server.py` | serves the page, cuts the video of each window on request, appends saves (standard library + ffmpeg) |
| `index.html`, `timemap.js` | the page; `timemap.js` = video frame ↔ recording time (the LED map) |
| `export_revisions.py` | `revisions.jsonl` → `paper/output/chase_review/NAME/revisions.csv` (one-chase reviews), `window_matches.csv` (blind window reviews matched to BORIS) + summary |
| `serve.sh` | `server.py` as a systemd user service (`chase-review-NAME`) |

A session lives in `/mnt/data1/chirp_v2/labels/chase_review_2019/NAME/`:
`session.json`, `revisions.jsonl` (one line per save, the latest line of a chase wins; never
edit or delete it) and `clips/` (the cut videos, a cache that may be deleted).
`build_session.py NAME --force` rebuilds `session.json` and keeps the revisions.

## What you see

- **Video** of the window: from 10 s before the BORIS onset to 10 s after the BORIS offset
  (recording seconds; `+10 s before` / `+10 s after` widen it). It is cut from the file BORIS
  was scored on (`<date>_N_mod.avi`). Brightness and contrast sliders only change the display.
- **Overview timeline** of the whole window and a **zoom** of ±2 s around the playhead with a
  tick per video frame. Lanes: *BORIS* (this chase light grey, other chases dark grey, contacts
  purple diamonds; dashed lines = BORIS onset and offset), *yours* (orange, with handles),
  *loser chirps* (teal) and *winner chirps* (green), as in Fig. 6. The axis is in recording
  seconds relative to the BORIS offset by default (as Fig. 6 C1); `t = 0 at` changes it.
- **Chirps are only shown, never edited**: they come from the electric recording
  (`chirp_times_rtdetr_gapfill_t080.npy`, the primary run), not from the video.
- The table gives BORIS vs your onset, offset and duration, and *yours − BORIS* in seconds
  and video frames.

## Reviewing

Your marks start on the BORIS marks. Step through the video, set the onset (`o`) and offset
(`f`) at the current frame or drag the orange handles, and save (`Enter`, opens the next
open chase). Saving unchanged marks records that you agree. `not a chase` (`x`), `onset
unsure` (`u`) and `offset unsure` (`i`) are saved with the marks, and so is an optional note.
`k` replays your offset (3 s before to 2 s after) at the chosen speed.

The list `Fig. 6 chases (first 3 h), random order` holds the 1,958 chases whose onset and
offset lie in the first 3 h (the centres the event rates use), in a fixed random order, so
the reviewed chases are an unbiased sample at every point. `all chases` adds the 42 later
ones; a single trial lists its chases in time order.

**Blind marking: every chase in the window.** To place onsets and offsets without the
annotator's times, untick `show BORIS markers` (`h`) *before* opening a chase: the BORIS
lane, times and durations are hidden, and the window is widened by 0-8 s at each end (fixed
per chase) so its edges do not give the BORIS times away. Chases come in bouts, so a window
often holds several, and blind you cannot know which one it was cut for. So blind you mark
**every chase you see in the window** (`o` at an onset, `f` at its offset, as often as
needed; drag handles to adjust, click a bar to select it, `Backspace` deletes the selected
one), or tick `no chase in this window` (`x`):

- a chase already running when the window starts: onset on the first frame (`Home`, `o`);
- a chase still running when it ends: `e` (offset on the last frame);
- `+10 s before` / `+10 s after` widen the window; a chase marked on the old edge moves to
  the new one, so mark its real end there.

Such a review is saved as `mode: "window"` with the list of chases (each with
`starts_before_window` / `ends_after_window` and the unsure flags). `export_revisions.py`
matches them to the BORIS chases in the window by overlap: `matched` (one-to-one), `merged`
(one of your chases spans several BORIS chases), `split` (several of yours in one BORIS
chase), `boris_only`, `yours_only`; shifts come from matched pairs only, and an end on the
window's edge (yours) or outside it (BORIS) is left out. The first blind reviews (one chase
per window, before 2026-11-23 13:00) stay as they were (`mode: "target"`).

To mark from the video alone, also untick `show chirps` (`c`); otherwise the chirp ticks can
pull your marks towards chirps, which would make a chirp-vs-offset comparison circular. Each
save records `blind` (BORIS markers hidden the whole time the window was open) and
`chirps_hidden`.

Keys: `space` play/pause, `←`/`→` one frame (`shift`: 10), `[`/`]` slower/faster, `o`/`f`
onset/offset here, `r` reset to BORIS, `q`/`w` jump to BORIS onset/offset, `1`/`2` to yours,
`k` replay offset, `u`/`i` unsure, `x` not a chase (blind: no chase in this window), `h` BORIS
markers, `c` chirps, `Enter` save and next open chase, `n`/`p` next/previous, `Esc` stop;
blind also `e` runs past the end, `Home`/`End` window start/end, `Backspace` delete the
selected chase.

## Times and frames

- **BORIS → frame → recording time** exactly as in the analysis: frame = `round(25 · t_BORIS)`,
  recording time by the LED map of `led_alignment.fit_led_alignment`. `build_session.py`
  checks every chase against `analyze_event_time_rates_trial.load_and_convert_boris_events`
  and stops on any difference. The same map converts your frames
  (`revisions.jsonl`: `onset_rec_s`, `offset_rec_s`, `d_onset_s`, `d_offset_s`).
- **The clips are frame-exact.** The `_mod.avi` files are h264 with B-frames in AVI, which
  stores no presentation times, so ffmpeg stamps frame k with time (k + 2) / 25: a plain
  seek to f / 25 lands on frame f − 2. The server seeks to (f + delay) / 25 with the delay
  read from each file (2 for all 15), and checks the frame count of every clip. Checked on
  the LED: in clips of 2020-05-15 the LED pixels match `LED_val.npy` (the per-frame LED trace
  behind `LED_frames.npy`) best at lag 0 (r = 0.994 vs 0.94 at ±1 frame); a synthetic test
  video covers it in `tests/test_chase_review.py`.
- **Caveat for the comparison.** If the player in BORIS stamped the `_mod.avi` frames the
  way ffmpeg does, it showed frame k at media time (k + 2) / 25, and every BORIS time is
  2 frames (77 ms) late relative to the frame the annotator saw. This tool shows the true
  frames, so a constant shift of about −2 frames in *yours − BORIS* would come from the
  player, not from the annotator's definition. Which player BORIS used in 2020 is not known.

## Known oddities in the BORIS files

- 2020-05-13 has 141 chase onsets but 140 offsets. The analysis pairs each onset with the
  first offset after it, so c019 (onset 1686 s) takes c020's offset (1756 s): a 70 s
  "chase", the longest in the data. The page flags such chases.
- The BORIS files do not record which fish chased ("No focal subject").
