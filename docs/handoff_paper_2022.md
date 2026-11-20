# Handoff: run the 2019 paper analyses on the 2022 data

Written 2026-11-20 at the end of the `data2022` session (branch `data2022`, this file).
For a fresh session. Read the binding ground rules of the project spec first:
`.claude/worktrees/spec-2022/docs/spec_2022_dataset.md` (branch `worktree-spec-2022`).

## Goal

Run the paper's analysis and plotting pipeline (`paper/` in the main checkout:
`paper/run_all.sh`, chains `counts status agonistic interchase coupling gamma rates
receiver hurdle echo descriptive allocation posteriors hazard`) on the 2022 data,
**without mixing code or outputs with 2019**: same scripts, 2022 inputs, 2022-only
output location. The result is a 2022 report next to the 2019 numbers
(`paper/REPORT_t080.md`), not manuscript text.

## Rules (in addition to the spec)

- Work in your own worktree branched from `main` (it has `paper/`). Never edit
  `paper/scripts/`, `paper/inputs/`, `paper/output/`, `manuscript/`; never run
  `paper/export_figures.sh`; never set any output variable to a 2019 location.
- All 2022 adapter code in its own folder (e.g. `experiments/paper2022/`), with tests.
- All outputs under `/mnt/data1/chirp_v2/paper2022/` (dataset mirror and results).
  Raw data read-only; nothing new in `/mnt/data2/...`.
- If a 2019 script cannot run on 2022 without changes: do not edit it. Write down
  why and ask the author (an override copy in `experiments/paper2022/` is an option,
  but only on the author's go).
- Long chains (`gamma` ~2 h, `hazard`) detached with `systemd-run` + `scripts/guarded`;
  no GPU needed; look at every figure before reporting it.

## How the 2019 pipeline finds its data

`paper/scripts/paper_config.py`: environment variables `CHIRP_REC_2019` (dataset
root), `PAPER_DETECTIONS` (chirp file suffix), `PAPER_OUT` (output root),
`PAPER_FIGURE_DIR`, `PAPER_PRIMARY` (the run whose event rates set the offset
window). Per recording folder the scripts read:

| file | 2019 content | 2022 source |
|---|---|---|
| `fund_v.npy`, `ident_v.npy`, `idx_v.npy`, `times.npy` (+ check `rise_idx.npy`, `fishgrid.cfg`, others) | wavetracker tracks | gap-filled tracks `/mnt/data1/chirp_v2/detections/competition2022_v2_t001_gapfill/<rec>/tracks_gapfill.npz`; originals `/mnt/data2/2022_tube_competition/interrim/<rec>/` |
| `LED_frames.npy`, `led_idxs.csv` | LED sync | `interrim/<rec>/LED_frames.npy`, `raw/<rec>/led_idxs.csv` |
| `YYYY-MM-DD.csv` (BORIS: `Start (s)` video time, `Behavior` 0 = chase onset, 1 = offset, 2 = contact, `FPS`) | hand-annotated | **locked** model chases `/mnt/data1/chirp_v2/video2022/chases/model/model_v2/events.csv` (recording time, winner/loser chaser); convert back to video time with the frame clock `/mnt/data1/chirp_v2/video2022/b1_sleap/tables/frame_time/<rec>.npy` so the scripts' LED map (`led_alignment`) brings them back; verify the round trip |
| `chirp_times_<suffix>.npy`, `chirp_ids_<suffix>.npy` | chirp detections | corrected 0.8 dataset `/mnt/data1/chirp_v2/detections/competition2022_v2_gapfill_t080_corrected/<rec>/` (`chirp_times_rtdetr_t080.npy`, ids, weights) |
| root `trial_summary.csv` (recording, group, win/lose IDs, sex, size, EODf, chirp/chase counts, draw, ...) | 2019 table | build from `/mnt/data2/2022_tube_competition/raw/meta.csv` (+ `meta2.csv`, sizes); 18 trials; 2022-07-07 is a control, leave out |

## Open decisions (ask the author at the start)

1. **Contacts.** 2022 has no contact detection; only 30 hand-labelled contacts in
   clips (`chases/labels/chase_v2/labels.jsonl`). Skip the contact parts, or build a
   contact detector first?
2. **Chirp counts.** The scripts count detections (>= 0.8) without weights; 2022 has
   the per-fish correction (`p_chirp`, author's decision 2026-11-19). Default proposal:
   run on plain >= 0.8 detections (same as 2019), report corrected counts beside them.
3. **Hard-coded 2019 recordings** in 15 scripts (example trials, exclusions): which
   figures need 2022 counterparts, and how (override copies?).
4. **Dyads.** 2022 has 6 pairs x 3 bouts (a, b, c, consecutive days); 2019 dyads are
   mostly unique. Check how `group` / random effects are used; the repeated bouts may
   need their own grouping.
5. **Chains to skip:** `v2figs` (pipeline performance, not dataset-specific); others
   by the author's choice.

## Context worth reading

- `experiments/video2022/README.md` (branch `data2022`): pose model, full pose run,
  chase model, identities, the locked run and the chirp results (incl. 2019 vs 2022).
- `/mnt/data1/chirp_v2/video2022/chases/model/model_v2/LOCKED.md` (what the chases are,
  scores, checksums), `/mnt/data1/chirp_v2/video2022/identity/poses_r3/README.md`
  (winner/loser identities; branch `link2022`, not merged).
- `experiments/data2022/README.md`: the 2022 chirp dataset (threshold 0.8, per-fish
  correction, gap-filled tracks).

## Deliverables

The dataset mirror (`/mnt/data1/chirp_v2/paper2022/dataset/`, with a README and a
check that every file the scripts read is there), the outputs of each chain
(`/mnt/data1/chirp_v2/paper2022/output/<run>/`), a report `REPORT_2022.md` with the
2022 numbers next to the 2019 ones (`paper/REPORT_t080.md`) and every chain that did
not run with the reason, and the adapter code committed on its own branch.
