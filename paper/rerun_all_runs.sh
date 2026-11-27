#!/usr/bin/env bash
# The paper's three detection sets, one after the other: the primary run
# (PAPER_PRIMARY, every chain), then the sensitivity runs of Table S1 (0.5 and
# 0.9; every chain but v2figs: the pipeline figures belong to the primary run,
# and the sensitivity runs need the primary's offset window, offset_window.py).
# About 4 h per run on polarbear (the Bayesian fits). As a user service, from
# the repository root (boost off first, see the machine notes):
#
#   systemd-run --user --unit=paper-rerun-$(date +%m%d-%H%M) --working-directory="$PWD" \
#     -E PATH="$PATH" -E HOME="$HOME" "$PWD/paper/rerun_all_runs.sh"
#   tail -F paper/output/rerun_all_runs.log
#
# PAPER_PRIMARY (default rtdetr_sw_gapfill_t080) and PAPER_SENSITIVITY (default
# "<primary without _t080> <primary with _t090>") override the runs; arguments
# name the runs to do (default: all three; e.g. only the sensitivity runs).
# Each run's log: paper/output/<run>/rerun.log. The script stops at the first
# failing run.
set -euo pipefail
PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMARY=${PAPER_PRIMARY:-rtdetr_sw_gapfill_t080}
base=${PRIMARY%_t080}
read -r -a SENS <<< "${PAPER_SENSITIVITY:-$base ${base}_t090}"
LOG=$PAPER/output/rerun_all_runs.log
mkdir -p "$PAPER/output"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# run_all.sh's chains, without v2figs for the sensitivity runs
read -r -a ALL <<< "$(sed -n 's/^ALL=(\(.*\))$/\1/p' "$PAPER/run_all.sh")"
(( ${#ALL[@]} )) || { log "FAILED: no ALL=(...) in run_all.sh"; exit 1; }
SENS_CHAINS=()
for c in "${ALL[@]}"; do [[ $c == v2figs ]] || SENS_CHAINS+=("$c"); done

runs=("$@")
(( ${#runs[@]} )) || runs=("$PRIMARY" "${SENS[@]}")
[[ $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 0) == 0 ]] \
  || log "WARNING: CPU boost is on (echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost)"
log "=== runs: ${runs[*]} (primary $PRIMARY)"
for d in "${runs[@]}"; do
  if [[ $d == "$PRIMARY" ]]; then chains=("${ALL[@]}"); else chains=("${SENS_CHAINS[@]}"); fi
  mkdir -p "$PAPER/output/$d"
  log "$d: chains ${chains[*]}"
  if PAPER_DETECTIONS=$d PAPER_PRIMARY=$PRIMARY "$PAPER/run_all.sh" "${chains[@]}" \
      > "$PAPER/output/$d/rerun.log" 2>&1; then
    log "$d: done"
  else
    log "FAILED: $d (log $PAPER/output/$d/rerun.log)"; exit 1
  fi
done
log "=== done: ${runs[*]}"
