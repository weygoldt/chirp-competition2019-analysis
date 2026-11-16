#!/usr/bin/env bash
# Regenerate every analysis, posterior and figure of the chirp competition
# manuscript, front to back, on the detections named by PAPER_DETECTIONS
# (default rtdetr = pipeline v2). Sequential; about 4 h on polarbear
# (32 cores), almost all of it in the Bayesian fits (see README.md).
#
#   paper/run_all.sh                 # everything
#   paper/run_all.sh status coupling # only some chains (names below)
#
# Every step goes through run.sh (paper env, cwd = $PAPER_OUT, Agg backend,
# log in $PAPER_OUT/logs/). Model fits and the long resampling steps run in a
# memory-capped systemd scope (scripts/guarded). The v2 pipeline figures
# need chirpdetector/gridtools: their extraction step runs in the upgrade
# repository's root environment (CHIRP_ROOT_PY, default <repo>/.venv/bin/python).
set -euo pipefail
PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PAPER/.." && pwd)"
: "${CHIRP_DATA:=/mnt/data1/chirp_v2}"
: "${PAPER_DETECTIONS:=rtdetr}"
mkdir -p "$PAPER/output"   # derived data stays inside the repository (git-ignored)
: "${PAPER_OUT:=$PAPER/output/$PAPER_DETECTIONS}"
: "${CHIRP_ROOT_PY:=$REPO/.venv/bin/python}"
export CHIRP_DATA PAPER_DETECTIONS PAPER_OUT

step() { "$PAPER/run.sh" "$@"; }
# heavy NAME MEM SCRIPT [ARGS...]
heavy() {
  local name=$1 mem=$2; shift 2
  "$REPO/scripts/guarded" --name "paper-$name" --mem "$mem" --vram 0 -- "$PAPER/run.sh" "$@"
}
tagged() { local tag=$1; shift; PAPER_LOG_TAG=$tag "$@"; }

check_disk() {
  local avail
  avail=$(df --output=avail -B1G "$CHIRP_DATA" | tail -1 | tr -d ' ')
  if (( avail < 50 )); then
    echo "run_all: only ${avail} GB free on $CHIRP_DATA; stopping (do not free space automatically)" >&2
    exit 3
  fi
}

# --- chains -------------------------------------------------------------------

chain_counts() {       # chirp counts per trial and status, v1 vs v2
  step compare_detection_counts.py
}

chain_status() {       # Fig. chirp_status (plot_chirp_winner_loser_trial_summary)
  step make_trial_summary.py
  step tool_pivot_trial_summary_to_long_format.py
  heavy status-glmm 28G stats_bayesian_glmm_chirps_status_v3.py
  step plot_chirp_winner_loser_trial_summary.py
}

chain_interchase() {   # obs.csv for coupling, receiver effect, random effects
  step summarize_submissive_chirp_effect_on_dominant.py
}

chain_coupling() {     # Fig. chirp_coupling
  heavy coupling 16G stats_chirp_coupling.py
  step plot_chirp_coupling.py
}

chain_gamma() {        # 59 Gamma GLMMs (~2 h): Fig. obliterate_receiver_effect, Fig. re_analysis
  heavy gamma-glmm 20G stats_chirp_affects_chasing.py
}

chain_rates() {        # Fig. event_time_rates_winner_loser_chirp (+ inputs of the receiver-effect figure)
  tagged chirps heavy etr-default 12G analyze_event_time_rates_trial.py
  tagged ag heavy etr-ag 12G analyze_event_time_rates_trial.py --ag-events
  heavy warped 12G analyze_warped_chase_chirp_rates_trial.py
  step plot_event_time_rates_winner_loser_chirp.py
  step plot_event_time_rates_winner_loser_chirp_src_paper.py
}

chain_receiver() {     # needs chain_gamma and chain_rates
  step plot_obliterate_receiver_effect.py
  step plot_re_analysis.py
}

chain_hurdle() {       # Fig. hurdle_decomposition (+ offset logistic regressions)
  step summarize_chirp_event_correlation_around_chasing.py
  heavy hurdle 24G stats_bayesian_hurdle_glmm_chirprate_deviation_during_chasing_offsets.py
  step plot_hurdle_decomposition.py
  heavy offset-logreg 24G stats_bayesian_logreg_loser_chirps_around_chasing_offset.py
  step make_chirps_around_chasing_obs.py   # reconstructed writer (audit O5)
  heavy before-after 16G stats_bayesian_logreg_loser_chirps_before_vs_after_chasing_offset.py
}

chain_echo() {         # echo-response numbers (no figure)
  step summarize_chirp_event_correlation_around_winner_chirps.py
  step summarize_chirp_event_correlation_around_loser_chirps.py
  heavy echo 16G stats_bayesian_logreg_echo_responses_new.py
}

chain_descriptive() {  # Figs. what is a chirp, trial overview, chirp distributions
  step collect_chirp_overview_plot_data.py
  step plot_what_is_a_chirp_and_why_use_spetrograms.py
  step make_competition_overview.py
  step plot_competition_trial_overview.py
  step plot_chirp_distributions_manuscript.py
}

chain_allocation() {   # Figs. temporal allocation, all fish (+ supplement) and 15 fish
  tagged all_fish heavy progression 12G analyze_event_trial_progression_trial.py
  tagged annotated_subset heavy progression-annotated 12G analyze_event_trial_progression_trial.py --annotated-only
  step summarize_all_timing_distributions.py
  step plot_event_trial_progression_correlations_all_fish.py
  step plot_event_trial_progression_correlations_15_fish.py
}

chain_v2figs() {       # new figures of pipeline v2 (replace the v1 assignment and performance figures)
  ( cd "$PAPER" && PAPER_OUT="$PAPER_OUT" "$CHIRP_ROOT_PY" scripts/v2_fig_extract.py snippets traces raw )
  ( cd "$PAPER" && "$REPO/scripts/guarded" --name paper-v2fig-assign --mem 16G --vram 0.2 -- \
      env PAPER_OUT="$PAPER_OUT" "$CHIRP_ROOT_PY" scripts/v2_fig_extract.py assign-baseline )
  step plot_v2_pipeline_explainer.py
  step plot_v2_performance.py
  step plot_v2_failure_modes.py
}

chain_posteriors() {   # every reported posterior number (needs all fits)
  step summarize_reported_posteriors.py
}

ALL=(counts status interchase coupling gamma rates receiver hurdle echo descriptive allocation v2figs posteriors)
chains=("$@")
(( ${#chains[@]} )) || chains=("${ALL[@]}")
for c in "${chains[@]}"; do
  declare -F "chain_$c" > /dev/null || { echo "run_all: unknown chain '$c' (known: ${ALL[*]})" >&2; exit 2; }
done
for c in "${chains[@]}"; do
  check_disk
  echo "=== [$(date -Is)] chain $c"
  "chain_$c"
done
echo "=== [$(date -Is)] done: figures in $PAPER_OUT/figures"
