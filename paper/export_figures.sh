#!/usr/bin/env bash
# Copy the primary run's figure PDFs into the manuscript (manuscript/figures/).
# The only way figures enter the manuscript; sensitivity runs never do.
#
#   paper/export_figures.sh [--dry-run]
#   PAPER_PRIMARY=rtdetr_sw_gapfill_t080 paper/export_figures.sh
set -euo pipefail
PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PAPER/.." && pwd)"
PRIMARY=${PAPER_PRIMARY:-rtdetr_sw_gapfill_t080}
SRC="$PAPER/output/$PRIMARY/figures"
DST="$REPO/manuscript/figures"
DRY=0; [[ ${1:-} == --dry-run ]] && DRY=1
FIGS=(
  plot_what_is_a_chirp_and_why_use_spetrograms
  plot_competition_trial_overview
  plot_chirp_distributions_manuscript
  plot_event_trial_progression_correlations_all_fish
  plot_event_trial_progression_correlations_all_fish_supplement
  plot_event_trial_progression_correlations_15_fish
  plot_chirp_winner_loser_trial_summary
  plot_event_time_rates_winner_loser_chirp
  plot_chirp_coupling
  plot_receiver_before_after
  plot_receiver_matched_checks
  plot_hurdle_decomposition
  plot_v2_pipeline_explainer
  plot_v2_performance
  plot_v2_failure_modes
  plot_chase_validation
)
[[ -d $DST ]] || { echo "no $DST (git submodule update --init manuscript)" >&2; exit 2; }
missing=0
for f in "${FIGS[@]}"; do [[ -f $SRC/$f.pdf ]] || { echo "missing: $SRC/$f.pdf" >&2; missing=1; }; done
(( missing )) && { echo "nothing copied" >&2; exit 1; }
for f in "${FIGS[@]}"; do
  if [[ -f $DST/$f.pdf ]] && cmp -s "$SRC/$f.pdf" "$DST/$f.pdf"; then echo "same     $f.pdf"; continue; fi
  state=$([[ -f $DST/$f.pdf ]] && echo updated || echo new)
  (( DRY )) || cp "$SRC/$f.pdf" "$DST/$f.pdf"
  printf '%-8s %s.pdf\n' "$state" "$f"
done
for g in "$DST"/*.pdf; do
  b=$(basename "$g" .pdf); [[ " ${FIGS[*]} " == *" $b "* ]] || echo "left     $b.pdf (not exported by this script)"
done
(( DRY )) && echo "(dry run: nothing copied)"
echo "source: $PRIMARY"
