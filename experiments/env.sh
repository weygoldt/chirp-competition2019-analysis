# Paths of this project for the shell scripts in experiments/ (source it).
# Same variables and defaults as paths.py; set them before sourcing to run
# elsewhere.
: "${CHIRP_DATA:=/mnt/data1/chirp_v2}"
: "${CHIRP_FAST:=$CHIRP_DATA}"   # datasets on the data drive; point at a fast-disk copy to retrain fast
: "${CHIRP_REC_2019:=/mnt/data2/2019_tube_competition}"
: "${CHIRP_REC_2022:=/mnt/data2/2022_tube_competition}"
: "${CHIRP_V1_BENCHMARK:=$HOME/wrk/analyses/chirp_competition/detector_benchmark}"
# gridtools has no template default since it went public: point it at library v4
: "${GRIDTOOLS_TEMPLATES:=$CHIRP_DATA/templates/v4}"
# field recordings for the background libraries (--root-* of gridtools.background build)
: "${CHIRP_FIELD_FLONA2023:=/mnt/data1/efish/fielddata/flona2023}"
: "${CHIRP_FIELD_FLONA2025:=/mnt/data1/efish/fielddata/flona2025_workspace/flona2025}"
: "${CHIRP_FIELD_IRIRI2026:=/run/media/weygoldt/data1/efish/fielddata/iriri2026/iriri2026_recordings}"
CHIRP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CHIRP_DATA CHIRP_FAST CHIRP_REC_2019 CHIRP_REC_2022 CHIRP_V1_BENCHMARK CHIRP_REPO \
    GRIDTOOLS_TEMPLATES CHIRP_FIELD_FLONA2023 CHIRP_FIELD_FLONA2025 CHIRP_FIELD_IRIRI2026
