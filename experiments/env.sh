# Paths of this project for the shell scripts in experiments/ (source it).
# Same variables and defaults as paths.py; set them before sourcing to run
# elsewhere.
: "${CHIRP_DATA:=/mnt/data1/chirp_v2}"
: "${CHIRP_FAST:=$HOME/chirp_v2_fast}"
: "${CHIRP_REC_2019:=/mnt/data2/2019_tube_competition}"
: "${CHIRP_REC_2022:=/mnt/data2/2022_tube_competition}"
: "${CHIRP_V1_BENCHMARK:=$HOME/wrk/analyses/chirp_competition/detector_benchmark}"
CHIRP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CHIRP_DATA CHIRP_FAST CHIRP_REC_2019 CHIRP_REC_2022 CHIRP_V1_BENCHMARK CHIRP_REPO
