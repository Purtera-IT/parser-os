# Sourced by the pod scripts.
set -euo pipefail
C3=${C3:-/root/c3}
# Ubuntu images (Azure) have python3 and no `python`.
if ! command -v python >/dev/null; then python() { python3 "$@"; }; export -f python; fi
REPO=${REPO:-$C3/repo}
export HF_HOME=${HF_HOME:-$C3/hf}
export PYTHONPATH=$REPO
# The pod sees the host's 96 cores but is given 24 vCPU, and two runs share
# them: uncapped BLAS threads oversubscribe the box (the test suite crawled).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$C3/train" "$C3/out" "$HF_HOME"
# The fixed split (ml/c3/holdout.py holds the holdout too).
# Run C (2026-10-14): no new deals to label, so two validation and two holdout
# deals train; 010043 checks the run and 010038 / 010387 stay the test.
TRAIN="000002 010042 010050 010082 010115 000123 010144 010003 010353 010127 000061 010045 010224 010087 010149 010172 010194"
VALIDATION="${VALIDATION:-010043}"   # several deals: space separated; VALID_FILE names their files
HOLDOUT="010038 010387"
files() { for d in "$@"; do echo "$C3/train/$d.json"; done; }
# Less fragmentation across the varied page and part sizes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
