# Sourced by the pod scripts.
set -euo pipefail
C3=${C3:-/root/c3}
REPO=${REPO:-$C3/repo}
export HF_HOME=${HF_HOME:-$C3/hf}
export PYTHONPATH=$REPO
mkdir -p "$C3/train" "$C3/out" "$HF_HOME"
# The fixed split (ml/c3/holdout.py holds the holdout too).
TRAIN="000002 010042 010050 010082 010115 000123 010144 010003 010353 010127 000061 010045 010224"
VALIDATION="010087 010043 010149"
HOLDOUT="010172 010038 010387 010194"
files() { for d in "$@"; do echo "$C3/train/$d.json"; done; }
