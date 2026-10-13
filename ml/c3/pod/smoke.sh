#!/usr/bin/env bash
# Every stage once on two average-size parts, A and B side by side: catches
# GPU-only errors in minutes and prints warm per-part times.
source "$(dirname "$0")/common.sh"
cd "$REPO"
ATOMS=${ATOMS:-8192}
export C3_PROGRESS_EVERY=1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 5 > "$C3/out/smoke_mem.log" &
MEM=$!
common=(--smoke --steps 1 --max-atoms "$ATOMS"
        --train $(files $TRAIN) --validation $(files $VALIDATION) --holdout $(files $HOLDOUT))
CUDA_VISIBLE_DEVICES=0 python -m ml.c3.loop "${common[@]}" --checkpoint "$C3/out/smokeA.pt" \
  > "$C3/out/smokeA.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 python -m ml.c3.loop "${common[@]}" --train-encoder-layers 2 \
  --checkpoint "$C3/out/smokeB.pt" > "$C3/out/smokeB.log" 2>&1 &
B=$!
status=0
wait $A || status=1
wait $B || status=1
kill $MEM
for r in A B; do echo "== $r"; grep -E "smoke_parts|progress|stage|validation|saved|Error|error" "$C3/out/smoke$r.log" | tail -60; done
python - "$C3/out/smoke_mem.log" <<'PY'
import sys, collections
peak = collections.defaultdict(int)
for line in open(sys.argv[1]):
    i, mib = [x.strip() for x in line.split(",")]
    peak[i] = max(peak[i], int(mib.split()[0]))
print("peak GPU memory MiB:", dict(peak))
PY
exit $status
