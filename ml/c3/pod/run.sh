#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cd "$REPO"
: "${STEPS:?set STEPS from the preflight timing}"
ATOMS=${ATOMS:-8192}
TEACHER_STEPS=${TEACHER_STEPS:-$STEPS}
# STEPS_B: B's own step count (default STEPS). B's student steps cost about
# 2.5x A's, so a smaller STEPS_B lets both runs end together.
STEPS_B=${STEPS_B:-$STEPS}
common=(--teacher-steps "$TEACHER_STEPS" --max-atoms "$ATOMS"
        --train $(files $TRAIN) --validation $(files $VALIDATION) --holdout $(files $HOLDOUT))
CUDA_VISIBLE_DEVICES=0 nohup python -m ml.c3.loop --steps "$STEPS" "${common[@]}" \
  --checkpoint "$C3/out/runA.pt" > "$C3/out/runA.log" 2>&1 &
echo "Run A pid $!"
CUDA_VISIBLE_DEVICES=1 nohup python -m ml.c3.loop --steps "$STEPS_B" "${common[@]}" --train-encoder-layers 2 \
  --checkpoint "$C3/out/runB.pt" > "$C3/out/runB.log" 2>&1 &
echo "Run B pid $!"
