#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cd "$REPO"
: "${STEPS:?set STEPS from the preflight timing}"
ATOMS=${ATOMS:-8192}
TEACHER_STEPS=${TEACHER_STEPS:-$STEPS}
common=(--steps "$STEPS" --teacher-steps "$TEACHER_STEPS" --max-atoms "$ATOMS"
        --train $(files $TRAIN) --validation $(files $VALIDATION) --holdout $(files $HOLDOUT))
CUDA_VISIBLE_DEVICES=0 nohup python -m ml.c3.loop "${common[@]}" \
  --checkpoint "$C3/out/runA.pt" > "$C3/out/runA.log" 2>&1 &
echo "Run A pid $!"
CUDA_VISIBLE_DEVICES=1 nohup python -m ml.c3.loop "${common[@]}" --train-encoder-layers 2 \
  --checkpoint "$C3/out/runB.pt" > "$C3/out/runB.log" 2>&1 &
echo "Run B pid $!"
