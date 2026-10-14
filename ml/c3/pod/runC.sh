#!/usr/bin/env bash
# Run C: one GPU, the lines stage retrained with free answer heads and the
# answers' share of the loss raised, started from run A plus the heads fitted
# on the Mac (ml.c3.head_fit). The holdout is not read here; it is scored once
# afterwards with `python -m ml.c3.predict --score`.
#   SMOKE=1   every stage once on two parts, no gates (catches GPU-only errors)
#   INIT      start checkpoint (default $C3/runA.heads.pt)
#   TEACHER   saved teacher (default $C3/runA.pt.teacher)
source "$(dirname "$0")/common.sh"
cd "$REPO"
ATOMS=${ATOMS:-8192}
INIT=${INIT:-$C3/runA.heads.pt}
TEACHER=${TEACHER:-$C3/runA.pt.teacher}
args=(--max-atoms "$ATOMS" --train $(files $TRAIN) --validation $(files $VALIDATION)
      --init-from "$INIT" --train-encoder-layers 0
      --free-head all --head-weight col:label_type=20
      --loss-weight why_align=0.1 --loss-weight entities=0.05 --loss-weight hindsight=0.1
      --loss-weight pointers=0.1 --loss-weight near_misses=0.2
      --teacher-from "$TEACHER" --teacher-steps 1)
if [[ "${SMOKE:-0}" == 1 ]]; then
  export C3_PROGRESS_EVERY=1
  python -m ml.c3.loop "${args[@]}" --smoke --steps 1 --checkpoint "$C3/out/smokeC.pt" \
    2>&1 | tee "$C3/out/smokeC.log"
  exit "${PIPESTATUS[0]}"
fi
nohup python -m ml.c3.loop "${args[@]}" --steps 1 \
  --stage-steps lines=24,links=0,cards=0,numbers=0,deal=0,open_links=4,teach=2,bind=0 \
  --check-every 4 \
  --gate 'lines@8.fit_exact>=0.5' --gate 'lines@8.valid_exact>=0.58' \
  --gate 'lines@16.valid_margin>=0.10' \
  --checkpoint "$C3/out/runC.pt" > "$C3/out/runC.log" 2>&1 &
echo "Run C pid $!"
