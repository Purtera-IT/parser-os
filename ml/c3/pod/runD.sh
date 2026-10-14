#!/usr/bin/env bash
# Run D: the teach stage done right, on top of run C's pick. Run A's teach
# stage read the teacher on 6 parts, trained only those parts for ten passes,
# and lost task tier (1.000 to .656). Here the saved teacher gets one more
# warmed-up pass, is bound to run C's deal graph, gives views of 24 parts, and
# each lesson pass also takes gold steps on parts it has no view of.
#   INIT        run C's chosen checkpoint (required)
#   PICK_EXACT  its valid_exact on 010043 (required): the start check must
#               reproduce it, and the lesson may not lose more than .02 of it
#   TEACHER     saved teacher (default $C3/runA.pt.teacher)
#   SMOKE=1     every stage once on two parts, no gates; with
#               C3_SMOKE_LARGEST=1 the largest part, for the memory peak
source "$(dirname "$0")/common.sh"
cd "$REPO"
: "${INIT:?set INIT to run C's chosen checkpoint}"
: "${PICK_EXACT:?set PICK_EXACT to that checkpoint's valid_exact}"
ATOMS=${ATOMS:-8192}
TEACHER=${TEACHER:-$C3/runA.pt.teacher}
floor() { python -c "print(round($PICK_EXACT - $1, 4))"; }
args=(--max-atoms "$ATOMS" --train $(files $TRAIN) --validation $(files $VALIDATION)
      --init-from "$INIT" --train-encoder-layers 0
      --lr "${LR:-1e-4}" --warmup-passes "${WARMUP:-1}"
      --free-head all --head-weight col:label_type=20
      --loss-weight why_align=0.1 --loss-weight entities=0.05 --loss-weight hindsight=0.1
      --loss-weight pointers=0.1 --loss-weight near_misses=0.2
      --teacher-from "$TEACHER" --teacher-continue --teacher-steps 1
      --view-parts "${VIEW_PARTS:-24}" --lesson-gold "${LESSON_GOLD:-2}"
      --stage-steps "lines=0,links=0,cards=0,numbers=0,deal=0,open_links=0,bind=1,teach=${TEACH_STEPS:-2}")
if [[ "${SMOKE:-0}" == 1 ]]; then
  export C3_PROGRESS_EVERY=1
  python -m ml.c3.loop "${args[@]}" --smoke --steps 1 --checkpoint "$C3/out/smokeD.pt" \
    2>&1 | tee "$C3/out/smokeD.log"
  exit "${PIPESTATUS[0]}"
fi
nohup python -m ml.c3.loop "${args[@]}" --steps 1 \
  --check-every 1 \
  --gate "start.valid_exact>=$(floor 0.005)" \
  --gate "teach@1.valid_exact>=$(floor 0.02)" --gate 'teach@1.valid_jdg_task_tier>=0.9' \
  --gate "teach.valid_exact>=$(floor 0.02)" --gate 'teach.valid_jdg_task_tier>=0.9' \
  --checkpoint "$C3/out/runD.pt" > "$C3/out/runD.log" 2>&1 &
echo "Run D pid $!"
