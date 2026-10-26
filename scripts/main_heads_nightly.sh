#!/bin/zsh
# Nightly correction loop for the C3 super heads' MAIN heads (dev only,
# report-only): PM corrections -> queue with atom path -> training rows ->
# refit of each main line head -> held-out report. It never serves a
# checkpoint; promoting a refit stays the reviewed step (held-out + full-line
# diff + golden 34/34). The SOW heads have their own loop (heads_nightly.sh).
#
#   scripts/main_heads_nightly.sh                    # what launchd would run
#   EXPORT_DIR=<dir> scripts/main_heads_nightly.sh   # offline: reuse an export (no storage)
#   NO_REFIT=1 ...                                   # queue + rows + report only
#   APPROVE=<approvals.json> ...                     # owner-approved drafts train
#
# Steps: 1 export (scripts/main_heads_export.sh, read-only)  2 queue
# (scripts/main_heads_queue.py)  3 rows (ml.c3.labels_from_corrections)
# 4 refit (ml.c3.slot_fit, one head at a time, epoch on val/)  5 held-out
# (ml.c3.main_heads_eval on holdout/)  6 report (ml.c3.main_heads_report).
#
# Fails LOUDLY: any missing storage, input or checkpoint ends the run with a
# "FATAL" log line and a non-zero exit (2 storage, 3 inputs, 4 checkpoint,
# 5 a step failed), never a quiet skip. The one deliberate skip (a dev deploy
# in progress; this Mac serves dev's Ollama) logs "SKIP" and exits 0.
set -u
ROOT=${MAIN_HEADS_ROOT:-$HOME/c3-local/main_heads_loop}
CKPT=${CKPT:-$HOME/c3-serve/c3_base/checkpoint.pt}
PY=${PY:-python3.11}
HEADS=${HEADS:-col:label_type col:admission col:supplier jdg:site_role jdg:facility read:address_level}
mkdir -p "$ROOT"
LOG=$ROOT/nightly.log
[ -n "${NO_LOG:-}" ] || exec >>"$LOG" 2>&1
STAMP=${STAMP:-$(date +%Y%m%d)}
RUN=$ROOT/runs/$STAMP
echo "== $(date -u +%FT%TZ) main_heads_nightly start run=$RUN"
fail() { echo "FATAL main_heads_nightly: $2"; echo "== $(date -u +%FT%TZ) main_heads_nightly FAILED ($1)"; exit "$1"; }
cd "$(dirname "$0")/.." || fail 3 "repo not found"

if [ -z "${EXPORT_DIR:-}" ] && gh run list -R Purtera-IT/parser-os-worker -w deploy-dev.yml -L 3 --json status \
     -q '.[].status' 2>/dev/null | grep -qE "in_progress|queued"; then
  echo "SKIP main_heads_nightly: dev deploy in progress"; exit 0
fi
[ -f "$CKPT" ] || [ -n "${NO_REFIT:-}" ] || fail 4 "no checkpoint at $CKPT"
command -v "$PY" >/dev/null || fail 3 "no $PY on PATH"

rm -rf "$RUN"; mkdir -p "$RUN"
EXPORT=${EXPORT_DIR:-$RUN/export}
if [ -z "${EXPORT_DIR:-}" ]; then
  PY=$PY zsh scripts/main_heads_export.sh "$EXPORT"; rc=$?
  [ $rc -eq 0 ] || fail $rc "export failed (see the export's FATAL line above)"
fi
CORR=$EXPORT/corrections.jsonl
[ -f "$CORR" ] || CORR=$EXPORT/_feedback/corrections
[ -e "$CORR" ] || fail 3 "no corrections in $EXPORT"
[ -d "$EXPORT/envelopes" ] || fail 3 "no envelopes folder in $EXPORT"

$PY scripts/main_heads_queue.py --corrections "$CORR" --envelopes "$EXPORT/envelopes" --out "$RUN/queue" \
  || fail 3 "queue step found nothing trainable"
$PY -m ml.c3.labels_from_corrections --queue "$RUN/queue/queue.jsonl" --envelopes "$EXPORT/envelopes" \
    --out "$RUN/rows" ${APPROVE:+--approve "$APPROVE"} || fail 3 "rows step wrote no record"

cur=$CKPT
if [ -z "${NO_REFIT:-}" ]; then
  for key in ${=HEADS}; do
    tag=${key#*:}
    tr=($($PY -c "import json,sys; h=json.load(open(sys.argv[1]))['heads'].get(sys.argv[2],{}); print(' '.join(h.get('train',{}).get('files',[])))" "$RUN/rows/index.json" "$key"))
    va=($($PY -c "import json,sys; h=json.load(open(sys.argv[1]))['heads'].get(sys.argv[2],{}); print(' '.join(h.get('val',{}).get('files',[])))" "$RUN/rows/index.json" "$key"))
    if (( ${#tr} == 0 || ${#va} == 0 )); then
      echo "no $key labels in train or validation; nothing to fit" > "$RUN/fit_$tag.log"
      echo "  $key: no trained labels in train (${#tr}) or val (${#va}); not fit"
      continue
    fi
    $PY -m ml.c3.slot_fit --checkpoint "$cur" --key "$key" --train $tr --validation $va \
        --out "$RUN/refit_$tag.pt" --epochs ${EPOCHS:-60} > "$RUN/fit_$tag.log" 2>&1 \
      || fail 5 "slot_fit $key failed (see $RUN/fit_$tag.log)"
    tail -1 "$RUN/fit_$tag.log" | sed "s/^/  $key: /"
    [ -f "$RUN/refit_$tag.pt" ] && cur="$RUN/refit_$tag.pt"
  done
  echo "$cur" > "$RUN/checkpoint.txt"
  ho=($RUN/rows/holdout/*.json(N))
  $PY -m ml.c3.main_heads_eval --base "$CKPT" --refit "$cur" --files $ho --keys ${=HEADS} --out "$RUN/holdout.json" \
    || fail 5 "held-out eval failed"
fi
$PY -m ml.c3.main_heads_report --run "$RUN" --heads ${=HEADS} > /dev/null || fail 5 "report failed"
echo "  report: $RUN/report.md"

# Keep two weeks of runs; refits are large.
ls -1d "$ROOT"/runs/* 2>/dev/null | sort -r | tail -n +15 | while read old; do rm -rf "$old"; done
echo "== $(date -u +%FT%TZ) main_heads_nightly done"
