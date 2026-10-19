#!/bin/zsh
# Nightly "heads decide, rules retire" run on the Mac Studio (dev only). Report-only:
# it refits the SOW heads into a dated folder and publishes the per-rule report
# SowSmith's Heads panel reads. It never serves a checkpoint; promoting a refit
# stays a reviewed step (held-out deals + full-line diff + golden 34/34).
#
#   scripts/heads_nightly.sh            # what launchd runs
#   DRY=1 scripts/heads_nightly.sh      # no publish
#
# Inputs, all outside the repo:
#   ~/c3-local/heads_loop/deals.txt     "<deal> <uuid>" per line (the graded set)
#   ~/c3-local/heads_loop/rubric/*.A.jsonl   rubric labels (gold)
#   ~/c3-local/heads_loop/real_sow/sem_g*.jsonl  real-SOW alignments (gold)
# Each envelope carries c3_sow.rule_decisions, so labels grow with every compile.
set -u
ROOT=$HOME/c3-local/heads_loop
LOG=$ROOT/nightly.log
exec >>"$LOG" 2>&1
echo "== $(date -u +%FT%TZ) heads_nightly start"
cd "$(dirname "$0")/.."

# The Mac Studio also serves dev's Ollama: never train while a dev deploy or a
# heavy local job is running.
if gh run list -R Purtera-IT/parser-os-worker -w deploy-dev.yml -L 3 --json status -q '.[].status' 2>/dev/null | grep -qE "in_progress|queued"; then
  echo "dev deploy in progress; skipping tonight"; exit 0
fi

C=$(az storage account show-connection-string -g purtera-dev-rg -n purpulsedevstg01 -o tsv 2>/dev/null)
[ -n "$C" ] || { echo "no storage access (az login?); skipping"; exit 1; }
ENV_DIR=$ROOT/envs/nightly
rm -rf "$ENV_DIR"; mkdir -p "$ENV_DIR"
while read d u; do
  [ -z "${u:-}" ] && continue
  az storage blob download --connection-string "$C" -c orbitbrief-artifacts \
    -n "deals/$u/orbitbrief/latest/envelope.json" -f "$ENV_DIR/final_${d}_env.json" -o none --max-connections 4 >/dev/null 2>&1 \
    || echo "  no envelope for $d"
done < "$ROOT/deals.txt"
echo "  envelopes: $(ls "$ENV_DIR" | wc -l | tr -d ' ')"

OUT=$ROOT/runs/$(date +%Y%m%d)
rm -rf "$OUT"
OUT=$OUT RUBRIC="$ROOT/rubric/*.A.jsonl" REAL="$ROOT/real_sow/sem_g*.jsonl" \
  PUBLISH=$([ -n "${DRY:-}" ] || echo 1) zsh scripts/heads_loop.sh "$ENV_DIR" || echo "  heads_loop failed"
[ -n "${DRY:-}" ] && echo "  DRY: not published"

# Keep two weeks of runs; checkpoints are large.
ls -1d "$ROOT"/runs/* 2>/dev/null | sort -r | tail -n +15 | while read old; do rm -rf "$old"; done
echo "== $(date -u +%FT%TZ) heads_nightly done"
