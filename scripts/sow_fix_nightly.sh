#!/bin/zsh
# Nightly SOW fix triage (dev only): classify each PM fix to the heads' SOW into a head label,
# a parser label, a standards proposal, a writer bug, or a question for the PM.
#
#   scripts/sow_fix_nightly.sh            # dry run: report + would_post.jsonl, nothing sent
#   POST=1 scripts/sow_fix_nightly.sh     # also send head labels to the DEV feedback route
#
# Inputs, all outside the repo:
#   ~/c3-local/sow_fix/exports/<deal>.json   SowSmith "Export fixes" (Heads drawer, admins), one per deal
#   ~/c3-local/heads_loop/deals.txt          "<deal> <uuid>" per line, to fetch each deal's latest envelope
# Output: ~/c3-local/sow_fix/runs/<date>/{report.md,report.json,would_post.jsonl,reviews.jsonl,
#   standards_queue.jsonl,writer_bugs.jsonl,parser_labels.jsonl,questions.jsonl}
# reviews.jsonl feeds `ml.c3.labels_from_sow --reviews`; standards_queue.jsonl waits for the owner.
set -u
ROOT=$HOME/c3-local/sow_fix
LOG=$ROOT/nightly.log
mkdir -p "$ROOT/exports"
exec >>"$LOG" 2>&1
echo "== $(date -u +%FT%TZ) sow_fix_nightly start"
cd "$(dirname "$0")/.."

ls "$ROOT"/exports/*.json >/dev/null 2>&1 || { echo "  no exports; nothing to triage"; exit 0; }
C=$(az storage account show-connection-string -g purtera-dev-rg -n purpulsedevstg01 -o tsv 2>/dev/null)
[ -n "$C" ] || { echo "no storage access (az login?); skipping"; exit 1; }
ENV_DIR=$ROOT/envs
rm -rf "$ENV_DIR"; mkdir -p "$ENV_DIR"
for f in "$ROOT"/exports/*.json; do
  d=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("deal",""))' "$f")
  u=$(awk -v d="$d" '$1==d {print $2}' "$HOME/c3-local/heads_loop/deals.txt" 2>/dev/null)
  [ -z "$u" ] && { echo "  $d: no uuid in deals.txt"; continue; }
  az storage blob download --connection-string "$C" -c orbitbrief-artifacts \
    -n "deals/$u/orbitbrief/latest/envelope.json" -f "$ENV_DIR/final_${d}_env.json" -o none >/dev/null 2>&1 \
    || echo "  $d: no envelope"
done

OUT=$ROOT/runs/$(date +%Y%m%d)
FLAGS=()
if [ -n "${POST:-}" ]; then
  export PARSER_OS_DEV_BEARER=$(az containerapp secret show -n parser-os-service-dev-eus2 -g purtera-dev-rg \
    --secret-name bang-internal-bearer --query value -o tsv 2>/dev/null)
  FLAGS=(--post)
fi
python3 scripts/sow_fix_triage.py --edits "$ROOT/exports" --envs "$ENV_DIR" --out "$OUT" $FLAGS || echo "  triage failed"

ls -1d "$ROOT"/runs/* 2>/dev/null | sort -r | tail -n +31 | while read old; do rm -rf "$old"; done
echo "== $(date -u +%FT%TZ) sow_fix_nightly done"
