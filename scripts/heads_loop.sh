#!/bin/zsh
# The "heads decide, rules retire" loop, one command (SowSmith's Heads panel
# shows the same five steps). Run on the Mac with the C3 base.
#
#   scripts/heads_loop.sh <envelopes-dir> [reviews.jsonl]
#
# 1 Collect   envelopes carry c3_sow.rule_decisions; reviews.jsonl = SowSmith
#             "Export labels" (PM reviews + PM edits)
# 2 Labels    ml.c3.labels_from_sow → gold (PM) + silver (rules) records
# 3 Refit     ml.c3.slot_fit per head on train deals, held-out deals pick the epoch
# 4 Verify    ml.c3.rule_retire scores the refit against every rule's decisions
#             on the held-out deals → report (ready = >=95% over >=30)
# 5 Publish   report → orbitbrief-artifacts/_sow_heads/rule_retire.json (dev)
# Retiring a ready rule is a reviewed commit plus a regrade; the refit itself is
# only served after the full-line diff and golden 34/34 pass.
set -e
ENV_DIR=${1:?envelopes dir}
REVIEWS=${2:-}
OUT=${OUT:-$HOME/c3-local/heads_loop/$(date +%Y%m%d_%H%M)}
CKPT=${CKPT:-$HOME/c3-serve/c3_base/checkpoint.pt}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
REAL=${REAL:-}
extra=()
[ -n "$REVIEWS" ] && extra+=(--reviews "$REVIEWS")
[ -n "$REAL" ] && extra+=(--real-sow "$REAL")
python3.11 -m ml.c3.labels_from_sow --envelopes "$ENV_DIR"/final_*_env.json "${extra[@]}" --out "$OUT/records"
deals=($(ls "$OUT/records" | sed 's/.json//' | sort))
mkdir -p "$OUT/train" "$OUT/val"
i=0; for d in $deals; do if (( i % 4 == 0 )); then cp "$OUT/records/$d.json" "$OUT/val/"; else cp "$OUT/records/$d.json" "$OUT/train/"; fi; i=$((i+1)); done
cur=$CKPT
for key in read:sow_section read:co_action; do
  tag=${key#read:}
  python3.11 -m ml.c3.slot_fit --checkpoint "$cur" --key "$key" --train "$OUT"/train/*.json \
      --validation "$OUT"/val/*.json --out "$OUT/refit_$tag.pt" --epochs 60 | tee "$OUT/fit_$tag.log"
  [ -f "$OUT/refit_$tag.pt" ] && cur="$OUT/refit_$tag.pt"
done
val_envs=()
for f in "$OUT"/val/*.json; do d=$(basename $f .json); val_envs+=("$ENV_DIR/final_${d}_env.json"); done
python3.11 -m ml.c3.rule_retire --checkpoint "$cur" --envelopes $val_envs ${REVIEWS:+--reviews "$REVIEWS"} --report "$OUT/rule_retire.json"
if [ -n "$PUBLISH" ]; then
  az storage blob upload --connection-string "$(az storage account show-connection-string -g purtera-dev-rg -n purpulsedevstg01 -o tsv)" \
     -c orbitbrief-artifacts -n _sow_heads/rule_retire.json -f "$OUT/rule_retire.json" --overwrite -o none && echo "published report"
fi
echo "OUT=$OUT checkpoint=$cur"
