#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cd "$REPO"
ATOMS=${ATOMS:-8192}
PARTS=${PARTS:-4}
# Every deal loads and plans (no GPU).
python - "$ATOMS" $(files $TRAIN $VALIDATION $HOLDOUT) > "$C3/out/load.log" 2>&1 <<'PY' || { tail -20 "$C3/out/load.log"; exit 1; }
import sys
from ml.c3.loop import load_deal, training_batches
from ml.c3.schema import load_schema
schema, atoms = load_schema(), int(sys.argv[1])
for path in sys.argv[2:]:
    deal = load_deal(path)
    parts, notes = training_batches(deal, schema, max_atoms=atoms)
    print("deal", deal.deal_id, "lines", len(deal.atoms), "parts", len(parts), "notes", len(notes), flush=True)
    for note in notes:
        print("note", note)
PY
grep -E "^deal" "$C3/out/load.log"; echo "notes: $(grep -c '^note' "$C3/out/load.log")"
# Every stage once, on the largest parts, both versions side by side.
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 10 > "$C3/out/pre_mem.log" &
MEM=$!
common=(--steps 1 --max-atoms "$ATOMS" --limit-parts "$PARTS"
        --train $(files $TRAIN) --validation $(files 010087) --holdout $(files $HOLDOUT))
CUDA_VISIBLE_DEVICES=0 python -m ml.c3.loop "${common[@]}" --checkpoint "$C3/out/preA.pt" \
  > "$C3/out/preA.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 python -m ml.c3.loop "${common[@]}" --train-encoder-layers 2 \
  --checkpoint "$C3/out/preB.pt" > "$C3/out/preB.log" 2>&1 &
B=$!
status=0
wait $A || status=1
wait $B || status=1
kill $MEM
for r in A B; do echo "== $r"; grep -E "progress|stage|validation|Error|error" "$C3/out/pre$r.log" | tail -40; done
python - "$C3/out/pre_mem.log" <<'PY'
import sys, collections
peak = collections.defaultdict(int)
for line in open(sys.argv[1]):
    i, mib = [x.strip() for x in line.split(",")]
    peak[i] = max(peak[i], int(mib.split()[0]))
print("peak GPU memory MiB:", dict(peak))
PY
exit $status
