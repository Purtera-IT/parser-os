#!/usr/bin/env bash
# Pack the chosen C3 base for the worker: c3_base.tgz holds the checkpoint,
# the ml/ code at this commit (the parser package does not ship ml/), and a
# manifest. Writes the file only; uploading it is a separate, deliberate step.
#
#   CHECKPOINT=~/c3-runC/runC.pt.open_links.slim TYPES=task,requirement OUT=~/c3_base.tgz \
#       [DTYPE=bf16] [ENCODER=answerdotai/ModernBERT-large] bash ml/c3/pod/bundle.sh
#
# The encoder goes in the bundle (bf16, as the base reads it), so a worker
# that starts cold every compile never downloads it from the hub. DTYPE=bf16
# only after ml.c3.lite_check says the bf16 copy agrees.
set -euo pipefail
: "${CHECKPOINT:?set CHECKPOINT to the picked slim checkpoint}"
: "${TYPES:?set TYPES to the list route_report printed (SOWSMITH_C3_TYPES)}"
OUT=${OUT:-c3_base.tgz}
SURE=${SURE:-0.7}
DTYPE=${DTYPE:-fp32}
ENCODER=${ENCODER:-answerdotai/ModernBERT-large}
BUDGET=${BUDGET:-90}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/c3_base/code"
cp "$CHECKPOINT" "$WORK/c3_base/checkpoint.pt"
python - "$ENCODER" "$WORK/c3_base/encoder" <<'PY'
import sys
import torch
from transformers import AutoModel, AutoTokenizer
name, out = sys.argv[1], sys.argv[2]
AutoTokenizer.from_pretrained(name).save_pretrained(out)
AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16).save_pretrained(out)
PY
# Code only: no caches, no pod outputs, no deal files.
(cd "$ROOT" && find ml -name '*.py' -o -name '*.json' | grep -v __pycache__ \
    | tar -cf - -T -) | tar -xf - -C "$WORK/c3_base/code"
SCHEMA=$(cd "$ROOT" && python -c "
from dataclasses import asdict
from ml.c3.readiness import digest
from ml.c3.schema import load_schema
print(digest([asdict(o) for o in load_schema().opportunities]))")
MD5=$(python -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$CHECKPOINT")
cat > "$WORK/c3_base/MANIFEST.json" <<EOF
{"commit": "$(git -C "$ROOT" rev-parse HEAD)", "checkpoint": "$(basename "$CHECKPOINT")",
 "checkpoint_md5": "$MD5", "schema_sha256": "$SCHEMA", "max_atoms": 8192,
 "types": "$TYPES", "sure": $SURE, "dtype": "$DTYPE", "encoder": "$ENCODER", "budget_s": $BUDGET}
EOF
# The worker's own parser must carry the same head files (schema_sha256):
# a different schema makes the checkpoint refuse to load, and the layer abstains.
cat > "$WORK/c3_base/env.shadow" <<EOF
SOWSMITH_C3_TYPE=shadow
SOWSMITH_C3_TYPES=$TYPES
SOWSMITH_C3_SURE=$SURE
SOWSMITH_C3_CHECKPOINT=/tmp/ml/c3_base/checkpoint.pt
SOWSMITH_C3_ENCODER=/tmp/ml/c3_base/encoder
SOWSMITH_C3_DTYPE=$DTYPE
SOWSMITH_C3_BUDGET_S=$BUDGET
SOWSMITH_C3_DEVICE=cpu
# Add /tmp/ml/c3_base/code to the worker's PYTHONPATH (ml/ is not in the parser package).
EOF
tar -czf "$OUT" -C "$WORK" c3_base
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)) schema $SCHEMA checkpoint md5 $MD5"
