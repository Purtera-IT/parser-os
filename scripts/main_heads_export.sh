#!/bin/zsh
# Export the PM corrections the main heads learn from, READ-ONLY, from dev blob.
#
#   scripts/main_heads_export.sh <out-dir>
#
# Where they live: POST /projects/{id}/feedback/correction (app.core.pm_feedback)
# writes the correction to the service's per-process SQLite FeedbackStore, which
# nothing outside that container can read and which is lost on recycle, and
# mirrors it to blob (app.core.feedback_blob, SOWSMITH_FEEDBACK_BLOB on):
#   purpulsedevstg01 / orbitbrief-artifacts / _feedback/corrections/<id>.json
# That mirror is the durable copy, so this downloads it and writes
#   <out>/corrections.jsonl          one Correction per line
#   <out>/envelopes/<deal uuid>.json the latest envelope of every deal a
#                                    main-head correction names
#
# Storage auth, first that is set (never interactive):
#   AZURE_STORAGE_CONNECTION_STRING
#   AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_SAS_TOKEN   (a read+list SAS on the container)
#   az CLI already logged in (az storage account show-connection-string)
# Exit 2 = no storage access, 3 = no corrections / no envelopes. Writes nothing
# anywhere but <out>; never writes to blob or any database.
set -u
OUT=${1:?out dir}
PY=${PY:-python3.11}
cd "$(dirname "$0")/.."
mkdir -p "$OUT/_feedback" "$OUT/envelopes"

if [ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" ] && [ -z "${AZURE_STORAGE_SAS_TOKEN:-}" ]; then
  export AZURE_STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
      -g purtera-dev-rg -n purpulsedevstg01 -o tsv 2>/dev/null)
fi
if [ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" ] && [ -z "${AZURE_STORAGE_SAS_TOKEN:-}" ]; then
  echo "FATAL main_heads_export: no storage access (no connection string, no SAS, az not logged in)"
  exit 2
fi
[ -n "${AZURE_STORAGE_SAS_TOKEN:-}" ] && export AZURE_STORAGE_ACCOUNT=${AZURE_STORAGE_ACCOUNT:-purpulsedevstg01}

az storage blob download-batch -s orbitbrief-artifacts -d "$OUT" --pattern "_feedback/corrections/*" \
   --max-connections 8 --no-progress -o none --only-show-errors >/dev/null || {
  echo "FATAL main_heads_export: listing _feedback/corrections failed (storage access?)"; exit 2; }
n=$(ls "$OUT/_feedback/corrections" 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -gt 0 ] || { echo "FATAL main_heads_export: no corrections under _feedback/corrections/"; exit 3; }

# corrections.jsonl + the deals the main heads' corrections name.
$PY - "$OUT" <<'EOF'
import glob, json, os, sys
sys.path.insert(0, os.getcwd())
from scripts.main_heads_queue import wanted_deals
out = sys.argv[1]
rows = [json.load(open(p)) for p in sorted(glob.glob(f"{out}/_feedback/corrections/*.json"))]
with open(f"{out}/corrections.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
open(f"{out}/deals.txt", "w").write("\n".join(sorted(wanted_deals(rows))) + "\n")
EOF
echo "  corrections: $n · deals named: $(grep -c . "$OUT/deals.txt")"

missing=0
while read d; do
  [ -z "$d" ] && continue
  az storage blob download -c orbitbrief-artifacts -n "deals/$d/orbitbrief/latest/envelope.json" \
     -f "$OUT/envelopes/$d.json" -o none --only-show-errors >/dev/null 2>&1 || missing=$((missing+1))
done < "$OUT/deals.txt"
got=$(ls "$OUT/envelopes" | wc -l | tr -d ' ')
echo "  envelopes: $got (no latest envelope: $missing)"
[ "$got" -gt 0 ] || { echo "FATAL main_heads_export: no envelope for any corrected deal"; exit 3; }
exit 0
