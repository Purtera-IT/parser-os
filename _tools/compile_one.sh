#!/usr/bin/env bash
# Re-compile one deal on whatever is deployed, and wait for it.
#   DEAL=<uuid> bash compile_one.sh
set -u
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")"   # this worktree's _tools, where .bloburl lives
BASE=https://parser-os-service-dev-eus2.whitehill-a3348ba5.eastus2.azurecontainerapps.io
say() { echo "[$(date +%H:%M:%S)] $*"; }

TOK=$(az containerapp secret show -n parser-os-service-dev-eus2 -g purtera-dev-rg \
        --secret-name bang-internal-bearer --query value -o tsv 2>/dev/null)
[ -n "$TOK" ] || { say "no bearer"; exit 1; }
MAN=$(DEAL="$DEAL" python - <<'PY'
import os
from azure.storage.blob import BlobServiceClient
conn=open(".bloburl").read().strip()
c=BlobServiceClient.from_connection_string(conn).get_container_client("orbitbrief-artifacts")
D=os.environ["DEAL"]
bs=[b for b in c.list_blobs(name_starts_with=f"deals/{D}/parser-manifests/") if b.name.endswith(".json")]
print(os.path.basename(max(bs, key=lambda b: b.last_modified).name)[:-5])
PY
)
say "deal $DEAL manifest $MAN"
C=$(python -c "import uuid;print(uuid.uuid4())")
curl -s -X POST "$BASE/v1/compile/async" -H "Authorization: Bearer $TOK" \
  -H "Content-Type: application/json" \
  -d "{\"compile_id\":\"$C\",\"deal_id\":\"$DEAL\",\"manifest_blob_url\":\"https://purpulsedevstg01.blob.core.windows.net/orbitbrief-artifacts/deals/$DEAL/parser-manifests/$MAN.json\",\"force\":true}" >/dev/null
say "compile $C"
for i in $(seq 1 100); do
  sleep 20
  ST=$(curl -s "$BASE/v1/compile/status/$C?deal_id=$DEAL" -H "Authorization: Bearer $TOK" \
       | python -c "import sys,json;print(json.load(sys.stdin).get('status'))" 2>/dev/null || echo "?")
  case "$ST" in completed|succeeded|failed|error) break;; esac
done
say "compile: $ST"
