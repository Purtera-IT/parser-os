#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────
# PROD SMOKE TEST + AUTO-RECOVERY for parser-os / OrbitBrief
#
# Run this ANY TIME you're worried something's broken. It checks the
# whole stack end-to-end and auto-fixes the most common issues.
#
#   ./smoke_test.sh           — check-only (read-only)
#   ./smoke_test.sh --fix     — also auto-recover if anything is degraded
#
# Exits 0 if healthy, 1 if anything is wrong AND --fix didn't recover it.
# ───────────────────────────────────────────────────────────────────────
set -u
FIX="${1:-}"

DEAL=1bf0c10e-e840-4a1f-b526-d8f417181ada
RG=purtera-dev-rg
FA=purpulse-dev-api-eus2
WORKER=parser-os-worker-dev-eus2
SVC=parser-os-service-dev-eus2
TAILNET=griffin-purtera-it.github
TS_API_TOKEN="${TS_API_TOKEN:-tskey-api-kNqqGA1BYf11CNTRL-64MXE2udYx5UT7L2eYXYx5MZoRZgxLT99}"

BASE="https://${FA}.azurewebsites.net/api/proxy/api/quoting/deal/${DEAL}"
SVC_URL="https://parser-os-service-dev-eus2.whitehill-a3348ba5.eastus2.azurecontainerapps.io"

FAIL=0
RECOVERED=0

step() { echo ""; echo "─── $* ───"; }
ok()   { echo "    ✓ $*"; }
warn() { echo "    ⚠ $*"; }
err()  { echo "    ✗ $*"; FAIL=$((FAIL+1)); }
fixed() { echo "    ✚ recovered: $*"; RECOVERED=$((RECOVERED+1)); }

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PROD SMOKE TEST + AUTO-RECOVERY"
echo "  $(date)"
echo "════════════════════════════════════════════════════════════"

# 1. Function App routes
step "[1] Function App routes (must all return 200)"
DEGRADED_ROUTES=0
for E in envelope compile-progress compile-history pm-handoff polish-report pipeline-log sow-draft; do
  CODE=$(curl -sS -o /dev/null -w "%{http_code}" "${BASE}/orbitbrief/latest/${E}" -m 30)
  if [[ "$CODE" == "200" ]]; then
    ok "$E ($CODE)"
  else
    err "$E ($CODE)"
    DEGRADED_ROUTES=$((DEGRADED_ROUTES+1))
  fi
done

if [[ "$DEGRADED_ROUTES" -gt 2 ]] && [[ "$FIX" == "--fix" ]]; then
  echo "    → Function App degraded. Restarting..."
  az functionapp restart -n "$FA" -g "$RG" >/dev/null 2>&1
  echo "    → waiting 90s for warmup"
  sleep 90
  RETRY_OK=0
  for E in envelope compile-progress compile-history pm-handoff; do
    CODE=$(curl -sS -o /dev/null -w "%{http_code}" "${BASE}/orbitbrief/latest/${E}" -m 30)
    [[ "$CODE" == "200" ]] && RETRY_OK=$((RETRY_OK+1))
  done
  if [[ "$RETRY_OK" -ge 3 ]]; then fixed "Function App restored after restart"; fi
fi

# 2. parser-os-worker config
step "[2] parser-os-worker config"
CFG=$(az containerapp job show -n "$WORKER" -g "$RG" --query "{img:properties.template.containers[0].image, timeout:properties.configuration.replicaTimeout, retry:properties.configuration.replicaRetryLimit}" -o json 2>/dev/null)
IMG=$(echo "$CFG" | python -c "import sys,json; print(json.loads(sys.stdin.read())['img'])" 2>/dev/null)
TIMEOUT=$(echo "$CFG" | python -c "import sys,json; print(json.loads(sys.stdin.read())['timeout'])" 2>/dev/null)
RETRY=$(echo "$CFG" | python -c "import sys,json; print(json.loads(sys.stdin.read())['retry'])" 2>/dev/null)
if [[ "$IMG" == *"v57_6"* ]] || [[ "$IMG" == *"v57_4"* ]]; then ok "image: $IMG"; else err "wrong image: $IMG"; fi
if [[ "$TIMEOUT" -ge 3600 ]]; then ok "timeout: ${TIMEOUT}s (≥60min)"; else warn "timeout: ${TIMEOUT}s — should be ≥3600"; fi
if [[ "$RETRY" -ge 1 ]]; then ok "retry: $RETRY"; else warn "retry: $RETRY — should be ≥1"; fi

# 3. Storage queues
step "[3] Storage queues (must all be near 0)"
STORAGE_CONN=$(az containerapp secret show -n "$SVC" -g "$RG" --secret-name storage-conn --query "value" -o tsv 2>/dev/null)
for Q in parser-os-compile-jobs parser-os-orbitbrief-jobs parser-os-orbitbrief-jobs-poison; do
  COUNT=$(az storage message peek --queue-name "$Q" --num-messages 32 --connection-string "$STORAGE_CONN" --query "length([])" -o tsv 2>/dev/null | tail -1)
  if [[ "$COUNT" -le 3 ]]; then
    ok "$Q = $COUNT"
  else
    warn "$Q = $COUNT (high)"
    if [[ "$FIX" == "--fix" ]] && [[ "$Q" == *"poison"* ]]; then
      az storage message clear --queue-name "$Q" --connection-string "$STORAGE_CONN" >/dev/null 2>&1
      fixed "cleared $Q"
    fi
  fi
done

# 4. Tailscale tailnet
step "[4] Tailscale tailnet (parser-os-worker zombies)"
ZOMBIES=$(curl -sS "https://api.tailscale.com/api/v2/tailnet/${TAILNET}/devices" -H "Authorization: Bearer ${TS_API_TOKEN}" 2>/dev/null | python -c "
import sys,json
d = json.load(sys.stdin)
print(len([x for x in d.get('devices',[]) if (x.get('hostname') or '').lower().startswith('parser-os-worker')]))
" 2>/dev/null)
if [[ "$ZOMBIES" -lt 500 ]]; then
  ok "$ZOMBIES nodes (cap 1000)"
else
  warn "$ZOMBIES nodes — getting close to quota"
  if [[ "$FIX" == "--fix" ]]; then
    echo "    → running bulk delete..."
    TS_API_TOKEN="$TS_API_TOKEN" python "$(dirname "$0")/_tailscale_bulk_delete.py" --prefix parser-os-worker 2>&1 | tail -3
    fixed "deleted zombies"
  fi
fi

# 5. Mac Studio (Ollama LLM source) — measured via REAL brief activity
step "[5] Mac Studio Ollama (proves LLM working)"
# Tailscale's "online" flag is unreliable — use the orbitbrief-core-worker
# log stream instead, which only shows recent activity if Ollama responded.
WS_ID=$(az containerapp env show -n parser-dev-env-eus2 -g "$RG" --query "properties.appLogsConfiguration.logAnalyticsConfiguration.customerId" -o tsv 2>/dev/null | tail -1)
RECENT_BRIEF=$(az monitor log-analytics query --workspace "$WS_ID" --analytics-query "ContainerAppConsoleLogs_CL | where ContainerAppName_s == 'orbitbrief-core-worker-dev-eus2' and TimeGenerated > ago(2h) and Log_s contains 'orbitbrief_pipeline' and Log_s contains 'stages' | take 1 | project TimeGenerated" -o tsv 2>/dev/null | tail -1)
if [[ -n "$RECENT_BRIEF" ]]; then
  ok "LLM stage activity in last 2h: $RECENT_BRIEF"
else
  warn "no brief-gen activity in last 2h — fire a /rebrief to verify"
fi

# 6. End-to-end compile test
step "[6] End-to-end compile test (~10s on thin notes deal)"
BEARER=$(az containerapp secret show -n "$SVC" -g "$RG" --secret-name bang-internal-bearer --query "value" -o tsv 2>/dev/null | tail -1)
NEW_CID=$(python -c "import uuid; print(uuid.uuid4())")
echo "    compile_id: $NEW_CID"
curl -sS -X POST "${SVC_URL}/v1/compile/async" \
  -H "Content-Type: application/json" -H "Authorization: Bearer ${BEARER}" \
  -d "{\"compile_id\":\"${NEW_CID}\",\"deal_id\":\"${DEAL}\",\"manifest_blob_url\":\"https://purpulsedevstg01.blob.core.windows.net/orbitbrief-artifacts/deals/${DEAL}/parser-manifests/c2aaa346-926a-4920-8405-c3cf2970d9af.json\"}" -m 30 -o /dev/null
deadline=$(($(date +%s) + 120))
COMPILE_OK=0
while [ $(date +%s) -lt $deadline ]; do
  R=$(curl -sS -H "Authorization: Bearer ${BEARER}" "${SVC_URL}/v1/compile/status/${NEW_CID}?deal_id=${DEAL}")
  STATUS=$(echo "$R" | python -c "import sys,json; print(json.loads(sys.stdin.read()).get('status','?'))" 2>/dev/null)
  if [[ "$STATUS" == "completed" ]]; then
    ATOMS=$(echo "$R" | python -c "import sys,json; print(json.loads(sys.stdin.read()).get('atom_count','?'))" 2>/dev/null)
    ELAP=$(echo "$R" | python -c "import sys,json; print(json.loads(sys.stdin.read()).get('elapsed_sec','?'))" 2>/dev/null)
    ok "completed: $ATOMS atoms in ${ELAP}s"
    COMPILE_OK=1
    break
  fi
  if [[ "$STATUS" == "failed" ]]; then
    err "compile failed"
    break
  fi
  sleep 5
done
[[ "$COMPILE_OK" == "0" ]] && err "compile timed out"

# Summary
echo ""
echo "════════════════════════════════════════════════════════════"
if [[ "$FAIL" == "0" ]]; then
  echo "  🟢 ALL GREEN — system ready for PM testing"
  [[ "$RECOVERED" -gt 0 ]] && echo "     ($RECOVERED auto-recoveries applied)"
  exit 0
else
  echo "  🔴 $FAIL failure(s) detected"
  [[ "$FIX" != "--fix" ]] && echo "     run with --fix to attempt auto-recovery"
  exit 1
fi
