#!/usr/bin/env bash
# v45.2 — verify the deployed parser-os image contains the expected commit.
#
# Catches the failure we hit during the v45.1 dev deploy: claimed to deploy
# d931a38, but /v1/version reported 1a8176c because ACR/docker cached an old
# layer.  Run this immediately after every `deploy-dev.sh`.
#
# Usage:
#   ./scripts/verify_deployed_sha.sh \
#     --url https://parser-os-service-dev-eus2.<region>.azurecontainerapps.io \
#     --expected-sha $(git rev-parse HEAD)
#
# Optional flags:
#   --expected-label v45.2         # also check the build label
#   --recompile-deal <deal-id>     # POST a recompile + assert entity count > 200
#   --token $PM_INTERNAL_TOKEN     # required if --recompile-deal is set
#   --proxy-base <azure-proxy-url> # required if --recompile-deal is set
#
# Exit codes:
#   0 — everything OK
#   1 — SHA mismatch (the showstopper)
#   2 — service unreachable / /v1/version returned non-200
#   3 — recompile failed
#   4 — recompile finished but entity count below v45 floor

set -euo pipefail

# ─── Args ─────────────────────────────────────────────────────────────────
URL=""
EXPECTED_SHA=""
EXPECTED_LABEL=""
RECOMPILE_DEAL=""
TOKEN=""
PROXY_BASE=""
MIN_ENTITIES=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)              URL="$2"; shift 2 ;;
    --expected-sha)     EXPECTED_SHA="$2"; shift 2 ;;
    --expected-label)   EXPECTED_LABEL="$2"; shift 2 ;;
    --recompile-deal)   RECOMPILE_DEAL="$2"; shift 2 ;;
    --token)            TOKEN="$2"; shift 2 ;;
    --proxy-base)       PROXY_BASE="$2"; shift 2 ;;
    --min-entities)     MIN_ENTITIES="$2"; shift 2 ;;
    -h|--help)
      grep -E '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$URL" || -z "$EXPECTED_SHA" ]]; then
  echo "ERROR: --url and --expected-sha are required" >&2
  echo "Try: $0 --help" >&2
  exit 2
fi

# ─── 1. Check /v1/version ─────────────────────────────────────────────────
echo "→ GET $URL/v1/version"
VER_JSON=$(curl -sS --max-time 10 "$URL/v1/version") || {
  echo "✗ FAIL — service unreachable at $URL/v1/version" >&2
  exit 2
}

# Pretty-print so the operator sees everything
echo "$VER_JSON" | jq .

ACTUAL_SHA=$(echo "$VER_JSON" | jq -r '.parser_os_sha // "missing"')
ACTUAL_LABEL=$(echo "$VER_JSON" | jq -r '.build_label // "missing"')

if [[ "$ACTUAL_SHA" == "missing" || "$ACTUAL_SHA" == "unknown" ]]; then
  echo "✗ FAIL — parser_os_sha not exposed by /v1/version." >&2
  echo "  The deployed image either: (a) is older than v45.2 and doesn't have"
  echo "  the SHA endpoint, OR (b) was built without --build-arg GIT_SHA=..." >&2
  exit 1
fi

# Allow short prefix match (first 7 chars) for convenience
EXPECTED_SHORT="${EXPECTED_SHA:0:7}"
ACTUAL_SHORT="${ACTUAL_SHA:0:7}"

if [[ "$EXPECTED_SHORT" != "$ACTUAL_SHORT" ]]; then
  echo "" >&2
  echo "✗ FAIL — SHA mismatch:" >&2
  echo "    expected: $EXPECTED_SHA" >&2
  echo "    deployed: $ACTUAL_SHA" >&2
  echo "" >&2
  echo "  Likely causes:" >&2
  echo "    1. Docker build used a cached layer — rebuild with --no-cache" >&2
  echo "    2. deploy-dev.sh didn't pass --build-arg GIT_SHA=\$(git rev-parse HEAD)" >&2
  echo "    3. ACR push silently used the previously-tagged image" >&2
  exit 1
fi

echo "✓ SHA matches: $ACTUAL_SHA"

if [[ -n "$EXPECTED_LABEL" ]]; then
  if [[ "$ACTUAL_LABEL" != "$EXPECTED_LABEL" ]]; then
    echo "✗ FAIL — build_label mismatch: expected=$EXPECTED_LABEL deployed=$ACTUAL_LABEL" >&2
    exit 1
  fi
  echo "✓ Build label matches: $ACTUAL_LABEL"
fi

# ─── 2. Optional end-to-end smoke ─────────────────────────────────────────
if [[ -z "$RECOMPILE_DEAL" ]]; then
  echo ""
  echo "✓ All version checks passed.  (Skipped recompile — no --recompile-deal)"
  exit 0
fi

if [[ -z "$TOKEN" || -z "$PROXY_BASE" ]]; then
  echo "ERROR: --recompile-deal requires --token and --proxy-base" >&2
  exit 2
fi

echo ""
echo "→ POST recompile for deal $RECOMPILE_DEAL"
RECOMPILE_JSON=$(
  curl -sS -X POST --max-time 30 \
    -H "x-internal-token: $TOKEN" \
    -H "Content-Type: application/json" \
    "$PROXY_BASE/api/quoting/deal/$RECOMPILE_DEAL/orbitbrief/recompile"
)
COMPILE_ID=$(echo "$RECOMPILE_JSON" | jq -r '.compileId // .compile_id // empty')

if [[ -z "$COMPILE_ID" ]]; then
  echo "✗ FAIL — recompile returned no compileId:" >&2
  echo "$RECOMPILE_JSON" >&2
  exit 3
fi
echo "  → compileId: $COMPILE_ID"

# ─── 3. Poll progress until done ──────────────────────────────────────────
echo "→ polling compile progress every 10s (timeout 30 min)..."
DEADLINE=$(( $(date +%s) + 1800 ))
while [[ $(date +%s) -lt $DEADLINE ]]; do
  PROG=$(
    curl -sS --max-time 5 \
      "$URL/projects/$RECOMPILE_DEAL/compile/progress" 2>/dev/null || echo "{}"
  )
  STATUS=$(echo "$PROG" | jq -r '.status // "unknown"')
  PCT=$(echo "$PROG" | jq -r '.percent_complete // 0')
  STAGE=$(echo "$PROG" | jq -r '.current_stage_label // .current_stage // "?"')
  ETA=$(echo "$PROG" | jq -r '.estimated_remaining_seconds // 0')
  printf "  [%s] %3s%%  %-44s  ETA %ss\n" "$STATUS" "$PCT" "$STAGE" "$ETA"
  if [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]]; then
    break
  fi
  sleep 10
done

if [[ "$STATUS" != "completed" ]]; then
  echo "✗ FAIL — compile didn't reach completed within 30 min (final status: $STATUS)" >&2
  exit 3
fi

# ─── 4. Fetch envelope + assert entity count ──────────────────────────────
echo ""
echo "→ checking entity count from envelope"
ENV_JSON=$(
  curl -sS --max-time 10 \
    -H "x-internal-token: $TOKEN" \
    "$PROXY_BASE/api/quoting/deal/$RECOMPILE_DEAL/orbitbrief/latest/envelope"
)
ENTITY_COUNT=$(echo "$ENV_JSON" | jq -r '.entities | length // 0')
ATOM_COUNT=$(echo "$ENV_JSON" | jq -r '.atoms | length // 0')

echo "  envelope reports: entities=$ENTITY_COUNT, atoms=$ATOM_COUNT"

if (( ENTITY_COUNT < MIN_ENTITIES )); then
  echo "" >&2
  echo "✗ FAIL — entity count $ENTITY_COUNT < floor $MIN_ENTITIES" >&2
  echo "  v45.2 target is 200-400 entities for an average deal." >&2
  echo "  Likely causes:" >&2
  echo "    1. Ollama unreachable — check polish_stage in pm-handoff.json" >&2
  echo "    2. Embedding model not loaded on Mac" >&2
  echo "    3. SOWSMITH_RETRIEVAL_DISABLE / _ZERO_MISS_DISABLE env vars set" >&2
  exit 4
fi

# ─── 5. Polish-stage sanity ───────────────────────────────────────────────
HANDOFF_JSON=$(
  curl -sS --max-time 10 \
    -H "x-internal-token: $TOKEN" \
    "$PROXY_BASE/api/quoting/deal/$RECOMPILE_DEAL/orbitbrief/latest/pm-handoff"
)
POLISHED=$(echo "$HANDOFF_JSON" | jq -r '.polish_stage.items_polished // 0')
FALLBACK=$(echo "$HANDOFF_JSON" | jq -r '.polish_stage.items_fallback // 0')

echo "  polish_stage: polished=$POLISHED fallback=$FALLBACK"

if (( POLISHED == 0 && FALLBACK > 10 )); then
  echo "" >&2
  echo "⚠ WARN — polish_stage shows 0 polished / $FALLBACK fallback." >&2
  echo "  Core worker can't reach Mac Ollama.  Fix OLLAMA_BASE_URL or proxy." >&2
  # Non-fatal: parser still worked, just no LLM polish.  Use exit 4 if you
  # want this to fail the deploy.
fi

echo ""
echo "✓ All deployment checks passed for $URL"
echo "  SHA      : $ACTUAL_SHA"
echo "  Label    : $ACTUAL_LABEL"
echo "  Entities : $ENTITY_COUNT (≥ $MIN_ENTITIES)"
echo "  Polished : $POLISHED / fallback $FALLBACK"
