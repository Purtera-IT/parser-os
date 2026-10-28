#!/usr/bin/env bash
# Fetch and unpack the bundle, then serve; Ollama pulls and loads the quote heads' embedding model beside it.
#   C3_BUNDLE_URL      c3_base.tgz: a plain blob URL read with the managed identity, or a read-only SAS URL
#   SOWSMITH_C3_TOKEN  the bearer token the worker sends (a secret)
#   C3_DIR             where the bundle and Ollama models live (mount a volume here to skip re-downloads)
set -euo pipefail
: "${C3_BUNDLE_URL:?set C3_BUNDLE_URL to the c3_base.tgz blob URL}"
DST=${C3_DIR:-/models}
# Some hosts run the container without root: fall back to a writable path.
mkdir -p "$DST" 2>/dev/null || DST=/tmp/c3 && mkdir -p "$DST"
if [[ ! -f "$DST/c3_base/checkpoint.pt" ]]; then
  if [[ "$C3_BUNDLE_URL" != *"sig="* && -n "${IDENTITY_ENDPOINT:-}" ]]; then
    # A plain blob URL: read it as the app's managed identity (Storage Blob Data Reader), no SAS to expire.
    TOK=$(curl -fsS -H "X-IDENTITY-HEADER: $IDENTITY_HEADER" \
      "$IDENTITY_ENDPOINT?resource=https://storage.azure.com/&api-version=2019-08-01${AZURE_CLIENT_ID:+&client_id=$AZURE_CLIENT_ID}" \
      | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
    curl -fsSL -H "Authorization: Bearer $TOK" -H "x-ms-version: 2021-08-06" "$C3_BUNDLE_URL" | tar -xzf - -C "$DST"
    unset TOK
  else
    curl -fsSL "$C3_BUNDLE_URL" | tar -xzf - -C "$DST"
  fi
fi
export OLLAMA_MODELS="$DST/ollama"
mkdir -p "$OLLAMA_MODELS"
ollama serve >/tmp/ollama.log 2>&1 &
# Only the quote heads use Ollama (one embedding call per quote). Pulling the 4.7 GB
# embedding model before serving held every /read behind it: on a cold start (dev,
# 2026-10-28) the pull took 04:17:22-04:21:40 of a heads service that had its
# model loaded seconds later. Pull and load it beside the server instead; /v1/quote
# answers 503 (retried by the worker like any warming service) until the marker exists.
export C3_EMBED_READY_FILE="$DST/.embed_ready"
rm -f "$C3_EMBED_READY_FILE"
(
  for i in $(seq 1 60); do curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1 && break; sleep 1; done
  # A failed pull stops the replica, as it did when the pull ran first ($$ is the server
  # after the exec below), instead of leaving /v1/quote at 503 for good.
  ollama list | grep -q "^${C3_EMBED_MODEL%%:*}" || ollama pull "$C3_EMBED_MODEL" \
    || { echo "embedding model pull failed: $C3_EMBED_MODEL" >&2; kill "$$"; exit 1; }
  if [[ "${SOWSMITH_C3_QUOTE_WORDING:-off}" != "off" ]]; then ollama pull "$SOWSMITH_C3_QUOTE_WORDING" || true; fi
  # Load the model now, so the first quote after a cold start does not pay the load
  # (45-57 s per quote on 2026-10-28, longer than the brief writer it runs beside).
  # Held 30 minutes, enough to reach the compile that woke the replica; each quote
  # after that keeps Ollama's own idle timer, as before.
  curl -fsS "$OLLAMA_URL/api/embed" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$C3_EMBED_MODEL\", \"input\": [\"warm\"], \"keep_alive\": \"30m\"}" >/dev/null 2>&1 || true
  touch "$C3_EMBED_READY_FILE"
) &
export SOWSMITH_C3_CHECKPOINT="$DST/c3_base/checkpoint.pt"
export SOWSMITH_C3_ENCODER="$DST/c3_base/encoder"
[[ -f "$DST/c3_base/question_refit.pt" ]] && export SOWSMITH_C3_QUESTION_REFIT="$DST/c3_base/question_refit.pt"
[[ -f "$DST/c3_base/quote_heads_bundle.json" ]] && export SOWSMITH_C3_QUOTE_HEADS="$DST/c3_base/quote_heads_bundle.json"
export PYTHONPATH="$DST/c3_base/code:${PYTHONPATH:-}"
# Fewer fragmentation OOMs on long deals (serve.py sets the same default if this is missing).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$DST/c3_base/code"
exec python -m ml.c3.serve --port "${PORT:-8765}"
