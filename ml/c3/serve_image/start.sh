#!/usr/bin/env bash
# Fetch and unpack the bundle, start Ollama with the quote heads' embedding model, then serve.
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
for i in $(seq 1 60); do curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1 && break; sleep 1; done
ollama list | grep -q "^${C3_EMBED_MODEL%%:*}" || ollama pull "$C3_EMBED_MODEL"
if [[ "${SOWSMITH_C3_QUOTE_WORDING:-off}" != "off" ]]; then ollama pull "$SOWSMITH_C3_QUOTE_WORDING" || true; fi
export SOWSMITH_C3_CHECKPOINT="$DST/c3_base/checkpoint.pt"
export SOWSMITH_C3_ENCODER="$DST/c3_base/encoder"
[[ -f "$DST/c3_base/question_refit.pt" ]] && export SOWSMITH_C3_QUESTION_REFIT="$DST/c3_base/question_refit.pt"
[[ -f "$DST/c3_base/quote_heads_bundle.json" ]] && export SOWSMITH_C3_QUOTE_HEADS="$DST/c3_base/quote_heads_bundle.json"
export PYTHONPATH="$DST/c3_base/code:${PYTHONPATH:-}"
# Fewer fragmentation OOMs on long deals (serve.py sets the same default if this is missing).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$DST/c3_base/code"
exec python -m ml.c3.serve --port "${PORT:-8765}"
