#!/usr/bin/env bash
# Fetch and unpack the bundle, start Ollama with the quote heads' embedding model, then serve.
#   C3_BUNDLE_URL      a read-only (SAS) URL to c3_base.tgz
#   SOWSMITH_C3_TOKEN  the bearer token the worker sends (a secret)
#   C3_DIR             where the bundle and Ollama models live (mount a volume here to skip re-downloads)
set -euo pipefail
: "${C3_BUNDLE_URL:?set C3_BUNDLE_URL to the c3_base.tgz blob URL}"
DST=${C3_DIR:-/models}
mkdir -p "$DST"
if [[ ! -f "$DST/c3_base/checkpoint.pt" ]]; then
  curl -fsSL "$C3_BUNDLE_URL" | tar -xzf - -C "$DST"
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
cd "$DST/c3_base/code"
exec python -m ml.c3.serve --port "${PORT:-8765}"
