#!/usr/bin/env bash
# Fetch and unpack the bundle, then serve. The bundle carries the checkpoint,
# the encoder and the ml/ + app/ code it was packed with.
#   C3_BUNDLE_URL      a read-only (SAS) URL to c3_base.tgz
#   SOWSMITH_C3_TOKEN  the bearer token the worker sends (a secret)
set -euo pipefail
: "${C3_BUNDLE_URL:?set C3_BUNDLE_URL to the c3_base.tgz blob URL}"
DST=${C3_DIR:-/models}
mkdir -p "$DST"
if [[ ! -f "$DST/c3_base/checkpoint.pt" ]]; then
  curl -fsSL "$C3_BUNDLE_URL" | tar -xzf - -C "$DST"
fi
export SOWSMITH_C3_CHECKPOINT="$DST/c3_base/checkpoint.pt"
export SOWSMITH_C3_ENCODER="$DST/c3_base/encoder"
export PYTHONPATH="$DST/c3_base/code:${PYTHONPATH:-}"
cd "$DST/c3_base/code"
exec python -m ml.c3.serve --port "${PORT:-8765}"
