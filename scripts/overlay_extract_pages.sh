#!/usr/bin/env bash
# Regenerate overlay PNG/JSON + extraction for selected PDF pages.
# Example (first three pages):
#   ./scripts/overlay_extract_pages.sh \
#     "/path/to/TSC and PTS Wireless Access Point Refresh - April 2026 V3.pdf" \
#     0 1 2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PDF="${1:?usage: $0 <pdf> <page> [page ...]}"
shift
: "${PYTHON:=python3}"
OUT_DIR="${OUT_DIR:-compiled_artifacts/overlay_runs}"
base="$(basename "$PDF" .pdf)"
safe="$(echo "$base" | sed 's/[^A-Za-z0-9._-]/_/g')"
SUB="${OUT_DIR}/${safe}"
mkdir -p "$SUB"
for p in "$@"; do
  stem="${SUB}/p$(printf '%04d' "$p")"
  "$PYTHON" -m orbitbrief_page_os.segmentation.detect_standalone \
    --pdf "$PDF" --page "$p" \
    --out "${stem}.png" --json-out "${stem}.overlay.json"
  "$PYTHON" -m orbitbrief_page_os.segmentation.extract_overlay_text \
    --pdf "$PDF" --json "${stem}.overlay.json" --out "$stem"
done
echo "Done → $SUB"
