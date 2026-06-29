#!/usr/bin/env bash
# Package CPU pdf_image gate for blob upload (gate_pdf_image.tgz).
set -euo pipefail
SRC="${1:-runs/pdf_image_gate/best}"
OUT="${2:-gate_pdf_image.tgz}"
if [[ ! -d "$SRC" ]]; then
  echo "Missing $SRC — run train_pdf_image_gate.py first" >&2
  exit 1
fi
rm -f "$OUT"
tar -czf "$OUT" -C "$(dirname "$SRC")" "$(basename "$SRC")"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "Upload: az storage blob upload --account-name purpulsedevstg01 \\"
echo "  --container-name ml-artifacts --name gate_pdf_image.tgz --file $OUT --auth-mode login"
