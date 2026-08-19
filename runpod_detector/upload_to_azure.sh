#!/usr/bin/env bash
# Push trained artifacts from RunPod back to Azure ml-artifacts.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RG="${AZURE_RG:-purtera-dev-rg}"
ACCOUNT="${AZURE_STORAGE_ACCOUNT:-purpulsedevstg01}"
CONTAINER="${ML_CONTAINER:-ml-artifacts}"

az account show -o table >/dev/null 2>&1 || az login
KEY="$(az storage account keys list --account-name "$ACCOUNT" -g "$RG" --query "[0].value" -o tsv)"

upload() {
  local file="$1" blob="$2"
  if [[ ! -f "$file" ]]; then
    echo "SKIP (not found): $file"
    return
  fi
  echo "==> Upload $blob"
  az storage blob upload \
    --account-name "$ACCOUNT" --account-key "$KEY" \
    -c "$CONTAINER" -n "$blob" -f "$file" --overwrite
}

upload span_heads_gpu.tgz span_heads_gpu.tgz
upload type_head_gpu.tgz type_head_gpu.tgz
upload gate_rubric_best.tgz gate_rubric_best.tgz
upload contrastive_type.tgz contrastive_type.tgz
upload contrastive_facet.tgz contrastive_facet.tgz
upload contrastive_router.tgz contrastive_router.tgz
upload gate_pdf_image.tgz gate_pdf_image.tgz

# Optional: push grown training log if you logged new rows on pod
upload _training_deepseek.db _training_deepseek.db

echo "Done. Next worker compile will fetch_ml.py pull fresh span_heads_gpu.tgz"
