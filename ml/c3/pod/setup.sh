#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
pip install -q "transformers>=4.48" pytest
python -m pytest -q tests/test_c3_*.py
python - <<'PY'
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
for name in ("answerdotai/ModernBERT-large",):
    AutoTokenizer.from_pretrained(name); AutoModel.from_pretrained(name)
name = "Qwen/Qwen2.5-7B-Instruct"
AutoTokenizer.from_pretrained(name); AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
print("models ready")
PY
