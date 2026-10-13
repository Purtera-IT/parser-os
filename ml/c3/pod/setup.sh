#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
# RunPod's image has Debian's Python, which refuses a system-wide pip (PEP 668).
# The pod is disposable and its torch is the CUDA build we want, so install
# next to it rather than in a venv that would hide it.
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
# The repo's own dependencies (python-docx and the rest) for the test suite's
# conftest. torch sits in an optional extra, so this keeps the image's torch.
pip install -q -e .
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
