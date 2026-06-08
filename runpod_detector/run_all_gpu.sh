#!/usr/bin/env bash
# One A100 session: train all 3 ceiling-bound heads, stream tqdm + per-epoch
# held-out metrics, then print a single combined verdict summary at the end.
set -uo pipefail

echo "Installing deps..."
pip install -U ultralytics "transformers>=4.44" datasets accelerate scikit-learn tqdm 2>&1 | tail -1

echo "===================== 1/3  SYMBOL DETECTOR (schematic) ====================="
python runpod_detector/train_detector.py 2>&1 | tee _gpu_detector.log

echo "===================== 2/3  ATOM_TYPE HEAD (#70) ==========================="
python runpod_detector/train_type_head_gpu.py 2>&1 | tee _gpu_typehead.log

echo "===================== 3/3  SPAN TAGGERS (#71) ============================="
python runpod_detector/train_span_tagger_gpu.py 2>&1 | tee _gpu_span.log

echo
echo "########################  FINAL VERDICTS  ########################"
echo "--- detector (held-out firms) ---"
grep -E "held-out mAP|VERDICT" _gpu_detector.log || echo "(see _gpu_detector.log)"
echo "--- atom_type #70 (vs 0.65 frozen) ---"
grep -E "fine-tuned held-out acc|cutover:|STRONG|BETTER|no gain" _gpu_typehead.log || echo "(see _gpu_typehead.log)"
echo "--- span taggers #71 (vs frozen, skip bar 0.93) ---"
grep -E "VERDICT|SKIP UNLOCKS" _gpu_span.log || echo "(see _gpu_span.log)"
echo "##################################################################"
echo "Pull trained weights back:  runpodctl send runs/"
