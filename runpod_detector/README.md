# GPU training assets (RunPod)

Training + labeling scripts for the eval-gated heads that run on a rented A100:

- **atom_type head** (`train_type_head_gpu.py`, `train_type_head_v2.py`) and the
  rubric labeling stack (`RUBRIC.md`, `rubric_*.py`, `taxonomy.py`, guides in
  `PM_GOLD_GUIDE.md` / `SETUP_LABELING.md`).
- **Span taggers** (`train_span_tagger_gpu.py`, `train_span_tagger_v2.py`).
- **Contrastive encoder + kNN** (`train_contrastive_encoder_gpu.py`,
  opt-in `train_contrastive_qwen3_lora.py`, see `VLLM_EMBEDDING_LORA.md`).
- **PDF embedded-image gate** (`train_pdf_image_gate.py`, `pack_gate_pdf_image.sh`).

One-session driver: `bash runpod_detector/run_all_gpu.sh` (see `GPU_HANDOFF.md`
at the repo root for the full flow: data from Azure blob, weights back via
`runpodctl`).

## Retired: universal symbol detector (2026-08-26)

This directory used to also hold the YOLO symbol-detector training assets
(`prepare_yolo_data.py`, `train_detector.py`, `verify_gold.py`,
`LABELING_GUIDE.md`). Pure construction schematics are no longer accepted into
Purpulse, so that effort was retired along with the schematic ML modules in
`app/core/`. The labeled `dataset/` remains on Azure blob
(`purpulsedevstg01/ml-artifacts`, `dataset/*`) if it is ever needed again; it
was never in git (`.gitignore` still excludes local copies).
