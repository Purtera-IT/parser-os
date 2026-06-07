# Universal Symbol Detector (RunPod)

**Goal:** a class-agnostic symbol detector that finds *where* every symbol is on
any schematic (firm-independent). Meaning is assigned per-document by LegendIndex,
so it's **universal** — new firms / new symbols need zero retraining.

## Why a detector (vs VLM / pixels)
- Object detection IS the task (localize + box every symbol). A YOLO detector is
  purpose-built, runs **local + fast**, and beats a general VLM and pixel-match
  at detection. It replaces `region_proposals` + the pixel objectness head.
- Class-agnostic ("symbol") = universal. Per-doc legend = the vocabulary.

## Steps
1. **Local (uses cached VLM labels, ~free):**
   `python -X utf8 runpod_detector/prepare_yolo_data.py`
   -> builds `runpod_detector/dataset/` (images + YOLO labels + data.yaml).
2. **Upload** `runpod_detector/` to the RunPod pod.
3. **On RunPod GPU:** `pip install ultralytics && python train_detector.py`
4. **Download** `runs/symbol_detector/weights/best.pt`.
5. **Local:** set `SOWSMITH_SYMBOL_DETECTOR=/path/best.pt` -> the pipeline loads it
   as the universal detector (see `app/core/schematic_detector.py`).

## Honest notes
- This is the GPU spend that pays off (vs VLM LoRA, which has small blast radius).
- More gold boxes = better. The prep script accumulates labels (cached) every run.
- After training, the local stack is: detector (where) -> LegendIndex (what) ->
  verify+abstain+human+self-heal (100%). No full-time VLM.
