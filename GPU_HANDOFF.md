# Parser-OS GPU training + system handoff

State as of v57.23. Use this to continue on another machine (the chat session
itself does NOT transfer — start a fresh Claude there and paste/point it here).

## What's deployed (parser-os main, tags v57.16 → v57.23)
- **Parser fixes**: stale-cache fingerprint (recovers dropped headline facts on
  every deal + worker), docx receipt whole-doc fallback (prod gate passes),
  quantity re-typing (Net30→payment_term, hours→access_window).
- **#70 type-head deflector** (CPU, learnable, eval-gated): deflects ~9% of type
  calls @ 92% precision. Flag `SOWSMITH_TYPE_HEAD_DEFLECT`.
- **#71 span-extractor**: recall engine + AUGMENT (live recall gain) +
  self-gating SKIP (safe; auto-enables a relation when recall≥0.93 & verbatim).
  Flags `SOWSMITH_SPAN_AUGMENT`, `SOWSMITH_SPAN_SKIP`.
- **50-thought PM head-start** (orbit dashboard `head_start`) + UI cards merged
  (purpulse-frontend#13).
- **Live-learning write-back** on the worker: after each compile it retrains the
  eval-gated heads on the grown log + pushes log+heads back to blob (cross-run).

## The deflectors are ON in dev (worker `parser-os-worker-dev-eus2`)
Env flags set: `SOWSMITH_TYPE_HEAD_DEFLECT, _ATOM_TYPE_DEFLECT, _SPAN_ADMISSION,
_FEEDBACK_STORE_DB, _ADMISSION_REGISTRY, _TYPE_HEAD_DIR, _SPAN_HEAD_DIR,
_TRAINING_LOG_DB, ML_ARTIFACT_DIR=/tmp/ml`. Artifacts in Azure blob
`purpulsedevstg01 / ml-artifacts` (fetched on startup by `fetch_ml.py`).

## The honest ceiling (why GPU)
CPU heads use FROZEN embeddings + linear/kNN → cap (atom_type ~0.65, requirements
~0.74). Adding gold does NOT help (demonstrated: it regressed; eval-gate rolled
back). The lever is **fine-tuning the representation = GPU**.

## GPU training — run on an A100 (RunPod)
Scripts in `runpod_detector/` (in git). Data is NOT in git — fetch from blob
`ml-artifacts`: `_training_deepseek.db` (the log) and the YOLO `dataset/`.

```bash
# from any machine with the repo + az login:
git clone https://github.com/Purtera-IT/parser-os && cd parser-os
KEY=$(az storage account keys list --account-name purpulsedevstg01 -g purtera-dev-rg --query "[0].value" -o tsv)
az storage blob download --account-name purpulsedevstg01 --account-key "$KEY" -c ml-artifacts -n _training_deepseek.db -f _training_deepseek.db
az storage blob download-batch --account-name purpulsedevstg01 --account-key "$KEY" -s ml-artifacts/dataset -d runpod_detector/dataset
# then on the A100:
bash runpod_detector/run_all_gpu.sh        # trains all 3, prints per-epoch held-out + final verdicts
```
Each trainer prints **tqdm progress + per-epoch HELD-OUT-by-deal metrics vs the
frozen baseline** (rising held-out = truly learning; held-out flat while train
climbs = overfit). Final verdicts: detector mAP@50, atom_type acc (beat 0.65?),
span recall (cross 0.93 → SKIP unlocks).

## Standing constraints (NEVER violate)
- Fixes UNIVERSAL (no per-deal/keyword/per-model hacks).
- Guess-free: skip rather than emit a wrong label; eval-gate + rollback.
- DO NOT enter API keys yourself; the user sets their own.
- Commit CODE + tests ONLY — never .db/.pkl/.npz/artifacts/_dbg_*/deal data.
- Models pinned: qwen3:14b, qwen3-embedding:8b (4096-d), qwen2.5vl:7b/32b.
  Remote Ollama http://100.114.102.122:11434 (Tailscale). Text teacher = DeepSeek
  (TEACHER_API_* — user's key).

## Running from a MacBook Pro (the new setup)
The control machine is now a MacBook Pro (macOS); GPU training runs on a RunPod
A100. This all works from the Mac — everything is cross-platform:
- **New Claude chat**: `git clone https://github.com/Purtera-IT/parser-os` then
  *"read BRAIN_DUMP.md and GPU_HANDOFF.md and continue."* (Chat history does not
  transfer between machines; these docs are the context.)
- **Tools to install on the Mac**: `git`, Azure CLI (`brew install azure-cli`,
  then `az login`), and `runpodctl` (https://github.com/runpod/runpodctl). That's it.
- **GPU training needs NO Ollama / NO Tailscale.** The fine-tuners embed on the
  RunPod GPU via `transformers`; the detector uses `ultralytics`. They only need
  the data (pulled from Azure blob) + the scripts (in the repo). The Mac Studio
  Ollama (`100.114.102.122`, Tailscale) is only for *local deal compiles* — not
  for GPU training.
- **Flow from the Mac**: `az login` → download data from blob (commands above) →
  `runpodctl send` the bundle to the A100 → on the pod `bash runpod_detector/run_all_gpu.sh`
  → `runpodctl send runs/` to pull trained weights back.
- **Local compiles on the Mac (optional)**: need Tailscale to the Mac Studio
  Ollama + your DeepSeek key in env (`TEACHER_API_BASE/KEY/MODEL`). Not needed
  for GPU training.

## Open next steps
1. Run the A100 session (above) → get the 3 verdicts; promoted heads slot into
   the eval-gated registry + the worker fetches them.
2. #71 SKIP auto-unlocks per relation when its fine-tuned recall ≥0.93.
3. PM-correction UI (close the human loop).
