# Brain dump — session 2026-06-08 (B type-head, schematic data, architecture)

Hand to a fresh Claude on another machine: *"Read BRAINDUMP_SESSION_2026-06-08.md,
ARCHITECTURE_PROPOSAL.md, RUBRIC.md, then continue."* The chat does not transfer; these docs do.
Builds on BRAIN_DUMP.md + GPU_HANDOFF.md (prior state). Control machine this session = Mac Studio
M3 Ultra (96GB, has Ollama qwen3:14b/32b/embedding:8b). GPU = RunPod RTX PRO 6000 (96GB).

================================================================
## 1. THE HEADLINE DISCOVERIES (read these first)
1. **The atom_type ceiling is LABEL AMBIGUITY, not the model.** Two strong independent LLMs
   (DeepSeek teacher vs local qwen3:32b) agree only **59%** on keep-vs-typed with no rubric. You
   cannot score above your label agreement. 70% of type-head errors touch the `_keep` boundary.
2. **A universal role-based RUBRIC (runpod_detector/RUBRIC.md) raised inter-model agreement 59% → 95%.**
   The boundary was never irreducibly ambiguous — it was *undefined*. This is the single biggest win.
3. **BUT cleaning labels did NOT break the gate past ~0.82.** Measured against a CLEAN test set
   (qwen3:14b∩32b agreement, ~95% reliable), the LoRA-classifier gate still caps ~0.79–0.82. So the
   remaining cap is the **model class** (an embedding+linear head *pattern-matches*; it can't
   replicate the labeler's multi-step *reasoning*) + ~15% irreducible ambiguity (32b vs 14b only
   agree ~85% on the hard cases).
4. **The detector is DATA-bound:** held-out-FIRM mAP@50 = **0.113** (trained 10 min on RunPod). 24
   firms is too few; symbols differ per firm. Lever = more labeled firms, NOT compute.
5. **Big objects are cheap to detect:** note_block/legend/room need ~30–50 examples; tiny devices need
   volume. (Informs the multi-class detector plan.)

================================================================
## 2. B — TYPE-HEAD: the full journey + numbers
Goal: gate (keep-vs-typed, the big LLM-deflector) ≥0.90 held-out + confident slice ≥0.90 precision +
fine-type as high as possible. Atom_type ~26.9k rows / 41 labels, dominated by `_keep` (53%).

Ladder (held-out-by-deal, all measured this session):
- frozen-embedding baseline (qwen3-embedding:8b + linear) ... 0.65
- v1 single-stage fine-tune (bge-small) .................... 0.573
- v2 two-stage (keep-vs-typed gate + fine-type, class-wt) .. 0.594
- v2 + universal label cleaning (clean_labels_universal) ... 0.619
- **8B-encoder LoRA gate (Qwen3-Embedding-8B), 3k labels ... 0.822** (still climbing @ ep4)
- 8B gate, 8k labels (noisy 14b expansion) ................. 0.773 (more data HURT = label noise)
- **8B gate, 6.6k CLEAN labels (14b∩32b agreement) ......... 0.791** (vs CLEAN test; load-best ep2)

Key sub-findings:
- bge-small (33M) < frozen qwen3-embedding:8b (8B) — that's why v1/v2 lost to the frozen baseline.
- 8B encoder is a real lever (0.76→0.82) but overfits fast on a few-thousand rows and over-predicts
  "typed" (recall ~0.97, precision ~0.74) — a precision/threshold problem.
- Cleaning measured: train labels 82% self-consistent (14b vs 32b), TEST 87% — so the noisy yardstick
  was capping measurable accuracy at ~0.87. Cleaning lifted the ceiling but the model still hit ~0.79.
- **Conclusion: the embedding-classifier approach caps ~0.82. To break it → ARCHITECTURE_PROPOSAL.md
  (contrastive space + kNN + cascade), or distill a small instruct model. Do NOT just add more data.**

Deployable NOW: the confident slice ~85% of atoms @ ~0.84 precision (guess-free deflection, LLM fallback).

Fine-type: collapse 41 micro-types → 7 universal FACETS (= dashboard sections), measured 0.846
two-model agreement (vs un-learnable 41-way). 7 fuzzy types have explicit rulings in RUBRIC.md.
NOT yet trained — staged as the next B experiment.

================================================================
## 3. THE TOOLING WE BUILT (all in runpod_detector/, committed)
- `RUBRIC.md` — universal keep-vs-typed rubric (role-based, all trades) + 7-facet fine-type rubric.
- `rubric_adjudicate.py` — apply rubric; `agree` mode proved 59→95%; `relabel` mode generates labels.
- `clean_labels_universal.py` — embedding confident-learning denoise (8B embeddings, drop contradictions).
- `gold_eval_keep_boundary.py` — independent qwen3 adjudication to build a clean yardstick.
- `type_agreement.py` — measures which of the 41 types are reproducible (defined the 7 facets).
- `train_type_head_v2.py` — two-stage + class-weight + LoRA + tokenizer-save (supersedes _gpu version).
- `train_span_tagger_v2.py` — honest val/test threshold + Wilson lower-bound for the SKIP gate.
- `train_gate_rubric.py` — the 8B-LoRA gate trainer (load-best added). Runs on RunPod.
- `expand_gate_labels.py`, `relabel_clean.py`, `diagnose_labels.py` — label expansion / ensemble-clean
  / noise diagnosis (relabel_clean keeps 14b∩32b agreement; runs on the pod with local Ollama+CUDA).
- Bug fixes: `train_detector.py` best-weights path (use model.trainer.best); tokenizer-save in the
  original _gpu trainers (they degenerate to all-UNK without it).
NOTE: data files (_rubric_gate_data*.json, _*.npz, _*.jsonl, runs/) are gitignored — code only.

================================================================
## 4. C — SPAN TAGGERS (status)
First pass (held-out by deal): requirements 0.789 (below 0.93 skip bar), site_clusters 1.000 (only 13
held-out positives), commercial 0.958 (24 positives, unstable). Real signal but tiny-N — `train_span_
tagger_v2.py` (honest threshold + Wilson bound) NOT yet run, and the same rubric/ensemble cleaning +
the contrastive-kNN architecture should apply. C is on deck after B's fine-type.

================================================================
## 5. DETECTOR (A) + the SCHEMATIC DATA PIPELINE + Luke & Danny
- Detector A: YOLOv8m, trained on RunPod (~10 min). Seen-firm val mAP@50 0.27; **held-out-firm 0.113**
  → data-bound. Needs more firms; compute is a non-issue.
- Corpus built: `~/Downloads/schematic_corpus.zip` — 24 public plan sets / 1,350 large-format pages,
  scraped from public permit/bid portals (WebSearch + curl, filtered to drawing sheets). Includes
  `scrape_more.sh` to grow it. (This is how the original dataset was sourced — public portals.)
- **Luke & Danny labeling packages** (`~/Downloads/LUKE_package.zip`, `DANNY_package.zip`, also in blob
  ml-artifacts with 7-day SAS links): each self-contained, idiot-proofed. Contents: START_HERE.md (3
  unbreakable rules), ROBOFLOW_STEPS.md (object-detection, **Tile 3×3 on export** = critical), LABELING_
  GUIDE.md, EXTRA_VALUE_JOBS.md (tiered metadata), examples (GOOD floor plan vs SKIP detail sheet),
  templates (legend_capture.csv, sheet_metadata.csv, device_type_classes.txt), PRACTICE_PAGE.png +
  its legend (Alameda T0.01). Split BY FIRM so each has their legends. **Golden rule: label IDENTICALLY
  — sync on the shared practice page first** (the 59→95% consistency lesson applied to boxes).
- **NEW expanded scope (this session): multi-class detection.** Luke & Danny now document rooms
  (polygons), room_tags, note_blocks (each page's notes/legend), arrows/callouts, devices, and
  device_tags (text above devices). See ARCHITECTURE_PROPOSAL.md "schematic multi-class detector":
  detect → OCR → geometric association = full schematic understanding. Phase 1 = device/device_tag/
  room/room_tag (plenty of examples); Phase 2 = arrow/callout/note_block/legend (need ~100+ pages).
  Relationships (tag→device, device→room, arrow→target, room#→note) are a SEPARATE association layer.

================================================================
## 6. THE NEW ARCHITECTURE PROPOSAL (the "perfect" build)
See ARCHITECTURE_PROPOSAL.md. Summary: solve the FAST/LEARNS-FROM-TEXT/ACCURATE trifecta at the SYSTEM
level via a confidence-routed cascade — (1) supervised-CONTRASTIVE encoder so the space is sorted by
the keep-vs-typed/facet boundary (fixes why kNN failed: a general-meaning space has mixed-label
neighbors), (2) kNN over that space + the feedback store = fast + learns-from-text instantly, (3)
confidence gate (guess-free), (4) LLM/distilled-instruct fallback on the ambiguous minority, whose
outputs feed the store → coverage grows. Eval-gated, monotonic, universal. Honest: ~15% is
irreducibly ambiguous and always abstains to reasoning — "perfect" = the SYSTEM, not one model at 1.0.

================================================================
## 7. NEXT STEPS (in priority order)
1. **Build the trifecta gate** (ARCHITECTURE_PROPOSAL.md): supervised-contrastive fine-tune the encoder
   on the CLEAN rubric labels (on RunPod, ~30 min) → kNN over the new space + feedback store → confidence
   gate → measure. Target: clear the 0.82 wall on the separable slice, with instant-learning.
2. **Fine-type facets**: relabel atom_type → 7 facets (rubric) → train on the contrastive space.
3. **C**: run train_span_tagger_v2.py (honest threshold) + rubric/ensemble clean.
4. **Detector**: ingest Luke & Danny's YOLOv8 export → self-supervised pretrain on 1,350 pages →
   bootstrap (detector pre-labels → human verifies) → measure held-out-firm lift → add more firms.
5. **Wire the deployable confident-slice gate now** (85% @ 0.84, guess-free) for real LLM savings while
   the trifecta is built.

## STANDING CONSTRAINTS (unchanged): universal fixes only; guess-free (skip > wrong); eval-gate +
## rollback; commit CODE+tests only (never .db/.pkl/.npz/artifacts/data); models pinned
## (qwen3:14b, qwen3-embedding:8b 4096-d, qwen2.5vl:7b/32b); user sets their own API keys.
