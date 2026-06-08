# Parser-OS — full brain dump (training, models, data, metrics, GPU goals)

Everything discovered + the way we now do training. Hand this to a fresh Claude
on any machine: *"Read BRAIN_DUMP.md and GPU_HANDOFF.md, then continue."*
(The chat session is machine-local and does NOT transfer; these docs do.)

---
## 1. What the system does
Deal docs (docx/pdf/xlsx/notes) → **parse into atoms** (one per clause/row/line)
→ classify + extract (atom types, entities, sites, commercials) → **OrbitBrief
PM dashboard** (scope, financials, sites, open questions, head-start checklist,
schematic review). Output = `PM_HANDOFF.json` (envelope) read by the purpulse
frontend Deal Artifacts page.

## 2. Models + infra (where things run)
- **LLM (text reasoning/classification/extraction):** DeepSeek API (`deepseek-chat`,
  the user's `TEACHER_API_*` key) — fast/cheap teacher. Fallback: `qwen3:14b` on
  the **Mac Studio** Ollama.
- **Embeddings:** `qwen3-embedding:8b` (4096-dim, L2-normalized) on the Mac Studio.
  NEVER swap the embed model (thresholds/heads are pinned to it).
- **Vision (schematics):** `qwen2.5vl:7b` (and `:32b`) on the Mac Studio.
- **Mac Studio Ollama** = `http://100.114.102.122:11434` over **Tailscale**. The
  dev worker joins the tailnet (entrypoint) to reach it directly.
- **Cloud:** Azure. Dev worker = Container App Job `parser-os-worker-dev-eus2`
  (RG `purtera-dev-rg`). Artifacts + data in blob `purpulsedevstg01/ml-artifacts`.
  Frontend = Azure Static Web App (purpulse-frontend). GPU = RunPod A100.

## 3. THE NEW WAY WE DO TRAINING (two-tier, eval-gated, self-learning)
Core principle: **guess-free** — emit only what's verified; abstain → fall back
to the LLM; a model is promoted only if it does NOT regress (eval-gate + rollback).

**Tier 1 — CPU, instant-learning** (no GPU):
- **kNN feedback store** (`_feedback.db`, decide() seam): PM corrections become
  instant kNN exemplars. 119 corrections seeded.
- **Admission heads** (5: acceptance/risk/quantity/milestone/stakeholder): linear
  heads on FROZEN embeddings, precision-first (~90%), abstain when unsure.
- **Type-head deflector** (#70): trained LR over frozen embeddings; deflects only
  its high-confidence subset off the LLM.
These learn instantly from the log and are great at HIGH-PRECISION deflection.

**Tier 2 — GPU, fine-tuned representation** (RunPod): the ceiling-breakers.
Frozen-embedding heads cap (see §6); fine-tuning an UNFROZEN encoder per task
breaks it. Same eval-gate + rollback registry.

**The learning loop (now live on the worker):** every compile logs the LLM's
decisions (+ any PM correction) to the training log → `retrain_if_stale` /
`retrain_span_heads` rebuild the eval-gated heads on the grown log → `write_back_ml`
pushes log+heads back to blob → next run loads improved heads. Monotonic (never
worse). On startup `fetch_ml.py` pulls all artifacts to `/tmp/ml`.

## 4. Data assets (all in blob `ml-artifacts`)
- **`_training_deepseek.db`** — the training log, **~26.8k rows**. relation
  `atom_type` (~26k, dominated by `_keep`), + extraction relations: requirements
  (~494), commercial_line_items (~872), site_clusters (~618), stakeholders (~289),
  quantities (~233), risks/acceptance/milestones. Teacher=llm (silver) + pm (gold).
  Schema: relation, label, raw_text, masked_text (delexicalized, generalization-
  safe), teacher, weight, deal_id, split (held-out-by-deal hash).
- **`_feedback.db`** — 119 kNN corrections.
- **`_admission_heads/`** — 5 trained binary heads (.pkl + index.json metrics).
- **`_type_head/`** — the #70 deflector (LR, ~0.92 precision on its confident slice).
- **`_span_heads/`** — 5 #71 span heads (recall-tuned, eval-gated).
- **`dataset/`** — the **schematic YOLO dataset**: 345 page-images + labels,
  **15,420 boxes**, 30 firms, **held-out-by-firm** test (uri_telecom, va_electrical,
  ildot_firealarm, analytix_av, rva_electrical). data.yaml + test.yaml.

## 5. Current state / success metrics (honest)
- **Quality on deals: real + deployed.** Yonah: was missing the headline "110 TVs"
  → now **18/18 key facts**. Parser/receipt/quantity fixes live on every deal.
- **Deflectors firing in dev** (read-only warm base; write-back makes them learn
  cross-run). Type-head deflects ~9% of type calls @ 92% precision.
- **#71**: recall engine + AUGMENT live (more complete extraction now); self-gating
  SKIP armed (no-op until a relation is certified — see §7).
- **Frozen-embedding ceilings (the reason for GPU):** atom_type held-out acc
  **~0.65**; span recall **requirements 0.74 / sites 0.69 / commercial 0.67**;
  stakeholders/quantities 1.00 but on log-values not atoms (see §6).

## 6. KEY DISCOVERIES (what we learned)
1. **Stale-cache bug** (systemic): `parser_version` was a hand-typed constant
   never bumped → every cached deal frozen on its first parser version → fixes
   never reached cached deals (incl. worker). Fix: **code-fingerprint in cache key**.
2. **docx receipt index-drift**: parser numbers paragraphs incl. table/SDT ones
   that python-docx omits → out-of-range → hard-fail. Fix: **whole-doc fallback**.
3. **Quantity mis-typing**: "Net 30 days"→payment_term, "8-5 business hours"→
   access_window (were `quantity`). Verbatim-safe re-typing.
4. **`_keep` is the noise**: 58% of atom_type rows; the LLM labels the typed↔_keep
   boundary INCONSISTENTLY → ~90% of head errors are `_keep↔typed` flips. A kNN/LR
   can't learn a boundary the teacher doesn't draw consistently → 0.65 ceiling.
5. **Gold volume is NOT the lever for diverse relations** (demonstrated): adding
   305 verified requirement examples DROPPED recall 0.74→0.23 (more diverse points
   in a frozen space hurt generalization). The eval-gate correctly rolled it back.
6. **Frozen embeddings are the ceiling.** The fix is fine-tuning the representation
   (GPU) — proper distillation, not frozen-embedding+linear.
7. **Eval distribution gotcha** for #71: heads were eval'd on the log's extracted
   VALUES (held-out), not the production ATOMS. Representative for verbatim
   relations (requirements: clause=source); NOT for stakeholders/quantities
   (value=parsed substring) — so their "100%" is optimistic; the GPU span tagger
   trains/evals on the real distribution.
8. **Skip safety**: a relation's LLM call is skipped ONLY if recall≥0.93 AND
   verbatim Norm (value=atom text, agreement=1.0). Today nothing qualifies →
   safe no-op; auto-unlocks as fine-tuning lifts recall.

## 7. GPU training — dataset + success metric + what we want from EACH
Run on RunPod A100: `bash runpod_detector/run_all_gpu.sh` (see GPU_HANDOFF.md for
data fetch). Each prints tqdm + per-epoch HELD-OUT metrics + final verdict.

| GPU training | trains on | success metric | what we want |
|---|---|---|---|
| **Symbol detector** (`train_detector.py`, YOLOv8) | `dataset/` (15,420 boxes, train/val) | **held-out-FIRM mAP@50** on 5 unseen firms | **≥0.70** = universal symbol detection generalizes; 0.50-0.70 usable; <0.50 = need more firms |
| **atom_type #70** (`train_type_head_gpu.py`, fine-tuned bge-small) | log `atom_type` rows, held-out-by-deal | held-out **accuracy** + cutover precision@conf≥0.85, **vs 0.65 frozen** | beat 0.65; **≥0.85** = ship the type cutover (kills the ~98s LLM stage on the confident slice) |
| **span taggers #71** (`train_span_tagger_gpu.py`, fine-tuned bge-small) | log requirements/sites/commercial rows, held-out-by-deal | held-out **recall @ precision-floor 0.80**, vs frozen (0.74/0.69/0.67) | recall **≥0.93** per relation = the LLM extractor call for it is safely skipped (kills part of the ~325s stage), zero recall loss |

**How we know it's truly learning (not overfitting):** every epoch shows held-out
(by-deal/firm — never trained on) metric + delta vs the frozen baseline + best-so-
far. Held-out rising with train = learning. Held-out flat/dropping while train
climbs = overfit (flagged). The final held-out number IS the honest ceiling, and
the eval-gated registry only promotes a model that beats the incumbent.

## 8. Standing constraints (never violate)
- UNIVERSAL fixes only (no per-deal/keyword/per-model hacks).
- Guess-free: skip rather than emit a wrong label; eval-gate + rollback.
- DO NOT enter API keys yourself — user sets their own (DeepSeek/Anthropic).
- Commit CODE + tests ONLY — never .db/.pkl/.npz/artifacts/_dbg_*/deal data.
- Embed model pinned to qwen3-embedding:8b (4096-d). Mac Studio Ollama on Tailscale.

## 9. Open next steps
1. Run the A100 session → 3 verdicts; promoted heads auto-slot into the registry
   the worker fetches → deflectors get stronger.
2. #71 SKIP auto-unlocks per relation as fine-tuned recall crosses 0.93.
3. PM-correction UI (close the human loop; the write-back loop already persists).
4. Merge/observe the frontend head-start + schematic cards live; run Yonah.
