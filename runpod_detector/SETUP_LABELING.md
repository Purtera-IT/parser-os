# Labeling system — clean setup (for hand-audit)

Everything needed to take atom-type from "looked good, was unusable" to a verified,
honestly-measured base. Built per the architecture review. Each piece is small and
auditable. **Nothing here can inflate a number** — the leak, the teacher-as-truth
grading, and the destructive cleaner are all fixed.

## The pieces (all committed)

| File | Role | Audit it by |
|---|---|---|
| `taxonomy.py` | ONE source of truth: 7 facets, 44 micro-types, VALUE_LIGHT, REVIEW sink | `python taxonomy.py` → covers 100% of DB labels (UNMAPPED=none) |
| `_split_util.py` | canonical train/holdout (recorded column, holdout-wins, hash fallback) | `python _split_util.py` → 17.9% / 97.2% hash-vs-recorded disagreement |
| `build_gold_eval.py` | the **keystone**: frozen, holdout-only, stratified gold set + scorer | `build` → CSV w/ ~39% _keep, all 7 facets, novel slice |
| `rubric_relabel_facets.py` | rubric+vote+context → facet_clean (micro preserved, holdout untouched) | reads `TEACHER_API_KEY`; reports AMBIGUOUS% + rubric-vs-teacher agreement |
| `clean_labels_universal.py` | confident-learning drop — **now split-safe** (was deleting 99% of holdout) | diff vs old: `split()` reads recorded column |
| `train_contrastive_encoder_gpu.py` | the "best build" — **now reads canonical split** | `split()` → `_split_util` |
| `rubric_relabel_deepseek.py` | binary gate relabeler — **now canonical split** | — |
| `app/core/type_head.py` | runtime head — canonical split + floor 0.85→**0.95** + abstain log | — |

## Canonical numbers (regenerated under the taxonomy + split)
```
_training_deepseek.db: 26,991 atoms | split train 22,504 / holdout 4,487
  _keep 47.4% · SITE 21.3% · META 10.9% · WORK 6.8% · PARTY 5.8%
  COMMERCIAL 5.2% · COMPLIANCE 2.4% · TIMING 0.2%
```
Top 3 (_keep+SITE+META) = ~80% of atoms → covering those three well = the coverage goal.

## The pipeline (run order)
```
1. python build_gold_eval.py build --db _training_deepseek.db --n 1800
   → PM adjudicates gold_eval_v1_TOADJUDICATE.csv (fills gold_facet). 2 PM-days.   [HUMAN]
2. TEACHER_API_KEY=sk-... python rubric_relabel_facets.py
   → _training_facet.db (facet_clean on train rows). ~$10-30, <1hr.               [NEEDS KEY]
3. (optional) python clean_labels_universal.py   → drop contradictory train rows
4. train the head/contrastive on facet_clean (RunPod for contrastive)             [NEEDS RUNPOD]
5. python build_gold_eval.py score --gold <adjudicated.csv> --pred <model.csv>
   → per-facet precision/coverage on TRUTH. The first honest number.
```

## What needs a human (everything else is automated)
- **DeepSeek key** (`TEACHER_API_KEY`) → unblocks step 2 (label-ceiling fix).
- **~2 PM-days** → step 1 adjudication (gold = truth). Granularity = **facets**; micro preserved.
- **RunPod** → step 4 contrastive train, but ONLY after steps 1-3 (else paying to measure garbage).

## Invariants this setup guarantees (the anti-self-deception list)
- Holdout is never trained on, relabeled, or cleaned (gold is the only holdout truth).
- Every metric is vs the human gold set, not the teacher.
- AMBIGUOUS is a first-class label (trains abstention; not dropped).
- No silent drops: unmapped micro-types → REVIEW; cleaner drops are split-checked.
- Micro labels preserved → fine heads later without redoing PM work.
```
