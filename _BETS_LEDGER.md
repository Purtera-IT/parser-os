# The five bets — status and GO conditions

Plan-of-record for the post-audit build, companion to `_HEAD_LEDGER.md`.
Each bet rests on the substrate (receipts, abstain census, deal-keyed
corrections, the eval that refuses) — clean provenance is the precondition
all five quietly share, and it is the part already built and tested.

| # | bet | code | state |
|---|-----|------|-------|
| 2 | Conformal abstention | `app/core/conformal.py` | **BUILT + tested.** Guarantee verified empirically; refuses below n=5, refuses unreachable coverage, fails closed. |
| 1 | Delta head | `app/core/delta_head.py` | **BUILT + tested** with injected embedder. Censoring handled: negatives require review evidence. Waiting on tap data. |
| 4 | Self-deriving registry | `app/core/calibration.py` | **BUILT + tested.** Two constants re-derive their bounds from the routing-eval corpus; `drift_report()` runs in CI. Caught a bug in its own derivation on first run. |
| 5 | One backbone, many heads | `app/learning/multitask_table.py` (CPU half) | **Table assembled from real DBs**: 29,881 deduped rows (103k duplicates collapsed), atom_type 43 classes / 2,545 held-out / 218 pm-gold. Trainer = GPU. |
| 3 | Deal-graph fixpoint | — | **Design below.** Phase-2 reconciliation work per `_HEAD_LEDGER.md`. |

## GO conditions — say "go" when the condition is true

**GO #2a (conformal on the router)** — when the PM router hold-out reaches
~10 deals. `fit()` on the hold-out scores, swap `sim_floor`/`tau` for
`gate.decide()`. Zero GPU. The gate will state the coverage n actually buys.

**GO #1a (delta head live)** — when the correction store holds ≥8 corrections
on reviewed deals (any head, not just router). `build_examples()` off the
store, fit with the bge-small embedder, and its ranking replaces the static
disagreement queue ordering. Zero GPU.

**GO #5 (backbone training run)** — ready now, needs the A100 box:
1. `python -m app.learning.multitask_table` → `_multitask_table.db` (done locally, rerun on the box for freshness)
2. train one encoder (bge-base init) with task-conditioned heads over `(task, text) → label`; hold out by the `split` column, never re-split
3. eval gate per task against the per-task hold-out, PM-gold reported separately; promotion via the same refuse-capable discipline as `router_eval`
4. the encoder replaces the per-store embedders in `ContrastiveTypeKNN` dirs — one embedding space, so feedback-store corrections transfer across heads

**GO #3 (deal-graph fixpoint)** — with Phase-2 reconciliation. Design intent:
each document is a message into a persistent deal graph; reconciliation is
belief propagation with authority ranks as priors and the neural reconciliation
head as the learned potential; every belief update carries the receipt of the
message that caused it ("quantity became 56 because this rank-90 email
arrived"). The COPPER contradiction-edge tests are the first fixture.

**GO #6 (suppression head — "should this have been kept?")** — when the
`suppression` head holds ≥150 judgments across ≥3 deals. Every drop stage asks
one question, so dedup, boilerplate, rollup and the substance gate all train
one head; the stage travels as a feature, not a separate head.

*Why it exists:* `CompileResult.suppressed_atoms` recorded every dropped atom
all along and nothing carried it to the envelope — 4,591 on COPPER_001, 1,203
on an email-heavy deal, none ever shown to a person. The SHAPE audit found
three real deletions in there by hand: 81 per-plate cable quantities, 39 folds
where the numbers differed (including a bid deadline), and one stage
(`table_rollup`) that turned out correct. Nobody should find the fourth by hand.

*Where the labels come from:* the **Dropped** stage of the labelling workspace.
A fold carries the survivor it folded into, because "is this a duplicate"
cannot be answered without the thing it was a duplicate of — so those verdicts
ARE the dedup pair-head's training set. **There is no separate dedup labelling
job to remember.**

*Wiring:* parser-os `SOWSMITH_SUPPRESSED_IN_ENVELOPE=1` → Platform-infra
`suppression` head (PR #373) → purpulse **Dropped** stage (PR #634).

*Gate:* same discipline as `router_eval` — a candidate scoring worse than the
current hand-written rules on held-out judgments does not promote. And two
invariants stay as HARD gates whatever the head says: it may never fold atoms
that state different numbers, or that belong to different places.

**GO #7 (rule thresholds)** — when the `rule` head holds ≥80 judgments per rule
on the ambiguous band. 15 SemanticRules decide how atoms FORM, their thresholds
are hand-tuned, and a live harvest put 435 of 847 distinct decisions (51%)
within 0.08 of their boundary. Tier A fits one number per rule and ships as a
JSON file — no model at runtime. Labels come from the **Rules** stage
(Platform-infra PR #370, purpulse PR #630); see `_STEP4_RULE_LABELLING.md`.


## The compound

\#1 + #2 together: a system that knows when it is wrong, with a coverage
guarantee, and spends PM attention exactly there. Both halves are built; both
wait only on labels — and the chip that produces labels shipped in
purpulse-frontend#117.

## Phase-2 reconciliation (bet #3, first slice) — 2026-08-24

**Built:** `app/core/reconcile.py` — every contradicts-cluster becomes a
conflict set; the authority lattice resolves it with full receipts (winner,
rank, every superseded claim, the binding edge ids), or REFUSES on a
top-tier tie (margin = 5, the lattice's narrowest real gap) and surfaces
both receipts as a PM judgment. No atom is mutated — a verdict layer, so
replay keeps verifying. Wired into the envelope as `reconciliation`
(additive key). 8 tests, COPPER shape as the founding fixture.

**Measured first:** the 447 labelled edge pairs are rule-proposal vs actual:
supports 90% precise, **contradicts 41%**, excludes 0/101 confirmed but
UNMEASURABLE (the label space never offered the labeller 'excludes'). The
envelope surface carries these numbers so the conflict list reads as leads,
not verdicts.

**The learned potential refused its own audition:** a deal-disjoint split of
5 deals leaves a 23-row single-class test side. GO condition: edge labels
from ≥12 deals with ≥2 classes on the held-out side — the same bar
router_eval enforces. Until then the potential is injectable and the
lattice is the incumbent.
