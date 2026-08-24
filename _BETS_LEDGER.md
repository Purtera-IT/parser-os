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

## The compound

\#1 + #2 together: a system that knows when it is wrong, with a coverage
guarantee, and spends PM attention exactly there. Both halves are built; both
wait only on labels — and the chip that produces labels shipped in
purpulse-frontend#117.
