# ml/c3: the C3 model, in code, untrained

This package is the model that docs/C3_HEADS.md describes, built so it can be
read, reasoned about and tested before anyone spends a GPU hour on it. It
builds, runs a forward pass, and computes every loss on a synthetic deal.
It has never been trained on real data.

- **Self-contained.** It reads `app/core/label_heads.json` and
  `app/core/atom_types.json` by path and never imports `app`. Nothing in `app/`
  imports it, and `pyproject.toml` does not package it, so the parser service
  and parser-os-worker gain no dependency (tests/test_c3_isolation.py).
- **Dependencies:** `torch` only (`requirements.txt`). `transformers` is
  optional, for the pretrained encoder.
- **No customer text.** The fixture is invented. Real deals enter only
  through `DealExample.from_training_blob` at training time, outside the repo.

```
pip install -r ml/c3/requirements.txt
pytest tests/test_c3_model.py tests/test_c3_isolation.py
python -m ml.c3.card --deal ml/c3/fixtures/synthetic_deal.json --params
```

## 1. How data gets in

| Step | File | What happens |
|---|---|---|
| Schema | `schema.py` | Every head in label_heads.json becomes a list of **labeling opportunities**: `col:label_type`, `read:sow_coverage`, `rel:answers`… Each carries the head's question, the reading's description and one description per answer. Universal descriptions are scrubbed of company words. 80 opportunities today. |
| Contract | `data.py` | One JSON per deal: atoms (text, entered_at, doc, section, speaker role and side), each with an optional label (`label_type`, `about`, `wants`, `reads_set`, `note`), edges, and an optional outcome. `from_training_blob` joins the real blob's labels onto atom rows by `label_key`. |
| Split | `data.featurize` | **Inputs:** the line, number-magnitude tokens, doc kind, role/side, time, same-document and same-section structure. **Targets:** every label field. The only label fields that are also inputs are the stage and geo context slots, dropped to null at random. A test relabels every atom and checks the outputs do not move. |
| Note | `notes.py` | `split_note` (same grammar as the ingest; a test checks they agree) gives the universal WHY and the company line. The WHY is verdict-masked before encoding. `extract_programs` turns "2 techs x 8 hours = 16 hours" into an executable program, kept only if it reproduces its own stated number. |

A missing reading on a labeled card is **unknown, not negative**, by default
(`absent_is_negative=False`, as multitask_table does). The training thread
can flip it once removed readings are recorded.

## 2. The architecture

```
atoms ─ AtomEncoder ─ DealGraph (foresight mask, same-doc/section biases) ─ h_i
  h_i ─► CONTENT z_c            (gradient-reversed company adversary)
  h_i ─► CONSEQUENCE q_i        mixture density, trained toward the EMA hindsight
                                encoder that reads the whole finished deal
  ctx ─► CONTEXT slots          stage, geo; null at random
  r_i = R(z_c, E[q_i], ctx) + penalized residual        RATIONALE
  r_i ─► universal DESCRIBED HEADS, one per space (folds.py)
  r_i ─► relations (described pair scorer, causal mask) and boxes for governs
  q_i ─► absence head: per claim slot, "empty now, filled later"
  [r_i; q_i] ─► CONDUCT = Σ_m α_k,m · gate_m · A_m  ─► company described head
```

| Design piece | Where | Status |
|---|---|---|
| Deal graph transformer, foresight mask, structural biases | `model.DealEncoder` | built; tested that line i never sees a later line |
| Content + company adversary | `model.C3Model.p_c`, `adversary` | built |
| Hindsight-JEPA, Consequence as a distribution | `q_head`, `hindsight_targets`, `losses.hindsight_loss` | built (mixture NLL + VICReg variance) |
| Rationale bottleneck + residual | `rationale`, `res_down/res_up` | built |
| Described heads (the new part) | `folds.py` | built; section 3 |
| Box containment for governs | `_box_containment` | built |
| Absence head, with free targets from the timeline | `absence`, `losses.absence_targets` | built |
| Policy genome: sparse codes, low-rank atoms, one readable threshold θ per atom | `conduct` | built; universal outputs tested identical under any code |
| New company: freeze all but its code, seed it from its written rules | `add_company`, `freeze_for_new_company` | built |
| Question engine input | `sample_consequence` | built; the engine is not |
| Rationale writer, neural program head, precedent memory, question engine | none | **not built**: each needs trained r_i or q_i first (C3_HEADS.md build order steps 6 to 8) |

`C3Config()` is small enough for CPU tests. `C3Config.design()` gives the
sizes in the design docs (1024-wide, 6 layers) and is meant to run with
`text.HFEncoder` (ModernBERT-large) on a GPU. The default `HashingEncoder` is
a stand-in so nothing downloads in CI.

## 3. The new part: the labeling text shapes the space

Every labeling opportunity comes with text a person wrote for a person: the
question, the reading's description, a description per answer. And every
labeled line comes with a WHY. A normal model throws all of that away and
learns one weight vector per class. Here the text does three jobs, all in
one space, because one encoder reads the lines, the WHYs and the
descriptions.

**1. Answers are points, not weights.** The logit for answer *v* is the
similarity between the line's latent and the encoded description of *v*,
plus a prior that is also read from *v*'s text. A new reading gets a working
head from its words alone, with no new parameters (tested), and two answers
described alike start out alike.

**2. The question folds the space.** From the opportunity's description, a
shared hypernetwork generates a few folds. A fold reflects the half-space on
one side of a hyperplane onto the other:

```
s = <x, n> − b        x ← x + 2·σ·relu(−s)·n          (σ = 1: a full fold)
```

A full fold is an isometry on each side and identifies mirror images, so
after it the head cannot tell which side a line was on (tested). That is the
mechanism for "this question doesn't care about that difference": "is it in
the SOW?" should not care who said it, while "who owns the task?" must. Each
question gets the invariances its wording implies, and because the folds are
generated from the text, questions worded alike get alike folds, so a rare
head borrows a common head's geometry through its description. The answer
prototypes are folded by the same folds, so line and answers sit on the same
sheet. σ between 0 and 1 is a partial fold (a crease).

**3. The WHY is the path from a line to its answer.** In training, the
encoded WHY does two things:
- `why_align`: r_i is pulled toward its own WHY and away from other lines'
  WHYs (InfoNCE, verdict words masked).
- `why_sufficiency`: the WHY *replaces* r_i, and the same universal heads,
  through the same folds, must still reach the gold answer from it.

So a WHY is not a side label. It is a displacement in the same space as the
descriptions, one that has to carry the line to its answer by itself. At
inference nobody writes a WHY; r_i has to predict that displacement. This is
why the label depth rule matters: a WHY that names the number, the site and
what it changes carries the line somewhere specific, while a boilerplate
WHY carries every line to the same place and teaches nothing.

Two consequences worth testing early:
- **Schema edits are model edits.** Rewording an answer's description changes
  that head's behaviour with no retraining (tested on shapes; the size of the
  effect after training is the open question).
- **Company rules can be text too.** `add_company(name, policy_text)` seeds a
  new company's sparse code from its rules in its own words, before its ~30
  corrections.

### Ablations that would show it pays

On top of v3's C1 to C8:

| | Remove | Predicts a drop in |
|---|---|---|
| D1 | Described answers → free per-class weights | rare readings, zero-shot on a new reading |
| D2 | Folds → identity | heads that should be invariant (sow_coverage across speakers); rare heads |
| D3 | `why_sufficiency` | leap set; faithfulness by intervention |
| D4 | Description text scrambled across opportunities | everything in D1/D2, which checks the gain comes from the words, not from the extra parameters |

D4 is the control a reviewer will ask for first.

## 4. Is it novel?

Honestly, partly. Each ingredient has close relatives:
- **Label text as the classifier:** zero-shot classification with label
  descriptions (DeViSE; entailment-style zero-shot; CLIP text prototypes;
  GLiNER for entity types).
- **Task text generating weights:** hypernetworks from task descriptions
  (Hypter; HINT), FiLM conditioning, and instruction tuning in general.
- **Learning from explanations:** annotator rationales (Zaidan et al.),
  e-SNLI, ExpBERT, BabbleLabble, concept and text bottleneck models,
  ERASER's sufficiency metric, and explanations as privileged information.
- **Explanation as a displacement:** the shape of TransE (head + relation ≈ tail).
- **Folding:** deep ReLU networks are known to fold input space (Montúfar et
  al.); Householder reflections appear in flows and orthogonal RNNs.

What I don't know of anywhere, as of this writing (from memory, not a fresh
literature search):
1. **Per-question fold operators generated from the question's own wording**,
   applied to both the line and the answer prototypes, so each head's
   invariances come from its text.
2. **The per-line explanation trained as a sufficient displacement through
   those same folds**, so the WHY, the descriptions and the line live in one
   geometry and the model must predict the WHY's displacement at inference.
3. Both inside a **factored schema** where the universal layer is provably
   blind to the company and the company layer can be seeded from the
   company's written rules.

So the components aren't new; the combination and the fold mechanism likely
are. Whether it *helps* is an empirical question that D1 to D4 answer. If D4
shows no drop, the words aren't doing the work, and the claim should be
dropped.
