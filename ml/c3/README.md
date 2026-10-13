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
| Schema | `schema.py` | Every head in label_heads.json becomes a list of **labeling opportunities**: `col:label_type`, `read:sow_coverage`, `rel:answers`… Each carries the head's question, the reading's description and one description per answer. Universal descriptions are scrubbed of company words. Registry-derived opportunities; count with the pinned schema. |
| Contract | `data.py` | One JSON per deal: atoms (text, entered_at, doc, section, speaker role and side), each with an optional label (`label_type`, `about`, `wants`, `reads_set`, `note`), edges, and an optional outcome. `from_training_blob` joins the real blob's labels onto atom rows by `label_key`. |
| Split | `data.featurize` | **Inputs:** the line, number-magnitude tokens, doc kind, role/side, time, same-document and same-section structure. **Targets:** fields with implemented consumers. Other saved supervision is retained and blocked by preflight until consumed. The only label fields that are also inputs are the stage and geo context slots, dropped to null at random. A test relabels every atom and checks the outputs do not move. |
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
| Explanation reader: WHYs, company lines and rule cards applied as votes | `explain.py`, `_read` | built; section 4 |
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

## 4. Learning the connection from the explanation, not from a pile of examples

Section 3 uses the text to *shape* the heads. That still learns slowly: a
connection like "over 55 inches means two technicians" would only appear
after many such lines. A person learns it from one sentence. `explain.py`
gives the model that route.

**Write the connection once, in words.** Three kinds of text go into an
explanation bank, each kept in its layer:

| Source | Layer | Example |
|---|---|---|
| every labeled line's WHY, with the card it led to | universal | "Cuts the site count from 3 to 2. Every quantity derived from 3 sites changes: 2 x 4 = 8." |
| every labeled line's company line | that company | "[purtera] reject: sellers do not own delivery tasks." |
| **rule cards** written once, no examples needed (`fixtures/rule_cards.json`) | either | "A display larger than 55 inches is mounted by two technicians, because one person cannot lift and level it safely." |

A rule card can name the question and answer it argues for, or name neither
and let the model decide where it applies. SAs and PMs can write company
cards in their own words ("how my SAs use this line").

**The model reads the bank and applies it.** For each line and each question,
a reader asks whether each explanation applies to this line (with a
"nothing applies" option), and which answer the explanation argues for, read
from its words and its stated conclusion. The votes are added to the heads.

**Why that speeds learning up.** The reader is trained on every pair of
(line, someone else's WHY). A line never sees its own WHY, so it has to
transfer reasoning written about a different line, and `explanation_only`
makes the votes reach the right answer by themselves. That is the skill of
*applying a stated reason*, learned once from all the WHYs. After it is
learned, a new rule card works the moment it's added: no retraining and no
new parameters (tested). The model changes because of what the sentence says,
the way a person does after being told.

**Guidance for every question and every answer.** `load_schema(guidance=...)`
adds a written "how to decide" to any question and a "when to pick it" to any
answer. It is folded straight into the descriptions the heads are built from.
`python -m ml.c3.card --guidance-template` prints a blank file with every
question and answer in the base. Company words in universal guidance are
dropped and reported, so the base stays portable.

**Firewall.** Universal heads read only universal explanations. Company heads
read only the active company's. A Purtera rule card cannot move a universal
output (tested bit-identical).

**What this does not do on day one.** The reader itself has to be trained
before a rule card means anything; until then its votes are noise. The WHYs
being written now are its training data, which is one more reason
line-specific WHYs matter.

Extra ablations:

| | Remove | Predicts a drop in |
|---|---|---|
| E1 | the reader (no bank) | few-shot accuracy on rare readings; new-company curve |
| E2 | leave-one-out off | inflated train scores, worse held-out deals (it copies its own WHY) |
| E3 | rule cards held out, then added at test time | the zero-shot gain from a new rule; the main result to show |

## 5. v6 (current): the long labels teach the heads in training

The product is the heads above. At run time nothing else runs: no WHY, no
language model, no text generation, and no gold links. The model proposes
its own links from the atoms and takes one foresight-safe hop over them
before the heads decide (`link_mix`, zero until the decision loss opens it).
A gold edge teaches the right target; a labeled line with no edge teaches
abstain, and that "no link" mass does not borrow another atom.
The WHY paragraphs do their work in training, as privileged information:

| File | What it does |
|---|---|
| `brain.py`, `lm.py` | The **teacher**, training only. A language model (tiny byte-level stand-in here; any pretrained causal LM via `HFCausalLM`) reads each line in its deal context with its WHY on the page, answers every head, and learns to write the WHY. No keyword or if/and rule on its path (tested). |
| `supercharge.py` | What the heads learn from it: `teach_heads` (every head matches the teacher's WHY-informed answer, so one paragraph teaches many heads), `teach_geometry` (lines alike for the same reason end up alike in r_i), `teach_unlabeled` (unlabeled lines get the teacher's read), `teach_pointers` and `teach_words` (which earlier lines a WHY rests on and which words of the line carry it, read from the teacher by `Brain.grounding`, not parsed). `mask_labels` keeps a chosen subset of labels. |
| `consolidate.py` | Teacher-side memory: a paragraph written into low-rank weights by context distillation, for when notes outgrow the page. |
| `notes.drop_meta` | Drops the optional last `[parser]` line of a note (parser remarks), so training never learns them. |
| `data.featurize` | Uses all of the labeling (heads-readthrough.md): skips old manual Deal Kit rows, makes Missed rows into lines, weights rows by weight_tier, teaches admission from the card's reject (a company reject with co_action stays in the company layer), teaches readings removed or ruled out as absent, routes Deal Kit lines' train_for to the company head `read:co_deal_kit_route`, keeps per-field notes for the teacher, and turns hint_refs and entity_keys into pointer and same-entity targets. |
| `judgment_heads.json`, `C3Model.judge`, `losses.judgment_loss` | The judgment tabs of the labeling page. Each tab's question is asked at its own size: one line (site role, gap, image, facet...), two lines (conflict, same site), a group of lines (document, table, sheet; attention pooling) or the whole deal (billing, scope, project tier). Layer and space come from the head that lists the judgment in label_heads.json; project tier pools the company layer, so the base never reads it. Dropped, admission and BOM owner verdicts fill existing questions. Verdicts find their lines by atom id, or by text after a re-parse; drafts written by a model never train. |
| `notes.sentences`, `data.suppositions`, `Brain.flip_read`, `supercharge.teach_flip` | Suppositions from WHYs, read by the teacher, not matched by words. Every sentence of a labeled line's WHY ("if the customer supplied the mounts, this would be a customer task", or just "the prices do not matter") is supposed in turn: the teacher reads the page with it assumed and answers. No word list or "if/unless" pattern decides which sentences describe a different case; the teacher's reading does. The heads, with the line moved a little by the sentence's text (`flip_proj`, training only), match that answer: where it changes, the boundary sits right beside the line; where it does not, the heads learn to hold. |
| `data.look_alikes`, `losses.near_miss_loss`, `losses.supposed_twins`, `efficiency.near_miss_accuracy`, `efficiency.twin_consistency` | Labeled lines that read almost alike (character-trigram Jaccard >= 0.6). Near misses answer differently: each must prefer its own answer by a margin. Twins answer the same: the wording differs where the answer does not care. Both give the teacher practice at supposing (line i's page with line j supposed must answer j's label), from labels alone. The efficiency check scores held-out near misses (both right) and twins (same answers): a keyword-spotting model fails the twin test. |
| `Brain.rewrite`, `supercharge.rewrite_lines`, `read_rewrites`, `teach_rewrites` | The meaning lock. The teacher writes each labeled line again twice, with its WHY on the page: in other words with the reason kept, and changed as the line's first WHY sentence says. It reads both. A reworded line it reads differently is dropped (the round trip). The heads run on the real rewritten text through the whole trunk: they hold their answer on the reworded line and take the teacher's reading on the changed one, so the encoder itself learns that wording is not meaning. Off unless asked (`teacher_view(rewrites=...)`, `efficiency --rewrites`): one short generation per rewrite, once per deal, in training only. |
| `lattice.py` (`C3Config.asked_layers`), `C3Model.suppose` | Asked attention. After the deal graph, each line carries one slot per question space, seeded from the space's written description. Slots of a line talk to each other, then each looks back over the deal (foresight-masked, same-document and same-section biases) for what its own question needs; its attention is its citation (`out.citations`). The firewall is wiring: base slots never read the company slot, so base answers are bit-identical with any company or none. `suppose` answers "what if this sentence held" at run time through the move teach_flip trains, with no language model. Off at 0 layers; `C3Config.design()` uses 2. |
| `efficiency.py` | The measurement: same heads, same labeled lines, with and without the WHYs, on held-out lines, across budgets and seeds. `python -m ml.c3.efficiency --deal ml/c3/fixtures/synthetic_deal.json` (synthetic: proves the harness only). |

The v5 explanation bank (reader votes, compiled operators, clause roles,
rule cards) is off by default (`C3Config.legacy_reasons=False`) and kept only
as the baseline to beat. Design and prior-work comparison: the project's
`labeling/base-architecture-v6.md`.

## 6. v5 (baseline): reasons compiled into operators, graded by outcomes

The full design and novelty write-up is `labeling/base-architecture-v5.md`
in the project files; this is the code map.

| Mechanism | File | What it does |
|---|---|---|
| Reason compiler | `operators.py` | Every explanation (WHY, company line, rule card) compiles into a region (where it applies), a topic (which questions), a fold and a move. Operators apply in order before the head reads the line; an overruling switches the older one off only inside its own region. |
| Change vector and outcome grading | `consequence.py` | Each line predicts what it changes (hours, crew, sites, price, tasks, schedule × down/none/up). Each explanation claims a change. Closed deals grade the claims, and `ReliabilityLedger` scales each explanation by its track record. |
| Planted-rule twins | `synthetic.py` | Same synthetic lines, two rule texts differing in one detail, labels following each text. Trains and tests following a rule from its words. |
| Explanation value | `ask.py` | Ranks lines by how much the model's own voiced reason for a line, compiled as a rule, would move its answers on the other lines: which WHY to ask for first. |
| Full paragraphs (v5.1) | `clauses.py`, `operators.py` | A paragraph is cut into clauses with roles (condition, exception, cause, consequence, evidence, quantity, statement). Each clause gets its own slot: conditions AND into the region, exceptions are cut out, causes shape the fold, consequences add to the move and the claims, arithmetic becomes programs. Every word stays readable by each line (`detail`). `clause_use` makes sure no clause is dead weight. `python -m ml.c3.clauses "<text>"` shows how a paragraph was read. |
| Proposed labeling fields | `data.py` (`changes`, `follows_rules`, `exception_to`, quoted/final outcome) | Read when present, ignored when absent. Not on the labeling card or in the registry until the user decides. |

Ablations F1 to F5 and results R1 to R6 are in the v5 doc.

## 7. Is it novel?

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
3. A **small model that learns to apply written rules**: a reader
   meta-trained on lines paired with *other* lines' WHYs, so a new rule card
   moves predictions with no retraining. Close relatives are BabbleLabble
   (explanations compiled into labeling functions), ExpBERT and CLUES
   (classifiers from explanations), and LLM in-context learning, which
   already does this but at frontier cost and with no firewall.
4. All of it inside a **factored schema**: the universal layer is provably
   blind to the company, and the company layer reads only that company's
   rules.

v5 adds the reason compiler with case-law overruling, outcome-graded
reasons, planted-rule following, and asking for the explanation worth the
most; the v5 doc compares each claim with its closest prior work, including
explanation-based learning (1986), which had the core idea symbolically.

So the components aren't new; the combination and the fold mechanism likely
are. Whether it *helps* is an empirical question that D1 to D4 answer. If D4
shows no drop, the words aren't doing the work, and the claim should be
dropped.
# Supervision preflight

Before a real corpus run, audit each frozen export with
`python -m ml.c3.readiness --blob /outside/repo/blob.json --atoms
/outside/repo/atoms.json --deal-id <id> --split-manifest
/outside/repo/split.json --output /outside/repo/report.json` (one shell line).
Exit 2 means blocked; exit 0 means the supplied contract and split checks
passed. This does **not** certify model quality, source-audit completion,
live-card coverage, GPU capacity, or production deployment readiness. Reports
contain categorical counts and hashes, not customer source text.

The split manifest has nonempty `train`, `validation`, and `holdout` lists.
Every entry has `deal_id`, `snapshot_sha256`, `audit_status: "verified"`, and
`source_groups` identifying shared source documents or related job versions.
Set `holdout_used_for_tuning: false` only when it is true. The final holdout
must not be used in teacher training, note retrieval, calibration or tuning.
Freeze all source snapshots and schema/model/tokenizer revisions before a
run; validate each snapshot hash against the manifest rather than treating
the existence of a manifest as proof of a frozen corpus.

The loader now selects named human labels before assistant mirrors and
rejects conflicting saved labels. Equal text is an occurrence fallback only
when unique; suppression keys resolve their precise dropped occurrence.
Retired-card links are filtered by author, **head and key**, so an excluded
hours card cannot erase a live task card sharing that key.

Multi-select hints and parser routing train independent sigmoid answers in
the heads, teacher, rationale-sufficiency and distillation paths. Scalar gold
targets remain unknown for these opportunities. The diagnostic efficiency
metric scores exact selected sets rather than their first member. Missing
selections remain unknown; they are not invented negative labels. Unlabeled
distillation is off by default and requires explicit warmup/calibration
enablement plus a confidence gate.

`DealExample.source_supervision` preserves the selected labels, raw cards,
links, Deal answers and missed history as a training-only packet. It is not
a model input or a claim that every field has an implemented loss. Preflight
blocks deferred fields/tabs, ambiguous occurrences, colliding targets or
card rationale, missing vision inputs and dense-attention budget overflow.
The 4,096-atom ceiling is a guard against accidental quadratic allocation,
not a GPU capacity guarantee. Do not truncate long deals to pass it.

Question reason codes, structured Values (hours, terms, kit shape), open
strings, Deal service and site-count, pixel and table-geometry inputs, and
relation-preserving long-deal windows now have targets and losses. See
`python -m ml.c3.coverage` for the surface table. Norm and terminology strings
are exact teacher-token targets plus a unique in-line span; they are not a
student string decoder. Rules remain a separate export. `ml.c3.gates.train`
refuses to start. The base experiment is a separate loop,
`python -m ml.c3.loop --train <deal.json> --holdout <deal.json> --checkpoint <path>`,
which on an H100 trains the design model in bf16. The line encoder
(ModernBERT-large) stays frozen. The teacher defaults to
`Qwen/Qwen2.5-7B-Instruct`, with its body frozen and a low-rank slot trained,
and reads long deals in row slices. The student schedule is lines, links,
pair and group cards, numbers, deal answers, then the link gate, then blank
cells filled by the teacher. AdamW with a cosine schedule is the H100 optimizer. After the link gate
opens, the teacher is shown the deal graph the student built, then a light
boundary pass runs, then only questions seen many times may fill a blank line.
The teacher does not invent new line text. `--tiny` is the CPU stand-in, not the base.

The labeling page's export becomes a training file with
`python -m ml.c3.export --page <export.json> --deal-id <deal> --output <file>`
(outside the repository; `--manifest` records its sha256). Every line gets the
time it reached the deal. A Dropped card about a copy of a kept line gets its
own line, so its verdict never lands on the line that was kept.

A deal past the atom budget (`ml.c3.longdeal`) becomes overlapping windows,
edge packs and one digest, never a truncated deal. Each line's labels train in
one window only. A link no window holds whole trains in a pack of its two lines
and their neighbours. Whole-deal cards and saved deal answers train on the
digest: the deal's commercial documents first, then labeled lines, latest
first. A long deal's parts are weighted so the deal counts sqrt(parts), not
parts.

`--validation <deal.json>...` scores those deals after every stage, which is
how steps are chosen. The labeling holdout (`010172`, `010038`, `010387`,
`010194`, in `ml.c3.holdout`) and the five kits never train in either run, and
the H100 run refuses to start unless all four labeling holdouts are passed as
`--holdout`. The holdout is scored once and is not used to stop or tune. A
frozen line encoder's vectors are computed once and reused. `--max-atoms`
sets the window (4,096 by default; 8,192 keeps most labeled deals whole and
needs a bigger GPU). `--train-encoder-layers K` lets the encoder's top K
blocks learn at `--encoder-lr`; keep it only if the validation deals say it
beats the frozen encoder. That loop does
not flip `READY_FOR_TRAINING`. The deal kit is a second model:
`python -m ml.c3.kit --train <kit json> --holdout <kit json>`. `--tiny` is the
CPU check. The live run is the H100: ModernBERT-large stays frozen, and a
slot reader trains on top. Every atom from before the kit was attached is a
token. When one atom is decided, every other atom stays in the attention,
tagged as evidence, the same fact, a lookalike, a sibling, or by its section.
A section the tab filled teaches the atoms left off it. A section left empty
teaches nothing. A reason sentence is stored and is not an input. A site the
tab kept is also a `job_site` mark for the atom model's site head. Billing
reads each present section once, plus the other training kits. The kits
`000051`, `010095`, `010164`, `010253`, and `010287` are the holdout and
never train. The run does not flip `READY_FOR_TRAINING`. `READY_FOR_TRAINING` is false until a frozen audited corpus,
an untouched holdout, a certified pretrained budget and a separate Rules
evaluation exist. Synthetic tests do not measure learned accuracy.

Rules preparation has an explicit separate caller:
`python -m app.learning.rule_feedback --manifest /outside/repo/rules-split.json
--output /outside/repo/rules.jsonl` (one shell line). Its manifest has the
same three verified deal partitions and untouched-holdout certification;
training entries additionally carry `blob_path` and a byte-level
`blob_sha256`. This exporter opens **only** the training blobs, rejects
evaluation-purpose sources and conflicting saved verdicts, filters
assistant proposals/excluded cards, and publishes the JSONL atomically with
private permissions. The existing semantic-rule trainer reads this shape
through `SOWSMITH_RULE_LOG`; exporting does not invoke that trainer. A
separate rule export/evaluation is still required before calling the whole
corpus ready: the C3 preflight cannot certify an external learner's artifact.

## Source-conditioned card supervision

Questions with source query text are separate subjects in `Batch.judged`.
Two questions sharing one evidence line retain independent answers and WHYs.
An unanchored question reads non-excluded deal evidence; a stale named anchor
is rejected rather than replaced by arbitrary equal text. Legacy query-less
line cards keep their existing line-target behavior.

`model.judge(..., queries=[...])` combines learned source-query embeddings
with evidence representations. Saved answers, fields and expert notes do not
enter that inference API. Card fields `deal_stage`, `needed_by` and `intake_gap`
train their registry heads on the same subject; `needed_by` is a set trained
with sigmoid BCE. Unknown values fail; blanks remain unknown. Company intake
heads require the company pass, while the universal stage head remains separate.

Card explanations have an explicit privileged embedding-alignment loss.
The training-only teacher also reads each card's evidence/query directly and
with its complete WHY, supervised by card-specific target losses. Its detached
reasoned logits reach the student through `teach_judgments`; subject identity
binds those targets to source text, question and expert annotation. These tests
verify wiring and gradients, not pretrained semantic quality.

Expert paragraph token loss now respects paragraph-quality weights and rejects
over-budget pages. Card teacher pages also reject over-budget source/WHY text.
Teacher tokenization and encoder paragraph/token readers also reject overflow
instead of silently clipping text. Real-model budget configuration and long-deal
context preparation still need validation before release. The default legacy clause/compiler
loss is off; explicit opt-in is retained for ablation tests.

Training readiness remains blocked by the frozen audited corpus, the untouched
holdout, pretrained-budget certification, separate Rules evaluation, and the
missing student decoder for normalized strings that are not spans.
`READY_FOR_TRAINING` is false. Neither synthetic tests nor saved raw JSON
certify learned accuracy or that every release gate has passed.
