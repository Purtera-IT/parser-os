# C3: the head architecture labels train

Status: design of record, 2026-10-02. The map from label fields to heads is
code: `app/core/label_heads.json`, checked by `tests/test_label_heads.py`.
This file is the why. Nothing here changes how today's heads train; it says
which head every label feeds, so the labels written now are the ones the
model below needs.

## 1. The idea

Language models learn what a sentence means from the sentences around it.
This model learns what a line in a deal means from **what it led to**: the
hours the job took, the crew that went, the money paid, the question somebody
had to ask later, the line that ended up in the signed SOW. Call it
*consequential semantics*. A deal timeline is a free training signal of that
kind, and no text-only model sees it.

Every line is factored into three learned spaces, and only the third one
knows which company is asking:

| Space | Question | Sees | Trained by |
|---|---|---|---|
| **Content** `z_c` | What does this line say? | the line and the deal so far | universal labels; a company adversary keeps company signal out |
| **Consequence** `z_q` | What will it turn out to mean? | **only the past** | the deal's own **future** (Hindsight-JEPA), then labels |
| **Conduct** | What does company *k* do about it? | Content, Consequence, a sparse company code | that company's labels |

Around them sit five supporting spaces: **Beliefs** (the current value of
every fact), **Links** (how lines stand to each other), **Program** (exact
hours, crew and totals), **Context** (stage and geography as optional slots)
and **Rationale** (the WHY every decision passes through).

## 2. The pipeline

```
deal timeline (atoms from mail, calls, BOMs, quotes, SOWs, POs, notes; each with when it entered the deal)
   │
ATOM ENCODER (ModernBERT-large) + number-magnitude tokens + role/side (never names)
   │
DEAL GRAPH TRANSFORMER, structural attention biases, FORESIGHT mask
   │ h_i
   ├─► CONTENT z_c ──► type, frame, claims, document role  ──► BELIEF TRACKER (current values)
   ├─► CONSEQUENCE q_i ~ p(q | past)  (a distribution, trained toward the EMA hindsight latent)
   └─► CONTEXT slots: stage, geo (teacher / predicted / live / null)
                 │
          RATIONALE LATENT r_i ──► rationale writer (on demand)
                 │
   ┌─────────────┼───────────────────────────────┐
PROGRAM HEAD   PRECEDENT MEMORY                CONDUCT = Σ_m α_k,m · A_m
(executes)     (cite, overrule, consolidate)   sparse policy genome; constants inside atoms
   └──────────► VALUE-OF-INFORMATION QUESTION ENGINE ◄──┘
                sample q → run Program + Conduct → spread in $ and hours → best question
```

## 3. The heads, and the labels that train them

`label_heads.json` is authoritative; this table is its summary. Every
reading and relation in `atom_types.json` belongs to exactly one head.

| Head | Space | Layer | Trained by (columns · readings · relations) |
|---|---|---|---|
| content.type | content | universal | label_type · small_talk, noise_class |
| content.frame | content | universal | about, wants, supplier · introduces_party, situational |
| content.claims | content | universal | entity_keys · job_scale, bom_role, scope_facet, option_selected, start_date, billing_type |
| content.document | content | universal | doc_kind, sow_fill, sow_section, derivation, unreliable_part, points_at_artifact |
| content.deal | content | universal | deal answers, router and deal-scope judgments |
| structure.formation | content | universal | admission, rule, suppression, document and sheet judgments |
| beliefs.tracker | beliefs | universal | superseded · same_as, contradicts, near_miss · conflict and site judgments |
| consequence.work | consequence | universal | commitment, task_owner_role, triggered_task, trigger_event, fires_task, blocked_on, urgency, expansion · blocked_by, triggered_by |
| consequence.questions | consequence | universal | needs_artifact, chase · answers · the Questions card |
| consequence.outcome | consequence | universal | weight_tier · sow_coverage, train_for |
| consequence.hindsight | consequence | universal | no labels: finished timelines |
| links.hierarchy | links | universal | deal_summary, opens_block · governs |
| links.support | links | universal | supports, context, derived_from |
| program.estimate | program | universal | hours_stated, crew_size, removes_cost · the arithmetic in WHY notes |
| context.stage | context | universal | stage_std, deal_stage, sow_available_at_stage |
| context.geo | context | universal | location_tier, address_level, remote_miles |
| rationale.evidence | rationale | universal | hints, hint_refs |
| rationale.why | rationale | universal | the note, up to the `[company]` line |
| conduct.action | conduct | company | co_action, co_reason, the `[company]` line, rejected |
| conduct.constants | conduct | company | co_crew_rule, co_hours_estimate, co_tech_base_miles |
| conduct.routing | conduct | company | scope_category, delivery_field, source_of_truth |
| conduct.intake | conduct | company | intake_gap, needed_by |
| conduct.stage | conduct | company | co_stage_raw |
| conduct.precedent | conduct | company | every saved label is a precedent; a later save overrules it |
| meta.bookkeeping | — | meta | co_company, deal_outcome, universal_type, why_author (never a task or input) |

## 4. The mechanisms

**Hindsight-JEPA (Consequence).** Cut a finished deal at time *t*. An EMA
encoder reads the whole deal and gives target latents per line and per deal
(final SOW lines, quote v1→v2 changes, final hours and price, the questions
asked after the PO). The foresight model sees only lines up to *t* and
predicts those latents from `q_i`. Loss in latent space (cosine + VICReg), so
it predicts meaning, not wording. In v4, `q_i` is a small mixture density, so
the model knows what it does not know yet.

**Belief tracker.** Each line emits `(slot, entity, value, confidence)`
claims: sites with address level, quantities per item class, display sizes,
hours, crew, rates, dates, parties and roles, scope items, materials, access
constraints, billing basis. A two-layer recurrent transformer over claims in
time order keeps the current value per slot. `same_as` is a quotient over
claims (same slot, same value); `contradicts` and `superseded` are same slot,
different value, later time. `near_miss` joins two lines that look alike and
answer differently; training adds each such link to the look-alike near misses
that character trigrams find, so lines alike in meaning but not in wording are
held apart too. An **absence head** predicts "empty now, filled
later" per slot from `q_i`: the unasked question, learned with no labels.

**Link geometry.** `governs` is box containment (transitive by
construction). `answers` pairs a later line that fills the question's slot.
`supports`, `context`, `derived_from`, `blocked_by`, `triggered_by` use a
bilinear scorer under a causal mask. A consistency pass removes governs
cycles and time-order violations.

**Program head.** A pointer decoder writes a small typed program over belief
slots (`mul, add, ceil, max, min, lookup(const)`, depth ≤ 4), e.g.
`hours = sites × visits × hours_per_visit`, `crew = 1 + [display_in > θ]`.
It is executed, so arithmetic is exact, and it re-runs when a slot changes.
Shapes are universal; constants come from Conduct. Supervised first by the
arithmetic in WHY notes, then by what the job actually took.

**Rationale bottleneck.** Universal decision heads read `r_i` plus a
penalized 64-d residual. `r_i` is pulled toward the encoded human WHY
(InfoNCE, verdict words masked). The writer decodes from `r_i`, so the WHY it
writes is the one the decision used; faithfulness is measured by editing
`r_i` and checking the decision and the text move together. Nearest human
rationales from other deals are cited as precedent.

**Policy genome (Conduct).** A dictionary of low-rank policy atoms over
`[r_i ; beliefs_i]`. A company is a sparse code α (3-6 active atoms), fitted
from about 30 corrections with everything else frozen. Company constants (the
display size above which two techs go) are learned scalars inside an atom, so
a rule is a readable number. Universal outputs are bit-identical under any α.

**Precedent memory.** Every gold label and correction is an entry
(company scope, `r_i`, decision, reason, line). Decisions attend over the
top-k; a correction applies to the very next case and overrules the old entry
(kept, marked, linked). Nightly consolidation folds precedents into the
policy atoms.

**Question engine (value of information).** Sample K=32 consequence
latents, run Program and Conduct on each, and read the spread in hours and
dollars. Candidate questions come from the absence head; an answer simulator
trained on past question → answer → estimate-change triples scores each. The
model asks the one question worth the most, with its worth.

## 5. Rules the labels follow so these heads can learn

1. **Inputs vs targets.** Inputs at inference: the line, table, section,
   lead-in, document kind, neighbors, page, file, source date, speaker role
   and side. Every label field is a target, never an input. Stage and geo are
   optional context slots; the model must work without them.
2. **Universal vs company.** A field is universal when any field-services
   company reading the same line would give the same answer. Anything that
   depends on our roles, pricing, products, crew rules or CRM is `conduct.*`.
   A policy decision never changes the type.
3. **The note is two arguments.** The universal WHY first (the actual number,
   site, person or step; what it changes; what it depends on or feeds), then
   `[purtera] <keep|reject|ignore>: <the rule and its effect on this line>`
   only when Purtera has a consequence. `human_labels.split_note` trains the
   first part into `rationale.why` and the second into `conduct.action`.
4. **Links carry the leaps.** `answers`, `superseded` + `contradicts`,
   `derived_from`, `triggered_by` are what the tracker, the absence head and
   the program learn from. A fact with a consequence gets its link.
5. **Arithmetic in the WHY.** When a line implies hours, crew or a total,
   write the computation (`4 sites × 52 days × $920 = $191,360`); the program
   head is weakly supervised from it.

## 6. Where it lives

| Piece | File |
|---|---|
| Head registry | `app/core/label_heads.json`, `app/core/label_heads.py` |
| Field registry | `app/core/atom_types.json` (types, readings, relations) |
| Format checks and transform | `app/learning/label_format.py`, `scripts/labels_to_heads.py` |
| Ingest | `app/learning/human_labels.py` (note split, policy rows, edges from every registered relation) |
| Training table | `app/learning/multitask_table.py` (rows carry `head` and `space`) |
| API | Platform-infra `azure-function-api/shared/label-heads.json` (vendored), `/types` serves it, saves return format checks |
| Page | purpulse-frontend `LabelingWorkspace.tsx`: one card section per space |
| Model (untrained) | `ml/c3/`: the spaces, heads and losses above as PyTorch, heads built from this registry's text; never imported by `app` (`ml/c3/README.md`). v6 (current): the heads run alone; the WHY paragraphs teach them in training through a teacher that reads them (`ml/c3/supercharge.py`), measured as label efficiency (`ml/c3/efficiency.py`); see the project's base-architecture-v6.md. v5 (compiled operators) is kept as a baseline |

## 7. Build order

1. Count complete deal timelines; it decides whether Consequence is viable.
2. Build the v2 baseline (deal graph transformer, latent context slots) and freeze it.
3. Claims + belief tracker, then the absence head.
4. Hindsight-JEPA pretraining and the Consequence space.
5. Policy genome in place of a dense company vector.
6. Rationale bottleneck and writer from `r_i`.
7. Link geometry and the program head.
8. Precedent memory and the question engine (v4).

Each step is kept only if its ablation clears its confidence interval on the
frozen test deals. Headline results to design for: beats a frontier LLM on
foresight (later questions, final hours, which figure ends current); recovers
written rules blind, thresholds included; ranks questions by dollar value;
zero company signal in the fact layer; corrections take effect at once;
accuracy that scales with closed deals, not with text.
