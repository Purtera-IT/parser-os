# Step 4 — labelling the semantic rules

**Status: not started. Queue generated, home identified, no migration needed.**

This is the last of the four determinism steps and the only one that needs a
person. It is also the cheapest: roughly an hour of yes/no, not a campaign.

---

## Why it exists

15 `SemanticRule`s decide how atoms FORM — is this line a section title, is
this a money column header, is this paragraph a table's lead-in. Each compares
a line's embedding to hand-written prototypes and fires when the nearest
positive clears a **hand-tuned threshold**.

Harvested over 6 deals / 61 artifacts:

| | |
|---|---|
| decisions logged | 1,125 |
| distinct (rule, text) | 847 |
| **within 0.08 of the threshold** | **435 — 51%** |

Half of every decision these rules make is close to a coin flip. Real content
is landing on the wrong side by a thousandth:

| rule | line | best_pos | threshold | fired |
|---|---|---|---|---|
| `meeting_section_header` | `SOW – Premise Wiring, Bldg. 704` | 0.549 | 0.550 | **no** |
| `qa_answer_block` | `Can existing in-wall pathways be reused…` | 0.549 | 0.550 | **no** |
| `image_field_label` | `2.2.20 As-Built Drawings. The Contractor shall…` | 0.530 | 0.530 | on the line |

That first line is the same one that was flipping between `Bldg. 704` and
`Bldg. 704 B-4` under the PyMuPDF bug. It is balanced on a knife edge twice.

So this is a **quality** job as much as a determinism one.

---

## What you label

One line, one rule, one yes/no: *should this rule have fired here?*

| | your atom labelling | rule labelling |
|---|---|---|
| unit | one atom | one line + one rule |
| you decide | type, facets, span, links | fired: **yes / no** |
| time each | minutes | ~3–5 seconds |
| when | after atoms exist | *before* — it decides how they form |

Only the ambiguous ones are worth your time. A line whose nearest prototype
sits far from the threshold was never in doubt, and labelling it teaches
nothing. The queue is ranked by `|best_pos - threshold|`, closest first.

**Volume: ~80–100 per rule × 15 rules ≈ 1,200 items.** Tier A fits one number
per rule, and a 1-D boundary does not need more than that.

---

## Where the labels go — no migration

`atom_label_judgments` already has the right shape and already carries three
heads (`gap` 152, `document_job` 54, `site_role` 2):

| column | holds |
|---|---|
| `head` | the rule name, e.g. `meeting_section_header` |
| `target` / `target_key` | the line being judged |
| `text` | the line's text |
| `parser_value` | what the rule decided (`true` / `false`) |
| `verdict` | what it should have been |
| `note`, `reason` | why, when it is not obvious |
| `labeler`, `judged_at` | provenance |

Nothing to create. It is the same table the other heads already use.

Note `label_rules` is a DIFFERENT thing — 12 rows of doctrine you coined while
labelling ("A role word is not a party"). Worth keeping separate: that table
holds principles, this one holds decisions.

---

## The pipeline, which already exists end to end

1. `SOWSMITH_RULE_LOG=<path>` → `SemanticRule.fires` appends every decision as
   JSONL: rule, text, best_pos, best_neg, threshold, decision. Off by default,
   zero cost.
2. `_tools/warm_embed_cache.py` runs the harvest and ranks the queue. It fills
   the persistent embedding cache in the same pass, because both want the same
   thing — the lines the rules actually get asked about.
3. You label the queue.
4. Trainer fits one threshold per rule from the labelled set.
5. `SOWSMITH_RULE_THRESHOLDS=<path>` → `_trained_threshold` reads it back at
   rule construction.

Step 5 is the payoff for determinism: **a shipped JSON file, no network, no
model at runtime.** The decision stops depending on anything but the text.

---

## Tier A and Tier B

**Tier A — refit the thresholds.** One number per rule. Everything above
already exists; only the labelling and a small trainer are missing. This is
what to do first.

**Tier B — distil the rules into a head.** Replace prototype-cosine with a
trained classifier. Structurally better and removes the network from parse
decisions entirely, but it consumes embeddings, so it needs the persistent
embedding cache (step 2, done) underneath it. A real project, not an hour.

---

## Still to build

- a small trainer: labelled queue → `SOWSMITH_RULE_THRESHOLDS` JSON
- a gate that refuses a threshold set that scores worse than the current one
  on held-out labels — the same rule as `_eval_gate.py`
- **a section in the labelling UI**, so the queue is somewhere you already
  look rather than a JSONL on a laptop

---

## The UI section — where it goes, and the one thing blocking it

Traced end to end. The mechanism is ALREADY GENERIC, which makes this much
smaller than it looked:

| piece | file | what it does |
|---|---|---|
| entry point | `DealArtifactsPage.tsx:1891` | `Label atoms →` sets `labelingMode` and opens the modal |
| host | `AtomQualityAuditModal.tsx:799` | renders the toolbar when `labelingMode && dealId` |
| the save | `AtomLabelingToolbar.tsx` | `useParserFeedback.mutate({ head: "type", ... })` |

`head` is just a string on the way to `atom_label_judgments`. A rule section
posts `head: "meeting_section_header"` with the line text, the rule's decision
as `parser_value` and yours as `verdict`. **No new endpoint, no new table, no
migration** -- the same route the type corrections already take.

Blocking it: **the parser does not surface its rule decisions**. `fires()`
logs them to `SOWSMITH_RULE_LOG`, a file on whichever box ran the compile, and
nothing carries them into the envelope. So a section built today would render
an empty panel.

The order is therefore:

1. parser-os: carry each atom's rule decisions (rule, best_pos, threshold,
   decision) through to the envelope, behind a flag so normal compiles do not
   grow. This is the real work and it is small.
2. purpulse: a section in `AtomLabelingToolbar` listing those decisions for the
   selected atom with a yes/no on each, posting `head: "<rule>"`.

Not started, deliberately: shipping an inert panel is worse than shipping
nothing, and the frontend's `main` deploys to dev, so this belongs on a branch
with a PR.

Worth knowing while labelling: the OTHER atom UI on that page, `Audit quality
→`, keeps its verdicts in **localStorage only** -- `useAtomAuditLabels` says
so in its own header ("never calls an API... no sanctioned mutation endpoint
today"). `Label atoms →` is the one that reaches Postgres.
