# What a lesson is keyed on

**Status: NEEDS DECISION.** Written for PUR-55 (parent PUR-14), milestone 5
"A correction on one deal changes another". The code in this PR follows
option B below. It is behind `SOWSMITH_LESSON_KEY=work_shape` and off by default.
A person needs to approve the recommendation and the date note at the end.

## The problem in one paragraph

A PM's lesson is a `Correction` row. Its exemplar is **the sentence the PM was
looking at**, embedded, and a later line matches it when its sentence embeds
nearby. So what a lesson matches on is the words. Taught heads reproduced the
deal they were taught on and moved other deals about 6% of the time. That is
what a wording key does. The next customer describes the same job differently
and misses the lesson. A deal that reuses the phrasing for different work gets
hit by it. Two earlier fixes were to plumbing (the embed cache keyed on the
wrong backend, and a merged `when` condition). This problem is the key itself.

## What a lesson is (the definition this doc proposes)

A lesson has three parts:

| part | what it is | what it is NOT |
|---|---|---|
| **field** | exactly one correctable field (`units`, `visits`, `hours_per_visit`) | a total, which cannot be attributed to any one field |
| **value** | a **rate** relative to the deal's own inputs (`units per stated count`, `visits per site`, `hours per unit`) | the other deal's number copied over |
| **key** | a **gate**: structural facets that must agree wherever both deals state them. A **key text**: a canonical `facet:value` string embedded instead of the sentence | the customer, the document title, the phrasing, the raw count |

All facets come from the work-order stage (`app/core/work_order.py`). That stage
states the job structurally: work, object, count, unit, site count, after hours,
no one on site, customer supplies the equipment. Values are normalised by
**structure**: lower case, the head noun of a phrase, light stemming. **No
vocabulary lists and no templates.** Nothing in the key knows what any piece of
equipment is called.

### Facets

| facet | source | normalised as |
|---|---|---|
| `action` | work line `action`, else the first word of `work` | stemmed word |
| `object` | work line `object` | head noun (`security cameras` → `camera`) |
| `unit` | work line `unit`, else `object` | head noun (`AP location` → `location`) |
| `size_band` | work line `count` | order of magnitude (`1e0`, `1e1`, `1e2`), never the number |
| `site_band` | work order `site_count` | `single` or `multi` |
| `after_hours`, `no_onsite_hands`, `customer_supplies_equipment` | work order | `yes`, `no`, or unstated |
| `delivery_model` | quote context (`onsite`, `remote`, `config_only`) | slug |
| `billing_type` | kit / deal | slug (recorded, not used in any gate yet) |

A facet that is not stated **never blocks** a gate. A query with **no shape at
all** fails every gated lesson: a lesson keyed on work has nothing to say about
a line whose work was never described.

## Field by field

`app/core/work_shape.py::FIELD_KEYS` is the source of truth. Change it and this
table together.

### `hours_per_visit`: how long one trip takes

- **Gate (must agree):** object, unit, after_hours, delivery_model, no_onsite_hands, customer_supplies_equipment
- **Soft (in key text, nearest wins):** action, size_band
- **Ignores:** site count, customer, document title, wording, exact count
- **Why:** time on site depends on what is being worked on and how the work is
  delivered. It does not depend on how many sites there are. Size stays soft
  because hours per unit can fall as jobs get bigger.
- **Stored as:** hours per unit = accepted h/visit × visits ÷ units.
- **Worked example.** A, "Install 12 cameras at the warehouse" (Northwind): the
  PM moves 12 h/visit to 18 h ("each camera needs a lift and a new home run"),
  which is 1.5 h per camera. **B**, "Mount and cable forty security cameras
  across the new distribution center" (Contoso, a different template):
  object `camera`, unit `camera`, onsite, daytime. It moves 40 → 60.
- **Counter-example.** **C** uses A's sentence, A's customer and A's
  document, but the work order says remote activation of camera licences
  (object `license`, delivery `remote`). The gate fails and C stays unchanged.

### `visits`: how many trips

- **Gate:** object, site_band, after_hours, delivery_model, no_onsite_hands, customer_supplies_equipment
- **Soft:** action
- **Ignores:** size (the rate is per site, so size is already divided out), unit, customer, title, wording
- **Why:** trips depend on the site pattern and the working window. The object
  is gated because, measured on the fixtures, a laptop visits lesson without it
  changed 17 unrelated deals (cameras, cabling, TVs) whose site pattern happened
  to match.
- **Stored as:** visits per site.
- **Worked example.** A, "Refresh POS terminals at 6 stores after close": 6 → 12
  visits ("two nights per store: stage then cut over"). **B**, "Replace
  point-of-sale registers overnight across 25 retail locations" should go
  25 → 50. **B currently FAILS**, because `terminal` and `register` are
  different head nouns (see option C).
- **Counter-example.** C uses A's sentence, but the refresh is a remote
  software push (delivery `remote`, no one on site). It stays unchanged.

### `units`: what the count becomes once priced

- **Gate:** object, unit, delivery_model
- **Soft:** action
- **Ignores:** size, site count, after hours, customer, title, wording
- **Why:** how many priced units one counted thing becomes (two drops per AP
  location, one reader each side of a door) is a property of the object and the
  unit it is counted in.
- **Stored as:** units per stated count.
- **Worked example.** A, "Install badge readers on 8 doors": 8 → 16 ("in and
  out reader per door"). **B**, "Fit access card readers to 14 entry doors":
  object `reader`, unit `door`. It moves 14 → 28.
- **Counter-example.** C uses A's sentence, but the object is `door strike`
  (head noun `strike`). It stays unchanged.

### Heads outside the estimator

Most of the other 21 heads judge **text**: a site's role, a document's
relevance, a sheet's kind. For those the wording is the thing being judged, and
they keep their current key. The heads that describe **work** should move to
the work-shape key next, in this order:

1. `task_hours`: same field shape as `hours_per_visit`
2. `commercial_terms`: billing type, PM hours, travel days
3. `task_tier`: parent or child

Moving each one is a separate change with its own gate triples. This doc does
not move them.

## Options

| | option | A→B (transfer) | A↛C (no leak) | cost | risk |
|---|---|---|---|---|---|
| A | **Wording** (today) | fails (6% real, 0/11 synthetic) | fails (11/11 leak on synthetic) | none | the ceiling on the whole loop |
| B | **Structural work shape**: gate on work-order facets, embed canonical key text | 9/11 synthetic | 11/11 synthetic, 0 collateral | small; built, behind a flag | object synonyms (`laptop` vs `notebook`) miss; depends on the work-order stage, which is off by default |
| C | **B + learned object identity**: a `same_work_object` head a PM teaches (like `same_physical_site`), used in place of head-noun equality for `object` and `unit` | expected to fix both synthetic misses | same as B | medium: new head, UI affordance, triples | cold start: needs a few taught object pairs, and a bad pair loosens the gate |
| D | **End-to-end learned key**: embed the whole work-order line JSON and fit the threshold per field | unknown | unknown | large; needs real pairs to fit | a learned key can quietly learn wording back; nothing to inspect |

## Recommendation

**Ship B behind the flag now. Build C next. Don't start D before real pairs
exist.**

- B is the smallest key that can be inspected and that passes both halves of
  the gate on most triples with zero leak. Every miss is visible and has a name.
- Both misses come from one cause: object identity by head noun. C fixes that
  with a learned judgment, not a synonym list, which keeps the no-vocabulary
  rule.
- D can't be trusted until there are real A/B/C pairs to fit and check against.

## Measured so far (synthetic only)

`python -m app.eval.transfer_gate` runs 11 frozen triples in
`tests/fixtures/paired_deals/triples.v1.json` with the offline embedder.

```
   wording: transfer 0%  leak 100%  pass 0% (0/11)
work_shape: transfer 82%  leak 0%   pass 82% (9/11)  collateral 0
            FAIL t04_pos_rollout_visits    (terminal vs register)
            FAIL t05_laptop_deploy_visits  (laptop vs notebook)
```

Read these numbers with three caveats:

1. **This is not the 6% re-measure.** The offline embedder is a bag of words,
   so the wording key scores 0% transfer here rather than 6%. The gate measures
   the **key**, not the production embedder.
2. **The fixtures and the key have the same author.** That biases the result
   upward. Real pairs chosen by someone else are the honest test.
3. **The work-order extraction is assumed to be correct.** Each fixture's
   `work_order` is written the way the stage *should* emit it. On real deals
   the stage is off by default (`SOWSMITH_WORK_ORDER`) and has not been measured
   on a full corpus. Wrong facets break the key.

## Format for real pairs

Real pairs use the fixture format, one file per batch, validated by
`app.eval.transfer_gate.validate_triples`:

```json
{
  "format_version": 1,
  "description": "who picked these, when, from which deals",
  "triples": [{
    "id": "r01_<short>",
    "field": "hours_per_visit | visits | units",
    "a": { "deal_id": "...", "customer": "...", "document_title": "...", "wording": "the line as the documents state it",
           "billing_type": "...", "delivery_model": "onsite|remote|config_only",
           "work_order": { "work_lines": [{"work": "...", "object": "...", "unit": "...", "count": 0}],
                           "site_count": 0, "after_hours": false, "no_onsite_hands": false,
                           "customer_supplies_equipment": false } },
    "b": { "...": "same shape as a" },
    "c": { "...": "same shape as a" },
    "correction": { "line_index": 0, "field": "...", "accepted": 0, "reason_code": "...", "reason_text": "..." },
    "why_b_same_shape": "one sentence",
    "why_c_different_work": "one sentence",
    "expect": { "b_changes": true, "c_unchanged": true }
  }]
}
```

Rules:

- Pick B for **shared work**, not a shared customer or template. The validator
  rejects a B with A's wording or A's customer.
- Pick C to be adversarial: A's phrasing, different work.
- `work_order` must be what the stage **actually emitted** for that deal under
  its as-of cutoff. It must not be hand-corrected, or the gate measures the
  fixture instead of the pipeline.
- Real pairs come from deals that **already have finished kits**. The standing
  rule stands: no training deals are added to get pairs. The measurement must
  run on deals **no head was taught from**.
- Real deal data stays in the private data store. It never goes into this repo.

## Is 29 September honest? (go / no-go)

- **Go:** the flagged key, the gate, the synthetic triples, the estimator as a
  correctable head, override records and the one-deal loop CLI. All are code
  and all are tested.
- **No-go** on calling PUR-14 or milestone 5 *done* by 29 September, unless all
  of these happen by about 24 September:
  1. At least one set of real A/B/C pairs, picked by a PM from deals that
     already have finished kits.
  2. The work-order stage run on those deals and its facets checked.
  3. The gate re-run with the production embedder on deals no head was taught
     from.

  PUR-14's own definition of done is "cross-deal transfer re-measured", and
  none of those three steps exists yet. The two known synonym misses also mean
  option C is likely needed before the gate passes on real `visits` pairs.
- **Suggested honest date:** 29 September for "key built and gated on synthetic
  pairs". The real-pair re-measure gets its own date once someone owns picking
  the pairs.
