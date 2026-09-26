# Training deals

One row per labelled deal, newest last. This is the ledger for the labelling
run: what went in, what it taught, and — the column that matters most — whether
the deal is **burned for evaluation**.

## Why "burned" is a column

A deal we open to fix the parser can never measure the parser. On 010288 I
shipped ten commits tuned to that deal: the substance gate was fixed because it
deleted *that* sentence, quoted names because *Albert* was deleted, Q&A pairing
because of *that* bullet. Its score would be fiction.

So the rule is: **any deal we debug against is burned, permanently.** Record it
here at the time, because nobody will remember later, and at the end the
holdout set is whatever is left unburned.

If we keep debugging against whatever we are labelling, eventually every deal
is burned and there is nothing clean to measure with. Leave some alone.

---

## 010288 — Access Control for Huzzard, Nesfield Performance Bethesda

| | |
|---|---|
| deal_id | `c2a3bdce-53bb-4da6-a840-a5260841685a` |
| shape | one external door, single site, sold through a reseller (CDW), drawn by a vendor (Huzzard) |
| documents | 12 (10 emails, 2 HubSpot notes, 1 linked drawing read as a surface) |
| atoms | 64 |
| **burned for eval** | **YES — #225–#234 were all tuned against it** |
| labelled by | developer@purtera-it.com |
| compile | `9cc94210` on manifest `c3166a06` |

### What it produced

| source | rows |
|---|---|
| atom labels (65) | `atom_type` 65 · `atom_type_coarse` 65 · `facet` 65 · `about` 58 · `wants` 58 |
| reasoning axes | `consumer` 65 · `weight_tier` 65 · `decided_by` 65 · `rejected` 25 |
| evidence spans | `evidence_span:*` 159 · `reads_value:*` 22 |
| readings | `reads:*` 37 (incl. one considered-and-rejected hard negative) |
| edges | `edge_relation` 80 — governs 38, supports 21, context 11, contradicts 5, same_as 3, answers 2 |
| questions | `gap_valid` 81 (21 valid / 60 invalid) + `gap_valid_reason` 81 |
| documents | `document_job` 12 (all `this_deal`) |
| sites | `site_role` 1 (`job_site`) |
| deal | router gold: `security_access`, 1 site, 2,675-char defect note |
| rationale | `rationale:*` 236 — atom 65, gap 81, edge 76, document_job 12, site_role 1, deal 1 |
| **total** | **1,241 rows** |

Every note written on this deal reaches a row: 65 of 65 atom notes, 94 of 94
judgment notes, 79 of 80 link notes (the last is the card's own boilerplate),
161 of 161 evidence spans. 137,000 characters of argument are carried as
generative targets under `rationale:*`, which `multitask_table` does not list
as a backbone task — so a classifier skips them untouched and a generative head
has all of it.

Weights: 3.0 × load-bearing, 1.0 × ordinary, 0.3 × slight.

### What this deal is good for

- **The relation-edge head**, which had no human gold anywhere. 80 edges, and
  the interesting half is `contradicts` — the same part claimed by two
  companies on two surfaces.
- **The gap head.** 81 judged questions with a reason each, and a 26% valid
  rate that matches the independent audit finding that most PM questions are
  junk. The contrast that teaches: `permits/AHJ` is valid on a maglock egress
  door while `Davis-Bacon`, `HIPAA` and `hoisting access` are not.
- **Router gold that is not circular.** The router returned `primary=None` with
  candidates `audio_visual, low_voltage_cabling, staff_augmentation, wireless`
  on a job that is nothing but access control. It routed on the materials (a
  Cat5e run) rather than the purpose.

### What this deal cannot teach

- `document_job`: 12 rows, all one class. No boundary.
- `site_role`: one row, one class.
- `expenses_scope` / `billing_type`: the documents carry no price, no PO, no
  rate. Not a labelling gap — the deal genuinely never says.
- Anything about scale. One door, one site, one visit. A head fed only deals
  like this will learn that enterprise vocabulary means "invalid".

### Parser defects it exposed (all shipped)

| | |
|---|---|
| #225 | a question built from the BOM lines it names was always "answered" and deleted as noise |
| #226 | "AJ- get the location and well confirm if we can do it." deleted as chatter on every compile |
| #227 | `"Albert Arzate" <albert@rd-systems.com>` deleted for being quoted |
| #228 | #226's rule resurrected "Received, thank you!" |
| #229 | `situational` — a third answer between "this is a fact" and "delete it" |
| #230 | a line asking twice and answering twice paired in order |
| #231 | …and merged into one atom, the way the unsplit form already was |
| #232 | do not ask where a site is when the deal states the address |
| #233 | a conflict the sender settled in writing is not a question |
| #234 | one question per topic — `work_hours` and `workhours` were asked separately |

### Still known-wrong on this deal

- The drawing's 18 parts are typed `deal_metadata` to keep them off the bill of
  materials. Defensible, and it still makes the sheet read as less important
  than it is.
- Two internal action_items are typed as outstanding work; both were done the
  same afternoon.
- 40 of 65 atoms carry no `rejected`, so they teach a point and not a boundary.

---

## Next deal

Pick a **different shape**. Not another access-control job: something with
several sites, a price, a rate card, or eighty documents. And if we do not need
to debug against it, leave it unburned and it becomes eval.
