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
| evidence spans | `evidence_span:*` 72 · `evidence_doc:*` 29 · `reads_value:*` 22 |
| readings | `reads:*` 37 (incl. one considered-and-rejected hard negative) |
| edges | `edge_relation` 80 — governs 38, supports 21, context 11, contradicts 5, same_as 3, answers 2 |
| attribution | `decided_from` 148 — which context field settled the label |
| questions | `gap_valid` 81 (21 valid / 60 invalid) + `gap_valid_reason` 81 |
| documents | `document_job` 12 (all `this_deal`) |
| sites | `site_role` 1 (`job_site`) |
| deal | router gold: `security_access`, 1 site, 2,675-char defect note |
| rationale | `rationale:*` 294 — atom 65, gap 81, edge 76, document_job 12, site_role 1, deal 1, evidence 58 |
| **total** | **1,389 rows — 972 backbone, 417 held back for the span and generative heads** |

Every note written on this deal reaches a row, counted rather than asserted
(`corpus_loss_audit.py` derives this and reports **0 losses**):

| written | chars | reaches |
|---|---|---|
| 65 atom notes | 96,717 | `rationale:atom` 65 |
| 76 edge notes | 22,493 | `rationale:edge` 76 |
| 81 question verdicts | 12,752 | `rationale:gap` 81 |
| 12 document verdicts | 2,016 | `rationale:document_job` 12 |
| 1 site verdict | 221 | `rationale:site_role` 1 |
| 1 deal note | 2,671 | `rationale:deal` 1 |
| 159 evidence pointers | — | span 72 + doc 29 + prose 58 |

137,000 characters of argument are carried as generative targets under
`rationale:*`, which `multitask_table` does not list as a backbone task — so a
classifier skips them untouched and a generative head has all of it. The only
text deliberately dropped is four copies of the card's own boilerplate, "drawn
while labelling the whole deal", which is not an argument.

The first version of that audit **hardcoded its eight findings as literal
strings**, because it was written to diagnose and then kept being run as if it
measured. After the fixes landed it went on reporting all eight, four of which
no longer existed. It now derives every line from the rows, and finding that out
cost two false alarms of its own — judgment prose lives in `note` while `reason`
is a short code, and the 94 judgments span three heads with a rationale
relation each.

Weights: 3.0 × load-bearing, 1.0 × ordinary, 0.3 × slight.

### A pointer is not always a span

Building the span head found the largest silent loss yet, and it was inside the
rows rather than between them. All 159 pointers were being emitted as
`evidence_span` for a head to locate, and only 72 of them are words on the page
that head is holding:

| | n | where it goes now | why |
|---|---|---|---|
| words in the atom or its context | 72 | `evidence_span:*` | the ranker can find them |
| an `atomId` on another surface | 29 | `evidence_doc:*` | real words, wrong page — retrieval, not extraction |
| `who_said_it`, `doc_type` | 56 | `decided_from` + `rationale:evidence` | a structured field — no reading of the page finds this text on it — but the rendering argues, so it is kept too |
| the labeler paraphrasing | 2 | `rationale:evidence` | argument, not a selection |

Nothing counted this, because nothing was missing: the rows were present, the
relations were listed, and the field check passed. The loss was a label its own
prompt did not contain — an instruction to produce words that are not there,
which is how a span head learns to invent one. `check_spans` now fails on it.

The routing lost nothing and gained a head: `decided_from` is free on all 159
pointers, it is learnable where a span is not, and it is the readable half of
"why did you say that" — *the heading above it*, *who sent it*, *its own words*.

Naming the field is not the whole of it, though. Among the 12 distinct
renderings behind those 56 pointers are *"purtera-it.com (internal, ours) —
internal only, never leaves our org"* and *"'us' in this list is the reseller,
not us"*. That is an argument about the envelope, so the text is carried to
`rationale:evidence` as well and only the **locating** job is dropped.

### Two more, in the last step of all

Counting the new head's rows into the table found the seam past every check
that existed: the rows leave `human_labels` complete, the relation **is** a
backbone task, and they vanish inside `assemble` — into a counter named
`duplicate`, which is the one word that stops anyone looking.

- **`decided_from`: 148 emitted, 65 arrived.** The dedupe key was
  `(relation, text)`, so a second label read as a contradiction. It usually is
  not one: an atom is decided by its own words *and* by who said it. Identity
  is the whole assertion, label included; a real conflict is two **teachers**
  on one assertion, and that is now resolved separately and per teacher — a
  weaker teacher's labels are dropped whole rather than merged, since a union
  nobody asserted is worse than either answer.
- **`edge_relation`: 80 emitted, 75 arrived.** An edge row's text is the
  sentence it comes from and its label is the relation, so the six atoms that
  *"Here are the details for the small job"* governs were six identical rows.
  The target lives in the provenance, and it is the rest of the assertion.

Both are now in `check_assembled`, which compares emitted against assembled per
relation. **972 backbone rows, up from 879, with nothing lost at any seam.**

### What the span head scores

**precision@1 95.1% (58 of 61), atoms held out, against an 80.3% baseline of
always returning the whole line.** Only atoms with both a right and a wrong
candidate are counted; three whose every candidate is correct — a phone number,
a one-line question — are excluded rather than allowed to inflate it.

Two things had to be right before the model mattered:

- **One pointer claims one candidate.** Containment marked the neighbour gold
  whenever it quoted the atom, and a ratio strict enough to stop that rejected a
  genuine 140-character selection for being 56% of its line. Token-F1 argmax
  needs no threshold.
- **Pairwise, not pointwise.** A pointwise classifier scored 85% and lost every
  bill-of-materials line: "Mag Lock Cable" is decided by the heading above it,
  and a global prior that the atom's own words win is right across the corpus
  and wrong there. Subtracting two candidates of the same atom cancels the
  prior; the same features then reach 95%.

Known limit, and the next feature when there are deals to fit it on: **the right
span depends on which reading is being asked for.** `Diagram: https://…` under
*Provided by Club/installer* is decided by the heading for a `supplier` reading
and by the URL for a `link` one, and the features never look at the label. One
of the three errors is exactly this. It is one example, so it is documented
rather than fitted.

These numbers come from the deal the features were designed on. Atom-level folds
hold out the *weights*, not the *choice of feature* — the interaction was added
after reading the errors. Treat 95.1% as the ceiling this design reaches when it
already knows the shape, and re-measure on the first unburned deal.

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

### A re-attachment risk, found while checking the parser fixes could not touch it

None of the three fixes below can: 010288 has **0 atoms** matching meeting-invite
vocabulary and no `.xlsx` or `.pdf` document, so its 130 labels are safe.

But while proving that: **31 of the 130 labels do not recompute their own
`label_key`** from the `(deal, filename, page, text)` they store. All 31 are on
one file, `010288-hs-email-116254345238.eml` — the email that linked the drawing
— so the surface's atoms were keyed under one filename and stored under another.
Those labels will **not re-attach** on a future re-compile of this deal.

Not urgent, because the deal is burned and finished. It is a lesson for every
deal after it: build the key from the same filename and page that get written
beside it, or the key stops being the thing that survives a re-parse.

### Still known-wrong on this deal

- The drawing's 18 parts are typed `deal_metadata` to keep them off the bill of
  materials. Defensible, and it still makes the sheet read as less important
  than it is.
- Two internal action_items are typed as outstanding work; both were done the
  same afternoon.
- 40 of 65 atoms carry no `rejected`, so they teach a point and not a boundary.

---

## 010180 — CDW FlexTrade Cabling, 7 Penn Plaza

| | |
|---|---|
| deal_id | `c79db726-323e-41f8-899d-1d8ca29a579a` |
| shape | one NYC high-rise floor, 106 workstations, **a priced quote**, a floor plan, 25 documents |
| **burned for eval** | **NO — keep it that way** |
| sibling | 010374 — CDW Site Survey Walkthrough FlexTrade NY, the same customer's second site |

Chosen because it fills the four holes 010288 left. Against 010288's 64 atoms and
12 documents it has ~250 atoms and 25 documents, and it carries the classes
010288 had none of: `vendor_line_item` (7), `quantity` (4, including **212**
workstation drops), `site_access_restriction`, and above all **money** —
$110,108 of it, which is the first `expenses_scope` and `billing_type` gold we
will have.

### Two parser defects fixed BEFORE labelling, and neither against 010180

Both were confirmed present in the **live** envelope (2026-09-24, 275 atoms) and
both were verified against the deals worst affected, **never against 010180**, so
it stays clean enough to measure with. That is the whole point of the burned
column.

| | measured | verified on |
|---|---|---|
| **A Teams join block parses as work.** "Dial in by phone" and "Reset dial-in PIN" as `scope_item`, "Need help?" as an `open_question`, and a bridge passcode as a `site_access_restriction` on a cabling job. | **87 atoms on 38 deals** across 461 envelopes — 42 `scope_item`, 22 `raw_utterance`, 14 `deal_metadata`, 7 `constraint`, 2 `site_access_restriction`. 7 still live on 010180. | deal 8aa9051a: **100 → 88 atoms, chrome 7 → 0**, and the 13 removed are all join details including the passcode `e23op74h`. The security banner they were fused with survives. |
| **A quote's money never reached its text.** `parse_money_cell("51092")` returns 51092.0 and the atom still read "Line item Cat 6A patch panels, 48-port" — no quantity, no price. $110,108 never became an atom at all. | 7 nameless line items and 4 bare `Quantity N` atoms, still live on 010180 | its own sheet, and the change is in the text builder rather than tuned to it: `Line item Cat 6A patch panels, 48-port — qty 6, unit price 900` |

Also fixed on the way past: **the test suite could not prove anything.** Five
files imported each other as `tests.X` with no `tests/__init__.py` and failed to
collect — those were the "known pre-existing failures" this session kept working
around, which is exactly how a real regression hides. Now a package, with the one
bare-import file converted to match.

### A third fix I built, measured, and threw away

The stale 2026-09-09 envelope showed a floor plan torn into fake table rows —
`WOM RESTR: EN'S ROOM`, `PE Floo: A`, `RECEPT: ION` — typed
`site_access_restriction`, so the deal's site facts were words torn in half. I
built a `page_is_drawing` detector (8,582 vector strokes over 153 words; a ruled
table is the opposite ratio), flattened the invented tables to notes rather than
dropping them so `7 PENN PLAZA` and `12,154 RSF` survived, and measured a clean
win locally: 9 torn atoms replaced by 6 honest label clusters, no words lost.

Then I checked the live envelope and it has **zero** of those atoms. The plan is
already read well — `site_room_mix` and `site_infrastructure`, 11 atoms each. My
local parse reproduced the tearing and the deployed parser does not, which means
I was fixing a local-environment artifact.

The `col_N` defect itself is still live, on three envelopes from 2026-09-22 — and
**not one of them is a drawing**:

```
col_1: L2 Network Technician | B: Per Hour | col_4: 6 | col_5: ( ) $72
DESCRIPTION: DESCRIP | col_2: PTION | STATED: RATE (USD) | BILLING: IN
col_0: 9 .0 to 10.0 | Severity Level: Critical | col_5: Any vulnerabil
```

A signed **rate card**, a pen-test severity table, an SHI quote. The real defect
is **header-row misdetection on genuine ruled tables**: the header gets torn
(`DESCRIPTION: DESCRIP | col_2: PTION`), so the first cell splits at its first
space and every other column falls back to a `col_N` placeholder.
`page_is_drawing` does nothing for any of them, so shipping it would have been a
fix aimed at the wrong thing — and a risk, since it could turn a good small table
into a note on any stroke-heavy page.

Reverted. **The thing worth fixing is the rate card**, because a mangled rate card
is precisely the `billing_type` gold we have none of.

### Known-wrong on 010180, and deliberately left for the labeler

- **41 atoms across 26 deals start with a bare colon** — but from three unrelated
  causes: a stripped rollup marker (`: 698 table rows (rolled up)`), a DKIM
  header fragment (`:date:message-id:reply-to;`), and a torn label. Three fixes
  for 41 atoms is worse value than getting this deal labelled, so they are
  labelled as noise instead. Three of them are on 010180.
- **The grand total has no name.** Row 10 of the sheet is a bare `110108` with no
  description, so it parses as `Line item — unit price 110,108`. The row-kind
  classifier needs a label in the row and there is none. A labeler marking it as
  the grand total is better gold than a parser guessing.
- The sheet's header says **"Unit Price"** over what are plainly line totals —
  212 drops for 51,092 is $241 each, not $51,092 each. Column names are carried
  through verbatim; deciding what the customer meant is a labeler's judgement.

---

## Next deal after 010180

Something with **eighty documents**, or a rate card, or a genuine multi-site
roster. 010374 is 010180's sibling site and would make the first cross-deal pair.
