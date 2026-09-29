# Phase 3 — ENRICH + CLASSIFY

The sixteen stages between "atoms worth enriching" and "atoms worth gating":
entity enrichment, typing, sanity, span admission, question resolution, geo
fallback, the second dedup pass, provenance joins. **32.5% of the compile** —
the largest phase by time, and the largest by rule surface by a wide margin.

Every number here was measured on live artifacts, on deals `4edf04d3`
(010180), `01491cca` (010238) and `c065bfc4` (010237).

---

## The finding that came first, because it invalidated the audit

**Four stages were deleting atoms with no suppression receipt.**

The suppression ledger is the instrument every content-loss audit in this repo
reads, `_tools/_phase3_audit.py` included. It is only a guarantee if every
departure is written to it. It wasn't, so the first pass over this phase
reported that **13 of 16 stages never suppress anything** — and that reading
was an artefact of the instrument, not a fact about the parser.

Two things had to be fixed before any of the rest could be trusted:

**The measurement.** `output_count` is not an atom count. For most stages it
reports what the stage *did* — packets built, atoms tagged, gates run — so
`packetize` reads 3753 → 52 and `document_job_scope` reads 853 → 0 without
either having deleted anything. Comparing it against the ledger produced 36
"holes" of which 32 were my own error. `_tools/_ledger_holes.py` reads the
compile's actual `atoms` list off the calling frame at each stage boundary
instead.

**The code.** Measured that way, four stages were genuinely silent:

| stage | atoms deleted with no receipt | why |
|---|---|---|
| `atom_type_sanity` | 37 on 010238, 293 on 010237 | no capture of any kind — not even a warning |
| `stakeholder_dedup` | 17 | recorded as telemetry warnings, which the audits do not read |
| `open_question_resolution` | 7 | filed under the sub-step's name, `open_question_quality_filter` |
| `pasted_note_dedup` | 3 | see below |

`pasted_note_dedup` is the sharpest of the four. It **had** a
`capture_suppressed` call, and that call omitted the required keyword-only
`reason`. Every invocation raised `TypeError` into the stage's
`except Exception` and was logged as *the stage failing*. The ledger call had
never once run. `tests/test_every_departure_leaves_a_receipt.py` now walks the
compiler's AST and fails on any `capture_suppressed` call missing a `reason`,
because a required keyword omitted inside a `try` is invisible: it raises, the
handler logs, and the atoms leave anyway.

After the fix: **zero atoms leave the compile without a receipt.**

---

## What was wrong

Seven defects, all fixed, all found by measuring rather than reading.

### 1. A totals block collapsed to one figure

`_value_key` keyed a `commercial_total` on its **category alone**:

```python
if atype == "commercial_total":
    cat = _first("category")
    amt = val.get("amount")     # computed, then never used
    if cat:
        return (atype, cat)
```

Live 010237's labour block is four atoms under one category:

| text | metric | value | fate |
|---|---|---|---|
| Total Labor Revenue: $121,519 | revenue | 121518.99 | kept |
| Total Labor Cost: $96,000 | cost | 96000.0 | **deleted** |
| Total Labor Margin: $25,519 | margin | 25518.99 | **deleted** |
| Margin % on Labor: 21% | margin_pct | 21.0 | **deleted** |

The cost, the margin and the margin percentage were deleted, on a deal whose
margin is the reason anyone reads it. The dead `amt` local says the intent was
right and only the return was wrong.

Then `_merge_values` made it worse than a deletion. It fills the survivor's
empty fields from the atom being deleted, so the kept revenue atom came out
carrying `metric="margin_pct"` beside `value=121518.99` — asserting that the
labour margin percentage is 121,519. Not a figure that went missing: a figure
that became a false statement about a different quantity.

Now keyed on `(category, metric, amount)`. The amount is rounded to **whole
units**, not to the cent — one figure reaches this branch rounded two ways
(the same atom carries `value=121518.99` beside `amount=121519`), and keying
to the cent split a genuine restatement back into two. My own test caught that.

**12 `value` clashes and 12 `metric` clashes → 0.**

### 2. A constant used as an identity

`deal_metadata` fell through to `_first("field_name", "value")`, and for two
kinds `field_name` is a **constant**. Every HubSpot note on the deal keyed as
`("deal_metadata", "hubspot_note_meta")` and every message header as
`("deal_metadata", "email_metadata")`, so each kind collapsed to a single atom
for the whole deal. 010237 lost four notes with four distinct note ids to one,
and seven message headers spanning 19 August to 29 September to one — same
sender, same subject, different message.

The branch immediately above already did this correctly for quoted headers
(sender + timestamp required, `None` otherwise, stay out of dedup rather than
fold on a guess). Extended to notes (`hubspot_note_id`) and headers
(`from` + `date`) on the same policy.

**26 `date`, 12 `subject` and 9 `hubspot_note_id` clashes → 0.**

### 3. Length was treated as authority

```python
elif isinstance(wval, str) and isinstance(lval, str) and len(lval) > len(wval):
    wv[k] = lval
```

`_merge_values`'s own docstring promises it "doesn't override populated winner
fields". This overrides them whenever the loser's string is one character
longer. A stakeholder named **"Chase Smith"** (11) was overwritten by one named
**"WIFI Example"** (12), and the placeholder was kept.

Longer only means "says more" when it says the same thing and more —
`"Rhonda Sharp"` → `"Rhonda Sharp <rhonda.sharp@cdw.com>"`. Now gated on
containment.

### 4. A cross-type group was a bin

```python
if len(members) == 1 or len({_atom_type_value(a) for a in members}) == 1:
    survivors.update(id(m) for m in members)   # all kept
winner = max(members, ...)
survivors.add(id(winner))                       # exactly ONE kept
```

A group that is all one type keeps everything. A group that spans **two** types
keeps exactly **one atom**, however many members it has — so one stray atom of
a second type joining four quotes of a thread turned "collapse a retyping" into
"keep one and delete the rest". The docstring already promised the right
behaviour ("groups that are all one type are left untouched"); it was true of a
group of two and false of every larger one.

Now only a member of a *different* type is folded. Same-type members are left
to `semantic_dedup`, which is what the docstring says.

### 5. A retyping took the figures with it

The cross-type key **strips digits** by design, so an atom that dropped the
numbers keys identically to one that kept them. `_not_at_the_cost_of_the_words`
guards this, but only counts added **prose** — `_SCAFFOLD_RE` strips digits and
punctuation, so an added money column reduces to nothing and the guard never
fires. That is correct for the case it names (a `raw_table_row`'s trailing
`| $21,560.00`, whose amount lives in the typed sibling's `value`) and wrong
when the winner's words are not in the member at all.

Now: a member is kept when it states figures the winner does not **and** the
winner's text is not contained in it. The money-column case is untouched, and
two existing tests pin that.

### 6. A contract row on the signature page was filed as a signature

`merge_signature_rows` groups every row on a page carrying a
by/name/title/date/signature label, rewrites the first as a merged record
holding `party/name/title/signed_at`, and deleted **every other row in the
group** — whether or not the merged record said anything of it.

Live 010238 kept a contract table on that page:

```
Effective Date:: Account # | 2022-10-01 00:00:00: 2701149/5698885
Effective Date:: Exp Date: | 2022-10-01 00:00:00: 2023-10-01 00:00:00
```

Both matched on their date label, both were deleted, and what survived read
**`"Effective Date: : Exp"`** — the labels with the values torn off. The deal's
account number and its contract effective and expiry dates left the compile and
nothing else in it stated them.

Two halves to the fix, because the row chosen to *become* the merged record is
overwritten in place and can carry figures too:

* a row is folded only when the merged record states every figure it states
* `keep` is chosen to be such a row, falling back to `rows[0]`

### 7. "A richer type exists at this cell" was taken on trust

Fixing defect 6 moved the loss one stage downstream rather than ending it —
which is the honest shape of this kind of bug, and only a stage-by-stage trace
showed it. `_suppress_table_row_blob_doubles` drops a table row's generic
`scope_item` blob when a richer-typed atom exists at the **same cell**, on the
reasoning that the richer atom is the classifier's reading of that row and the
blob therefore adds nothing.

`merge_signature_rows` breaks that reasoning: it retypes one row of the
signature page to `signatory` and rewrites it as a merged party record. So the
"richer" atom at 010238's cell `Lift:r11` read `"Effective Date: : Exp"` while
the blob it displaced held the account number and both contract dates — and the
blob was dropped for being a duplicate of it.

A blob is now folded only when the rich atom states every figure it states.

---

This is the same invariant as phases 1 and 2 — **a fold may not delete what
only the loser held** — and phase 3 needed it in five separate places
(defects 1, 5, 6, 7 and the ledger work that made them visible). That is no
longer a series of bugs; it is a missing rule, and it belongs above the stages
rather than inside each of them.

---

## What was RIGHT, and had to be proved

Three things looked like defects and were not. Each cost a real investigation,
which is the point of writing them down.

**`11720 Amber Park Dr, Suite 350` vanishes from 010237** — three atoms carrying
it are dropped and no survivor states it. That address is **PurTera's own
corporate office**, on `vendor_site_ban`'s list. The vendor's HQ is never a job
site, and dropping it is deliberate.

**A `deal_metadata` atom dated `Tue, 29 Sep 2026 18:19:15 +0000`** — the compile
date — beating the real August message dates. It looked like a `now()` stamped
onto a missing header. It is genuinely the `Date:` header of
`010237-hs-email-117681720241.eml`, a HubSpot email that really did sync that
day.

**21 `quoted_message_header` folds per deal, clashing on `message_index`** —
same sender, same `sent_at`, one message quoted in many files. The index is a
per-document position, and the fold is correct and documented.

---

## The measurement, before and after

Same three deals, same artifacts.

| | before | after |
|---|---|---|
| `commercial_total` folded into `commercial_total` | 18 | **0** |
| `commercial_total.value` / `.metric` clash | 12 / 12 | **0 / 0** |
| `deal_metadata.date` / `.subject` / `.hubspot_note_id` clash | 26 / 12 / 9 | **0 / 0 / 0** |
| `scope_item` folded into `scope_item` | 26 | **0** |
| damaging folds via `_merge_values` | 140 | 95 |
| atoms deleted with no suppression receipt | 64 | **0** |

The 95 that remain are the `quoted_message_header` folds and the stakeholder
folds, both correct.

---

## Two upstream defects this phase revealed

Neither belongs to phase 3; both were found by it and fixed in
`app/parsers/signature_block.py`.

**A sign-off became a job title.** `_is_titleish` rejected a sign-off only when
it ended in a comma. `"Thank you"` on its own line is two words with a leading
capital and passed every other test. So 010237 filed a person whose job title
was "Thank you" — and, because the cluster began at the line *above* the
sign-off, that person also took Chase Smith's email and phone and was named
after a document heading. A cluster that has collected nothing now stops at the
next name-shaped line; the line directly under a name stays exempt, because
that is where a title lives and `"Sr. Account Manager"` is name-shaped.

**A non-breaking space rode into a name.** `_clean` stripped only the ends of a
line, so Outlook's U+00A0 between a first and last name survived every check —
`str.split()` treats it as whitespace and `\s` matches it, so it never broke the
name *shape*, it just rode along into the stored value. Dedup then correctly
folded `"Tim\xa0Penney"` with `"Tim Penney"` and kept the non-breaking one, and
every later match on the plain spelling missed a person the deal had already
identified.

---

## The shape of the remaining problem

Phase 3's rule surface, counted against phase 2's:

| | phase 2 | phase 3 |
|---|---|---|
| lines | 5,919 | **20,671** |
| regexes | 25 | **288** |
| hand-kept vocabularies | 4 | **113** |
| magic thresholds | 20 | **97** |
| `SemanticRule` uses | 0 | **0** |

Worst three files: `entity_extraction` (6,244 lines, 126 regexes, 36
vocabularies), `multi_entity_llm` (55 vocabularies), `atom_type_sanity` (64
regexes).

`atom_type_sanity` is where defect 6 lived, and its shape is why: fourteen
demote/strip passes run in sequence over every atom, each with its own notion
of what a row is, and until this pass none of them filed a receipt. It is the
strongest candidate in the phase for replacement by a head — the question it
answers ("is this atom's type reconcilable with its text?") is exactly a
classification, and a wrong answer is recoverable where a deletion is not.

---

## Next in this phase

* **`atom_type_sanity` as a head.** 64 regexes answering one classification
  question. Now that it files receipts, its drops are labellable.
* **The 113 vocabularies.** Same migration path as phase 2's `SemanticRule`,
  and the same reason: a hand-kept word list is a threshold nobody can see.
* **`entity_extraction` at 6,244 lines** is unaudited at this granularity.
  It was profiled (13.84s → 11.52s, identical entity-key digest) and never
  read for content loss.
* **`_merge_values`' empty-field fill.** Defect 1 showed it can build a
  franken-atom out of two real ones. Containment fixed the override; filling an
  empty field from a deleted atom is still a guess that no invariant covers.

---

## On the instrument

Three tools were written for this phase and all three found things the
previous one could not:

* `_tools/_crosstype_loss.py` — every fold that deleted a field, or left the
  survivor stating a **different** value, attributed to the calling function.
  The attribution mattered: my first reading blamed `cross_type_dedup_atoms`
  for 140 folds that belonged to `_merge_values`, and only a stack walk showed
  it.
* `_tools/_last_seen.py` — the stage after which a given string stops appearing
  in the compile's atom list. This is what found `atom_type_sanity`, which the
  ledger could not see, and then found defect 7 when fixing 6 moved the loss
  one stage downstream instead of ending it.
* `_tools/_why_dropped.py` — the cross-type group an atom fell into and who won
  it. It had to capture keys BEFORE the call: the function merges provenance as
  it goes, so a key recomputed afterwards describes a group that never existed,
  and the first version of this probe reported a "group of 1 that survived 0".
* `_tools/_ledger_holes.py` — every stage whose atom list shrank by more than
  it filed.

The lesson of this phase is the second one. **An audit that reads the ledger
can only be as honest as the ledger**, and for four stages the ledger was
silent. Two of the six defects above were invisible until the instrument was
fixed, and one of them — the account number and the contract dates — was a
deletion the compile had been performing, unrecorded, on every run.
