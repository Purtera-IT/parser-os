# Cross-deal knowledge

**Status:** design, agreed 2026-08-31. Part 1 in progress; Parts 2 and 3 not started.

## The thing to get right first

This looks like one problem and is two, pulling in opposite directions.

- **Keeping other deals OUT.** Octavian's `Sodexo Breakdown.xlsx` is attached to
  010215 and covers the whole Sodexo programme. Read as 010215's evidence, its
  totals become 010215's price.
- **Bringing other deals IN.** We have done eleven Sodexo deals. The last four
  time-clock installs know what this work costs, and nothing surfaces that.

Build one mechanism for both and you get a system that either leaks other deals'
numbers into this quote, or forgets everything the company has ever done.

The distinction that separates them: **what a document is evidence FOR versus
what it is evidence ABOUT.** The Breakdown is evidence *about* the Sodexo
programme. It is evidence *for* 010215 only in the rows that name 010215.

## The worked example

`Sodexo Breakdown.xlsx`, on deal 010215, produces ten atoms. Four of them are
the whole problem:

```
[deal_metadata]    Oppty: PO# 00034150      <- not this deal
[scope_item]       Oppty: PO# 00033068      <- not this deal either
[scope_item]       Total | Total: 4750      <- aggregate, no deal attribution
[commercial_total] Total | Total: 535       <- aggregate
```

010215's own PO is `PO-00034965`. The document spans at least three POs, and two
unattributed totals sit in the evidence set as if they were this deal's
commercial figures.

Its covering email says so out loud: *"Please see the attached breakdown for all
Sodexo sites."* The document announces its own scope, and nothing reads it.

Scale: **82 files corpus-wide** share a content hash across more than one deal.
The Breakdown is not one of them -- it is attached to exactly one deal while
covering eleven. Attachment count does not detect this. Content does.

---

## Part 1 — Scope, as a third axis

Every document already carries two axes. Scope is the third.

| axis | question | source |
|---|---|---|
| stage | WHEN did it arrive | deal's HubSpot stage transitions |
| direction | WHOSE is it | HubSpot sent/received |
| **scope** | **HOW WIDE is it** | **the document's own content** |

### Values

| scope | meaning | example |
|---|---|---|
| `deal` | this deal only (default) | Marion County site list |
| `program` | several named deals | Sodexo Breakdown |
| `account` | the customer generally | Sodexo rate card, MSA |
| `global` | standing reference | Kronos install instructions |

### Detection

Ordered by how much each signal proves. All are content signals; none rely on
the filename.

1. **Deal keys in the content.** `Oppty:` values, PO numbers, `0\d{5}` deal
   numbers. More than one distinct key that is not this deal -> `program`.
   This is the strongest signal and the one that catches the Breakdown.
2. **Scope language.** "all sites", "all locations", "nationwide", "programme",
   "each site", "per site" in a delivering message or a heading.
3. **Same content hash on more than one deal.** Catches the 82; misses the
   Breakdown. Necessary, not sufficient.
4. **Customer-level document types.** RATE_CARD, MSA, THIRD_PARTY_POLICY route
   to `account` from the existing taxonomy.

### Treatment — narrowing, not discarding

Discarding the Breakdown loses real information: some of its rows *are* 010215.

- **Admit** rows that resolve to this deal — name its deal number, its PO, or
  one of its sites (Marion, Mullens, Johnakin, Easterling, McCormick, Palmetto,
  Creek Bridge, Academy of Early Learning).
- **Demote** rows resolving to another deal to context: readable for
  orientation, never quotable as this deal's scope.
- **Never admit aggregates.** See below.

### The aggregate rule

From a document whose scope exceeds the deal, rollup atoms are **structurally
inadmissible** — not low-confidence, inadmissible.

A total, a count, a per-site average from a multi-deal document is the most
dangerous atom in the system: perfectly plausible, precisely wrong, and it lands
straight in a price. A missing row is a gap somebody notices. A wrong total is a
number somebody quotes.

Aggregate atoms are `commercial_total`, or a `scope_item` whose text is a bare
`Total`, or any atom carrying a sum with no deal key in its row or section.

### When scope is uncertain

Demote aggregates, keep rows. Losing a rollup costs a re-read; admitting a wrong
one costs a mispriced quote. Scope detection will over-fire on documents that
merely mention another deal in passing, and that asymmetry is the safe way to be
wrong.

---

## Part 2 — Row narrowing

Part 1 marks a document. Part 2 acts on it row by row.

Each atom already carries `locator`, `section_path` and `structured`. Resolution
walks outward from the atom until it finds a deal key:

1. the atom's own row (an `Oppty` cell on the same line)
2. the enclosing section or table header
3. the nearest preceding deal key in document order

Found and it is this deal -> **admit**. Found and it is another deal ->
**demote to context**. Not found, in a `program` document -> **demote**, because
in a multi-deal document an unattributed row is not evidence of anything.

The last rule is what makes it work: silence in a multi-deal document is not
neutral. That is the same abstention principle already used elsewhere -- a
silent zero and a real zero must never look the same.

---

## Part 3 — Precedent, as its own channel

The upside. "We do a lot like this with this company."

**Sibling deals contribute priors, never evidence.** This is the hard wall.

### Mechanism

Embed each deal's scope profile — work type, customer, site count, equipment
mix, stage timeline shape. Given a live deal, retrieve the k nearest historical
deals. Surface what they actually cost and contained.

### The boundary

Precedent gets its own authority class, ranked below every piece of deal
evidence, and is **excluded from the receipt chain entirely**. It can never
manufacture a receipt for this deal's scope. The abstention doctrine survives
because precedent cannot be quoted.

### Output shape is the safeguard

Not "the scope includes X" but:

> The last 4 Sodexo time-clock installs priced $340–$410 per site. This one is
> quoting $290. Nothing in this deal's evidence explains the difference.

A **check**, not an input. A PM can act on it. A model cannot launder it into
scope, because it never enters the evidence set.

### Bootstrapping

No new labelling needed: 11 Sodexo deals, 194 deals with timelines, 574 verified
events, 1,114 classified documents, the lifecycle dataset.

---

## Order, and why

1. **Scope + aggregate suppression** — stops active harm, small, uses data we have
2. **Row narrowing** — recovers the rows that genuinely belong here
3. **Precedent channel** — the real upside, a proper project

1 and 2 are small. 3 is not, and should not start until 1 and 2 are proven,
because a precedent channel built on a corpus that still leaks cross-deal
aggregates would learn from contaminated priors.
