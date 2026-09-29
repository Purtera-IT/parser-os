# Phase 2 — SHAPE

The eleven stages between "the compile has atoms" and "the compile has atoms
worth enriching": threading, dedup, splitting, rollup. **20.1% of the compile.**

Every number here was measured on live artifacts.

---

## What was wrong

Three defects, all fixed, all found by measuring rather than reading.

### 1. `source_replay` did twice the work for the same answer

96% of the phase. Profiling found ONE generator expression running
**42,451,772 times** — 17.6 of the stage's 41.7 seconds — with
`unicodedata.combining` alone accounting for 42.3 million calls.

`_replay_norm` strips combining marks so "cafe" matches "café". Replay asks it
about the same strings relentlessly: every atom against every candidate line.
It is a pure function of the string, so none of that work bought a different
answer.

Cached, with an ASCII fast path (NFKD and combining-strip are provably no-ops
on ASCII — checked over 4,017 strings including random Unicode).

| | time | receipts |
|---|---|---|
| before | 9.80s | `c0a166a88483` |
| after | **4.62s** | `c0a166a88483` |

**2.12×, byte-identical receipts.** Worth ~10% of the whole compile.

### 2. Dedup deleted a location's quantities

`collapse_duplicate_atoms` keeps the higher-confidence copy and drops the other
**outright — it does not merge**. Its key was (artifact, type, normalised text).

On COPPER_001 that folded 4,615 pairs, and **81 dropped a twin whose identity
differed**. Every one was an `atom_type=quantity` row:

    "Quantity RJ45 2"     plate AVL 3  ->  folded into plate AVL 2's copy
    "Quantity Cat6 UTP 2" plate AVL 3  ->  folded into plate AVL 2's copy

On a cabling job a plate is a location. Two plates needing the same number of
jacks produce the same WORDS and are not the same FACT.

Fixed by putting an identity in the key — the value fields that say WHICH THING
the atom is about (`plate_id`, `site`, `room`, `location`) plus its entity keys.
Deliberately **not** `sheet`/`row`: those differ for every table row and keying
on them would dedup a spreadsheet not at all.

**81 fewer folds, identity loss 0, the other 4,534 folds unchanged.**

### 3. Dedup deleted a bid deadline

The near-duplicate pass scores with `fuzz.ratio` — character edit distance — at
92. That is the wrong instrument for "is this the same fact", and it fails in
BOTH directions:

    "Provide 10 drops" vs "Provide 100 drops"        ~94%  -> folded
    "Contractor shall X" vs "The contractor will X"   low  -> kept

Too loose on digits, too tight on paraphrase. Measured: **39 of 58
near-duplicate folds dropped an atom whose numbers differed**, including:

    dropped : ... Bid and "January 10, 2025". The bidder must also ...
    survivor: ... Bid and "January 20, 2025". The bidder must also ...

97% of those characters match. The original and the addendum state different
deadlines. Folding them deletes one, and with it any chance of a conflict stage
seeing a contradiction — there is nothing left to contradict.

Fixed by putting the figures in the bucket key, so atoms stating different
numbers are never compared. Thousands separators and trailing decimal zeros are
normalised first, because "1,200" and "1200" are the same quantity.

**Near-dup folds 58 → 19, number-differing folds 39 → 0.** The 19 that remain
are genuine reprints.

---

## What was RIGHT

`table_rollup` folds 94 homogeneous table rows into 2 summary atoms, and it
does it correctly: **all 27 distinct figures survive**, and `value.rows` holds
all 48 rows with their cells. A presentation fold, not a deletion. No change
needed.

Worth recording. Three bugs in one phase makes it tempting to assume the rest
is broken too, and this one is not.

---

## The shape of the remaining problem

Both fixes above are **invariants, not lists**:

* different identity → different fact
* different numbers → different fact

They are general, cheap, and need no maintenance. That matters, because the
phase is otherwise held together by hand-kept rules.

| module | lines | regexes | hand vocabularies | magic thresholds | uses SemanticRule |
|---|---|---|---|---|---|
| `entity_resolution` | 1,418 | 8 | 2 | 10 | **0** |
| `email_threading` | 628 | 5 | 0 | 1 | **0** |
| `table_rollup` | 492 | 0 | 0 | 6 | **0** |
| `entity_hygiene` | 336 | 5 | 0 | 0 | **0** |
| `prose_list_splitter` | 273 | 5 | 0 | 0 | **0** |
| `pasted_note_dedup` | 258 | 2 | 2 | 3 | **0** |

**25 regexes, 4 hand-kept vocabularies, 20 magic thresholds, and not one use of
`SemanticRule`** — in a codebase that has that pattern specifically to replace
them.

Dedup alone carries `_GENERIC_TYPES`, `fuzzy_dedup_types` (8 types), `0.92`,
`_STOP_WORDS`, `max_reps` and an 8-token bucket. Every one is a guess somebody
typed.

---

## Where a head belongs, and where it does not

Most of this phase should NOT be a head. Saying so plainly matters as much as
the two that should.

| stage | head? | why |
|---|---|---|
| `email_threading` | **no** | `Message-ID` / `References` are exact. A model could only add error. |
| `source_replay` | **never** | It IS the provenance proof. Exact by definition. |
| `quoted_history_dedup` | no | Structural markers; deterministic. |
| `table_rollup` | no | Structural, and already correct. |
| **`duplicate_atom_collapse`** | **yes — the strongest case in the phase** | "Same fact?" is semantic equivalence with numeric sensitivity. Exactly what an embedding pair-head does and a character ratio cannot. |
| **`execution_boilerplate_drop`** | **yes** | "Is this boilerplate?" is textbook classification, currently 5 regexes. |
| `prose_list_split` | measure first | Boundary detection, and the sentence splitter already does this well. |
| `confidence_floor` | **should already be** | `LOW_CONFIDENCE_FLOOR = 0.50` — one hardcoded number deciding what a person reviews, when a trained calibration head exists. |

The dedup head is the cheapest of these to supervise: the 298 existing edge
labels already include **`same_as` (21)**, which is exactly the signal for
"these two atoms are one fact".

Whatever replaces the similarity measure, **the two invariants stay as hard
gates**. A head must never be permitted to fold atoms that state different
numbers or belong to different places. A learned score is an opinion; those
two are not.

---

## Next in this phase

1. `confidence_floor` → the calibration head. One magic number currently gates
   review status.
2. `execution_boilerplate_drop` → `SemanticRule`. Smallest migration, follows
   the established pattern, removes 5 regexes.
3. Dedup similarity → an embedding pair-head, with the invariants as gates.

Unaudited so far: `pasted_note_dedup`, `quoted_history_dedup`,
`prose_list_split`, `candidate_adjudication`, `supply_conflicts`. The last two
produced nothing on this deal, which is not evidence that they never do — one
deal cannot tell a dead stage from an inapplicable one.
