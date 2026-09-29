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

## The email stages, audited on email

`pasted_note_dedup` and `quoted_history_dedup` act on mail, and COPPER_001 is
PDF-heavy, so on that deal they never fired. Audited on 4edf04d3 (30
artifacts, 235 emails) instead:

| stage | in → out | dropped | text lost entirely | figures lost |
|---|---|---|---|---|
| `pasted_note_dedup` | 2406 → 2406 | 0 | — | — |
| `dedup_quoted_history` | 2406 → **1203** | 1203 | **13** | **0** |
| `collapse_duplicate_atoms` | 1210 → 1069 | 141 | **13** | **0** |
| `drop_execution_boilerplate` | 1069 → 1069 | 0 | — | — |

`dedup_quoted_history` removes HALF the atoms, which is what it is for — a
twenty-message thread quotes the same paragraph twenty times. The only text
that leaves the compile entirely is one sender signature line,
`Zach Burdick | zach_burdick@shi.com`, which the email headers carry anyway.
No figure is lost by either stage. **Both pass.**

A note on how that number was reached, because the first answer was wrong.
Comparing dropped text to surviving text EXACTLY reported 72 losses, including
what looked like a PM naming a reference document::

    I have an example here form the past for a passive survey:  PurTera WiFi
    Validation - Report.docx

It survives twice. The dropped copy had two spaces after the colon and the
survivor has one. Whitespace-normalising both sides takes 72 to 13, and the
13 are all the same signature line. A fold that changes only spacing has not
deleted anything, and an audit that says otherwise is worse than no audit.

---

## Next in this phase

1. `confidence_floor` → the calibration head. One magic number currently gates
   review status.
2. `execution_boilerplate_drop` → `SemanticRule`. Smallest migration, follows
   the established pattern, removes 5 regexes.
3. Dedup similarity → an embedding pair-head, with the invariants as gates.

Still unexercised: `drop_execution_boilerplate` and `pasted_note_dedup` dropped
nothing on either deal tried, and `candidate_adjudication` and
`supply_conflicts` produced nothing on COPPER. That is not evidence they never
do — one or two deals cannot tell a dead stage from an inapplicable one, and
saying which would need the multi-deal profile that nothing currently records.

---

## `pre_classify_dedup` — the biggest dropper, and it is correct

238 atoms on COPPER, more than every other SHAPE stage combined. It collapses
one sentence emitted under several atom types, keying on a text with money and
quantities STRIPPED — which is exactly the shape that deleted a bid deadline
one stage earlier, so it was the obvious place to look next.

Of the 238, 114 pair with a survivor, and **18 have a survivor stating
different numbers**. All 18 are one shape:

    dropped : ... RFP-250007521 12/5/2024  2 Any other use of this property ...
    survivor: ... RFP-250007521 12/5/2024 15 Any other use of this property ...

A page footer reprinted on every page, differing only in the page number —
which the locator already carries. Folding them is right. **No change needed.**

The contrast with the bid-date case is the whole lesson: there the differing
number WAS the fact, here it is furniture. "Numbers differ" is a good alarm and
a bad verdict; something has to look at which number.

### A process note

The first pass on this stage reported it dropping **2** atoms, and concluded it
was clean. That was wrong. `cross_type_dedup_atoms` is called TWICE in a
compile — `pre_classify_dedup` at compiler:1622 and again at 2019 — and the
capture wrapper kept only the last call. The suppression ledger said 238 and
the wrapper said 2; the ledger was right.

That is the fifth measurement error in this audit, after the stale artifact
cache, the half-empty worktree bisect, the loose fold pairing, and the
whitespace-exact text comparison. Every one was caught by a second signal
disagreeing with the first. **When two measurements disagree, neither is
evidence until the disagreement is explained** — and in this pass the
instrument was wrong four times out of five, not the code.

---

# The full pass — all eleven stages, three deals

Run as one compile per deal with every stage watched, because one deal cannot
tell a dead stage from an inapplicable one.

| stage | ran | suppressed | text gone | figures gone |
|---|---|---|---|---|
| `quoted_history_dedup` | 3 | 1,486 | 235 | 21 |
| `duplicate_atom_collapse` | 3 | 639 | 16 | 10 |
| `execution_boilerplate_drop` | 3 | 15 | 15 | 0 |
| the other eight | 3 | **0** | 0 | 0 |

**Only three of eleven stages ever remove an atom.** The rest verify
(`source_replay`), flag without dropping (`confidence_floor`), add
(`prose_list_split`), restructure (`email_threading`, `table_rollup`), or had
nothing to act on (`candidate_adjudication` received 0 atoms on all three
deals, `pasted_note_dedup` and `supply_conflicts` dropped nothing).

## `quoted_history_dedup` — correct, and it took three tries to prove it

It removes HALF the atoms in a mail-heavy deal, so it deserved the scrutiny.

On 010237 it drops **103 of 112 `quoted_message_header` atoms**, and ten
distinct `sent_at` values appear on no surviving atom — including
`Thursday, August 6, 2026 10:23 AM`, the customer answer that mattered on that
deal. That looked like the loss of *when a quoted message was sent*, which
would break as-of compiles and take the "when" out of "who said what when".

It is not. Every one of those dates survives on
``value.email_thread.date``, which travels with every atom in the thread
alongside ``sender``, ``in_reply_to``, ``replied_to`` and ``answered_by``. The
probe was reading ``value.sent_at`` on the atom — the wrong field.

The stage's own key is ``(sender_address, minute_stamp)`` and it drops a quoted
header only when the original message is in the deal or an earlier quoted copy
was already kept. Its comment states the rule exactly: *a quoted routing atom
exists so attribution survives when the original is missing; when the original
is right here it is pure repetition.* **No change needed.**

## The 0.92 that turned out not to matter

Dedup's near-duplicate cutoff is the most-cited magic number in the phase.
Measured across 6,467 candidate buckets holding 9,214 atoms:

| cutoff | folds |
|---|---|
| 80 | 2,738 |
| 90 | 2,730 |
| **92 (shipped)** | **2,730** |
| 95 | 2,729 |
| 100 | 2,691 |

Moving it from 80 to 100 changes the outcome by **47 folds — 1.7%**. Between
88 and 97 it moves **four**. The number is not load-bearing: the bucket key
(type + first eight tokens + the figures) does the work, and 2,691 of the 2,730
folds are exact-text matches that need no threshold at all.

That reframes the head question for this stage. A better similarity SCORE has
almost nothing to win here, because scoring is not what decides. If dedup is
ever learned, the thing to learn is the **key** — what makes two atoms the
same fact — not the distance between two strings.

## What the constants ledger already knows

`app/core/calibration.py` is a self-deriving registry: each constant carries
its derivation, the corpus size behind it, what makes it stale, and often a
`re_derive` that recomputes its bounds as the corpus grows. Six constants are
registered.

None of phase 2's are: `LOW_CONFIDENCE_FLOOR = 0.50`, dedup's `0.92`,
`max_reps`, the eight-token bucket. The registry's own rule is *"Small is fine;
hidden is not."* The 0.92 measurement above is the kind of evidence an entry
needs — and it says the honest entry would record that the number barely
matters.

---

# On the instrument

Six times in this audit a measurement of mine produced a finding that did not
survive checking:

1. `cat6_utp` "regression" — a stale 118 MB artifact cache in my working tree
2. the commit bisect — worktrees with 664 of 1,233 files, reused path
3. 20 `quantity` + 20 `unit` fold losses — pairing looser than the stage's key
4. 59 text losses — exact-string compare against a double space
5. `pre_classify_dedup` "drops 2" — the function is called twice; I kept the
   last call
6. ten message dates "lost" — read `value.sent_at`, the date is on
   `value.email_thread.date`

Each was caught because a second signal disagreed with the first. **When two
measurements disagree, neither is evidence until the disagreement is
explained.**

The three real defects survived that same scrutiny, each corroborated
independently: folds dropping by exactly 81, number-differing folds going
39 → 0, and receipts hashing identical across a 2.12x speedup.
