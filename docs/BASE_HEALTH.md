# Base Health

Standing gates on the compiled corpus. One page; receipts, not marketing.

## Why

On 2026-08-27 two hand-run audits found real, large defects: **4,135 atoms**
sitting at rank-100 contract authority with no verified support, across 63%
of deals; and **964 of 1,100** `physical_site` atoms carrying a street
address with no city. Both were found because a person decided to look.

Worse, a fix that shipped the same morning **silently dropped ~50 real
sites** from one deal (438 → 388). Nothing failed. It was caught only
because someone re-measured by hand and noticed the number had moved.

A defect that is only visible when a human chooses to re-measure is not
monitored. `tools/base_health.py` turns those one-off audits into committed,
thresholded numbers that exit nonzero, so a regression becomes a red job
within hours instead of a weird brief noticed weeks later.

## Shape

```
blob envelopes -> stream one at a time -> per-deal counts -> thresholds -> exit code
                                                          \-> baseline drift
```

* `tools/base_health.py` — the CLI. Streams `deals/<deal>/orbitbrief/latest/
  envelope.json` one envelope at a time (download → measure → discard), so
  the ~1.6GB corpus never lands on disk.
* `tools/base_health_baseline.json` — committed per-deal expected site and
  atom counts. The only thing `site_count_drift` compares against.
* `.github/workflows/base-health.yml` — daily schedule + `workflow_dispatch`.
* `tests/test_base_health.py` — the measurement rules, hermetically, on
  synthetic envelopes. Never touches blob, so it runs on every PR via
  `tests.yml`.

```
python tools/base_health.py                    # table, 40-deal sample
python tools/base_health.py --sample 0 --json   # whole corpus, machine output
python tools/base_health.py --update-baseline   # deliberate act, see below
```

Exit codes: **0** clean · **1** threshold breached · **2** could not measure.

## The metrics

| metric | threshold | what it catches |
| --- | --- | --- |
| `unsupported_contract_authority` | ≤ 5 corpus-wide | rank-100 atoms with `verified != "verified"` |
| `rank100_from_noncontract_source` | 0 | rank-100 atoms whose source file is an estimate/quote/calc |
| `fabricated_names` | 0 | a shown name absent from its own source text |
| `street_without_city` | ≤ 65% of site atoms **(placeholder)** | the site-geo bug signature |
| `site_count_drift` | ±5% per deal, both directions | silently dropped **or** invented sites |
| `self_ingested_high_authority` | 0 | our own serialized output outranking real documents |

### Why each threshold is what it is

**`unsupported_contract_authority` ≤ 5.** Rank 100 is contractual scope — the
authority that wins every conflict. An atom holding it without verified
support is an unearned claim. This was 4,135 corpus-wide on the morning of
2026-08-27; applying the shipped demotion gate in projection took it to 1.
The bar is 5 rather than 0 because a handful of atoms can legitimately sit at
rank 100 while replay is still pending — but 5 is far enough below the
post-gate value that any real regression in the demotion pass clears it.

**`rank100_from_noncontract_source` = 0.** A ROM spreadsheet, a deal-kit
calc, an RFP, or a pricing workbook can never be contractual scope. There is
no legitimate instance, so any count above zero is a defect rather than a
tuning question. A filename carrying contract evidence (`agreement`,
`executed`, `purchase order`, `amendment`, …) is *never* flagged even when it
also carries a word like "draft" — a metric that fires on
`MSA Amendment 2 (draft cover).pdf` trains people to ignore it.

**`fabricated_names` = 0.** A name shown to a PM must be a string a human
actually wrote in the document the atom came from. A plausible fabrication is
worse than no name at all, because it survives review. Zero, permanently.
See the honesty caveat below — this metric deliberately reports a population
it *cannot* judge.

**`street_without_city` ≤ 65% — this is a placeholder, not the target.**
The target is 2%. You cannot parse a street out of a roster row and not know
its city; they are adjacent columns in the same row, so a street with an
empty city is an unambiguous parse bug. But **most envelopes on blob were
produced by an older deployed build that predates the roster geo fix.**
Measured 2026-08-27 over 38 deals: **56% corpus-wide (233/416 site atoms)**,
while a locally recompiled envelope of the same deal shows **0.3% (1/388)**.
The fix works; the corpus is stale. Setting 2% today would make this gate
permanently red, and a permanently red gate is an ignored gate.

> **RECOMPILE REQUIREMENT.** Once the corpus is recompiled on a build
> containing the roster geo fix (`app/parsers/site_roster_extractor`), set
> this threshold to `0.02` in `tools/base_health.py` and delete the
> placeholder comment. This threshold must not outlive the stale corpus.

**`site_count_drift` ±5% per deal, both directions.** This is the check that
would have caught the 438 → 388 loss. It flags **gains as well as losses**
deliberately: a fix that silently drops real sites is exactly as bad as one
that invents them, and only one of those two is something anyone would
notice by reading a brief. An absolute floor of 2 sites keeps tiny deals
(3 → 4 sites is +33%) from firing on rounding.

**`self_ingested_high_authority` = 0.** Serialized key paths
(`context.deal.sites[0]: …`) are our own machine output read back into a
compile as if a customer had written it. At `approved_site_roster` or above
it outranks real documents. There is no correct instance. The test is a
*shape* test, not a vocabulary list — it does not know the string
`context.`, so any future key that round-trips machine output back in is
caught by the same rule.

## Two honesty rules this tool holds

**1. A broken audit must not look like a clean base.** Same silent-zero
doctrine as `docs/IMAGE_GATE_LOOP.md`. Every path that cannot measure —
missing `az`, no auth, a download failure, an unparseable envelope, an
envelope with zero atoms — is an **error** and exits nonzero. Zero findings
from zero envelopes is never a pass. An envelope with an empty atom list is a
broken compile, not a clean deal, and is reported as such.

**2. The audit does not import the gate it audits.** The measurement rules
are reimplemented from the invariants in `app/core/atom_type_sanity.py`
rather than imported from it. An audit that calls the same function it is
checking cannot detect that function breaking — both sides move together and
the metric stays green. The two definitions drifting apart is itself a
finding.

### What `fabricated_names` cannot see

The envelope does not serialize an atom's `raw_text`. For roster-derived site
atoms the `text` field is an extractor-composed summary
(`site_id: HC-65 | facility: HC 65 | address: 6655 US 23`) — the roster's
facility-name column is consumed into `structured` and never appears in
`text`.

Checking those atoms anyway reported **all 1,518 site names on one real deal
as fabrications**, when every one of them was a verbatim string in the source
workbook. So atoms whose text is a composed field summary are excluded from
the metric and counted under **`names_undecidable`**, which is printed on
every run. Suppressing a check silently would be the exact failure this tool
exists to prevent.

The real fix is upstream: serialize the source row (or a replayable source
locator) onto the atom, and this exclusion can be deleted.

## Updating the baseline deliberately

`site_count_drift` only means something if the baseline is a considered
statement about what the corpus *should* contain. Regenerating it to make a
red job green is how this gate becomes decoration.

Update it when a change is **supposed** to move site counts — a new
extractor, a roster-parsing fix, deals added or removed from the corpus:

```
python tools/base_health.py --sample 40 --update-baseline
git diff tools/base_health_baseline.json     # READ THIS
```

Then, before committing:

1. **Read the diff per deal.** Every moved number should be explainable by
   the change you just made. A deal that moved for a reason you cannot name
   is the regression this gate is for.
2. **Say why in the commit message.** Name the change and the direction you
   expected. "regen baseline" is not a reason.
3. **Never regenerate as part of an unrelated commit.** A baseline bump
   hidden inside a feature diff is invisible in review.

The sample is size-stratified and deterministic — the same deals are measured
run to run — so a diff is a real comparison rather than sampling noise. Deals
absent from the baseline are reported as new, not as breaches; errored deals
are never written into it.

## Current state (2026-08-27, 38 deals, 37,782 atoms)

The corpus on blob predates today's fixes, so most metrics are red on day
one. That *is* the correct signal: the deployed corpus is stale and carries
the defects the fixes address. The first green run requires a corpus
recompile.

| metric | value | threshold | |
| --- | --- | --- | --- |
| `unsupported_contract_authority` | 1,598 | ≤ 5 | FAIL |
| `rank100_from_noncontract_source` | 1,636 | 0 | FAIL |
| `fabricated_names` | 84 | 0 | FAIL |
| `street_without_city` | 56.0% (233/416) | ≤ 65% | ok |
| `self_ingested_high_authority` | 421 | 0 | FAIL |
| `site_count_drift` | baseline established | ±5% | — |

Two envelopes (`991cc9f4`, `60601ebb`) carry zero atoms and are reported as
errors, not as clean deals. `self_ingested_high_authority` is concentrated
entirely in one deal (`01a499f9`, 421 atoms).
