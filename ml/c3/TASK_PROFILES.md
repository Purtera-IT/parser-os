# Bounded training-only task profiles (loader version 2)

This API supports source-local extraction, not causal forecasting. It does not
certify a receipt boundary, date, outcome exclusion, source freshness, reasoning
coverage, or Pilot C readiness. Existing version 1 holds retain their behavior:
target masking alone does not isolate their model inputs.

## Wire contract

Use the labeling-blob loader `DealExample.from_training_blob(blob, atoms,
deal_id=...)`, followed by `featurize` or `plan_long_deal`. One emitted blob is
one physical auxiliary context, never an envelope of tasks. The exact contract is:

```json
{
  "supervision_holds": {
    "version": 2,
    "groups": [],
    "task_profile": {
      "task_id": "source_local_literal",
      "execute": true,
      "as_of": null,
      "context": {
        "document_id": "source-artifact",
        "source_keys": ["source-occurrence"],
        "source_fingerprints": {"source-occurrence": "<64 lowercase hex characters>"},
        "evidence_ref": "reviewed-artifact-or-cell-group-admission"
      },
      "allow_heads": ["read:equipment_qty"],
      "target_provenance": [{
        "source_key": "source-occurrence",
        "head": "read:equipment_qty",
        "evidence_keys": ["source-occurrence"],
        "evidence_ref": "reviewed-direct-target-eligibility"
      }]
    }
  }
}
```

Fingerprint every selected **complete raw atom mapping**, before projection,
with `ml.c3.tasks.source_fingerprint(atom)` (sorted compact UTF-8 JSON SHA256,
`ensure_ascii=True`, no NaN). Keys must uniquely identify actual occurrences.
At most 2048 distinct atoms from exactly one nonempty `doc_id` are supported.
A workbook cell group must be selected explicitly; an entire workbook is not
implicitly an eligible group. Evidence keys cannot reach outside that context.
Do not assemble the `atoms` argument from separately pooled task contexts.

`source_local_literal` permits only `LITERAL_HEADS` in `ml/c3/tasks.py`;
`source_local_noise` permits only `NOISE_HEADS`. The per-blob list can narrow
these sets, never broaden them. Current literal cap:

- `col:label_type`, `col:about`, `col:supplier`, `col:admission`;
- `read:doc_kind`, `read:hours_stated`, `read:labor_hours`, `read:crew_size`,
  `read:equipment_qty`, `read:display_size`, `read:run_length`, `read:work_window`,
  `read:start_date`, `read:site`, `read:address_level`, `read:price_basis`,
  `read:rate`, `read:rate_for`, `read:billing_type`, `read:stage_std`, `read:deal_stage`.

Noise cap: `col:label_type`, `col:admission`, `read:noise_class`.
Every admitted head must also be a current universal, line-sized, nonrelation
registry opportunity. Each supervised occurrence/head needs its own reviewed
provenance entry; a head allowlist alone grants no supervision. Promoted labels
are rejected. Missing or derived `_absent` targets train nothing. This preserves
explicit source quantities, labor hours, sites, and work windows without admitting
company hours estimates or converting an arbitrary duration into labor hours.
Eligibility is a reviewed declaration, not a semantic theorem inferred by code.

Unknown profiles, quote/planning forecasts (even `certified:true` claims), any
non-null `as_of`, extra profile fields, misplaced top-level task metadata, v2
metadata in v1, cross-document evidence and changed fingerprints are errors.
`execute:false` forecast declarations are also errors at this API: the builder
must omit those task instances and emit explicit external skip receipts. Never
retry a rejected v2 artifact through the v1/unrestricted loader. The earlier
proposal JSON is a specification, not an accepted runtime envelope.

## Physical and supervision closure

Projection and raw source eligibility/conflict checks happen before derivation, featurization, windows and
encoder execution. The loader retains only selected atoms and direct admitted
label fields; embedded labels, global stage/view, notes, judgments, links,
policy, outcome and deferred source supervision are removed from its copy.
Observed source `stage_at` may be retained only with optional
`context.observed_stage_evidence={source_key: "review-reference"}` for that exact
fingerprinted source occurrence. A missing admitted stage is an error. Without
that explicit eligibility it stays blank. The universal `stage_std`/`deal_stage`
targets need their own normal target provenance. The loader never copies gold
stage targets into runtime context slots. Known company `co_stage_raw` extraction
is unsupported by this minimal universal-only profile; requesting it is an error,
not a claim that the source fact is illegitimate.
Existing source exclusions (exclude weight tier, exclusion note, ignore consumer,
`skip`/`exclude_from_training` reads) and nonexcluded row weight tiers are resolved
before sanitization. Their effect is retained without encoding the excluded note.
Conflicting human rows cannot become identical simply because their notes were
removed. Provenance eligibility never overrides these source exclusions.
Original customer atoms/labels are never modified. Materialization fingerprints
prevent rewriting inputs/profile before featurization; windows can subset the
same admitted context, not widen or rewrite it. Task identity includes a profile
hash, and each featurized part binds its inputs and profile with a digest. A second digest binds actual scalar/set/numeric/literal targets, row identities and head scales after all holds; eligible-pair membership cannot restore a held target or change an admitted number.

The loss route runs only direct universal head CE/set loss, eligible numeric
regression and eligible literal span pointers. All other loss terms are disabled
without evaluating them: company/forecast heads, deal numbers, hindsight/EMA,
absence, adversary, cards, teacher/lessons, reasons, external banks/rules,
relations, derived cards, graph/node/document pointers and layout objectives.
General/field/link/card notes and the corresponding target carriers are cleared
and checked. Anchor caches, KL, parameter pull and EMA updates skip these parts. Ordinary v1 number and literal
losses now honor zero and fractional `head_scale` too.

`run_base` rejects any auxiliary training set with teacher loading or positive
teacher/bind/teach/flips/spread budgets, before model construction. Set those
budgets explicitly to zero; default settings are rejected. Auxiliary samples
retain blank view tags and only explicitly evidenced source stage inputs even when the combined model explicitly enables
`saved cfg.view_tag=True`; quoting/planning tags on legacy samples still require
validation. Existing checkpoint semantics are unchanged. Serving via `read_deal`
rejects auxiliary profiles; no causal serving API is supplied here. Refit inputs,
refit implementation and active checkpoint-producing worktrees are untouched.
File API: `load_deal(path, task_consumer="bounded_auxiliary")` or
`load_split(paths, schema, task_consumer="bounded_auxiliary")`; CLI uses explicit
`--task-consumer bounded_auxiliary`. Default `legacy` refuses v2 before features,
cache or postfit forward; the auxiliary consumer conversely refuses legacy
unrestricted files. Existing head-fit, company-refit and question-fit call the
legacy defaults and therefore cannot execute on v2 files. Draft-promotion maps
are refused in this auxiliary file loader. None of those refit implementations
or their inputs was edited. In-memory `c3_loss`/guarded `run_base` remain direct
consumers, and auxiliary evaluation validates the same profile/target snapshot,
uses company=None, and scores only direct targets, never inferred no-link cards.
The 46 loss terms and 12 configuration fields have explicit census membership;
an added unclassified weight fails closed until its disposition is reviewed.

## Freeze-owner adapter obligations

1. Keep every unproven quote/planning task disabled with explicit skip receipts;
   preserve all existing source exclusions and all relation/read-field holds.
2. Reconcile artifact/cell-group admission and target eligibility with the actual
   reviewed source occurrences. Narrow the broader proposed head list to this
   implementation's cap. Unsupported semantic cases require further design,
   never a fallback profile or a guessed request/receipt date.
3. Emit separate auxiliary artifacts, retaining existing `groups` in each v2
   contract. Use exact source fingerprints and per-target evidence refs. Apply
   this before promotion/repeats/global aggregation; never share full-deal
   contexts across tasks or relabel missing values as absence.
4. Bind exact integrated code SHA, source compile/registry, policy/selector/hold
   hashes, emitted task/profile/output hashes, and skip receipts into the owner's
   approval manifest. This loader does not write or approve that manifest.
5. Validate actual rebuilt parts: physical inputs and context identity; all target
   carriers; allowed-target completeness; literal quantities/windows/sites;
   absence/teacher/reason/card/pointer/number/literal closure; original rows
   unchanged; windows and balances; exact-SHA CPU smoke and contract evidence.

Passing synthetic tests is implementation evidence only. No actual approved-data
causal audit or builder adaptation has been performed by this code change.

## Separate passes: preserve the main WHY schedule

The mixed-run refusal is intentional. Do not set the certified main RunE teacher,
card, reason or lesson budgets to zero in order to admit auxiliary samples.

1. Main certified pass: use only its approved certified/legacy parts, unchanged
   teacher/card/reason schedule and preregistered selection protocol. No held
   forecast instances or v2 auxiliary files join that pass or its postfits.
2. Optional separate extraction pass: fork the exact main candidate's **weights**
   into a different candidate/checkpoint path; preserve the original main model,
   teacher artifact and their hashes. Validate auxiliary-only v2 parts, use
   direct `c3_loss` steps with a **fresh** optimizer, and disable teacher, banks,
   anchor KL/pull, EMA updates and postfits on those steps. Generic auxiliary
   `run_base(resume=True)` is refused. No reuse of main momentum, EMA targets,
   checkpoints or pick state is certified. No automatic stage/pick merges.
3. Treat the result as a new candidate: record immutable parent checkpoint hash,
   integrated code/schema/config/view_tag, ordered task/input/supervision hashes,
   fresh optimizer settings and new checkpoint hash in the owner's manifest.
   Keep the main winner if auxiliary adaptation fails selection; never overwrite
   it or call an auxiliary extraction loss an improvement in WHY reasoning.
4. Rerun all main golden/generalization/site/reasoning/link/bypass/company and
   serve-parity gates at comparable operating points, with real nonempty audit
   coverage. Any auxiliary weight update invalidates previous candidate metrics
   and matching refit identities; obtain new matching refit/hash evidence through
   its owner without touching an active run. No automatic INIT/pick/refit bypass.

This is **supervised-target isolation**, not strict parameter/update isolation:
valid extraction updates shared trunk, description projections and q-derived
representations. Dedicated unused parameters receive no new supervised gradient;
with fresh optimizer and grad=None they do not move from decay/stale momentum.
There is no claim that all forecast-related parameters stay fixed, or that an
old teacher/refit/EMA becomes compatible with adapted weights. A strict frozen-q
option, broader source-local heads, and per-part mixed teacher scheduling are
outside this bounded implementation. Positive extraction tests and closure
receipts do not establish full-base WHY coverage or generalization.

The synthetic checkpoint test checks immutable parent bytes, fresh optimizer,
view_tag load/save and a distinct reloadable candidate after one small CPU step.
It does not run a customer pass, select a winner, validate a real checkpoint or
approve either main or auxiliary data.

## Teacher artifact and active-loss receipts

Auxiliary `run_base` skips final teacher initialization/finalization and does not
write an untrained `.teacher` sidecar. The normal main-run teacher finalization
behavior is unchanged. Any teacher sidecars from earlier auxiliary probes are
unsupported artifacts, not compatible teachers or WHY coverage evidence.

A lines-only auxiliary schedule has `numbers=0` and `literals=0` by the existing
curriculum. Nonempty numeric/span targets do not demonstrate value learning.
An adaptation claiming that learning must enable the numbers stage (or explicit
positive direct-loss weights) and report positive named numeric/literal weighted
contributions. The actual-loop synthetic regression checks this with the numbers
stage, verifies no teacher write/step, and preserves a legacy finalization control.
