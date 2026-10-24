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
  `read:rate`, `read:rate_for`, `read:billing_type`.

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

Projection happens before label selection/derivation, featurization, windows and
encoder execution. The loader retains only selected atoms and direct admitted
label fields; embedded labels, global stage/view, notes, judgments, links,
policy, outcome and deferred source supervision are removed from its copy.
Original customer atoms/labels are never modified. Materialization fingerprints
prevent rewriting inputs/profile before featurization; windows can subset the
same admitted context, not widen or rewrite it. Task identity includes a profile
hash, and each featurized part binds its inputs and profile with a digest.

The loss route runs only direct universal head CE/set loss, eligible numeric
regression and eligible literal span pointers. All other loss terms are disabled
without evaluating them: company/forecast heads, deal numbers, hindsight/EMA,
absence, adversary, cards, teacher/lessons, reasons, external banks/rules,
relations, derived cards, graph/node/document pointers and layout objectives.
General/field/link/card notes and the corresponding target carriers are cleared
and checked. Anchor caches/losses skip these parts. Ordinary v1 number and literal
losses now honor zero and fractional `head_scale` too.

`run_base` rejects any auxiliary training set with teacher loading or positive
teacher/bind/teach/flips/spread budgets, before model construction. Set those
budgets explicitly to zero; default settings are rejected. Auxiliary samples
retain blank stage/view tags even when the combined model explicitly enables
`saved cfg.view_tag=True`; quoting/planning tags on legacy samples still require
validation. Existing checkpoint semantics are unchanged. Serving via `read_deal`
rejects auxiliary profiles; no causal serving API is supplied here. Refit inputs,
refit implementation and active checkpoint-producing worktrees are untouched.
Only the documented loader + direct `c3_loss`/guarded `run_base` route is validated;
other consumers are not implicitly certified by their ability to load a Batch.

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
