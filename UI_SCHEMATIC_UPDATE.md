# UI Update Spec — new schematic + verification capabilities

Hand this to the Lovable project. The backend (parser-os) now emits new data in
**`PM_HANDOFF.json`** (the primary UI payload). The UI should add the sections
below. Everything is additive — existing tabs are unchanged.

## 1. NEW: `PM_HANDOFF.json.schematic_review`  (the 100% hand-check)
Only render when `schematic_review.present === true` (doc-only deals omit it).

```jsonc
"schematic_review": {
  "present": true,
  "coverage_pct": 62,            // verified coverage of the legend
  "confident_count": 18,         // symbols count-verified (safe to trust)
  "needs_review_count": 7,       // the PM hand-check queue size
  "review_queue": [              // <- THE PM NOTES LIST (flag-don't-guess)
    {"type": "smoke detector", "status": "flagged",
     "reason": "count mismatch: legend 5 vs found 3"},
    {"type": "card reader", "status": "missing",
     "reason": "legend declares 2, found 0"}
  ],
  "summary": "18 verified & safe to deliver; 7 need review -> emitted accuracy ~100%"
}
```
**Display:** a "Schematic Review" card with a coverage bar + two lists:
- ✅ **Confident** (count) — verified, no action.
- ⚠️ **Needs review** (`review_queue`) — each row = a one-line PM note: type +
  status chip (FLAGGED/MISSING) + reason. A "Confirm / Correct" button per row
  feeds the correction loop (self-heal). **This is the no-silent-errors guarantee
  surfaced to the PM.**

## 2. COMING (after detector trains + connectivity wires): takeoff
Will appear under schematic_review:
- `device_counts` — BOM/takeoff per device type (whole-project rollup)
- `cable_runs` — from → to → type → **length** (material takeoff)
- `quantity_contradictions` — legend vs detected mismatches (already partly in queue)

## 3. Audit/PM fields already present (unchanged)
`parser_quality_score`, `blockers`, `cross_doc_contradictions`, `stakeholders`,
`money_summary`, etc. — see OUTPUTS_FOR_UI.md.

## Notes
- The UI only needs to fetch `PM_HANDOFF.json` (this field is embedded).
- Safe to ship before the detector: `schematic_review` works off whatever
  schematic atoms exist today; it gets richer as the detector lands — no UI change
  needed then.
