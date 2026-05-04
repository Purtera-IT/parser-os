# `real_data_cases/` — stress-test corpus

The corpus that gates parser regressions. Each subdirectory is one
self-contained **case** — a real-world project (RFP / addendum / vendor
quote bundle) plus a `labels/gold_standard.json` describing what a
successful compile should produce.

`parser-os matrix --cases-dir real_data_cases ...` runs every case
through `compile` and `compare`, then aggregates the results into a
green/red grid.

## Directory layout per case

```
STRESS_<NAME>/
├── SOURCE_NOTES.md                  # provenance + why this case is interesting
├── case_manifest.json               # (optional) fixture metadata
├── README_CASE.md                   # (optional) human-readable case notes
├── project.yaml                     # (optional) pinned domain pack / customer
├── artifacts/                       # source documents (PDF, XLSX, DOCX, …)
│   ├── *.pdf
│   └── *.xlsx
└── labels/
    ├── gold_standard.json           # the comparison target
    ├── gold_standard.md             # human-readable version
    ├── gold_packets.json            # (some cases) per-packet expectations
    └── packet_reviews_template.json # (some cases) review-workflow seed
```

The `artifacts/` directory is `.gitignore`d (`real_data_cases/*/artifacts/*`).
Bootstrap it with `bash scripts/fetch_stress_test_corpus.sh`.

## Gold-standard schema (`labels/gold_standard.json`)

```json
{
  "case_id": "STRESS_VT_CAM",
  "service_line": "security_camera",
  "recommended_domain_pack": "security_camera",
  "bundle_summary": "Virginia Tech RFP #0016531 Addendum #2 — Q&A from pre-proposal conference, 67 numbered Q/A pairs ...",

  "expected_artifacts": [
    {"filename": "...", "artifact_type": "pdf"}
  ],

  "expected_min_atom_count": 60,
  "expected_min_packet_count": 12,
  "expected_min_quantity_atoms": 5,
  "expected_min_distinct_sites": 3,
  "expected_min_unique_vendors_referenced": 2,
  "expected_min_constraint_atoms": 8,
  "expected_min_compliance_atoms": 15,
  "expected_min_cross_artifact_edges": 0,
  "expected_quantity_conflict_edges_within_artifact": 1,
  "expected_quantity_conflict_edges_within_artifact_min": true,

  "expected_packet_families": [
    "scope_inclusion", "scope_exclusion", "customer_override", "site_access",
    "missing_info", "meeting_decision", "action_item"
  ],

  "expected_entity_keys_must_include": [
    "site:virginia_tech", "site:perry_street_parking_deck",
    "device:ip_camera", "device:ups",
    "vendor:t2_systems", "vendor:thyssenkrupp", "vendor:esri"
  ],

  "expected_authority_class_distribution": { ... },
  "expected_exclusion_patterns_fired": [ ... ],
  "expected_constraint_patterns_fired": [ ... ],
  "expected_ontology_gap_candidates": [ ... ],
  "stress_test_attributes": [ ... ],
  "known_failure_modes": [ ... ]
}
```

The fields the comparator (`app/core/gold_compare.py`) uses are:

| Field | Compares against |
|---|---|
| `expected_min_atom_count` | `len(result.atoms) >= N` |
| `expected_min_packet_count` | `len(result.packets) >= N` |
| `expected_min_quantity_atoms` | atoms with any `quantity:*` entity_key |
| `expected_min_distinct_sites` | distinct `site:*` keys across atoms |
| `expected_min_unique_vendors_referenced` | distinct `vendor:*` keys |
| `expected_min_unique_part_numbers` | distinct `part_number:*` keys |
| `expected_min_constraint_atoms` | atoms with `atom_type=constraint` |
| `expected_min_compliance_atoms` | atoms with any `requirement:*` key |
| `expected_quantity_conflict_edges` | edges with `edge_family=part_number_quantity_conflict` |
| `expected_min_cross_artifact_edges` | edges spanning two artifact_ids |
| `expected_packet_families` | every listed family must appear ≥ once |
| `expected_entity_keys_must_include` | every listed key must appear in some atom |

Any field absent from the JSON is reported as `"verdict": "skipped"` —
useful for cases where you don't want to enforce a particular check.

The remaining fields (`stress_test_attributes`, `known_failure_modes`,
etc.) are advisory metadata for humans and are ignored by the
comparator.

## Adding a new case

1. **Create the directory:**

   ```bash
   mkdir -p real_data_cases/STRESS_MY_NEW_CASE/{artifacts,labels}
   ```

   Or scaffold via:

   ```bash
   parser-os init real_data_cases/STRESS_MY_NEW_CASE --service-line wireless
   ```

2. **Drop your source artifacts** into `artifacts/` (PDFs, XLSXs, etc.).
   Don't commit them if they're large or sensitive — the dir is gitignored.
   If you need them in the repo, add them to a fetch script under
   `scripts/`.

3. **Write `SOURCE_NOTES.md`** — provenance, source URL, why this case
   is interesting (what failure mode does it stress?). The pack
   auto-router scores this content for keywords.

4. **Run the compile** to see what the parser produces:

   ```bash
   parser-os compile real_data_cases/STRESS_MY_NEW_CASE \
       --out /tmp/result.json --review-out /tmp/review/
   ```

5. **Inspect the review folder + result JSON.** Tune the pack (or pick
   a different one via `--domain-pack`) until the output looks right.

6. **Snapshot what good looks like into `labels/gold_standard.json`.**
   Start conservative — only set thresholds for metrics you're confident
   the case should clear. The comparator skips absent thresholds.

7. **Verify the case shows up in the matrix:**

   ```bash
   parser-os matrix --cases-dir real_data_cases \
       --out /tmp/matrix.json --markdown-out /tmp/matrix.md
   ```

## Stress-test categories

Cases under `real_data_cases/` are intentionally varied — each one
exercises a different failure mode the parser has historically broken
on. Tags appear in each gold's `stress_test_attributes`:

- `single_artifact_no_BOM` — no canonical bill-of-materials reference
- `color_coded_customer_answers_blue` — Q&A transcript with embedded
  customer-authored answers
- `multi_artifact_addenda_chain` — original + addendum 1 + addendum 2
- `phased_scope` — quantities expressed as ranges or "up to N"
- `embedded_scanned_pages` — image-only pages mixed with text
- `xlsx_table_no_canonical_header` — first non-empty rows aren't headers
- `cross_artifact_quantity_conflict` — same SKU, different quantities
  across vendor quotes
- … and ~30 more.

These tags are advisory — the comparator doesn't read them — but they
help operators decide which cases to add when reproducing a bug.
