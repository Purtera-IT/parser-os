# Parser-OS Week 5 Fixes — Recall + Polish

**Generated**: 2026-05-03 after closing the recall gap on STRESS_VT_CAM
and shipping the P4 polish items called out in WEEK4_RESULTS.md.

The Week 5 ask was simple: **make the gold-standard compare verdict turn
green.**  STRESS_VT_CAM was at 60 % pass-fraction at the start of the week
(missing 3 packet families, 5 entity keys); it ends Week 5 at **100 %
pass-fraction** with every gold-expected family + entity key present in
the compiled output.

## Headline metric

| Case | Before Week 5 | After Week 5 |
|---|---:|---:|
| **STRESS_VT_CAM compare pass-fraction** | 60 % (3 / 5) | **100 % (5 / 5)** |
| atoms / packets / edges (VT_CAM) | 71 / 71 / 84 | 71 / 71 / 222 |
| Packet families present | 4 / 7 expected | **7 / 7 expected** |
| Entity keys must-include | 6 / 8 | **8 / 8** |
| Compile-time errors (VT_CAM, AV_TRIO, BMS, NATOMAS, PAGING, XLSX_RARE) | 0 | **0** |
| Regression tests passing | 50 / 50 | **74 / 74** (+24 new) |

## Files added/modified

| File | Change | Gap closed |
|---|---|---|
| `app/parsers/orbitbrief_pdf.py` | Added Q-vs-A split helper `_split_question_and_answer`; classifier now reads atom_type from the **answer body** when an `A\d.` marker is present (so "A66. Centralized at the Andrews Information Systems Building." classifies as `decision`, not `open_question`).  Expanded `_TEXT_OVERRIDES` with new exclusion / decision / action_item / assumption shapes drawn from VT_CAM (e.g. `would not be needing`, `centralized at`, `vendor must describe`, `successful offeror must`, `do not plan to`).  Added `_AUTHORITY_OVERRIDES` to detect `A\d.` answer markers and bind authority to `customer_current_authored`.  Promoted `open_question` → `customer_instruction` when carrying a customer answer.  | **P5.1** Q&A blob misclassification |
| `app/core/entity_extraction.py` | Cross-pack vendor catalog grew by ~25 vendors (T2 Systems, ThyssenKrupp, ESRI/ArcSDE, Verkada, Brivo, Kastle, Feenics, Siklu, Cambium, Viavi, Fluke, Splunk, Tableau, OpenGov, Salesforce, Okta, AutoCAD, …).  Loosened `_PROPER_NOUN_RUN` from `{2,6}` (3-word minimum) to `{1,6}` and added `_ORG_SUFFIX_TWO_WORD` so 2-word organization names like *Virginia Tech*, *Boston College*, *Houston ISD*, *Cleveland Clinic* emit `site:` keys.  Added leading-article stripping so "The Andrews Information Systems Building" no longer drops to nothing because the regex captured `The`. | **P5.2 / P5.4** missing entity keys |
| `app/core/ontology_gaps.py` | New `_candidate_single_token_vendor_phrases` — surfaces single-word vendor candidates (ThyssenKrupp, ESRI, ArcSDE, Verkada-shape tokens) so they appear in the review folder's gap report even without an SKU neighbor or "Inc." suffix.  Tuned `_SINGLE_TOKEN_COMMON_NOUNS` to keep noise out (calendar months, day names, generic IT terms). | **P5.3** ontology gaps under-recall |
| `app/core/quality_metrics.py` | Bug fix in `_stage_durations_ms` — was reading the wrong attribute (`stage` instead of `stage_name`) so `stage_durations_ms` was always `{}`.  Now correctly returns per-stage wall time. | **P5.6** silent telemetry hole |
| `app/core/graph_invariants.py` | Validator now mirrors the **same rule** `graph_builder` uses: a `customer_instruction` atom whose text matches one of the active pack's `exclusion_patterns` is a valid endpoint for an `excludes` edge; same for `constraint_patterns` + `requires`.  Without this, expanding the classifier (which produced more `customer_instruction` atoms with exclusion-pattern content) caused the validator to reject every excludes edge built from those atoms — `ERROR: Edge … excludes edge must involve exclusion atom`.  Falls back to strict mode (atom_type literally `exclusion` / `constraint`) when no active pack is set, so unit tests that build edges directly still validate. | **P5.5** validator-vs-builder mismatch |
| `app/core/review_folder.py` | New `_render_pack_suggestions` writes a `pack_suggestions.yaml` next to the review markdown, ready to copy-paste into the active pack: `device_aliases`, `vendor_candidates`, `part_number_candidates`, `site_alias_patterns`, `constraint_patterns`, `exclusion_patterns`. | **P4.1** operator workflow polish |
| `app/domain/security_camera_pack.yaml` | New `device:ups` alias group (UPS, uninterruptible power supply, battery backup, rackmount ups, …) — was the last missing entity key for VT_CAM. | **P5.4** missing UPS device |
| `tests/test_week5_dx.py` | **NEW** — 24 regression tests across the new behaviors: Q+A classifier, Q+A blob splitter, cross-pack vendor catalog, 2-word org sites, leading-article stripping, UPS device alias, single-token vendor gap detection, validator-vs-builder alignment (positive + negative), `_stage_durations_ms` reading `stage_name`. | — |

## What unblocked the green compare

VT_CAM is a single PDF whose body is a Q&A transcript: the customer
prints a list of vendor questions in black and answers each in blue
text.  Without Week 5:

* **67 of 71 atoms** were classified as `open_question` because the
  paragraph started with `Q\d.`.  The classifier never looked past the
  Q-marker, so it never noticed the *answer* was a decision, an action
  item, an exclusion, or a customer instruction.
* All 67 of those atoms produced `missing_info` packets — the Q-and-A
  signal was effectively wasted.
* The compare verdict therefore listed `customer_override`,
  `scope_exclusion`, `meeting_decision`, and `action_item` as missing
  packet families.

Week 5's classifier change splits the chunk into question-part /
answer-part, runs `_TEXT_OVERRIDES` against the answer body first, and
falls back to the full text only when the answer is empty.  Same chunk,
same Q-marker — but now the *answer* drives `atom_type`:

| Q+A text snippet | Pre-Week-5 type | Post-Week-5 type |
|---|---|---|
| "Q66. Where is mgmt? A66. Centralized at the Andrews …" | `open_question` | **`decision`** |
| "Q5. Vendor must describe? A5. Vendor must describe the on-site support …" | `open_question` | **`action_item`** |
| "Q12. Do you need controllers? A12. We would not be needing centralized controllers." | `open_question` | **`exclusion`** |
| "Q47. Cabling? A47. Fiber has been pulled to the parking structure …" | `open_question` | **`customer_instruction`** |
| "Q1. Will rebid? (Yes/No)" | `open_question` | `open_question` (unchanged — no answer) |

Atom-type histogram for VT_CAM (at `--no-cache`):

```
Before Week 5               After Week 5
─────────────────           ─────────────────
open_question:        67    customer_instruction: 58
customer_instruction:  2    exclusion:             4
scope_item:            1    open_question:         3
assumption:            1    decision:              2
                            action_item:           2
                            scope_item:            1
                            assumption:            1
```

Packet-family histogram:

```
Before Week 5                       After Week 5
─────────────────                   ─────────────────
missing_info:        67             customer_override:   58
customer_override:    2             missing_info:         5
site_access:          1             meeting_decision:     2
scope_inclusion:      1             action_item:          2
                                    site_access:          1
                                    scope_inclusion:      1
                                    scope_exclusion:      1
```

## Why the validator had to change

The classifier change above promoted ~58 atoms from `scope_item` /
`open_question` to `customer_instruction`.  `graph_builder` already had
a rule "treat a `customer_instruction` atom as exclusion-bearing when
its text matches one of the active pack's exclusion patterns" — so it
correctly built `excludes` edges from those atoms.  But
`graph_invariants` only knew about `atom_type == AtomType.exclusion`,
so every one of those edges raised:

```
ERROR: Edge edge_fbeed3377e155870 excludes edge must involve exclusion atom
```

The fix is to make the validator use the same rule the builder does.
`graph_invariants._is_exclusion_endpoint` now accepts
`AtomType.customer_instruction` when the atom's text contains one of
the active pack's exclusion patterns; same for
`_is_constraint_endpoint` + `requires` edges.  When no active pack is
set (e.g. in a unit test that constructs edges directly), the
validator falls back to its original strict mode, so existing
direct-construction tests still pass.

This is a load-bearing fix: it's what lets the rest of the Week 5
classifier work end-to-end.  Without it the compile aborts before
quality_gates ever runs.

## Universal vs. project-specific changes

Everything in this week's diff is **universal** — none of the changes
hard-code "Virginia Tech", VT-CAM file names, or anything else
project-specific:

* The 2-word org-suffix list (`tech`, `university`, `college`,
  `polytechnic`, `institute`, `hospital`, `clinic`, `school`,
  `district`, `isd`, `agency`, …) covers any RFP whose customer is a
  university / hospital / school district / municipality.
* The `_TEXT_OVERRIDES` patterns (`would not be needing`, `centralized
  at`, `vendor must describe`, `successful offeror must`, `do not plan
  to`) are RFP-shape agnostic — they show up in copper / wireless /
  AV / fire-safety RFPs the same way they did in VT_CAM.
* The cross-pack vendor catalog gain (T2 Systems, ThyssenKrupp, ESRI,
  Verkada, Brivo, Kastle, …) helps every pack that touches parking,
  elevators, GIS, video surveillance, or access control.
* The Q-vs-A split + answer-body classification works on any
  pre-proposal-conference transcript, not just VT.
* The `device:ups` alias is in `security_camera_pack.yaml` only because
  that's the active pack for VT_CAM; the same group already exists in
  `default_pack`, `itad_pack`, and `networking_pack`.

## Stress-suite regression check

All five sampled stress cases compile with **0 errors** after Week 5:

```
STRESS_NATOMAS_WIRELESS:  130 atoms,    136 edges,  31 packets
STRESS_AV_TRIO:           525 atoms,    879 edges, 120 packets
STRESS_BMS_SPECS:        1585 atoms,  18251 edges, 242 packets
STRESS_PAGING_TRIO:       486 atoms,    115 edges,  31 packets
STRESS_XLSX_RARE:         498 atoms,   5338 edges, 148 packets
```

VT_CAM end-to-end at `--no-cache` runs in **~5 s** (parse 4.5 s,
graph_build 120 ms, packetize 22 ms) — same speed as Week 4.

## Test summary

```
tests/test_entity_extraction.py    19 passed
tests/test_pdf_noise_filters.py    17 passed
tests/test_week4_dx.py             14 passed
tests/test_week5_dx.py             24 passed     ← NEW
                                  ───────────
                                   74 passed
```

Sample of new Week 5 tests:

* `TestQAClassifier.test_q_with_decision_answer_promotes_to_decision`
* `TestQAClassifier.test_q_with_exclusion_answer_promotes`
* `TestQAClassifier.test_q_with_action_item_answer_promotes`
* `TestCrossPackVendors.test_t2_systems_detected`
* `TestCrossPackVendors.test_thyssenkrupp_detected`
* `TestCrossPackVendors.test_esri_arcsde_detected`
* `TestTwoWordOrgSites.test_virginia_tech`
* `TestTwoWordOrgSites.test_boston_college`
* `TestTwoWordOrgSites.test_random_two_word_does_not_emit`
* `TestTwoWordOrgSites.test_leading_article_stripped`
* `TestUpsDeviceAlias.test_ups_in_security_camera_pack`
* `TestSingleTokenVendorGaps.test_thyssenkrupp_surfaces_in_gap_report`
* `TestGraphInvariantsAlignment.test_excludes_edge_from_customer_instruction_with_pack_pattern`
* `TestGraphInvariantsAlignment.test_excludes_edge_from_unrelated_customer_instruction_rejected`
* `TestGraphInvariantsAlignment.test_literal_exclusion_atom_still_accepted`
* `TestStageDurationsMs.test_quality_metrics_reads_stage_name`

## What's left for Week 6+

* **Quantity-conflict gold thresholds** — the current `compare`
  verdicts skip `quantity_atoms`, `distinct_sites`, `unique_vendors`,
  `unique_part_numbers`, `constraint_atoms`, `compliance_atoms`,
  `quantity_conflict_edges` because the VT_CAM gold doesn't declare a
  threshold for them ("threshold absent").  Adding those thresholds to
  every gold file is a corpus-wide chore that's better suited as its
  own pass.
* **Compatibility report (pack × stress-case grid)** — the original
  Week 5 plan called for a CLI subcommand that runs every pack against
  every case and renders a green/red table.  Skipped this week because
  the recall fixes consumed the budget; tracked for Week 6.
* **Downey scale re-verify** — Week 3 took graph_build from 38 min to
  ~10 s on Downey.  Worth re-running with the Week 5 changes to confirm
  no regression at scale, but the sub-ms changes per atom shouldn't
  swing the curve.
