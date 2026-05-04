# Gold standard — STRESS_NET_MAINT (Mobile portion)

This case has 3 artifacts: Mobile + OCTA + MS ITS. **This document covers the Mobile RFP only**; OCTA and MS ITS XLSX have separate gold sheets.

## Artifact: `mobile_camera_maint_RFP5954.pdf` (10 pages)

**City of Mobile, AL — RFP 5954: Security Camera Maintenance Services**

- 6-page main RFP body + 4-page **Exhibit A Camera Inventory** (a real device-count table embedded in PDF)
- Proposals due 4:00 p.m., July 18, 2025
- Mandatory pre-proposal conference June 26, 2025 (recorded as a `meeting_decision` if minutes exist; we don't have them in this bundle)
- One-year contract, renewable for two additional one-year periods (3 years max)

### Service line: `security_camera` (with a *strong* networking + electrical adjacency)

The Mobile RFP is a **maintenance contract**, not a new install. It stresses the parser's ability to differentiate:
- *Maintenance* scope (repair, replace, install in response to existing failures) from
- *New install* scope (initial cabling/mounting/programming).

It also introduces **service-level criticality classes** (Red/Yellow/Green/White) — a constraint pattern not in any current pack.

### Expected entity_keys (must include)

The Exhibit A camera inventory contains **80+ named sites** with explicit camera quantities. Every one should produce a `site:*` entity_key + a `quantity` atom. Sites include:

| Site name | Current cams | Expected | Total |
|---|---|---|---|
| MMOA | 75 | — | 75 |
| Saenger Milestone | 50 | — | 50 |
| Convention Center | 148 | — | 148 |
| Cruise Terminal | 154 | — | 154 |
| Civic Center | — | 150 | 150 |
| Canal Parking Garage | — | 125 | 125 |
| GulfQuest | 30 + 80 | — | 110 |
| Public Services | 98 | — | 98 |
| ... (+70 more sites) | ... | ... | ... |
| **Total** | | | **1,523** |

Expected `device:ip_camera` should appear with `aggregate=true` value flag for the row total (1,523), with individual site rows attached as supporting atoms.

### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `device:ip_camera` (aggregate 1,523) | active | Real device count from a contractual roster (Exhibit A). |
| `scope_inclusion` | `site:mmoa`, `site:cruise_terminal`, `site:convention_center`, etc. | active | Per-site quantities. |
| `scope_inclusion` | `service:repair_replacement` | active | Section II.1 — repair/replace/install scope. |
| `scope_inclusion` | `service:event_support` | active | Section II.2 — pre-event survey + temporary cameras. |
| `scope_inclusion` | `service:inventory_database_entry` | active | Section II.3 — Genetec or Nexgen inventory updates. |
| `site_access` | `site:secure_facility_pdq` | needs_review | "All working on City contract must be able to pass MPD criminal background check". Background check requirement applies city-wide. |
| `missing_info` | `vendor:vms_software_decision_genetec_or_similar` | needs_review | Section II.5: "may be Genetec Security Center or a similar product" — VMS decision is open. |
| `missing_info` | `service:event_list_72_hours_notice` | needs_review | Section II.2.a: "Events to be identified by City with a minimum 72-hours notice" — events not enumerated. |
| `customer_override` | `pricing:hourly_with_threshold` | active | Section IV.2: "service calls beyond two hours or requiring equipment exceeding $250" require pre-approval. Constraint on vendor cost autonomy. |
| `meeting_decision` | `decision:phased_growth_3000_cameras` | needs_review | Section I: City expects camera count to grow to ~3,000 within a year. |

### Expected constraint patterns (gap candidates)

The **Red/Yellow/Green/White criticality response classes** are NOT in any current pack. The gap detector should flag:
- `Red: Two hour on-site response` → `constraint:sla_response_2hr`
- `Yellow: Six-hour on-site response` → `constraint:sla_response_6hr`
- `Green: 24-hour on-site response` → `constraint:sla_response_24hr`
- `White: 48-hour on-site response (or next business day)` → `constraint:sla_response_48hr`

Recommend adding a new `constraint_patterns.sla` bucket to `default_pack.yaml` with these.

Other constraint atoms expected:
- "Master electrician or equivalent on staff or on retainer" → `constraint:license_master_electrician`
- "Alabama Electrical Contract Board Electrical Contractor License" → `constraint:license_al_electrical_contractor`
- "CompTIA Network +" → `constraint:certification_network_plus`
- "MPD criminal background check" → `constraint:background_check`
- "City of Mobile business license" → `constraint:license_city_of_mobile`
- "AL Code 34-1A" → `constraint:license_al_electronic_security_board` (optional)
- Insurance: $1M GL, $1M auto, $1M professional, $2M umbrella → `constraint:insurance_*` for each line
- "A.M. Best rating of A-VII or better" → `constraint:insurance_carrier_rating`

### Expected vendor mentions
- Genetec Security Center
- Nexgen (city's existing software)

### Stress-test attributes

- **Spreadsheet-in-PDF (Exhibit A)** — the camera inventory is an 80-row, 3-column table embedded in pages 7–10 of the PDF. The OrbitBrief PDF table extractor MUST reconstruct this as a `block.kind = "table"` with all rows preserved, and the structured.json should round-trip.
- **GIS map on page 7** — first page of Exhibit A is a map graphic (bitmap), not a table. The parser should handle the page-mode transition gracefully without emitting noise atoms.
- **Total row at the very end** — "**Total** | | **1,523**" — classic mid-sheet total-row pattern. The xlsx_parser handles this; testing whether the PDF table extractor does too.
- **Two columns of quantities** (Current Cameras + Expected Growth) — many sites have BOTH a current count and an expected growth count. The parser must NOT sum them blindly; the "Total" column already does the sum. Expected behavior: emit two atoms per site (current vs expected growth), or one atom with `value.current_cameras` and `value.expected_growth` separately.
- **Extensive insurance/legal boilerplate (pages 4–6)** — pages 4, 5, 6 are nearly all insurance, indemnification, subrogation. Many are noise; only the dollar-amount minimums are useful. Atoms emitted from these pages should be flagged `low_confidence_atom` or filtered as boilerplate.
- **Service-level codes (Red/Yellow/Green/White)** — color-coded SLA tiers. Without pack support, the gap detector should flag all four phrases.
- **License plate detail not specified** — contrast with VT (which DID specify license plate at entrances/exits). Mobile RFP doesn't say either way; missing_info candidate.

### Expected metrics

```
expected_min_atom_count: 110     # 80+ site rows + ~30 narrative atoms
expected_min_packet_count: 8
expected_min_quantity_atoms: 80   # one per site row, plus aggregate
expected_aggregate_quantity_atom: {value: 1523, key: "device:ip_camera"}
expected_total_row_handled: true
expected_min_unsupported_receipts: 1  # the GIS map page
```
