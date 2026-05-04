# Gold standard — STRESS_AV_TRIO (AMBAG portion)

This bundle has 3 artifacts: ICMA + Hayward + AMBAG. **This document covers AMBAG/MBARD only**; ICMA and Hayward have separate gold sheets.

## Artifact: `ambag_mbard_av_addendum1.pdf` (~50+ pages, 1982 lines pdftotext)

**Association of Monterey Bay Area Governments (AMBAG) — Audio Visual Design & Installation, Monterey Bay Air Resources District (MBARD) Main Conference Room — Addendum #1**

- Issued: June 29, 2023
- Site Visit: July 12, 2023, 1:00–3:00 PM (mandatory pre-proposal walk-through)
- Questions Due: July 17, 2023
- **Proposals Due: August 4, 2023 → August 11, 2023** (deadline extended; both dates appear with strikethrough in the source — testing parser's ability to detect deadline change)
- Contract location: 24580 Silver Cloud Court, Monterey, CA 93940 (3rd floor main conference room)
- Two-agency joint use: AMBAG (MPO for tri-county region) + MBARD (air-quality agency for Monterey/San Benito/Santa Cruz)
- Federal funding (DBE/Title VI/Caltrans/FHWA references)

### Service line: `av` with strong `procurement_compliance` adjacency

This is an **addendum**, not the original RFP — note the `addendum_supersedes` lattice rule applies. Without the original RFP, the addendum looks like an open-scope solicitation; with the original, the addendum's deadline extension and clarifications govern.

This is also a **federal-funded contract** (Caltrans, FHWA via DBE Section G), which adds a layer of compliance entities that the av_pack alone won't recognize.

### Expected entity_keys (must include)

- `customer:ambag` (Association of Monterey Bay Area Governments)
- `customer:mbard` (Monterey Bay Air Resources District) — joint use
- `address:24580_silver_cloud_court_monterey_ca_93940`
- `site:ambag_mbard_main_conference_room_3rd_floor`
- `region:tri_county` (Monterey, San Benito, Santa Cruz — 21 jurisdictions, 5800 sq mi)
- `region:caltrans_140_miles_2_5_hr` (driving distance reference)
- **Configurations** (3 named):
  - `room_config:north_south` (with seating chart shown)
  - `room_config:east_west`
  - `room_config:classroom`
- **Devices** (mostly generic, no proprietary parts):
  - `device:wireless_handheld_mic`
  - `device:wireless_lapel_mic`
  - `device:podium_mic`
  - `device:ceiling_mounted_camera`
  - `device:wall_mounted_camera`
  - `device:tripod_mounted_camera`
  - `device:video_switcher`
  - `device:recording_system`
  - `device:distribution_equipment`
  - `device:av_control_system`
  - `device:wireless_pc_connection`
  - `device:tv_screen` (existing, 2 units mentioned as "may incorporate")
  - `device:speaker_system` (existing)
  - `device:equipment_rack` (cabinet with cooling, in main conference room)
- `service:av_design_drawings`
- `service:procurement_and_delivery`
- `service:configuration_and_site_prep`
- `service:installation`
- `service:testing` (audio + video + virtual platform connectivity)
- `service:training` (AMBAG + MBARD staff)
- `service:ongoing_equipment_service_5yr` (hourly rate basis)
- `service:two_board_meeting_attendance` (vendor must attend 1 AMBAG + 1 MBARD board meeting)
- **Federal compliance**:
  - `requirement:dbe_disadvantaged_business_enterprise` (49 CFR Part 23)
  - `requirement:title_vi_civil_rights`
  - `requirement:eeo` (Equal Employment Opportunity)
  - `requirement:ada` (Americans with Disabilities Act)
  - `requirement:prevailing_wage` (CA Department of Industrial Relations)
  - `requirement:caltrans_audit_access`
  - `requirement:fhwa_audit_access`
- `vendor:project_manager_errol_osteraa` (AMBAG Director of Finance)
- `pricing:firm_fixed_price`

### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `deadline:proposals_due_aug_11_2023` | active | Original Aug 4 was struck through; new deadline Aug 11. **Customer-current-authored override of original RFP.** |
| `scope_inclusion` | `service:av_design_drawings` (Task 1) | active | "Full design drawing(s) and equipment specifications". |
| `scope_inclusion` | `service:procurement_delivery` (Task 2) | active | "All equipment and licenses will be registered to MBARD as the owner". |
| `scope_inclusion` | `service:installation` (Task 4) | active | "Contractor will supply experienced, certified AV engineers". |
| `scope_inclusion` | `service:testing` (Task 5) | active | Multiple audio + video + virtual platform tests required. |
| `scope_inclusion` | `service:training` (Task 6) | active | Includes "(1) AMBAG and (1) MBARD board meeting" attendance. |
| `scope_inclusion` | `service:ongoing_support_5yr` (Task 7) | active | 5-year hourly support rate as Year 1 / 2 / 3 / 4 / 5 line items. |
| `scope_inclusion` | `room_config:three_configurations` | active | "AV solution should be designed to function optimally under both the North/South configuration, East/West configuration, and the classroom configuration". |
| `customer_override` | `decision:incorporate_existing_optional` | active | "Bidders may choose to incorporate these existing capabilities into their proposals, but this is not a requirement" — incumbent equipment optional. |
| `scope_exclusion` | `service:warranty_minimum_1yr` | active | Statement of minimum warranty (1 year) — implicit exclusion of <1yr offerings. |
| `missing_info` | `device:specific_camera_count` | active | "combination of ceiling mounted, wall mounted, and tripod mounted cameras" — count unspecified. |
| `missing_info` | `device:specific_microphone_count` | active | "combination of wireless, handheld, and lapel microphones" — count unspecified. |
| `missing_info` | `pricing:total_amount` | active | Cost Proposal Attachment A has empty "Total Project Not-to-Exceed Amount" cell. |
| `meeting_decision` | `decision:mandatory_pre_proposal_walkthrough_jul_12` | active | Site visit was mandatory; non-attendance is grounds for disqualification. |
| `action_item` | `vendor:firm_fixed_price` | active | "firm offer for at least a ninety (90) day period". |
| `action_item` | `vendor:dbe_obligation` | active | DBE participation required, names + addresses + dollar amounts must be in proposal. |
| `action_item` | `vendor:affirmative_action_policy` | active | "(1) A copy of the consultant's affirmative action policy (applicable for firms with 50 or more employees)". |
| `meeting_decision` | `decision:protest_bond_10pct` | active | Alternative Protest Process — 10% protest bond. |
| `scope_inclusion` | `requirement:dbe_obligation_49_cfr_23` | active | DBE obligation explicit. |
| `scope_inclusion` | `requirement:caltrans_audit_3yr` | active | "three years from the date of final payment under the contract". |
| `meeting_decision` | `decision:contract_extends_to_jun_30_2024` | active | "CONTRACTOR shall complete all tasks on or before June 30, 2024". |
| `meeting_decision` | `decision:single_award_anticipated` | active | "AMBAG anticipates awarding one (1) single award" (implied; standard contract). |

**Expected packet count**: ≥ 18

### Expected ontology gap candidates

- `tri_county` (regional concept — Monterey/San Benito/Santa Cruz)
- `metropolitan_planning_organization` / `mpo`
- `air_resources_district`
- `vehicle_miles_travelled` / `greenhouse_gases` (the *justification* for the project)
- `north_south_configuration` / `east_west_configuration` / `classroom_configuration` (room layout terminology)
- `seating_chart` (block term)
- `disadvantaged_business_enterprise` / `dbe`
- `prevailing_wage` (CA Dept of Industrial Relations)
- `alternative_protest_process` (CA procurement-specific)
- `protest_bond_10_percent`
- `oaks_initiative` (Santa Monica's parallel — should NOT fire here, but trigger `local_law` ontology)
- `caltrans` / `fhwa` / `fta` (state/federal funding agencies — gap candidates for a generic AV pack)
- `levine_act_disclosure` (likely; CA-specific)
- `49_cfr_18_39_i_11` / `49_cfr_part_23` (regulatory citations)

### Expected exclusion patterns (boilerplate likely fires)

- "but this is not a requirement" → optional/may-be-excluded scope (existing equipment incorporation)
- "Errors and ambiguities in proposals will be interpreted in favor of AMBAG" → vendor-disadvantage rule
- "may be added or deleted during contract negotiations" → scope-fluidity caveat (low-information for parser)
- "AMBAG also reserves the right to award the contract without oral briefings" → procurement-flexibility (boilerplate)

### Expected constraint patterns

- "ninety (90) day" firm offer period
- "30 calendar days" termination notice
- "within ten (10) days" prompt-payment-to-subcontractor
- "fifteenth day of each month" invoicing deadline
- "1 year warranty" → minimum equipment warranty
- "5 year ongoing support" → support contract duration
- "100 points" evaluation total / weighting (25/25/20/30)
- "$1,000,000 per occurrence" professional general liability (boilerplate)
- "8.5 inches x 11 inches, 45 pages" — proposal document format constraint
- "all books, records, and documents... three years from the date of final payment" — retention constraint

### Stress-test attributes

- **Strikethrough text in deadlines** — pdftotext flattens "August 4, 2023 August 11, 2023" as both dates on one line. Parser should detect deadline change. The first date is the original; the second is the addendum-amended version. **Customer-override packet should fire on the date.**
- **No equipment quantities specified** — narrative-only scope. Tests parser's ability to NOT generate quantity atoms when none are given (no false positives).
- **Three room configurations** — multi-modal scope with seating-chart "art" embedded as text in pdftotext output (the seating chart on page 8 produces messy alignment). Parser should treat seating-chart content as low-confidence atoms or unsupported.
- **Federal funding compliance section is dense** — DBE, Title VI, EEO, ADA, prevailing wage, audit rights — pages 16–20 are nearly all regulatory boilerplate. Atoms emitted from these pages should be classified `procurement_compliance` not `scope_inclusion`.
- **Empty cost proposal table (Attachment A)** — "Total Labor Costs for Tasks 1-6 [blank]". Parser must NOT generate quantity atoms for empty cells.
- **Draft Agreement (Attachment C)** is 10+ pages of placeholder contract language with `xxx`, `XXXX`, `($XX,XXX)` placeholders. Parser should detect template state and NOT generate atoms.
- **Two customer entities** (AMBAG + MBARD) sharing one room — dual-customer atom resolution. Parser should produce two `customer:*` entities and link both to the project.
- **Old-style document conventions** — single-spaced, narrow margins, footer "Page N of M". The parser's structural extractor should detect page boundaries cleanly.
- **Mandatory pre-proposal walkthrough** — non-attendance is disqualifying. **Should NOT be confused with pre-bid recommended visits** (different lattice tier).

### Expected metrics

```
expected_min_atom_count: 80       # narrative-only, mostly task descriptions + compliance
expected_min_packet_count: 18
expected_min_devices_in_atoms: 8  # camera, mic, switcher, recording, distribution, control, PC connect, rack
expected_min_compliance_atoms: 10 # DBE, Title VI, EEO, ADA, prevailing wage, etc.
expected_min_quantity_atoms: 0    # genuinely no quantities — parser should not invent any
expected_customer_override_packets: 2  # deadline change, optional-existing-equipment
expected_template_unsupported: 5+  # Attachment A (blank), Attachment C (placeholder draft agreement)
```

### Known difficulties & where the parser will likely fail

1. **Strikethrough detection** — without color-aware extraction, the parser sees "August 4, 2023 August 11, 2023" as two dates. The packet "deadline" should be the second; the first should be flagged as `superseded`. If only one date is extracted, recall metric fails.
2. **Two-customer joint use** — every "ICMA does X" should resolve to BOTH AMBAG + MBARD. Cross-customer entity linking required.
3. **No BOM, no quantities** — most current parsers will hallucinate quantities from the device-name list. Stress test for restraint.
4. **Federal regulatory boilerplate is high-volume but low-information** — pages 16–20 should be filtered. If the parser emits 50 compliance atoms, it's overproducing; if it emits zero, it's missing the DBE/Title VI/prevailing-wage requirements that are real scope constraints.
5. **Three room configurations** — the parser should produce 3 `room_config:*` entities, not collapse to one "conference room" entity. Each configuration may have different microphone/camera placement requirements.
6. **Five-year support split into 5 line items** — Year 1 / Year 2 / Year 3 / Year 4 / Year 5 are 5 separate cells in the cost proposal. Parser should treat as one packet (`service:ongoing_support_5yr`) with 5 quantity atoms, not 5 separate scope packets.
