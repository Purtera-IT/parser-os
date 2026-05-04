# Gold standard — STRESS_NET_MAINT (MS ITS Managed VPN portion)

This case has 3 artifacts: Mobile + OCTA + MS ITS. **This document covers the MS ITS XLSX only**; Mobile and OCTA have separate gold sheets.

## Artifact: `ms_its_managed_vpn_RFP4080_attachA.xlsx` (3 sheets)

**State of Mississippi ITS (Information Technology Services), RFP 4080, Category XI: Managed VPN — Attachment A**

This is a **standalone XLSX attachment** — the rare real-world spreadsheet artifact in this corpus. Tests the parser's `xlsx_parser` (not `pdf_parser`) and the universal small-table extraction logic.

### Service line: `networking` (Managed VPN service)
**Recommended domain pack**: `networking_pack`

### Workbook structure (3 sheets)

| Sheet | Rows | Cols | Purpose |
|---|---|---|---|
| `12-Managed VPN` | 281 | 5 | Technical Specifications & Requirements Matrix (M = Mandatory; vendor responds A/E/X) |
| `12-Cost Submission` | 41 | 5 | Cost tables: Managed Remote Access Service, Hourly Rate, Change Order Rates |
| `12-Scoring Methodology` | 19 | 5 | 100-base + 5 value-add = 105 max points |

### Sheet 1: `12-Managed VPN` (Technical Specifications & Requirements Matrix)

#### Header schema (rows 1–7)
- Row 1: "Technical Specifications – Requirements Matrix"
- Rows 3–5: Response key (A = acknowledges/will comply; E = unable to meet but offers alt; X = not capable)
- Row 6: Column header — `Ref #` | `Description` | `Mandatory Requirement "M"` | `Vendor Response: A, E, X` | `Vendor Comments (if response is "E", please provide details)`
- Row 7: "MANAGED VPN--ATTACHMENT A"

#### Section structure (Ref# numbering — hierarchical)

- **1. Vendor Qualifications and Experience**
  - 1.1 Organization Description (corporate info, parent corp, state of incorporation, principal office, servicing office, restructurings/mergers, annual report, lines of business %, last full-year revenue, FTE in line of business, years in business)
  - 1.2 Experience (similar engagements, capacity demonstration, state/local govt understanding, paid customer counts: site-to-site VPNs, traditional workstation remote-access VPNs, mobile device VPNs)
  - 1.3 Staff Qualifications (executive/professional personnel, resumes/refs, years of directly-related experience, relevant certifications/security clearance)
- **2. General Requirements**
  - 2.1 Termination of...(VPN end-points/methods) [marked **M** = Mandatory]
  - 2.2 Billing options (direct to agent of record, monthly, etc.)
- (Sections 3+ continue but truncated in initial read — typical RFP technical-spec sheet has 200–300 line items)

### Sheet 2: `12-Cost Submission`

- Table 1 — Managed Remote Access Service: `Service Description | SLA | Per Instance Cost`
- Table 2 — Hourly Rate Service: `Service Description | Hourly Rate`
- Table 3 — Change Order Rates: `Item Description | Base Rate | Fully-Loaded Rate`

All three tables are **vendor-fillable templates with no values pre-populated** (similar to AMBAG cost proposal). Parser should produce `template_field:*` entities for each blank cell, not `quantity:0`.

### Sheet 3: `12-Scoring Methodology`

100-base + 5 value-add = 105 max points:

| Category | Possible Points | Type |
|---|---|---|
| Local State Account Team | 5 | Non-Cost |
| General Requirements & References | 20 | Non-Cost |
| Technical Specifications | 45 | Non-Cost |
| **Total Non-Cost Points** | **70** | |
| Lifecycle Cost | 26 | Cost |
| Hourly Rate | 2 | Cost |
| Change Order Rate | 2 | Cost |
| **Total Cost Points** | **30** | |
| **Total Base Points** | **100** | |
| Value Add | 5 | Bonus |
| **Maximum Possible Points** | **105** | |

## Expected parser routing

| Artifact | Parser | Confidence | Why |
|---|---|---|---|
| `ms_its_managed_vpn_RFP4080_attachA.xlsx` | `orbitbrief_xlsx` | ≥ 0.95 | `.xlsx` extension + Office Open XML magic bytes. Multi-sheet workbook. |

## Expected entity_keys (must include)

- `customer:state_of_mississippi_information_technology_services` (alias `customer:ms_its`)
- `rfp:rfp_4080`
- `category:category_xi_managed_vpn`
- **VPN service categories**:
  - `service:managed_vpn`
  - `service:site_to_site_vpn`
  - `service:traditional_workstation_remote_access_vpn`
  - `service:mobile_device_vpn`
- **Vendor evaluation framework**:
  - `requirement:vendor_response_a_acknowledges_will_comply`
  - `requirement:vendor_response_e_unable_to_meet_but_alt_available`
  - `requirement:vendor_response_x_not_capable`
  - `requirement:mandatory_requirement_m`
- **Cost categories**:
  - `pricing:per_instance_cost_for_managed_remote_access`
  - `pricing:hourly_rate`
  - `pricing:base_rate_change_order`
  - `pricing:fully_loaded_rate_change_order`
- **Scoring weights** (105-point scale):
  - `scoring:local_state_account_team_5pts`
  - `scoring:general_requirements_references_20pts`
  - `scoring:technical_specifications_45pts`
  - `scoring:lifecycle_cost_26pts`
  - `scoring:hourly_rate_2pts`
  - `scoring:change_order_rate_2pts`
  - `scoring:value_add_5pts`
- **Vendor qualification categories** (from §1):
  - `requirement:org_description_corporate_parent`
  - `requirement:state_of_incorporation`
  - `requirement:principal_office_location`
  - `requirement:servicing_office_location`
  - `requirement:disclosure_of_restructurings_mergers_acquisitions`
  - `requirement:annual_report_most_recent`
  - `requirement:lines_of_business_with_percentages`
  - `requirement:last_full_year_revenue_in_related_line`
  - `requirement:fte_count_in_related_line`
  - `requirement:years_in_business`
  - `requirement:state_local_government_experience`
  - `requirement:paid_customer_count_per_vpn_type`
  - `requirement:executive_professional_personnel_resumes_references`
  - `requirement:directly_related_years_per_resume`
  - `requirement:certifications_or_security_clearance_per_resume`
- **State approval rights**:
  - `requirement:state_reserves_approval_of_individuals_assigned`
  - `requirement:proposed_staff_must_agree_to_approval_process`

## Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `service:managed_vpn_with_3_vpn_types` | active | Site-to-site, remote workstation, mobile device — 3 distinct VPN types per §1.2.4. |
| `scope_inclusion` | `requirement:mandatory_termination_of_vpn_endpoints` | active | §2.1 marked **M** — non-negotiable. |
| `scope_inclusion` | `service:billing_options_to_agent_of_record_monthly` | active | §2.2 — billing flexibility scope. |
| `scope_inclusion` | `requirement:vendor_response_matrix_a_e_x_with_comments` | active | The XLSX is itself a structured response form. |
| `customer_override` | `pricing:scoring_70_30_non_cost_to_cost_split` | active | 70/30 weighting prioritizes technical capability over price. **Higher than typical 60/40 — strong customer signal.** |
| `customer_override` | `pricing:value_add_bonus_5pts_above_100` | active | Vendors can score >100% via Value-Add — incentivizes innovation. |
| `customer_override` | `pricing:lifecycle_cost_dominates_cost_score_26_of_30` | active | Lifecycle cost dwarfs hourly rate (26 vs 2 vs 2). Discourages low-bid-with-high-change-order strategy. |
| `scope_exclusion` | `requirement:non_compliant_with_mandatory_m_items` | active | "M" requirements are pass/fail; non-compliance disqualifies. |
| `missing_info` | `pricing:per_instance_cost_table_1` | active | Vendor-fillable; no defaults. |
| `missing_info` | `pricing:hourly_rate_table_2` | active | Vendor-fillable. |
| `missing_info` | `pricing:change_order_base_and_fully_loaded_table_3` | active | Vendor-fillable. |
| `missing_info` | `quantity:total_estimated_user_count` | active | Not specified in the visible portion of the spec sheet — depends on agency-level rollout. |
| `meeting_decision` | `decision:state_reserves_approval_of_individuals_assigned` | active | §1.3.4 — State has unilateral staff-approval rights. |
| `meeting_decision` | `decision:rfp_category_xi_separate_from_other_categories` | active | "Category XI" = MS ITS multi-category contract structure. |
| `action_item` | `vendor:resume_format_year_count_at_top_of_each_resume` | active | §1.3.3 — required resume formatting. |
| `action_item` | `vendor:annual_report_most_recent_required` | active | §1.1.6 — financial transparency. |
| `action_item` | `vendor:disclose_restructurings_mergers_acquisitions` | active | §1.1.5 — corporate change disclosure. |

**Expected packet count**: ≥ 14 for MS ITS

## Expected ontology gap candidates

The `networking_pack` should know VPN/IPsec/IKE/SSL/TLS/AnyConnect. But these are likely gaps:

- `managed_vpn_service` (vs. self-managed VPN)
- `site_to_site_vpn` (specific topology variant)
- `traditional_workstation_remote_access_vpn` (specific use case)
- `mobile_device_vpn` (specific use case)
- `category_xi` (MS ITS multi-category contract structure)
- `state_of_mississippi_information_technology_services` / `ms_its`
- `lifecycle_cost` (procurement-specific TCO terminology)
- `value_add_bonus_points`
- `agent_of_record_billing` (federal/state procurement billing model)
- `change_order_base_rate` vs `change_order_fully_loaded_rate` (rate-card distinction)
- `vendor_response_matrix_a_e_x` (Mississippi-specific format)

## Stress-test attributes

- **Standalone XLSX (no PDF)** — only XLSX file in the corpus that ISN'T paired with a parent RFP. The xlsx_parser must extract structure even without a PDF context.
- **3-sheet workbook** — typical RFP attachment shape. Each sheet has different schema and purpose. Parser should detect sheet boundaries and produce per-sheet structured projections.
- **Hierarchical Ref# numbering (1, 1.1, 1.1.1, 1.1.2, ...)** — multi-level outline. Parser should preserve the hierarchy in the structured projection (parent/child relationships).
- **A/E/X response key** — short codes that have rich semantics. Parser should expand A/E/X to full meanings.
- **Mandatory column "M"** — flagging requirements as pass/fail. Parser should produce `requirement_mandatory:true` boolean.
- **Empty cost cells across all 3 cost tables** — vendor-fillable template state. Parser must NOT generate `quantity:0` for empty cells.
- **Empty Vendor Response columns** — same template-state issue. Parser should produce `template_field:vendor_response_per_row` entities.
- **Scoring methodology in separate sheet** — distinct from technical specs. Parser should treat as procurement-evaluation metadata, not scope.
- **70/30 non-cost-to-cost split** — relatively high non-cost weighting. Worth flagging as `procurement_emphasis:technical_over_price`.
- **Value Add bonus** — 5 points above 100. Parser should detect bonus-point structure separate from base-100 scoring.

## Expected metrics (MS ITS only)

```
expected_min_atom_count: 60        # ~50 visible Ref# rows in Sheet 1 + ~10 in Sheets 2/3
expected_min_packet_count: 14
expected_min_distinct_sheets: 3    # all 3 sheets must produce sheet-level entities
expected_min_template_unsupported_atoms: 5  # blank cost cells + blank vendor response cells
expected_mandatory_m_atoms: 1+     # at least §2.1 is marked M
expected_scoring_atoms: 7          # 7 weighted categories
expected_min_constraint_atoms: 5   # 70/30 split, 105 max, 26/30 cost dominance, etc.
```

## Known difficulties & where the parser will likely fail

1. **Multi-sheet workbook** — many xlsx parsers concatenate all sheets into one flat output. Parser should preserve sheet boundaries as `sheet:*` entities.
2. **Hierarchical Ref# (1.1.1, 1.1.2, 1.2.4.1, etc.)** — parser should recognize this is an outline and produce parent→child relationships, not 100 flat rows.
3. **Empty cells in cost tables** — false-positive risk: `Per Instance Cost = ?` could become `quantity:0`. Parser must detect template state.
4. **Vendor Response column** — header is "A, E, X" but values are blank. Should produce `template_field:vendor_response` per row, NOT `vendor_response:none`.
5. **Mandatory "M" flag** — single-character flag in column 3. Parser must detect "M" as a boolean modifier on the requirement.
6. **Scoring methodology embedded as table** — Sheet 3 is structured as a points table with categories. Parser should produce per-category scoring atoms with weight values.
7. **Cross-sheet references** — "Category XI" appears in all 3 sheets and refers to the same procurement category. Parser should deduplicate.
8. **State-of-Mississippi-specific procurement language** — "State reserves the right to approve all individuals assigned" is a strong customer-override pattern that parsers may miss if not explicitly trained on.
