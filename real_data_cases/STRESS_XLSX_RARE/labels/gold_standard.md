# Gold standard — STRESS_XLSX_RARE

**Bundle**: 2 standalone XLSX attachments — the rare real-world spreadsheet artifacts in this corpus. Both serve as bonus stress tests for the parser's `xlsx_parser` (universal small-table extraction). Neither has a paired PDF in this case directory.

| File | Sheets | Customer | Shape |
|---|---|---|---|
| `calsaws_qa_log.xlsx` | 1 sheet, 486 rows × 8 cols | California Statewide Automated Welfare System (CalSAWS Consortium) | Q&A log for RFP #01-2022 (Maintenance & Operations). Each row is one vendor question + customer response with section reference, page number, dates. |
| `njeda_fee_schedule.xlsx` | 1 sheet, 30 rows × 10 cols | New Jersey Economic Development Authority (NJEDA) | Fee Schedule template for Investment & Cash Management Consultant. Tiered fee structure with breakpoints. |

**Service line**: cross-cutting (these are general-procurement structured documents, not service-line-specific)
**Recommended domain pack**: `default_pack` for both (no service-line specialization).

These are NOT typical "core service-line" RFPs — they're **structured procurement attachments**. The gold reference here is for **xlsx_parser correctness on real-world non-trivial sheets**:
- CalSAWS = vendor Q&A log (similar to VT-CAM's PDF Q&A but as XLSX)
- NJEDA = financial-product fee schedule with tiered breakpoints

## Per-artifact gold

### CalSAWS — `calsaws_qa_log.xlsx`

**Setting**: California Statewide Automated Welfare System (CalSAWS) Consortium. RFP #01-2022 for Maintenance & Operations services. The Q&A log records all vendor questions submitted during the procurement Q&A period (July 2022) with customer responses.

#### Sheet schema

- Row 1: Title — "CalSAWS M&O RFP Question and Answer Log"
- Row 2: Column headers — `ID | Section | Page Number | Question/Concern | Date Submitted | Response to Question/Concern | Date Response Posted` (7 columns; 8th may be blank)
- Rows 3+: 484 numbered Q&A entries

#### Expected entity_keys

- `customer:calsaws_consortium` (alias `customer:calsaws`, full: California Statewide Automated Welfare System)
- `rfp:rfp_01_2022_maintenance_and_operations`
- **Cited document sections** (from Section column — every distinct section is a `spec_section:*` entity):
  - `spec_section:1_3` (Eligible Contractors)
  - `spec_section:1_12` (Bidder's Conferences)
  - `spec_section:calsaws_aws_footprint` (existing AWS infrastructure)
  - `spec_section:4_2_procurement_scope_table_25` (IVR Support/Maintenance scope)
  - `spec_section:5_2_1_1_infrastructure_firm_mandatory_qualifications`
  - `spec_section:5_3_1_1_maintenance_and_enhancements_firm_mandatory_qualifications`
  - `spec_section:6_2_proposal_submission`
  - `spec_section:6_3_1_proposal_format` (font requirement)
  - (plus dozens more — every distinct value in column "Section" of all 484 rows)
- **Date entities**:
  - `date:question_period_2022_07_07_to_2022_07_22` (visible from sample rows)
  - `date:response_posted_2022_07_15` and `2022_07_27` (response batches)
- **Existing infrastructure entities (from CalSAWS AWS footprint Q&A)**:
  - `service:aws_ec2_workloads_calsaws_existing` (one Q asks about EC2 instance applications)
  - `software:welfare_management_application_existing`
- **Scope items modified by Q&A responses**:
  - `decision:section_1_3_revised_eligible_contractors_clarified`
  - `decision:section_1_12_revised_bidders_conferences_dual_format`
  - `decision:table_25_revised_ivr_support_maintenance_scope_clarified`
  - `decision:section_5_2_1_1_firm_experience_clarified_subcontractors`
  - `decision:section_5_3_1_1_firm_experience_clarified_subcontractors`
  - `decision:section_6_2_revised_electronic_copies_instruction_removed`
  - `decision:section_6_3_1_revised_9_point_font_acceptable_in_tables` (was 11pt Century Gothic)
- **Vendor activity** (484 rows = ~484 vendor questions submitted):
  - `vendor_activity:484_questions_submitted_jul_2022`
  - `vendor_activity:multiple_addenda_required_to_clarify_scope`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_current_authored` | Each Response cell (484 cells) | active | Every Response cell is customer-authored (Consortium-issued). Should produce ~484 customer_current_authored atoms. |
| `vendor_quote` | Each Question/Concern cell (484 cells) | active | Every Q is vendor-asked. Should produce ~484 vendor_quote atoms. |
| `customer_override` | `decision:section_1_3_revised_eligible_contractors` | active | Customer-issued revision via Q&A. |
| `customer_override` | `decision:section_6_3_1_font_revision_9pt_acceptable_in_tables` | active | Q&A-issued formatting change. |
| `customer_override` | `decision:section_6_2_electronic_copies_instruction_removed` | active | Q&A-issued instruction removal. |
| `customer_override` | `decision:table_25_ivr_scope_clarified` | active | Q&A-issued scope clarification (potentially scope-changing). |
| `meeting_decision` | `decision:bidders_conferences_dual_format_per_q_2` | active | Two conferences in different formats (per Q&A). |
| `missing_info` | `service:aws_ec2_workloads_specific_per_q_3` | needs_review | Vendor asked "what applications are you using your EC2 instances for"; Consortium responded with list — should generate workload entities. |
| `scope_inclusion` | `service:ivr_support_maintenance_per_table_25_revised` | active | Originally out-of-scope; revised to include IVR support/maintenance. |

**Expected packet count for CalSAWS**: ≥ ~14 high-level packets + 484 atoms (one per Q/A cell). This is the densest single-artifact atom output in the corpus.

#### Expected ontology gap candidates (CalSAWS)

- `calsaws_consortium` (California Statewide Automated Welfare System)
- `welfare_management_system_wms`
- `m_and_o_maintenance_and_operations`
- `eligible_contractors_section_1_3`
- `bidders_conferences_dual_format`
- `aws_ec2_workloads_existing_calsaws`
- `ivr_interactive_voice_response`
- `century_gothic_font_11_point_proposal_format` (very specific format constraint)

---

### NJEDA — `njeda_fee_schedule.xlsx`

**Setting**: New Jersey Economic Development Authority. RFP #2022-RFP-IPM-051 — Investment & Cash Management Consultant. The Fee Schedule template defines tiered pricing breakpoints by Assets Under Management (AUM).

#### Sheet schema

- Row 1: "NEW JERSEY ECONOMIC DEVELOPMENT AUTHORITY"
- Row 2: "RFP #2022-RFP-IPM-051 - Investment & Cash Management Consultant"
- Row 3: "FEE SCHEDULE"
- Row 4: "INSTRUCTIONS TO PROPOSERS"
- Row 5: "1. Proposers shall not alter this Fee Schedule and must provide..."
- Rows 6+: Section 1A-1C "Monthly Percentage Costs of Assets Under Management" with breakpoint structure
- (then Tasks/Services rate schedule with hourly rates by position)

#### Tiered Fee Structure (key entity)

| Section | Asset Class | Breakpoint Tier |
|---|---|---|
| 1A | Fixed Income | Up to $150 Million |
| 1A | Fixed Income | $150 - $250 Million |
| 1A | Fixed Income | $250 Million Plus |
| 1B | Retiree Benefit Trust | Up to $20 Million |
| 1B | Retiree Benefit Trust | $20 - $40 Million |
| 1B | Retiree Benefit Trust | $40 Million Plus |
| 1C | If Applicable: Indicate Minimum Monthly Fee | Fixed Income Portfolio |
| 1C | If Applicable: Indicate Minimum Monthly Fee | Retirement Benefit Trust Portfolio |

#### Hourly Rate Schedule (separate section)

- `position:senior_executive_manager`
- `position:mid_level_manager`
- `position:low_level_or_similar_title`
- `position:administrative_support_staff`

(All-inclusive hourly rates — vendor-fillable.)

#### Expected entity_keys

- `customer:new_jersey_economic_development_authority` (alias `customer:njeda`)
- `rfp:rfp_2022_rfp_ipm_051_investment_cash_management_consultant`
- **Asset classes**:
  - `service:fixed_income_management`
  - `service:retiree_benefit_trust_management`
- **Tiered fee structure breakpoints** (6 tiers — every breakpoint is a constraint):
  - `breakpoint:fixed_income_up_to_150m`
  - `breakpoint:fixed_income_150m_to_250m`
  - `breakpoint:fixed_income_250m_plus`
  - `breakpoint:retiree_benefit_trust_up_to_20m`
  - `breakpoint:retiree_benefit_trust_20m_to_40m`
  - `breakpoint:retiree_benefit_trust_40m_plus`
- **Optional minimum monthly fee tiers**:
  - `pricing:minimum_monthly_fee_fixed_income_portfolio_optional`
  - `pricing:minimum_monthly_fee_retirement_benefit_trust_optional`
- **Hourly rate positions** (4 tiers):
  - `position:senior_executive_manager_with_hourly_rate`
  - `position:mid_level_manager_with_hourly_rate`
  - `position:low_level_with_hourly_rate`
  - `position:administrative_support_staff_with_hourly_rate`
- **Pricing semantics**:
  - `pricing:monthly_percentage_costs_of_assets_under_management`
  - `pricing:all_inclusive_hourly_rates_for_change_orders`
  - `requirement:no_alteration_of_fee_schedule_template`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `pricing:tiered_fee_structure_with_6_breakpoints` | active | Customer-defined fee structure; vendors fill in percentages. |
| `customer_override` | `requirement:no_alteration_of_fee_schedule_template` | active | "Proposers shall not alter this Fee Schedule" — strict template-form rule. |
| `scope_inclusion` | `service:fixed_income_management_with_tiered_pricing` | active | Section 1A. |
| `scope_inclusion` | `service:retiree_benefit_trust_management_with_tiered_pricing` | active | Section 1B. |
| `scope_inclusion` | `pricing:hourly_rate_per_position_4_tiers` | active | All-inclusive hourly rate schedule. |
| `scope_inclusion` | `pricing:minimum_monthly_fee_optional_per_portfolio` | needs_review | Section 1C — vendor-discretionary. |
| `missing_info` | `pricing:percentage_values_per_breakpoint` | active | Vendor-fillable; no defaults. |
| `missing_info` | `pricing:hourly_rate_dollar_values_per_position` | active | Vendor-fillable. |
| `missing_info` | `pricing:minimum_monthly_fee_dollar_values` | active | Vendor-fillable; optional indication. |
| `meeting_decision` | `decision:rfp_2022_rfp_ipm_051_two_asset_classes` | active | Fixed Income + Retiree Benefit Trust. |

**Expected packet count for NJEDA**: ≥ 10

#### Expected ontology gap candidates (NJEDA)

- `assets_under_management_aum`
- `monthly_percentage_costs_of_aum`
- `tiered_breakpoint_pricing_structure`
- `fixed_income_portfolio_management`
- `retiree_benefit_trust`
- `minimum_monthly_fee_floor`
- `all_inclusive_hourly_rate`
- `change_order_hourly_rate_schedule`
- `state_economic_development_authority`
- `rfp_ipm_investment_program_management`

---

## Cross-artifact bundle expectations

### Expected cross-artifact edges

- **0 cross-customer edges** — these are 2 unrelated state agencies' RFPs (CA welfare consortium vs NJ economic development authority). Different service lines too (M&O of welfare system vs. investment management consulting).
- **Both are XLSX-only** — no PDF context. Tests `xlsx_parser` standalone correctness.
- **Both are vendor-fillable templates** — every cost cell is blank, awaiting vendor response. Parser must NOT hallucinate values.

### Expected aggregate metrics

```
expected_min_atom_count: 1100  # ~970 from CalSAWS Q&A pairs alone (484 × 2) + ~30 NJEDA + headers
expected_min_packet_count: 25
expected_min_distinct_customers: 2
expected_min_template_unsupported_atoms: 30  # blank cost cells in NJEDA + blank Q rows in CalSAWS
expected_xlsx_artifacts: 2
expected_pdf_artifacts: 0
expected_min_q_a_pairs: 484  # CalSAWS alone
expected_min_breakpoint_atoms: 8  # 6 tier breakpoints + 2 minimum fee tiers (NJEDA)
```

## Stress-test attributes

1. **CalSAWS = the densest single-artifact in the corpus** — 484 Q&A rows = ~970 atoms. Tests parser scaling.
2. **CalSAWS Q&A row structure** — each row has `ID | Section | Page | Question | Date Submitted | Response | Date Posted`. The parser should produce 7 atoms per row (or 1 atom with 7 attributes), preserving the schema.
3. **CalSAWS section references** — `1.3`, `1.12`, `4.2 Procurement Scope, Table 25`, `5.2.1.1`, etc. — the parser should detect these as references to a *parent* RFP document not present in the corpus. Generate `missing_info:parent_rfp_section_reference` for each.
4. **CalSAWS dates** — both submission dates (Jul 7, 8, 21, 22 2022) and response posted dates (Jul 15, 27 2022). Parser should produce date atoms with timezone PST/PDT context implied.
5. **CalSAWS section 6.3.1 font revision** — the response says "Section 6.3.1 has been revised to note acceptable use of 9-point [in tables]". Original was 11-point Century Gothic. Parser should detect this as a customer-override of formatting requirements.
6. **NJEDA tiered pricing** — 6 breakpoint tiers across 2 asset classes. Parser should produce `breakpoint:*` entities with dollar-amount thresholds (`$150M`, `$250M`, `$20M`, `$40M`).
7. **NJEDA "If Applicable: Indicate Minimum Monthly Fee"** — conditional fee floors. Parser should produce `optional:*` packets, not assume all vendors fill them in.
8. **NJEDA hourly rate by position level (4 tiers)** — Senior Executive / Mid-level / Low-level / Admin-support — typical position-rate matrix. Parser should preserve hierarchy.
9. **Both XLSX have title rows + instruction rows + data rows** — non-trivial structure. Parser should detect that rows 1–4 are headers/metadata, not data.
10. **CalSAWS Q&A may include attached files reference** — e.g., "fillable PDF" mentions. Parser should detect file-attachment references and flag as `missing_info:attached_file_not_in_corpus`.
11. **NJEDA "1A | Fixed Income | Up to $150 Million"** is a single row with multi-cell value. Parser should preserve the breakpoint structure (Section ID + Asset Class + Tier).
12. **Both files are real-world XLSX** — they have charset issues (CalSAWS extraction failed at row 8 due to `` which is a Microsoft private-use character — likely a checkmark symbol). Parser must handle these gracefully.

## Known difficulties & where the parser will likely fail

1. **CalSAWS row count (484 rows)** — many parsers cap atom output at lower limits. Parser should produce ~970 atoms (Q + A per row) without truncation.
2. **CalSAWS multi-line cells** — Q and A often contain multiple sentences with line breaks. Parser must preserve cell content as a single block per cell.
3. **CalSAWS "Section" column has heterogeneous formats** — `1.3`, `CalSAWS AWS footprint`, ` 6.2 Proposal Submission` (with leading space). Parser should normalize but preserve original.
4. **NJEDA's row structure with merged-looking cells** — visual cells in pdftotext-style extraction may collapse. Parser must read the actual openpyxl cell positions.
5. **NJEDA's optional-tier indication** — Section 1C says "If Applicable: Indicate Minimum Monthly Fee". Parser must detect that this is conditional, not required.
6. **Charset issue (``)** — character encoding edge case. Parser must use proper Unicode handling.
7. **Headers vs. data rows** — Row 1 is title, Row 2 is column header, Row 3+ is data. Parser must skip headers but preserve them as metadata atoms.
8. **No service-line classification** — these documents don't fit security_camera/access_control/wireless/etc. Parser should route to `default_pack` and not force-fit into a service-line pack.
9. **CalSAWS Q&A is in chronological response-batch order** — not by section. Parser should NOT reorder rows; preserve as-is.
10. **NJEDA "all-inclusive hourly rates"** vs. typical "fully-loaded" or "burdened" rates — terminology variation. Parser should treat as equivalent.
11. **"Investment & Cash Management Consultant"** as service line — this is a financial-advisory procurement, not infrastructure. Parser should route to `default_pack` (general procurement) not any technical pack.
