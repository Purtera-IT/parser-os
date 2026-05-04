# Gold standard — STRESS_PAGING_TRIO

**Bundle**: 2 mass-notification / paging RFPs from higher-ed customers (Manchester was attempted but downloads 403'd, so it's not in artifacts).

| File | Pages | Customer | Shape |
|---|---|---|---|
| `umaine_mass_notification_RFP_24-18.pdf` | 68 | University of Maine System (7 universities, 28,997 students, 4,220 FTE staff, 18,664 active users) | RFP for higher-ed emergency mass notification SaaS, multi-institution master agreement |
| `sanjac_mass_comm_PR5.pdf` | 11 | San Jacinto College (TX) Board of Trustees | Purchase recommendation showing 4-vendor evaluation tabulation; recommends Rave Mobile Safety |

**Service line**: `paging` (mass notification / emergency communication)
**Recommended domain pack**: `paging_pack` (with `clery_act` + `it_security` adjacency)

The two artifacts together stress different shapes of paging procurement:
1. UMaine — pre-decision RFP (vendor unknown, scope-broad, multi-institution)
2. San Jacinto — post-decision board approval (vendor selected: Rave Mobile Safety; pricing comparison in source)

Manchester is a documented gap (file unavailable on the public web at fetch time) — see SOURCE_NOTES.md.

## Per-artifact gold

### UMaine — `umaine_mass_notification_RFP_24-18.pdf`

#### Expected entity_keys

- `customer:university_of_maine_system` (alias `customer:ums`)
- **7 universities** (each with FTE/enrollment/staff counts):
  - `site:university_of_maine_orono` (UM, flagship, 11,240 students, 2,192 FTE staff, 7,258 active users)
  - `site:university_of_maine_at_augusta` (UMA, 4,014 students, 335 FTE, 1,985 users)
  - `site:university_of_maine_at_farmington` (UMF, 2,080 students, 310 FTE, 1,726 users)
  - `site:university_of_maine_at_machias` (UMM, 701 students, 75 FTE, 241 users)
  - `site:university_of_southern_maine` (USM, 7,794 students, 1,034 FTE, 5,000 users)
  - `site:university_of_maine_at_fort_kent` (UMFK, 1,760 students, 123 FTE, 1,496 users)
  - `site:university_of_maine_of_presque_isle` (UMPI, 1,408 students, 151 FTE, 958 users)
- **Aggregates** (must be computed correctly, not hallucinated):
  - `quantity:total_students_28_997`
  - `quantity:total_fte_staff_4_220`
  - `quantity:total_active_users_18_664` (this is the user-license sizing)
- **Multi-campus structure**:
  - `region:multi_institution`
  - `division:university_services` (shared services org)
  - `division:office_of_strategic_procurement`
- **Service requirements**:
  - `service:mass_notification_solution_emns`
  - `service:maintenance_optional`
  - `service:technical_support_optional`
  - `service:training_optional`
  - `service:single_point_of_entry_for_account_management`
  - `service:multiple_campus_emergency_notifications`
  - `service:central_management_capability`
  - `service:campus_management_of_messaging`
  - `service:streamlined_simplified_procurement_per_campus`
  - `service:conversion_from_existing_separate_contracts`
  - `service:multi_year_renewable_offering`
- **Compliance**:
  - `requirement:maine_freedom_of_access_act_foaa_1_mrsa_401`
  - `requirement:trade_secret_must_be_marked_at_submission` (failure to mark = no exemption)
  - `requirement:no_provision_of_defense_or_indemnity` (UMS is a public entity)
  - `requirement:no_waiver_of_statutory_constitutional_immunity`
  - `requirement:maine_law_only_no_other_state`
  - `requirement:debarment_certification_3yr_lookback`
  - `requirement:non_collusion_certification`
  - `requirement:economic_impact_disclosure` (Maine-specific)
  - `requirement:vpat_voluntary_product_accessibility_template`
  - `requirement:it_security_pass_fail`
  - `requirement:accessibility_pass_fail`
- **Contract structure**:
  - `contract_term:initial_with_renewals` (Section 2 of Appendix E — actual term in master agreement)
  - `pricing:firm_for_term`
  - `pricing:no_bafo` (no best-and-final-offer round; vendors must give best-value at submission)
  - `pricing:per_fte_basis`
  - `pricing:tiered_per_campus_optional`
  - `pricing:enterprise_bundle_pricing_encouraged`
- **Key dates**:
  - `date:rfp_issued_jan_15_2018`
  - `date:questions_due_jan_23_2018_2pm_est`
  - `date:proposal_due_feb_8_2018`
  - `date:agreement_start_target_jul_1_2018`
- **Scoring**: 110-point scale
  - `scoring:cost_30pts`
  - `scoring:economic_impact_10pts` (5 recent + 5 projected)
  - `scoring:contract_for_services_10pts`
  - `scoring:org_quals_experience_references_10pts`
  - `scoring:business_matrix_presentation_20pts`
  - `scoring:general_implementation_training_support_10pts`
  - `scoring:accessibility_pass_fail`
  - `scoring:information_technology_security_pass_fail`
  - `scoring:information_technology_20pts`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `service:enterprise_emergency_mass_notification_per_fte` (18,664 active users) | active | 7-campus enterprise scope. |
| `scope_inclusion` | `service:single_point_of_entry_account_management` | active | "members (students, faculty and staff) with a single point of entry to manage their account". |
| `scope_inclusion` | `service:multi_year_renewable_design` | active | "renewable multiple-year offering". |
| `scope_inclusion` | `service:central_management_with_per_campus_messaging` | active | "central management" + "campus management of messaging" — both must coexist. |
| `scope_exclusion` | `requirement:no_indemnity_no_attorney_fees_no_unilateral_changes_no_auto_renewal` (10 enumerated) | active | Section 1.2.1.2: 10 enumerated terms UMS will NOT accept. **Strong customer-override pattern.** |
| `customer_override` | `requirement:section_1_2_1_2_governs_in_event_of_conflict` | active | "above Agreement provisions (Section 1.2.1.2) will govern the interpretation of such agreement notwithstanding the expression of any other term". |
| `customer_override` | `requirement:section_1_2_1_2_will_not_be_modified` | active | Hard-line lockout of vendor-proposed changes to those 10 enumerated terms. |
| `customer_override` | `pricing:no_bafo_round` | active | "The University will NOT seek a best and final offer (BAFO) from any Respondent". |
| `customer_override` | `pricing:firm_for_entire_term` | active | "All prices provided shall remain firm for the entire term of the agreement". |
| `customer_override` | `quantity:approximate_only_actual_needs_govern` | active | "The quantities shown on the cost response form are approximate only. The Contractor shall cover the actual needs of the University". |
| `missing_info` | `pricing:initial_term_and_renewal_periods_per_appendix_e` | active | Section 1.2.1.1 — actual term in Appendix E (referenced but not pre-specified in this RFP body). |
| `missing_info` | `vendor:final_award_decision` | active | RFP-only doc; vendor not yet selected. |
| `meeting_decision` | `decision:multi_institution_authorization_optional` | active | "may authorize other University Institutions to use the Agreement(s)". |
| `meeting_decision` | `decision:may_award_to_one_or_multiple_respondents` | active | "may include awards to Respondents for a geographical area". |
| `meeting_decision` | `decision:appendix_e_master_agreement_govern` | active | Appendix E is the Master Agreement form. |
| `action_item` | `vendor:debarment_certification_appendix_b` | active | Mandatory for response. |
| `action_item` | `vendor:economic_impact_form_appendix_d` | active | Maine-specific scoring component. |
| `action_item` | `vendor:vpat_for_accessibility_appendix_k` | active | Pass/fail. |
| `action_item` | `vendor:business_requirements_matrix_appendix_h_a` | active | Required scoring component. |
| `action_item` | `vendor:cost_excel_in_addition_to_pdf` | active | "An MS Excel Version must be included in your final submission for all of these tables." |
| `site_access` | `site:multi_campus_data_isolation` | needs_review | Multi-tenant SaaS: must separate per-campus messaging. |

**Expected packet count**: ≥ 18 for UMaine

#### Expected ontology gap candidates (UMaine)

- `mass_notification_solution` / `emns` (emergency mass notification system)
- `clery_act` (Jeanne Clery — implied for higher-ed but not explicit in UMaine doc)
- `cleert_act_higher_ed` (variant)
- `multi_institution_master_agreement` (procurement structure)
- `enterprise_solution_with_per_campus_management`
- `university_of_maine_system_ums`
- `freedom_of_access_act_foaa_maine` (1 MRSA §401 et seq.)
- `economic_impact_evaluation_state_of_maine` (procurement scoring)
- `vpat_voluntary_product_accessibility_template`
- `bafo_no_bafo` (Best and Final Offer policy)
- `pre_qualified_or_pre_approved_list_of_vendors` (procurement structure)
- `lap_lifecycle_costs` (TCO requirement)

---

### San Jacinto College — `sanjac_mass_comm_PR5.pdf`

This is a Board of Trustees Purchase Request, not the original RFP. It contains the post-evaluation tabulation and the recommendation to award.

#### Expected entity_keys

- `customer:san_jacinto_college` (alias `customer:sjcd`)
- `division:office_of_emergency_management_oem`
- `division:police_department`
- `division:information_technology_services`
- `division:marketing` (proposal evaluation team member)
- **Vendors evaluated** (4):
  - `vendor:rave_mobile_safety` (incumbent + recommended winner; final score 63.69, 7yr price $527,143)
  - `vendor:blackberry_corporation` (final score 59.84, $423,317; price includes additional costs not in scope)
  - `vendor:regroup_mass_notification` (final score 46.99, $398,740 — lowest price)
  - `vendor:alertus_technologies` (final score 41.18, $566,400; price includes additional costs not in scope)
- **Pricing entities** (extracted from tabulation):
  - `pricing:rave_3_year_initial_$226_227`
  - `pricing:rave_7_year_max_$527_143`
  - `pricing:blackberry_7_year_$423_317` (with asterisk: includes additional costs not in scope)
  - `pricing:regroup_7_year_$398_740` (lowest)
  - `pricing:alertus_7_year_$566_400` (with asterisk)
- **Compliance**:
  - `requirement:jeanne_clery_act_emergency_timely_notifications`
  - `requirement:texas_education_code_44_031_a` (competitive procurement)
- **Contract structure**:
  - `contract_term:3yr_initial_plus_4_one_year_options_max_7yr`
  - `pricing:funded_oem_2022_2023_operating_budget`
  - `pricing:no_implementation_cost_with_incumbent`
- **Scoring breakdown** (from tabulation):
  - `scoring:qualifications_30pts`
  - `scoring:presentations_40pts`
  - `scoring:pricing_30pts`
  - `scoring:total_100pts`
- **Resource personnel** (named):
  - `person:ali_shah_281_998_6311`
  - `person:karen_allen_281_998_6106`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `decision:award_rave_mobile_safety_$527_143` | active | Recommended for award. |
| `customer_override` | `pricing:incumbent_no_implementation_cost` | active | "Since Rave is the incumbent providing these services, there is not an implementation cost included with this new contract." |
| `meeting_decision` | `decision:contract_term_3yr_plus_4_one_year_options_max_7yr` | active | Standard SJCD procurement. |
| `meeting_decision` | `decision:rfp_23_25_issued_mar_9_2023_4_responses_received` | active | 4 vendor responses evaluated. |
| `meeting_decision` | `decision:cross_functional_evaluation_team` | active | Marketing + OEM + Police + IT representatives. |
| `meeting_decision` | `decision:initial_term_starts_jul_1_2023` | active | Initial 3-year award commencement. |
| `scope_inclusion` | `service:mass_communication_emergency_notification` | active | Core scope. |
| `scope_inclusion` | `service:multi_modal_voice_text_email_digital_signage_computer_screens_social_media` | active | "voice, text, e-mail, digital signage, computer screens, and social media". |
| `scope_inclusion` | `requirement:jeanne_clery_act_compliance` | active | Hard regulatory driver. |
| `scope_inclusion` | `service:safety_security_threat_notifications_all_locations_all_audiences` | active | Students, employees, visitors, contractors at all College locations. |
| `missing_info` | `pricing:line_item_breakdown_for_blackberry_alertus_excluded_costs` | active | Both have asterisks indicating "additional costs to the College not included in scope". |
| `action_item` | `vendor:rave_continues_service_no_data_migration_needed` | active | Incumbency benefits flagged. |

**Expected packet count**: ≥ 12 for San Jacinto

#### Expected ontology gap candidates (San Jacinto)

- `jeanne_clery_act` (Higher Ed safety act — federally mandated)
- `office_of_emergency_management_oem`
- `texas_education_code_44_031_a` (competitive procurement)
- `mass_communication_emergency_notification`
- `rave_mobile_safety` (vendor)
- `blackberry_at_hoc` (BlackBerry's mass notification product, possibly)
- `regroup_mass_notification` (vendor)
- `alertus_technologies` (vendor)
- `multi_modal_notification_voice_text_email_digital_signage` (delivery channels)

---

## Manchester (missing artifact)

The `STRESS_PAGING_TRIO/SOURCE_NOTES.md` documents the bundle was intended to include a third paging RFP from Manchester, but the public download was unavailable at fetch time (403/404). For completeness, expected attributes if Manchester were available:

- Likely a New Hampshire / municipal mass-notification RFP
- Smaller scope than UMaine, larger than a single-school SJCD
- Probable mention of FCC WEA (Wireless Emergency Alerts) integration
- Probable mention of NIMS / ICS interoperability

This third artifact is a known coverage gap — the parser should not generate atoms for non-existent files, and the case_id-level expected metrics below assume only the 2 present artifacts.

---

## Cross-artifact bundle expectations

### Expected cross-artifact edges

- **`vendor:rave_mobile_safety`** appears in San Jacinto only (named winner). UMaine doesn't name vendors but Rave is one of the most likely respondents. The graph builder should NOT auto-link Rave to UMaine without evidence; this is a single-customer reference.
- **0 cross-customer `quantity_conflict` edges** — different customers, different scopes, different user counts (UMaine 18,664 vs SJCD population not given).
- **Service-line consistency**: both should resolve to `paging_pack`. The `paging_pack` should know mass notification platforms.
- **Compliance overlap**: both reference Jeanne Clery Act (UMaine implicitly via higher-ed obligations; SJCD explicitly). The pack should treat Clery as a higher-ed-specific compliance entity.

### Expected aggregate metrics

```
expected_min_atom_count: 200
expected_min_packet_count: 30
expected_min_distinct_customers: 2  # UMS, SJCD
expected_min_distinct_sites: 8       # UMaine 7 campuses + SJCD 1
expected_min_unique_vendors_referenced: 4  # Rave, BlackBerry, Regroup, Alertus (all from SJCD tabulation)
expected_min_constraint_atoms: 20
expected_min_compliance_atoms: 12  # FOAA, Clery, Title VI, debarment, etc.
expected_pricing_comparison_packets: 1  # SJCD's tabulation generates a comparison packet
expected_min_unsupported_receipts: 5  # blank cost-form fields in UMaine
```

## Stress-test attributes (cross-bundle)

1. **Pre-decision vs. post-decision artifacts** — UMaine is a "vendors please bid" RFP; SJCD is a "we already chose" board memo with full tabulation. Different lattice tier (UMaine: customer-asked; SJCD: customer-current-authored decision).
2. **Per-FTE pricing in UMaine** — 18,664 active users × per-FTE-cost = annual licensing. Tests parser's ability to recognize FTE-based pricing model.
3. **4-vendor pricing tabulation in SJCD** — same line item, 4 different prices. Parser should extract all 4 and produce a `pricing_comparison` block, NOT a quantity_conflict (different vendors, not different sources for the same vendor).
4. **Lowest-price did NOT win in SJCD** — Regroup at $398,740 was lowest but scored 46.99/100; Rave at $527,143 won with 63.69/100. The parser should detect that "lowest price" is NOT the determining factor and flag this as a `selection_pattern:best_value_not_lowest_cost`.
5. **Maine FOAA / trade-secret exemption** — UMaine has a strict "mark at submission or lose exemption" rule. This is a procurement compliance pattern that's stronger than typical "trade secret" language.
6. **Section 1.2.1.2 enumerated 10 prohibited terms** — UMaine has 10 specific terms it WILL NOT accept. Parser should produce a `customer_override` packet with `requirement:no_indemnity_section_1_2_1_2` containing all 10 enumerated items as sub-atoms.
7. **Multi-institutional shared services** — UMaine has "University Services" as a shared org; the procurement is for a master agreement that other UMS-affiliated institutions can opt-into. Tests multi-customer-from-one-RFP semantics.
8. **Asterisks on pricing in SJCD tabulation** — "Price includes additional costs to the College not included in scope" — note about scope-creep adjustment. Parser should detect the asterisk semantics and flag those prices as `pricing_with_caveat`.
9. **Tiered pricing optional in UMaine** — Table 1.2 is for tiered pricing if a vendor offers it; otherwise blank. Parser should NOT generate atoms for blank tiered table.
10. **No-BAFO rule in UMaine** — vendors must give their best price at submission; no negotiation round. Parser should detect "no BAFO" as a procurement-mode constraint.

## Known difficulties & where the parser will likely fail

1. **UMaine campus FTE arithmetic** — student + staff + active-user counts are 3 different aggregations. The parser should NOT sum all three (28,997 students + 4,220 staff = 33,217 ≠ 18,664 active users). Each aggregate is a different population.
2. **UMaine cost form is empty** — Exhibit 1 Tables 1, 2, 3, 4 are blank in the RFP (vendor-fills-in). Parser should produce `template_field:*` entities for each blank cell, NOT `quantity:0` atoms.
3. **SJCD pricing-by-asterisk** — "* Price includes additional costs to the College not included in scope" — easy to miss if the asterisk reference isn't connected to the relevant vendors (BlackBerry and Alertus, both with `*`).
4. **SJCD's Rave incumbent advantage** — "no implementation cost included with this new contract" is a *because-incumbent* clause. Parser should flag the relationship between vendor incumbency and pricing structure.
5. **Manchester missing** — the bundle is documented as a "trio" but only 2 artifacts present. Parser should detect the case-level mismatch and not hallucinate Manchester atoms.
6. **Different vendor scoring methods**:
   - UMaine: 110-point with cost as ratio formula + economic impact + IT pass/fail
   - SJCD: 100-point with separate qualifications/presentation/pricing
   The parser should treat the scoring schema as a per-customer entity, not assume one universal model.
