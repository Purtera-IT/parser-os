# Gold standard — STRESS_DOWNEY_CABLING

**Bundle**: Two-document Downey Unified School District CAT6 cabling bid (Bid 23/24-20). The bundle is the gold reference for **CSI MasterFormat 00/01-XX procurement-document boilerplate** and **addendum-clarifies-original-bid** lattice tier interactions.

| File | Pages | Customer | Shape |
|---|---|---|---|
| `Bid-23_24-20-Various-Site-CAT6-Cabling-for-IP-Phone-Project.pdf` | 100+ | Downey Unified School District | Full Project Manual: Division 00 (procurement) + Division 01 (general requirements). Issued Dec 13, 2023. |
| `Addendum-1-Various-Sites-Cabling-1.pdf` | ~50 | Downey Unified School District | Addendum #1 with 21 pre-bid RFIs answered + revised Division 01 + revised plans. Per-site drop counts updated. |

**Service line**: `copper_cabling`
**Recommended domain pack**: `copper_cabling_pack` (with `networking_pack` adjacency for IP phone integration)

The bundle stresses:
1. **CSI MasterFormat 00/01-XX procurement boilerplate** (50+ document section references like 00 01 01, 00 01 10, ..., 01 91 00)
2. **Project-Manual-as-bound-PDF** with embedded Bid Form, Bid Bond template, Designated Subcontractors List, etc.
3. **Addendum-supersedes-original** lattice rule (DIV 01 "Rev. 1 - Per Addendum 1" stamp)
4. **Customer-authored full BOM** in Section 01 11 00 — Superior Essex + Leviton, both with model numbers and Green color
5. **19-site cabling project** with after-school work hours (3 PM–11:30 PM)

## Per-artifact gold

### Main document — `Bid-23_24-20-Various-Site-CAT6-Cabling-for-IP-Phone-Project.pdf`

#### Expected entity_keys

- `customer:downey_unified_school_district` (alias `customer:dusd`)
- `address:11627_brookshire_ave_t_4_downey_ca_90241` (District Office Facilities Department)
- **19 sites** (full list — every row should produce a `site:*` entity):
  - `site:alameda_elementary_school` — 8613 Alameda St., Downey, CA 90242
  - `site:carpenter_elementary_school` — 9439 Foster Rd., Downey, CA 90242
  - `site:gallatin_elementary_school` — 9513 Brookshire Ave., Downey, CA 90240
  - `site:gauldin_elementary_school` — 9724 Spry St., Downey, CA 90242
  - `site:imperial_elementary_school` — 8133 Imperial Hwy., Downey, CA 90242
  - `site:lewis_elementary_school` — 13220 Bellflower Blvd., Downey, CA 90242
  - `site:old_river_elementary_school` — 11994 Old River School Rd., Downey, CA 90241
  - `site:pace_education_center` — 9625 Van Ruiten St., Bellflower, CA 90706 (note: Bellflower not Downey)
  - `site:price_elementary_school` — 9525 Tweedy Ln., Downey, CA 90240
  - `site:rio_hondo_elementary_school` — 7731 Muller St., Downey, CA 90241
  - `site:rio_san_gabriel_elementary_school` — 9338 Gotham St., Downey, CA 90241
  - `site:unsworth_elementary_school` — 9001 Lindsay St., Downey, CA 90240
  - `site:ward_elementary_school` — 8851 Adoree St., Downey, CA 90242
  - `site:williams_elementary_school` — 7530 Arnett St., Downey, CA 90241
  - `site:doty_middle_school` — 10301 Woodruff Ave., Downey, CA 90241
  - `site:sussman_middle_school` — 12500 Birchdale Ave., Downey, CA 90242
  - `site:columbus_high_school` — 12330 Woodruff Ave., Downey, CA 90241
  - `site:downey_high_school` — 11040 Brookshire Ave., Downey, CA 90241
  - `site:warren_high_school` — 8141 De Palma St., Downey, CA 90241
- **Devices (customer-authored BOM in Section 01 11 00)**:
  - `device:cat6_cable` → `vendor:Superior Essex`, model `66-240-5A` ("Category 6 Cable DataGain Category 6+ CMR, Green")
  - `device:cat6_quickport_connector` → `vendor:Leviton`, model `61110-RV6 eXtreme Cat 6 Quickport Connector` (Green)
- **Cabling rooms / locations within sites**:
  - `room:idf_room` (Intermediate Distribution Frame — origin for runs)
  - `room:classroom` (primary destination)
  - `room:custodial_closet` (added per Addendum 1)
  - `room:multi_purpose_cafeteria` (added per Addendum 1)
  - `room:staff_lounge` (added per Addendum 1)
- **Existing infrastructure (NOT new scope)**:
  - `device:cat3_existing_cable` (must remain operational — NOT demolished)
  - `device:cat3_jack_existing` (in same outlet box, behind new CAT6 jack)
  - `device:cat3_faceplate_existing` (remains)
- **Document sections** (50+ — Division 00 + Division 01, every section is an entity):
  - Division 00: 00 01 01 (Title Page), 00 01 10 (TOC), 00 01 15 (List of Drawings), 00 11 16 (Notice to Bidders), 00 21 13 (Instructions to Bidders), 00 31 19 (Existing Info), 00 41 13 (Documents Bidder Must Submit), 00 43 13 (Bid Form), 00 43 36 (Bid Bond), 00 43 40 (Designated Subcontractors List), 00 43 50 (Noncollusion Declaration), 00 45 00 (Notice of Award), 00 45 10 (Agreement), 00 45 40 (Certifications), 00 45 85 (Criminal Background/Fingerprinting), 00 54 70 (Storm Water Pollution Prevention Plan), 00 61 14 (Performance Bond), 00 61 15 (Payment Bond), 00 63 40 (Allowance Expenditure Directive), 00 63 57 (Proposed Change Order), 00 63 63 (Change Order), 00 70 00 (General Conditions), 00 71 00 (Special Conditions), 00 91 13 (Addenda)
  - Division 01: 01 11 00 (Summary of Work), 01 12 10 (Contract Forms), 01 20 00 (Price and Payment), 01 21 00 (Allowances), 01 23 00 (Alternates and Unit Pricing), 01 25 10 (Product Options and Substitutions), 01 26 00 (Contract Modification), 01 26 10 (Requests for Information), 01 31 00 (Coordination), 01 32 16 (Construction Schedule), 01 33 00 (Submittals), 01 40 00 (Quality Requirements), 01 42 13 (Abbreviations), 01 42 16 (General Definitions), 01 45 29 (Testing Laboratory), 01 50 00 (Temporary Facilities), 01 52 10 (Site Standards), 01 56 39 (Tree Protection), 01 57 10 (SWPPP), 01 60 00 (Materials and Equipment), 01 66 10 (Delivery/Storage), 01 73 00 (Execution), 01 73 10 (Cutting and Patching), 01 77 00 (Closeout), 01 78 23 (O&M Data), 01 78 36 (Warranties), 01 78 39 (Record Documents), 01 91 00 (Commissioning)
- **Compliance / regulatory**:
  - `requirement:c_7_contractor_license_active` (CA State — low voltage systems)
  - `requirement:bid_bond_10pct_or_cash_or_cashier_check`
  - `requirement:performance_bond_100pct`
  - `requirement:payment_bond_100pct`
  - `requirement:prevailing_wage_dir_labor_code_1770`
  - `requirement:contractor_registration_dir_labor_code_1725_5_1771_4`
  - `requirement:iran_contracting_act_certification`
  - `requirement:public_contract_code_3400_c_specified_brands`
  - `requirement:public_contract_code_5100_bid_error_claims`
  - `requirement:public_contract_code_22300_security_substitution`
  - `requirement:storm_water_pollution_prevention_plan_swppp_construction`
  - `requirement:criminal_background_investigation_fingerprinting`
- **Schedule**:
  - `date:contract_documents_available_dec_13_2023`
  - `date:mandatory_pre_bid_conference_dec_28_2023_at_9am_alameda_elementary`
  - `date:bid_due_jan_5_2024_at_10am`
  - `date:bid_validity_90_days_after_opening`
  - `constraint:work_hours_mon_fri_3pm_to_1130pm` (after-school)
  - `constraint:work_hours_saturday_7am_to_4pm`
  - `constraint:notice_to_proceed_within_3_months_of_award`
  - `constraint:notice_of_award_to_documents_submission_7_calendar_days`
  - `constraint:bid_protest_3rd_business_day`
  - `constraint:utility_shutdown_3_days_advance_notice`
- **Pricing**:
  - `pricing:lowest_responsive_responsible_bidder_base_bid_only`
  - `pricing:base_bid_includes_all_work_required`
  - `pricing:no_alternates_specified`
  - `pricing:0_5pct_designated_subcontractors_threshold`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `device:cat6_superior_essex_66_240_5a_green` | active | District Standard CAT6 cable. **Brand-specific BOD with Green color.** |
| `customer_override` | `device:cat6_quickport_leviton_61110_rv6_green` | active | District Standard connector with Green color. |
| `customer_override` | `pricing:lowest_responsive_responsible_bidder_base_bid_only` | active | Award method: lowest base bid. |
| `customer_override` | `decision:no_substitutions_without_owner_approval` | active | "No substitutions, deletions, changes, or additions of access point locations shall be permitted without written approval". |
| `customer_override` | `decision:c_7_license_required` | active | Single license type (CA C-7 low voltage). |
| `customer_override` | `decision:after_school_work_hours_3pm_1130pm_mon_fri` | active | All work outside school instructional hours. |
| `customer_override` | `pricing:bid_bond_10pct` | active | Bid security required. |
| `customer_override` | `pricing:performance_bond_100pct_payment_bond_100pct` | active | Both 100% bonds required from awarded bidder. |
| `scope_inclusion` | `service:cat6_install_idf_to_classroom_19_sites` | active | Section 01 11 00 — core scope. |
| `scope_inclusion` | `service:cat6_install_to_custodial_multipurpose_staff_lounge` | active | Per Addendum 1 — additional locations beyond classrooms. |
| `scope_inclusion` | `service:patch_panel_at_idf_terminate_new_drops` | active | Per Addendum 1 RFI 7 — contractor provides patch panel. |
| `scope_inclusion` | `service:cat6_jack_in_existing_outlet_box_behind_cat3` | active | Per Addendum 1 RFI 6, 14 — service loop ~10', new jack in same box as CAT3. |
| `scope_inclusion` | `service:reuse_existing_pathway` | active | Per Addendum 1 RFI 9 — all pathway is existing. |
| `scope_inclusion` | `service:lift_for_high_ceilings_cafeteria` | active | Per Addendum 1 RFI 15 — most elementary sites with cafeteria need lift. |
| `scope_exclusion` | `device:cat3_demolition` | active | Per Addendum 1 RFI 3 — existing CAT3 NOT demolished, remains operational. |
| `scope_exclusion` | `device:patch_cords` | active | Per Addendum 1 RFI 5 — District provides patch cords if/when needed. |
| `scope_exclusion` | `device:faceplate_or_surface_mount_box` | active | Per Addendum 1 RFI 14 — only new jack in existing box. |
| `scope_exclusion` | `device:osp_outside_plant_cable` | active | Per Addendum 1 RFI 21 — no OSP cables needed. |
| `scope_exclusion` | `room:sussman_middle_school_2_story_building_already_upgraded` | active | Per Addendum 1 RFI 17 — eliminated from scope. |
| `scope_exclusion` | `device:non_district_standard_cat6` | active | District Standard is specifically Superior Essex 66-240-5A. |
| `scope_exclusion` | `requirement:project_labor_agreement_pla` | active | Per Addendum 1 RFI 2 — no PLA. |
| `missing_info` | `quantity:total_drop_count_per_site` | active | Per Addendum 1 — **provided in revised plans, but not visible in this PDF text** (drawings indicated but not in text body). |
| `missing_info` | `device:underground_facility_subsurface_conditions` | active | Section 00 31 19 — limited reliance on subsurface info. |
| `missing_info` | `device:as_built_drawings_accuracy` | active | Section 00 31 19 — District does not warrant accuracy. |
| `meeting_decision` | `decision:mandatory_pre_bid_conference_jan_5_2024` | active | "Failure to attend or tardiness will render bid ineligible". |
| `meeting_decision` | `decision:multiple_sites_simultaneously_allowed` | active | Per Addendum 1 RFI 16 — Yes. |
| `meeting_decision` | `decision:patch_panel_provided_by_contractor` | active | Per Addendum 1 RFI 7. |
| `meeting_decision` | `decision:t_bar_ceiling_most_sites` | active | Per Addendum 1 RFI 20. |
| `action_item` | `vendor:bid_bond_or_cash_or_cashier_check_at_submission` | active | 10% of base bid. |
| `action_item` | `vendor:designated_subcontractors_list_at_submission` | active | Subs >0.5% of bid must be listed. |
| `action_item` | `vendor:fingerprinting_certification_at_submission` | active | Required form. |
| `action_item` | `vendor:iran_contracting_act_certification_at_submission` | active | CA-specific. |
| `action_item` | `vendor:noncollusion_declaration_at_submission` | active | Required form. |
| `action_item` | `vendor:agreement_documents_within_7_calendar_days_of_award` | active | Performance bond, payment bond, insurance certificates, certifications. |
| `site_access` | `site:after_school_only_during_instructional_year` | active | Schedule constraint applies to all 19 sites. |

**Expected packet count for Downey main**: ≥ 32

---

### Addendum 1 — `Addendum-1-Various-Sites-Cabling-1.pdf`

This is a separate but tightly-coupled artifact. Tests **addendum-supersedes-original** lattice with 21 numbered Q&A pairs (similar to VT_CAM addendum pattern).

#### Expected packets specific to Addendum 1

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `decision:rev_1_per_addendum_1_supersedes_original_div_01` | active | Every revised page stamped "Rev. 1 - Per Addendum 1". **The addendum's DIV 01 supersedes the original.** |
| `customer_override` | `scope:additional_locations_custodial_multipurpose_staff_lounge` | active | RFIs 1, 11, 12 — locations expanded beyond classrooms. |
| `customer_override` | `quantity:total_drops_per_site_per_revised_plans` | active | RFIs 13, 17, 18 — drop counts now in revised plans (gold expects this from drawings, not text). |
| `customer_override` | `scope:exclude_sussman_2_story_building_already_upgraded` | active | RFI 17 — explicit removal from scope. |
| `customer_override` | `decision:patch_panel_required_at_each_idf_provided_by_contractor` | active | RFI 7 — clarification of original scope. |
| `customer_override` | `pricing:insurance_4m_general_aggregate_6m_excess_umbrella` | active | RFI 10 confirms specific insurance limits. **Strong constraint pattern.** |
| `customer_override` | `decision:second_shift_work_only` | active | RFI 8 confirms 2nd shift = 3PM-11:30PM. |
| `customer_override` | `decision:1_cat6_per_drop` | active | RFI 4 — clarification. |
| `customer_override` | `decision:service_loop_10_feet_above_each_drop` | active | RFI 6 — install constraint. |
| `customer_override` | `decision:cat3_jack_remains_in_box_behind_new_cat6` | active | RFI 6 — physical install constraint. |
| `meeting_decision` | `decision:lift_required_most_elementary_sites_cafeteria` | active | RFI 15 — physical access constraint. |

**Expected packet count for Addendum 1**: ≥ 11

---

## Cross-artifact bundle expectations

### Expected cross-artifact edges

- **Lattice precedence**: Addendum 1's DIV 01 OVERRIDES the original DIV 01. All "Rev. 1 - Per Addendum 1" sections are `customer_current_authored`; superseded text is `quoted_old_email`-equivalent.
- **The 21 RFIs are answered by VT-CAM-style blue-text customer-authored A's** (though Downey's may not be color-coded). Each Q&A pair generates a `customer_current_authored` atom.
- **Cross-artifact `addendum_supersedes` edge** between the original Project Manual and Addendum 1 should be the primary graph relationship.
- **`vendor:superior_essex` and `vendor:leviton`** are named brand-specific vendors with model + color requirements. The packs should know these as approved vendors.
- **0 cross-artifact `quantity_conflict` edges expected** — the original main bid says "classrooms"; the addendum expands to "classrooms + custodial + multi-purpose + staff lounge". These are not contradictions; they're scope-expansions.

### Expected aggregate metrics

```
expected_min_atom_count: 280
expected_min_packet_count: 43
expected_min_distinct_sites: 19
expected_min_unique_vendors_referenced: 2  # Superior Essex, Leviton (brand-specific)
expected_min_constraint_atoms: 30
expected_min_compliance_atoms: 15
expected_min_csi_section_atoms: 50
expected_addendum_override_packets: 11
expected_min_unsupported_receipts: 8  # blank Bid Form fields, Bid Bond template, etc.
expected_no_quantity_conflict_edges_cross_artifact: true
```

## Stress-test attributes

1. **CSI MasterFormat 00/01-XX procurement boilerplate** — the entire main document is structured around CSI section numbers. The parser must recognize section boundaries (00 01 01, 00 01 10, ..., 01 91 00) and use them as outline anchors.
2. **Brand-specific BOD with color** — Superior Essex 66-240-5A *Green* + Leviton 61110-RV6 *Green*. Color is a meaningful constraint (jacket color = service identification). Parser should produce `device:cat6` atoms with `value.color = "green"`.
3. **Project Manual = bound PDF with embedded forms** — Bid Form, Bid Bond, Designated Subcontractors List, Noncollusion Declaration, Iran Contracting Act Certification, Performance Bond, Payment Bond, Conditional/Unconditional Waiver/Release forms — all blank templates. Atoms emitted from these blank forms should be `template_field:*`, not `quantity:0`.
4. **Addendum-supersedes-original** — Rev. 1 stamps on every revised page. Parser must detect the Rev. 1 marking and apply customer-current-authored precedence.
5. **21 numbered RFIs answered** — each Q (vendor-asked) and A (customer-authored) pair tests question/answer role separation. Distinct from VT-CAM's color-coded blue-text — Downey may or may not have color cues.
6. **19-site list with addresses** — every row should produce a `site:*` entity with full street address. Tests entity normalization.
7. **Pace Education Center** is in *Bellflower* CA, not Downey — tests address-vs-customer-name disambiguation. The site is operated by Downey Unified but located in another city.
8. **Existing CAT3 + new CAT6 in same outlet box** — overlapping infrastructure constraint. Parser should detect both as distinct devices, not collapse them.
9. **After-school work hours** (3 PM–11:30 PM) — operational constraint distinct from typical 8 AM–5 PM. Parser should produce `constraint:after_school_hours` packet, not generic "8-5 work hours".
10. **No drawings in text PDF** — drawings are referenced ("Addendum 1 updated DIV 01 and plans indicated added locations and total drops per site") but the actual drawings are *image attachments* in the PDF that pdftotext can't extract. Parser should produce `missing_info:drawing_extraction` for the drawing pages.
11. **Insurance limits in Addendum 1** — $4M general aggregate, $6M excess umbrella — strong constraint pattern. Easy to miss because it's confirmed via RFI 10, not in the main body.
12. **Sussman 2-story exclusion** — single-site exclusion via RFI 17. Parser should produce a `scope_exclusion` packet anchored to `room:sussman_middle_school_2_story_building`.
13. **No project drawings = "N/A" entry under DRAWINGS** in 00 01 15 — the original document explicitly states drawings are N/A for the main bid (drawings are added via Addendum 1).
14. **Iran Contracting Act + Public Contract Code 5100, 22300, 3400(c)** — multiple CA-specific procurement statutes referenced. Parser's compliance pack should know these.

## Known difficulties & where the parser will likely fail

1. **CSI MasterFormat section numbering at multiple levels** — `00 01 10`, `00 21 13`, `01 11 00`, `01 91 00` — parser must preserve the hierarchy (Division → Section → Subsection).
2. **The 19-site list is in a flat narrative** in section 1 of the Notice to Bidders — not a structured table. Parser should detect the numbered list pattern and produce 19 separate `site:*` atoms.
3. **Each address has format "Site Name, Street, City, State Zip"** — typical address parsing but with potential confusion (Downey vs. Bellflower for Pace).
4. **Addendum 1 is a separate file but logically one document** — parser must produce cross-artifact edges, NOT treat them as two unrelated bundles.
5. **The 50+ CSI sections are mostly boilerplate** ("CONTRACTOR SHALL COMPLY WITH ALL APPLICABLE PROVISIONS IN THE AGREEMENT, GENERAL CONDITIONS, AND SPECIAL CONDITIONS, IF USED") — parser should detect boilerplate-only sections and produce minimal atoms (don't generate scope packets from sections that just point to other sections).
6. **Form 700-style waiver/release templates** include placeholder fields ("Name of Claimant: __", "Through Date: __"). Parser should produce `template_field:*` for each blank, not `name:_underscore_`.
7. **Existing CAT3 / CAT3 jack / CAT3 faceplate** — the parser should NOT generate `quantity_conflict` between the existing CAT3 and the new CAT6 (they coexist in the same box).
8. **"District Standard" vs "or equal"** — Section 01 25 13 lists substitution rules that allow "or approved equal". The parser should detect that the BOM is *brand-specific basis-of-design* with substitution rules, not absolute exclusion.
9. **2nd shift work hours** — non-standard. Parser should detect "3:00 PM - 11:30 PM" as an after-school constraint, not a typo.
10. **Superior Essex 66-240-5A "Green" color** — color is part of the part number suffix in some catalogs. Parser may misparse "Green" as a separate atom rather than a color attribute of the cable.
