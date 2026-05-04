# Gold standard — STRESS_NET_MAINT (OCTA portion)

This case has 3 artifacts: Mobile + OCTA + MS ITS. **This document covers OCTA only**; Mobile has its own gold sheet (`gold_standard_mobile.md`); MS ITS has its own (`gold_standard_ms_its.md`).

## Artifact: `octa_security_systems_repair_RFP_4-2293.pdf` (~30+ pages, 3824 lines pdftotext)

**Orange County Transportation Authority (OCTA) — RFP 4-2293 Security Systems Repair and Maintenance**

- Issued: June 4, 2024
- Pre-Proposal Conference (hybrid in-person + teleconference): June 11, 2024 at 10:30 a.m.
- Question Submittal: June 14, 2024
- Proposal Submittal: July 2, 2024 at 2:00 p.m.
- Interview Date: July 24, 2024
- **Budget: $480,454 for a three-year initial term** (with two 1-year option terms — 5 yr max)
- Contact: Luis Martinez, Senior Contract Administrator (lmartinez1@octa.net)
- Submission via: http://www.octa.net/Proposal Upload Link (80 MB max)
- CAMM NET vendor portal: https://cammnet.octa.net

### Service line: `security_camera` + `access_control` (multi-pack: ACS + VMS + IP intercom maintenance)

This is a **maintenance/repair contract** for an existing multi-vendor security stack, with explicit transition from Lenel + Milestone to Genetec underway. Stresses the parser's ability to handle:
- Multi-vendor existing stack (Lenel ACS + Milestone VMS + IP intercoms)
- Transition-in-progress to Genetec (replacing Lenel + Milestone simultaneously)
- 5-tier business-impact SLA matrix (Critical / Urgent / Minimal / Normal)
- Multi-bus-base geographic distribution (Anaheim, Garden Grove, Irvine, Santa Ana, Orange HQ)

### Expected entity_keys (must include)

- `customer:orange_county_transportation_authority` (alias `customer:octa`)
- `division:security_and_emergency_preparedness` (SEP)
- `division:contracts_administration_and_materials_management` (CAMM)
- `address:550_south_main_street_orange_ca_92863_1584` (HQ)
- **Sites/Locations**:
  - `site:headquarters_orange` (550 South Main Street)
  - `site:bus_base_anaheim`
  - `site:bus_base_garden_grove` (also Mobile Surveillance Unit / Camera Trailer station)
  - `site:bus_base_irvine`
  - `site:bus_base_santa_ana`
  - `site:transportation_centers` (×4)
  - `site:park_and_ride_lots` (×2)
  - `site:mobile_surveillance_unit_camera_trailer` (deployable)
- **Existing ACS infrastructure**:
  - `software:lenel_version_7_6_382_271` (currently in use, 6 users, 6 concurrent licenses)
  - `device:microsoft_windows_server_2012_r2_virtual_server`
  - `device:windows_10_desktops`
  - `device:network_switches`
  - `device:access_control_panels`
  - `device:video_intercoms`
  - `device:duress_devices`
  - `device:card_readers`
  - `device:locksets`
  - `device:wireless_doors`
  - `device:ada_compliant_door_openers`
  - `software:sql_server_2015_shared_virtual_cluster_for_lenel_database`
  - `network:isolated_vlan_for_acs_traffic`
- **Existing VMS infrastructure**:
  - `software:milestone_xprotect_corporate_2018_r1` (currently in use, 5 users, 5 concurrent licenses)
  - `device:master_video_server` (qty 1)
  - `device:physical_video_storage_servers` (qty 7)
  - `device:windows_server_2012_r2` (on all 8 video servers)
  - `device:windows_10_desktops` (qty 25 for VMS)
  - `device:poe_network_switches`
  - `device:axis_ip_cameras` (mixed)
  - `device:pelco_ip_cameras` (mixed)
  - `network:isolated_vlan_for_vms_traffic`
- **Future systems (transition target)**:
  - `software:genetec_security_center` (replacement for Lenel + Milestone, project underway)
  - `software:nedap` (potentially adopted)
- **Vendor / certification entities** (firm requirements):
  - `vendor:lenel_var_authorized_reseller_required`
  - `vendor:lenel_elite_partner_status_required` (must maintain throughout contract)
  - `vendor:lenel_certified_associate` (technician requirement)
  - `vendor:lenel_certified_professional` (technician requirement)
  - `vendor:lenel_certified_expert` (technician requirement)
  - `vendor:milestone_var_authorized_reseller_required`
  - `vendor:milestone_elite_partner_status_required`
  - `vendor:milestone_certified_master_technician` (with 5+ yrs experience)
  - `vendor:genetec_authorized_reseller_required` (for transition)
  - `vendor:nedap_authorized_reseller_required`
- **IP intercom systems**:
  - `device:zenitel_ip_intercom`
  - `device:grandstream_ip_intercom`
  - `device:aiphone_ip_intercom` (typo in source: "Airphone")
- **Camera vendors** (existing fleet):
  - `vendor:axis_cameras`
  - `vendor:pelco_cameras`
  - `vendor:sony_cameras`
- **California licensing**:
  - `requirement:state_of_california_c_10_contractor_license` (electrical low voltage)
  - `requirement:state_of_california_c_7_contractor_license` (low voltage systems)
  - `requirement:50_miles_of_orange_california_office`
- **SLA Business Impact Levels** (4 tiers — major gap candidate):
  - `constraint:critical_business_impact_4hr_response_emergency_charge`
  - `constraint:urgent_business_impact_5_to_6hr_response_emergency_charge`
  - `constraint:minimal_business_impact_24_to_48hr_response`
  - `constraint:normal_business_impact_3_to_5_business_days_response`
- **Compliance/contract**:
  - `contract_type:time_and_expense_with_fully_burdened_labor_rates`
  - `contract_term:3yr_initial_plus_2_one_year_option_terms` (5 yr max)
  - `pricing:firm_fixed_for_special_projects_within_contract`
  - `pricing:t_and_m_minor_repairs`
  - `pricing:pre_approval_required_above_$2_000`
  - `pricing:other_direct_costs_odc`
  - `pricing:emergency_call_out_charge_outside_business_hours`
  - `requirement:campaign_contribution_disclosure_form_political_reform_act_84308` (Ti.2 CA Code Reg. 18438-18438.8)
  - `requirement:statement_of_economic_interests_form_700` (Political Reform Act, GC §81000+)
  - `requirement:ukraine_russia_economic_sanctions_certification` (EO 13660, 13661, 13662, 13685, 14065)
  - `requirement:california_public_records_act_government_code_7920_000`
  - `requirement:no_proposal_copyright`
  - `requirement:proposal_120_day_validity`
  - `requirement:50_pages_max_excluding_appendices_resumes_forms`
- **OCTA Business Hours**:
  - `constraint:normal_hours_mon_fri_8am_5pm_excluding_holidays`
  - `constraint:outside_hours_mon_fri_5_01pm_to_7_59am_or_holidays`
  - **Holidays**: New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas (with Sat→Fri / Sun→Mon shift rule)
- **Performance reporting**:
  - `requirement:monthly_invoicing_packet_first_friday`
  - `requirement:quarterly_financial_report_first_friday_of_quarter`
  - `requirement:account_manager_single_point_of_contact`
  - `requirement:dedicated_project_manager_for_complex_projects`
- **Non-compliance escalation** (3-tier):
  - `escalation:level_1_3_instances_in_12mo_internal_memo`
  - `escalation:level_2_4_instances_remediation_plan_in_writing`
  - `escalation:level_3_5_instances_formal_proceedings_to_mitigate`

### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `decision:transition_lenel_milestone_to_genetec` | active | "A project is underway to transition from Lenel and Milestone to Genetec." Active org-level decision. |
| `customer_override` | `pricing:budget_$480_454_3yr_initial` | active | Hard budget cap. |
| `customer_override` | `pricing:pre_approval_required_above_$2_000` | active | Vendor cost autonomy ceiling. |
| `customer_override` | `decision:single_award_anticipated_with_t_and_e_pricing` | active | Section K — anticipated contract type. |
| `customer_override` | `requirement:firm_must_maintain_lenel_elite_partner_status` | active | Must remain current throughout contract — strong vendor lock-in. |
| `scope_inclusion` | `service:repair_maintenance_install_acs_vms_intercom` | active | Core scope. |
| `scope_inclusion` | `service:annual_acs_preventative_maintenance` (12 sub-tasks) | needs_review | Optional — only if OCTA Project Manager requests. |
| `scope_inclusion` | `service:annual_vms_preventative_maintenance` (10 sub-tasks) | needs_review | Optional — only if OCTA Project Manager requests. |
| `scope_inclusion` | `service:quarterly_inspection_high_use_acs_portals` | needs_review | Optional. |
| `scope_inclusion` | `service:battery_replacement_every_3_years` | active | "Check ACS power supply back up batteries to ensure replacement every three (3) years". |
| `scope_inclusion` | `service:lenel_genetec_milestone_authorized_reseller_var_status` | active | All 4 vendor relationships required. |
| `scope_inclusion` | `service:account_manager_single_point_of_contact` | active | Section 3.a — required role. |
| `scope_inclusion` | `service:client_facing_dashboard_or_portal` | active | Section 2.a — required deliverable. |
| `scope_inclusion` | `service:24_7_365_contact_for_dashboard_outage` | active | Backup channel for service requests. |
| `scope_inclusion` | `service:duress_button_alarm_verification` | active | Section 5.g — duress protection systems. |
| `scope_exclusion` | `vendor:ukraine_russia_sanctioned_entities` | active | Hard exclusion per EO 13660-14065. |
| `scope_exclusion` | `service:joint_ventures` | active | "Authority intends to contract with a single firm and not with multiple firms doing business as a joint venture" — joint ventures excluded. |
| `scope_exclusion` | `service:offerors_advocating_for_competing_firms` | active | "Offerors hired to perform services for the Authority are prohibited from concurrently acting as an advocate for another firm". |
| `missing_info` | `pricing:vendor_specific_hourly_rates` | active | Vendor must propose fully-burdened labor rates in Exhibit B. |
| `missing_info` | `quantity:specific_camera_count_per_bus_base` | active | Attachments A, B, C describe components but quantities are aggregate. |
| `missing_info` | `decision:genetec_transition_timeline` | active | Transition is "underway" — no exact dates. |
| `meeting_decision` | `decision:hybrid_pre_proposal_conference_jun_11_2024` | active | In-person + teleconference. |
| `meeting_decision` | `decision:interview_date_jul_24_2024_keep_open` | active | "All prospective Offerors will be asked to keep this date available." |
| `meeting_decision` | `decision:bafo_round_possible` | active | "Best and Final Offer (BAFO)" may be requested. |
| `meeting_decision` | `decision:single_or_multiple_award_at_authority_discretion` | active | Section C — Authority's discretion. |
| `action_item` | `vendor:campaign_contribution_disclosure_form_at_submission_and_15_days_before_committee` | active | Ongoing reporting obligation through Board selection. |
| `action_item` | `vendor:status_of_past_present_contracts_form_5yr_lookback` | active | Litigation/claims history disclosure. |
| `action_item` | `vendor:proposal_exceptions_deviations_form_at_submission_only` | active | Cannot submit exceptions after due date. |
| `action_item` | `vendor:irs_w9_form_pre_work_commencement` | active | Required pre-Notice-to-Proceed. |
| `action_item` | `vendor:50_miles_office_within_orange_ca` | active | Geographic constraint. |
| `site_access` | `site:occupied_facilities_safety` | active | Bus bases are operational; service must minimize disruption. |

**Expected packet count**: ≥ 28 for OCTA

### Expected ontology gap candidates (OCTA)

- `lenel` (vendor — Honeywell-owned ACS) — should be in `access_control_pack`
- `milestone_xprotect_corporate` (Milestone's flagship VMS) — should be in `security_camera_pack`
- `genetec_security_center` (and "transition target") — should be in both packs
- `nedap` (ACS vendor — Dutch) — should be in `access_control_pack`
- `lenel_certified_associate` / `lenel_certified_professional` / `lenel_certified_expert` (3-tier certification ladder)
- `milestone_certified_master_technician` (manufacturer credential)
- `mcm_master_certified_milestone` (variant)
- `lenel_elite_partner_status` (channel partner tier)
- `milestone_elite_partner_status` (channel partner tier)
- `var_value_added_reseller`
- `iso_8802_3_ethernet` (referenced)
- `c_10_contractor_license_california` (electrical low voltage)
- `c_7_contractor_license_california` (low voltage systems)
- `aiphone` (typo in source: "Airphone" — IP intercom vendor)
- `zenitel` (IP intercom vendor)
- `grandstream` (IP intercom vendor)
- `business_impact_levels_critical_urgent_minimal_normal` (4-tier SLA)
- `escalation_level_1_2_3` (non-compliance ladder)
- `mobile_surveillance_unit` / `camera_trailer` (deployable surveillance asset — unusual)
- `pelco_camera` (camera vendor — Schneider Electric-owned)
- `sony_camera` (camera vendor)
- `axis_camera` (already in pack)
- `bus_base` (transportation-specific facility type)
- `transportation_center`
- `park_and_ride_lot`
- `camm_net` (procurement portal — OCTA-specific)
- `political_reform_act_84308` (CA campaign contribution rule)
- `form_700_statement_of_economic_interests` (CA-specific)
- `ukraine_russia_economic_sanctions_certification`
- `executive_order_13660_13661_13662_13685_14065` (Russia/Ukraine sanctions stack)
- `california_public_records_act_government_code_7920_000` (Prop 59 / California Public Records Act of 2023)
- `time_and_expense_t_and_e_with_fully_burdened_rates`
- `t_and_m_time_and_materials`

### Expected exclusion patterns

- "Authority intends to contract with a single firm and not with multiple firms doing business as a joint venture" — joint venture exclusion
- "Authority shall not, in any event, be liable for any pre-contractual expenses" — pre-contract expense exclusion
- "Authority will not be bound to any modifications to or deviations from the requirements set forth in this RFP as the result of oral instructions" — oral-instruction exclusion
- "Faxed or emailed RFPs will not be accepted" → procurement-mode exclusion (this RFP requires upload portal)
- "After the date and time specified above will be rejected" — late-submission exclusion
- "Subcontractors must adhere to the same requirements and restrictions" — sub-con scope must equal prime
- Ukraine/Russia sanctioned entity exclusion (EO 13660-14065)

### Expected constraint patterns

- "Within four (4) hours from the time reported" — Critical SLA
- "Five (5) to six (6) hours" — Urgent SLA
- "Twenty-four (24) to forty-eight (48) hours" — Minimal SLA
- "Three (3) to five (5) business days" — Normal SLA
- "$2,000.00" — pre-approval threshold for repairs
- "120 days" — proposal validity period
- "50 miles of Orange, California" — geographic office requirement
- "every three (3) years" — battery replacement cycle
- "five (5) years of enterprise-level experience" — minimum vendor experience
- "five (5) installations of said security systems within the last two (2) years" — alternative qualification
- "six (6) months of historical data" — ACS data retention minimum
- "twelve (12) months of historical data" — VMS data retention minimum
- "Within ninety (90) days of contract execution" — first preventative maintenance deadline
- "$480,454" — total 3-year budget

### Stress-test attributes

- **Currently using vs. transitioning to vs. may later adopt** — three temporal lattices in one document. Lenel + Milestone (current); Genetec (in transition); Nedap (potential future). Tests parser's temporal entity-state extraction.
- **Multi-vendor existing inventory** with specific version numbers (Lenel 7.6.382.271, Milestone XProtect Corporate 2018 R1) — version-specific atom extraction.
- **4-tier business impact SLA matrix** — Critical/Urgent/Minimal/Normal with hour-based response time tiers. Strong constraint extraction test.
- **3-tier non-compliance escalation ladder** — Level 1/2/3 with specific instance counts and durations. Should produce `escalation:*` packets.
- **Russia/Ukraine sanctions certification** — federal sanctions intersect with state procurement. Tests parser's ability to detect federal regulatory references in state procurement language.
- **Hybrid pre-proposal conference** — both in-person + teleconference. Distinct from purely-virtual or purely-physical meetings.
- **CAMM NET commodity codes (multi-category)** — vendor must register under specific commodity codes. Tests parser's ability to extract procurement-platform metadata.
- **Mobile Surveillance Unit / Camera Trailer** — a deployable surveillance asset (not a fixed installation). Unusual entity type.
- **Optional preventative maintenance** — 12 ACS sub-tasks + 10 VMS sub-tasks listed but only performed "if OCTA Project Manager requests". Optional scope packets should be `needs_review` not `active`.
- **Multiple required vendor certifications** stacked — Lenel (3-tier), Milestone Elite Partner + Master Technician with 5 yrs, Genetec, Nedap, Zenitel, Grandstream, Aiphone, Axis, Pelco, Sony. Highest concentration of vendor-credential constraints in the corpus.

### Expected metrics (OCTA only)

```
expected_min_atom_count: 200
expected_min_packet_count: 28
expected_min_distinct_sites: 8       # HQ + 4 bus bases + 4 transportation centers + 2 park&ride + mobile unit
expected_min_unique_vendors_referenced: 12  # Lenel, Milestone, Genetec, Nedap, Zenitel, Grandstream, Aiphone, Axis, Pelco, Sony, Microsoft, OCTA itself
expected_min_constraint_atoms: 25
expected_min_compliance_atoms: 15  # Form 700, sanctions, public records, political reform act
expected_business_impact_level_atoms: 4  # Critical, Urgent, Minimal, Normal
expected_escalation_level_atoms: 3  # Level 1, 2, 3
expected_optional_scope_packets: 22  # 12 ACS + 10 VMS preventative maintenance sub-tasks
expected_temporal_state_packets: 3  # current Lenel/Milestone, transitioning Genetec, future Nedap
```

### Known difficulties & where the parser will likely fail

1. **Vendor-state lattice** — Lenel (current, transitioning out), Milestone (current, transitioning out), Genetec (current install, transition target), Nedap (potential future). The parser must produce 4 distinct vendor entities with different temporal states, NOT collapse them.
2. **"Lenel and/or Genetec systems"** — appears repeatedly. Parser should recognize this as inclusive disjunction (both might be present), not exclusive choice.
3. **Multi-tier SLA business impact** — 4 levels with overlapping language. The parser should produce 4 `constraint:sla_*` atoms with hour-range values.
4. **"Section A" vs "Section A.5" nested numbering** — the document uses both "5. PREVENTATIVE MAINTENANCE" and "g. Duress Protection Systems" as different list types in the same document. Parser must preserve the hierarchy.
5. **Holiday-shifting rules** ("When a holiday falls on a Saturday, the previous day is observed; Sunday → next day, unless otherwise designated by OCTA's CEO.") — multi-conditional date-shift logic. Parser should detect the rule but not normalize specific dates.
6. **Service-level vs. maintenance-level scope distinction** — most parsers conflate "repair" with "preventative maintenance". OCTA explicitly separates them. Parser should produce different scope packets for each.
7. **3-tier non-compliance ladder** — easy to flatten to "non-compliance is bad". Parser should preserve the 3 tiers and the 12-month rolling window.
8. **EO 13660-14065 chain** — 5 executive orders cited. Parser should produce 5 separate compliance atoms, not 1 generic "sanctions certification".
9. **Lenel certifications are 3-tier; Milestone is single-tier** — different vendor cert ladders. Parser should preserve.
10. **Mobile Surveillance Unit / Camera Trailer** — deployable, not fixed. Parser should NOT emit it as a `site:*` entity; it's a `device:*` (trailer) that moves between `site:*` entities.
