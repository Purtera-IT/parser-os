# Gold standard — STRESS_ACS_USC_PIEDMONT

**Bundle**: Two contrasting access-control RFP/spec artifacts that stress different shapes of the access_control vocabulary.

| File | Pages | Customer | Shape |
|---|---|---|---|
| `usc_access_control_master_spec.pdf` | ~120 | University of Southern California | **CSI MasterFormat narrative master spec.** No quantities, no prices. Site is parameterized as `[Indicate Site and Building]`. Spans 27 32 26 + 28 05 00 + 28 05 53 + 28 07 00 + 28 08 00 + 28 13 00 + 28 16 00 + 28 23 00. Existing systems: Lenel OnGuard (EACS) + Genetec VSS. |
| `piedmont_genetec_rfp.pdf` | ~50 | City of Piedmont, CA | **Tightly-scoped basis-of-design RFP** with full BOM. Genetec Synergis access control for City Hall + Fire Station + Police Department (~38 doors). Vendor was selected: Applied Video Solutions (AVS) at $248,782 + $52,126 maintenance/5yr. |

**Service line**: `access_control`
**Recommended domain pack**: `access_control_pack` (with `security_camera_pack` adjacency for the VSS sections of USC + Genetec entries in both)

This bundle is the gold reference for **narrative-only vs. quantified-BOM contrast**: USC has no quantities (parameterized), Piedmont has every door numbered (C1–C18 in City Hall, P1–P19 in PD).

## Per-artifact gold

### USC — `usc_access_control_master_spec.pdf`

**Setting**: USC GUIDELINE security specifications, revision 2024.03 (07-30-2024). This is a MASTER SPEC — it's a template referenced by USC project teams, NOT a single procurement.

#### Expected entity_keys

- `customer:university_of_southern_california` (alias `customer:usc`)
- `division:usc_lock_shop` (ultimate authority for access control deviations)
- `division:facility_management_services`
- `division:career_and_protective_services_caps`
- `division:department_of_public_safety_dps_police_dispatch_center`
- `division:uscard_services_department`
- `division:administrative_operations`
- **Existing systems** (carries through scope as integration constraint):
  - `software:lenel_onguard_pro_i_edition` (existing EACS — central system)
  - `software:genetec_vss` (existing video surveillance)
  - `device:hid_proximity_card_reader_existing` / `device:xceedid_card_reader_existing` / `device:aptiq_card_reader_existing` (must support all)
  - `software:uscard_credentialing_system_database`
- **CSI sections referenced**:
  - `spec_section:27_32_26` (Emergency Phone System)
  - `spec_section:28_05_00` (Security Systems General Requirements)
  - `spec_section:28_05_53` (Identification for Electronic Safety and Security)
  - `spec_section:28_07_00` (Security System Integration)
  - `spec_section:28_08_00` (Security Testing and Commissioning)
  - `spec_section:28_13_00` (Electronic Access Control System)
  - `spec_section:28_16_00` (Electronic Intrusion Detection System)
  - `spec_section:28_23_00` (Video Surveillance System)
- **Devices** (acceptable manufacturers):
  - `device:talk_a_phone_etp_mte_72_eco_tower` (Exterior Tower Phone)
  - `device:talk_a_phone_etp_mte_wp_arm` (CCTV+WAP mounting)
  - `device:talk_a_phone_etp_500c_single_button_faceplate` (Exterior)
  - `device:talk_a_phone_etp_wm_phone` (Exterior Wall Phone)
  - `device:talk_a_phone_etp_400c_phone` (Interior Wall Phone)
  - `device:axis_cctv_camera` (mounted on each emergency phone)
  - `device:blue_strobe_light_acrylic_housing`
  - `device:emergency_phone_chilean_red_color`
  - `device:lenel_card_reader` (HID/XceedID/AptiQ proximity)
- **Cabling**:
  - `cable:belden_5302ge_1_pair_twisted_shielded_18awg` (emergency phone)
  - `cable:belden_5500fe_1_pair_shielded_22awg` (alarm monitoring)
- **Codes/Standards** (applicable publications — gap candidates if not in pack):
  - `requirement:ul_294` (access control units)
  - `requirement:ul_1076` (proprietary alarm units)
  - `requirement:ulc` (UL Canada)
  - `requirement:nec_national_electric_code`
  - `requirement:nfpa_101_life_safety_code`
  - `requirement:ccr_title_24_california_building_code`
  - `requirement:ccr_title_24_california_electric_code`
  - `requirement:ada` (Americans with Disabilities Act)
  - `requirement:fcc_part_15` / `requirement:fcc_part_68`
  - `requirement:ieee_rs_170` (NTSC color camera broadcast)
  - `requirement:oshpd` (Office of State Health Planning Department)
  - `requirement:nema` `requirement:neca` `requirement:eia` `requirement:ntsc`
- **Owner-only restrictions**:
  - `requirement:no_proprietary_interface_modules` (security_integrator cannot use proprietary controllers)
  - `requirement:owner_sole_judge_substitutions` ("Or Equal" decisions belong to USC)
  - `requirement:contractor_subcontract_with_owner_eacs_service_provider`
  - `requirement:integration_with_lenel_onguard` (mandatory)
  - `requirement:integration_with_genetec_vss` (mandatory)
- **Warranty / service** (referenced through 28 05 00):
  - `requirement:1yr_warranty_minimum`
  - `requirement:as_built_documentation`
  - `requirement:training_per_28_05_00`
  - `requirement:burn_in_period`
- **Site placeholders** (template state):
  - `template_field:[indicate_site_and_building]` (appears 20+ times)
  - `template_field:[indicate_low_voltage_version_where_required]`
  - `template_field:[indicate_flush_or_surface_mounting]`
  - `template_field:[indicate_color]` (with default "Chilean Red")

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `service:eacs_expansion_lenel_onguard` | active | "Provide an Electronic Access Control System (EACS) expansion to the Lenel system" |
| `scope_inclusion` | `service:vss_expansion_genetec` | active | "Video Surveillance System (VSS) expansion of the Genetec system" |
| `scope_inclusion` | `service:eids_complete_per_contract_schedule` | active | EIDS — Electronic Intrusion Detection System (per 28 16 00) |
| `scope_inclusion` | `service:emergency_phone_system_27_32_26` | active | Section 27 32 26 covers exterior tower + wall phones, interior wall phones, all with Axis CCTV. |
| `scope_inclusion` | `service:integration_with_uscard_credentialing` | active | "Modifications to the existing credentialing system are not a part of this work" but new EACS must integrate. |
| `scope_inclusion` | `device:axis_cctv_camera_per_phone` | active | Every emergency phone gets an Axis camera. |
| `scope_inclusion` | `device:talk_a_phone_emergency_phones` | active | Multiple Talk-A-Phone models for exterior/interior. |
| `scope_inclusion` | `service:project_planning_through_warranty` (12 services) | active | Section 1.7 lists 12 services. |
| `scope_exclusion` | `service:owner_proprietary_interface_modules` | active | "Contractor may not use contractor proprietary interface modules" — explicit. |
| `scope_exclusion` | `software:credentialing_system_modifications` | active | "Modifications to the existing credentialing system are not a part of this work". |
| `scope_exclusion` | `service:owner_general_provisions_governs` | active | If conflict, General Terms and Conditions take precedence. |
| `customer_override` | `requirement:owner_sole_judge_substitutions` | active | Critical: USC has unilateral substitution authority. |
| `customer_override` | `requirement:lock_shop_authority_over_eacs_deviations` | active | "USC Lock shop's decision will precede what is acceptable for the Access Control Systems." |
| `customer_override` | `requirement:days_means_calendar_days_including_weekend` | active | Calendar-day definition shifts schedule semantics. |
| `customer_override` | `requirement:provide_means_furnish_install_connect_program_test_commission_warranty` | active | Definitional: a single "provide" line item bundles 7 actions. |
| `missing_info` | `template_field:site_and_building` | active | All 20+ instances are placeholder — every project that uses this spec must fill in. |
| `missing_info` | `device:exterior_tower_phone_low_voltage_or_120v` | active | "[Indicate low voltage version where required by project]" |
| `missing_info` | `device:wall_phone_flush_or_surface_mount` | active | "[Indicate flush or surface mounting]" |
| `missing_info` | `device:emergency_phone_color_default_chilean_red` | active | "[Color: Chilean Red]" — default-with-override-allowed |
| `meeting_decision` | `decision:lenel_remains_central_eacs` | active | Existing system continues; expansion-only. |
| `meeting_decision` | `decision:genetec_remains_central_vss` | active | Existing VSS continues; expansion-only. |
| `meeting_decision` | `decision:dps_dispatch_central_monitoring` | active | DPS Police Dispatch Center is the command-and-control point. |
| `action_item` | `vendor:integrator_pre_qualified_supplier_status` | active | Must be pre-qualified by USC purchasing, FMS, CAPS. |
| `action_item` | `vendor:california_state_contractor_license` | active | Required. |
| `action_item` | `vendor:manufacturer_3yr_3_installations` | active | Manufacturer's products must have been in satisfactory operation on at least 3 similar installations for not less than 3 years. |
| `action_item` | `vendor:visit_site_verify_existing_conditions` | active | Mandatory pre-bid site visit. |
| `action_item` | `vendor:autocad_format_shop_drawings` | active | Latest AutoCAD format required. |
| `site_access` | `requirement:credentialing_background_checks_for_contractors` | active | "Contractor's personnel...rules and regulations concerning Access Control at the University, including but not limited to those relating to credentialing, background checks". |

**Expected packet count**: ≥ 24 for USC

#### Expected ontology gap candidates (USC)

- `csi_masterformat_division_28` (Electronic Safety and Security)
- `eacs` (Electronic Access Control System — explicit acronym)
- `vss` (Video Surveillance System — explicit acronym)
- `eids` (Electronic Intrusion Detection System — explicit acronym)
- `lenel_onguard_pro_i_edition` (existing EACS family)
- `uscard` (USC's credentialing system name)
- `dps_police_dispatch_center` (USC's monitoring location)
- `lock_shop` (USC's authority for ACS deviations)
- `talk_a_phone_etp_*` (emergency phone family — manufacturer-specific gap)
- `eco_tower` (Talk-A-Phone product line)
- `chilean_red` (color reference)
- `proximity_multi_technology_card_reader`
- `uscard_credentials_aptiq_xceedid_hid` (multi-tech format support)
- `oshpd` (CA-specific code)
- `ccr_title_24` (CA building code)
- `ul_294` (access control unit standard)
- `ul_1076` (proprietary alarm units)
- `ieee_rs_170` (NTSC variable color standard)
- `nfpa_101` (life safety code)
- `auto_dialer_second_number_dial_on_first_no_answer`
- `request_to_exit_motion_sensor` (DS160 Bosch — though that's in Piedmont, USC also references PIR sensors)

---

### Piedmont — `piedmont_genetec_rfp.pdf`

**Setting**: City Council Agenda Report (March 3, 2025) recommending $248,782 contract award to Applied Video Solutions for City Hall, Fire Station, Police Department access control. **The full RFP + AVS proposal + signed contract are bundled in one PDF.** Distinct from USC: this is post-award; both bidders' scores are public.

#### Expected entity_keys

- `customer:city_of_piedmont` (CA, charter city, ~11,000 residents)
- `address:120_vista_avenue_piedmont_ca_94611` (City Hall)
- `division:public_works`
- `division:police_department`
- `division:fire_department`
- `division:dispatch_center` (under Police, currently in remodel — coordinated schedule)
- `division:kcom_tv` (gov/educational access)
- **Sites**:
  - `site:city_hall_fire_station` (combined building)
  - `site:police_department`
- **Specific rooms**:
  - `room:basement_mdf_1` (City Hall)
  - `room:new_it_server_room_103` (Police, part of Dispatch remodel)
  - `room:lower_level` (City Hall — 5 card readers)
  - `room:main_level` (City Hall — 12 card readers)
  - `room:upper_level` (City Hall — 1 card reader)
  - `room:dispatch_center` (Police — 7 card readers from remodel)
  - `room:holding_area_interior` (P12, P13)
  - `room:holding_area_exterior` (P14, P15)
- **Card reader IDs (numbered series)**:
  - `id:c1` through `id:c18` (City Hall, 18 readers)
  - `id:p1` through `id:p19` (Police Department, 19 readers)
- **Devices (full BOM)**:
  - `device:lifesafety_power_24_door_enclosure` → `vendor:LifeSafety Power (LSP)`, model `FPO250/250/250-5D8P3M8PNLXE12M`, qty 2 (1 City Hall + 1 PD)
  - `device:mp1502_mercury_intelligent_controller` → `vendor:Mercury/Genetec`, model `SY-MP1502`, qty 1 (City Hall)
  - `device:lp1502_mercury_intelligent_controller` → `vendor:Mercury/Genetec`, model `SY-LP1502`, qty 1 (PD)
  - `device:mr52_3s_mercury_dual_card_reader_interface_panel` → `vendor:Mercury/Genetec`, model `SY-MR52-S3`, qty 8+9 = 17
  - `device:power_supply_battery_enclosure` → model `IPROMC-CS012-E`, qty 2
  - `device:battery_shelf` → `vendor:LifeSafety Power`, model `BS1`, qty 2
  - `device:cloudlink` → `vendor:Mercury/Genetec`, model `SY-CLOUDLINK`, qty 2
  - `device:streamvault_sv_1040e_rs2_40t_20_434` → **Updated by AVS to 48T variant** (per AVS notes — NOT 40T, AVS chose 48T model). Qty 1, in Police IT Server Room 103.
  - `device:hid_access_card` → qty 200 total (100 City Hall + 100 PD)
  - `device:hid_20_card_reader` → qty 19 new (PD)
  - `device:bosch_pir_ds160_request_to_exit_motion_sensor` → `vendor:Bosch`, model `DS160`, qty 18+15 = 33 (NOTE: PD has 4 fewer because Holding Area readers C12/C13 and C14/C15 don't get PIR per spec)
  - `device:securitron_dps_m_gy_door_position_switch` → `vendor:Securitron`, model `DPS-M-GY`, qty 18+17 = 35
  - `device:spider_security_dst_1k2k_respk_end_of_line_resistors` → `vendor:Spider Security`, model `DST-1K2K-RESPK`, qty 36+34 = 70
  - `device:schlage_mortise_lock_hardware_kit` → `vendor:Schlage`, qty 2 (PD doors P14/P15 and P16)
  - `device:panic_bar_hardware` → manufacturer TBD, qty 5 (City Hall: 3 gates C1+C3+C16 + 2 exterior C12+C15)
  - `device:double_maglock_hardware` → qty 1 (City Hall main entrance C17 — custom wood door)
  - `device:fire_alarm_relay_facp_basement_mdf` → qty 1
  - `device:hubbell_2400_metallic_raceway` → model `HBL2400BCIV`
  - `device:hubbell_2410_entrance_end_fitting` → model `HBL2410CIV`
  - `device:hubbell_2448_standard_device_box` → model `HBL2448IVA`
- **Cables**:
  - `cable:genesis_32061112_22_6_stranded_shielded_plenum` (access control reader cable)
  - `cable:genesis_31141109_18_4_stranded_plenum` (strike/latch monitoring)
  - `cable:genesis_31021112_22_2_stranded_plenum` (door position control — daisy-chained)
- **Software/Licensing**:
  - `software:genetec_synergis_module_door_add_ons`
  - `software:genetec_administration_and_operations_training`
- **Pricing/contract entities**:
  - `pricing:installation_$248_782` (AVS award)
  - `pricing:5yr_maintenance_$52_126` (Alternate 1)
  - `pricing:1yr_maintenance_in_base` (coterminous with warranty)
  - `pricing:contingency_10pct_$24_878`
  - `pricing:total_construction_budget_$273_660`
- **Compliance**:
  - `requirement:cjis_criminal_justice_information_service_policy` (Police Department cyber security)
  - `requirement:high_encryption_door_access_control_components`
  - `requirement:prevailing_union_rates_for_all_locations`
  - `requirement:certified_payroll`
  - `requirement:executive_order_11246_equal_employment_opportunity` (amended by 11375)
  - `requirement:genetec_certified_installer_at_least_one_technician`
  - `requirement:fac004_facilities_capital_budget`
  - `requirement:fac006_facilities_capital_budget`
  - `requirement:cpi_san_francisco_bay_area_for_yearly_cost_increases` (5-year maintenance)
- **Vendor info**:
  - `vendor:applied_video_solutions_avs` (winning bidder, 20+ year company, Motorola/Avigilon Alta Elite Partner)
  - `vendor:eyep_solutions` (losing bidder, $604,136.57 — 2.4x higher)
  - `vendor:clientfirst_technology_consulting` (City's design consultant)
- **Aesthetic constraint**:
  - `constraint:historic_city_hall_aesthetic_preservation` (key evaluation criterion)
  - `constraint:no_substitution_without_owner_approval`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `device:lsp_24_door_enclosure_qty_2` | active | One per building (City Hall + PD). |
| `scope_inclusion` | `device:mr52_3s_card_reader_interface_panel_qty_17` | active | 8 City Hall + 9 PD. |
| `scope_inclusion` | `device:hid_access_card_qty_200` | active | 100 each. |
| `scope_inclusion` | `device:hid_20_card_readers_qty_19_new_pd` | active | 19 new readers in PD; 12 existing readers being removed. |
| `scope_inclusion` | `service:remove_12_existing_card_readers_at_10_doors` | active | "Provide removal of (12) twelve existing card readers at (10) ten doors". |
| `scope_inclusion` | `service:new_cabling_at_10_doors_pd` | active | New cable for PIR + DPS. |
| `scope_inclusion` | `service:fire_alarm_relay_to_facp_qty_1` | active | C17 main entrance ties into FACP. |
| `scope_inclusion` | `service:reuse_8_existing_door_handles_strikes_pd` | active | Hardware reuse where possible. |
| `scope_inclusion` | `service:up_to_8_hours_of_training_if_required` | active | "Provide up to 8 hours of training, if required" — capped scope. |
| `scope_inclusion` | `service:5yr_alternate_maintenance_$52_126` | active | Alternate 1 priced at $52,126; 5 years; CPI-capped. |
| `scope_inclusion` | `service:remote_monitoring_8x5_nbd_repair` | active | "8x5 / Next Business Day basis" remote monitoring + on-site repair. |
| `scope_inclusion` | `service:1yr_warranty_workmanship_materials` | active | Section 4.8. |
| `scope_inclusion` | `service:as_built_documentation` | active | Section 4.6.6. |
| `scope_inclusion` | `service:installation_certification_documentation` | active | Section 4.6.6. |
| `scope_exclusion` | `service:dispatch_center_door_hardware_p1_p7` | active | "Door hardware will be provided by the Dispatch Center Contractor for the new Dispatch Center doors (Card Readers P1 through P7)" — different contractor. |
| `scope_exclusion` | `service:network_switches_provided_by_city` | active | "City shall provide network switches and switch ports". |
| `scope_exclusion` | `service:ip_addresses_provided_by_city` | active | "City shall provide IP addresses for Contractor as needed". |
| `scope_exclusion` | `service:no_substitutions_or_changes_without_written_approval` | active | "No substitutions, deletions, changes, or additions of access point locations shall be permitted without written approval". |
| `customer_override` | `decision:streamvault_48t_instead_of_40t` | active | **AVS proposal explicitly notes: "Streamvault SV-1040E-RS2-40T-20-434 is an old part #. Genetec offers this model with 32T and 48T of storage, we went with the 48T model."** This is a vendor-suggested substitution accepted by City. Customer-override packet. |
| `customer_override` | `pricing:eyep_2_4x_higher` | active | Bid contradiction analyzed and resolved (City confirmed both pricing was correct). |
| `customer_override` | `decision:cjis_high_encryption_drives_design` | active | CJIS policy is the design driver for the PD side. |
| `missing_info` | `vendor:panic_bar_hardware_manufacturer_named_by_city` | active | "Provide and install 'panic bar' hardware supplied by manufacturer to be named by City" — vendor-name TBD. |
| `meeting_decision` | `decision:mandatory_pre_proposal_walkthrough_jan_21_2025` | active | Mandatory site visit prior to proposal. |
| `meeting_decision` | `decision:two_proposals_received_engineer_estimate_$250k` | active | Engineer's estimate; AVS came in at $248,782 (under estimate); EyeP at $604,137 (2.4x). |
| `meeting_decision` | `decision:awarded_avs_mar_3_2025` | active | Council approval scheduled for March 3, 2025. |
| `meeting_decision` | `decision:project_completion_may_2025` | active | Construction window March → May 2025. |
| `meeting_decision` | `decision:coordinated_with_dispatch_center_remodel` | active | Schedule alignment with Police Dispatch general contractor. |
| `action_item` | `vendor:genetec_synergis_certified_installer_required` | active | At least one certified tech on staff. |
| `action_item` | `vendor:certified_payroll_required` | active | Prevailing union rate compliance. |
| `action_item` | `vendor:cjis_compliant_components` | active | Required for PD subsystem. |
| `site_access` | `site:city_hall_historic_aesthetic_preserve` | active | Aesthetics is a graded evaluation criterion (35 points = same weight as price after pricing). |
| `site_access` | `site:occupied_city_hall_fire_pd` | active | Buildings remain operational during work. |

**Expected packet count**: ≥ 26 for Piedmont

#### Expected ontology gap candidates (Piedmont)

- `genetec_synergis` (Genetec access control product family)
- `streamvault_sv_1040e_rs2` (specific model with storage variants 32T/40T/48T)
- `lifesafety_power_lsp` (vendor)
- `cloudlink` (Mercury/Genetec product)
- `mr52_3s_mercury_dual_card_reader_interface_panel`
- `mp1502_mercury_intelligent_controller`
- `lp1502_mercury_intelligent_controller`
- `mortise_lock_hardware_kit_schlage`
- `panic_bar_hardware` (life-safety door hardware)
- `double_maglock_hardware` (specific door type)
- `fire_alarm_relay_facp_tie_in` (cross-system integration)
- `request_to_exit_motion_sensor` (DS160 Bosch)
- `door_position_switch_dps`
- `end_of_line_resistor_eol`
- `acx_4x8_fire_retardant_treated_plywood_backboard` (mounting)
- `nema_1_gutter_chase` (cable routing)
- `genesis_22_6_stranded_shielded_plenum_cable` (specific cable)
- `cjis_policy` (Criminal Justice Information Service — federal)
- `applied_video_solutions_avs` (vendor name)
- `motorola_avigilon_alta_elite_partner` (channel relationship)
- `clientfirst_technology_consulting` (consulting firm)
- `cpi_san_francisco_bay_area` (escalator index)
- `executive_order_11246_eeo_amended_by_11375`
- `holding_area_card_readers_no_pir` (specific exception to PIR-on-every-reader rule)

---

## Cross-artifact bundle expectations

### Expected cross-artifact edges

- **`vendor:genetec`** is the **shared vendor** across both — but applied differently:
  - USC: VSS (Genetec Video Surveillance) integrated with Lenel OnGuard EACS
  - Piedmont: Synergis (Genetec Access Control) — Genetec is the EACS basis-of-design itself
  - Graph builder should produce a `vendor_overlap` edge but with `application_difference` annotation: "Genetec VSS integrated with Lenel" (USC) vs "Genetec Synergis as primary EACS" (Piedmont).
- **0 cross-customer `quantity_conflict` edges** — different customers, different scopes.
- **Different access-control architectures**:
  - USC: Lenel OnGuard PRO-I (Lenel-primary, Genetec-secondary integration)
  - Piedmont: Genetec Synergis (Genetec-primary)
  - The parser should detect the architectural difference and not assume both customers use the same pack.
- **CSI MasterFormat 28-XX section anchors should appear in both** — Piedmont references "Section 281300 — ACCESS CONTROL SYSTEM" at end of RFP (page 38+); USC has full 28-XX series. Both should produce `spec_section:28_13_00` entities.
- **Both bundles reference Mercury/Genetec MR52-3S** — but USC doesn't specify quantity (template), Piedmont specifies qty 17. Tests parser's ability to handle quantity-known vs. quantity-template states.
- **`vendor:bosch`** appears in Piedmont (DS160 PIR motion sensors). USC doesn't name Bosch but references PIR sensors generically. Cross-artifact: Bosch is a likely match for USC's PIR requirement.

### Expected aggregate metrics

```
expected_min_atom_count: 250
expected_min_packet_count: 50
expected_min_distinct_customers: 2
expected_min_distinct_sites: 4   # USC site (parameterized) + Piedmont's 2 buildings
expected_min_unique_vendors_referenced: 14   # Genetec, Lenel, HID, XceedID, AptiQ, Talk-A-Phone, Axis, Bosch, Mercury, LifeSafety Power, Schlage, Securitron, Spider Security, Hubbell, Genesis, Belden, Streamvault
expected_min_constraint_atoms: 30
expected_min_compliance_atoms: 25  # USC has UL/NEC/NFPA/ADA stack; Piedmont has CJIS/EEO
expected_min_template_unsupported_receipts: 20  # USC parameterized fields
expected_min_substitution_packets: 1  # AVS's Streamvault 40T → 48T substitution accepted
expected_vendor_overlap_edges_via_genetec: 1+
expected_dual_vendor_naming: ">=1"  # "Mercury/Genetec" appears as compound vendor
```

## Stress-test attributes (cross-bundle)

1. **Master spec vs. project-specific RFP** — same service line, totally different documents. The parser should NOT generate scope_inclusion atoms with `quantity:?` from the USC master spec where the value is parameterized; it should generate `template_field:*` atoms instead.
2. **Lenel vs. Synergis as primary EACS** — same service line, different product families. The parser must not collapse Lenel and Synergis into a single anchor.
3. **Aesthetic preservation as evaluation criterion** — Piedmont scored aesthetics 35/100 points. The parser should detect "aesthetics" as a non-functional constraint distinct from technical specs.
4. **Bid tabulation included in Piedmont** — both bidders' scores are in the source PDF. The parser should detect this is post-award analysis (table on page 3) and not attempt to score the proposal.
5. **AVS's narrative agreement statements** — "AVS agrees with these requirements" appears 50+ times throughout the Piedmont PDF (it's the *winning vendor's compiled response*, not the standalone RFP). The parser should detect that the bundled PDF is a *combined RFP + Proposal + Contract* and not generate atoms from the agreement statements as scope.
6. **Mercury/Genetec compound vendor naming** — the slash in "Mercury/Genetec" requires special handling. Mercury is the underlying hardware, Genetec is the integrator/branding. Both should be recognized.
7. **Streamvault model substitution** — AVS proactively substituted 40T storage with 48T (because the original is EOL). This is a real vendor-suggested substitution that the City accepted. Generate a `vendor_suggested_substitution` packet that's `customer_override`-resolved.
8. **CJIS policy as primary design driver** — Piedmont's PD side requires high encryption per CJIS. CJIS is a federal law-enforcement standard the access_control_pack should know.
9. **38-door RFP with sub-numbering (C1–C18, P1–P19)** — every reader has a stable ID. The parser should produce one entity per reader ID, not collapse them.
10. **Holding Area exception to PIR rule** — "(The 4 x Holding Area card readers C12/C13 and C14/C15 will not receive a PIR)". This is a specific exception that breaks the otherwise-uniform pattern. The parser should detect the exception and produce a `scope_exclusion` for those 4 readers' PIRs without breaking the general PIR atoms.

## Known difficulties & where the parser will likely fail

1. **USC's parameterized fields** — every "[Indicate Site and Building]" is a template field, not a missing site name. The parser should NOT generate `site:[indicate_site_and_building]` as an entity; it should generate a `template_field:site_and_building` instead.
2. **USC's CSI MasterFormat structure** — multiple specifications stacked in one PDF. The parser must detect section boundaries (27 32 26, 28 05 00, etc.) and produce `spec_section:*` entities to organize atoms.
3. **Piedmont's combined RFP + AVS Proposal + Contract** — three documents in one PDF. The parser must detect the document-mode transitions (page 1 = council report, page 5 = contract, page 13 = RFP, page 19 = AVS narrative responses, page 38 = Section 281300 spec). Each section has different lattice tiers:
   - Council Report = `customer_current_authored`
   - Contract = `customer_current_authored` (executed)
   - RFP body = `customer_current_authored`
   - AVS narrative responses = `vendor_quote`
   - Section 281300 = `formal_rfp` (cited spec section)
4. **The "AVS agrees with these requirements" pattern** — 50+ instances. Each is a compliance ack from vendor; should NOT be treated as customer-authored.
5. **Mercury/Genetec compound vendor** — could parse as "Mercury" OR "Genetec" OR both. The pack should know that "Mercury" and "Genetec" are co-branded for ACS hardware.
6. **HID 20 vs HID generic** — "Provide and install (19) nineteen new HID 20 card readers" — "HID 20" is a specific reader product (HID Signo 20 ENT). Without that knowledge, the parser may treat "HID 20" as quantity 20 of HID, not as the model name.
7. **Streamvault SV-1040E-RS2-40T-20-434 → 48T variant** — the model number is parameterized by storage capacity (32T/40T/48T). The parser must recognize the family while preserving the specific variant.
8. **City Hall's "main entrance custom wood door"** — special hardware (double maglock with FACP integration) for one door. Easy to miss as a one-off.
9. **C17 vs P16** — both are "main entrance" doors, both get card readers, but in different buildings with different IDs. The parser must keep the IDs distinct.
10. **Piedmont's 4-criteria 100-point scoring** with weights 40/35/20/5 — should produce evaluation_criteria atoms preserving the weights.
