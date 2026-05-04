# Gold standard — STRESS_BMS_SPECS

**Bundle**: 2 BMS (Building Management System) construction standards from major US universities. Both are master specs that govern any new BMS install on the campus. Macquarie (originally intended) was unavailable at fetch time.

| File | Pages | Customer | Shape |
|---|---|---|---|
| `wayne_state_bas_construction_standards_2025.pdf` | ~50+ | Wayne State University | Division 25 Integrated Automation construction standard, rev 1.0 (9/26/2024). Tridium Niagara 4 / JACE / Vykon JACE 9000 explicit. WSU-specific controller hierarchy: BAS Manager → NAC → SNC → PEC → AUC. |
| `uh_division_25_master_spec.pdf` | ~80+ | University of Houston | Master Construction Specification — Section 25 50 00 Integrated Automation Facility Controls (BACnet). Was previously SECTION 23 06 06 BACNET BAS DDC GUIDE SPECIFICATION (renumbered to 25 50 00). Tridium JACE explicit (3E/6E/7), Niagara 4 Framework. B-OWS/NAC/EAC/UC controller hierarchy. |

**Service line**: `bms`
**Recommended domain pack**: `bms_pack` (with `electrical_pack` adjacency for power/UPS, and `networking_pack` adjacency for BACnet/IP)

This bundle is the gold reference for **CSI MasterFormat Division 25 + Tridium Niagara/JACE vocabulary**. Both customers use Tridium Niagara 4 + JACE hardware as the basis-of-design — but their controller hierarchy naming differs (UH uses B-OWS/NAC/EAC/UC; WSU uses BAS/NAC/SNC/PEC/AUC). The parser should resolve both to a unified `bms_controller_hierarchy` ontology.

## Per-artifact gold

### Wayne State University — `wayne_state_bas_construction_standards_2025.pdf`

#### Expected entity_keys

- `customer:wayne_state_university` (alias `customer:wsu`)
- `division:bas_manager` (decision-making authority)
- **Controller hierarchy** (WSU-specific naming):
  - `device:fcms_facility_management_control_system` (top-level umbrella)
  - `device:bas_building_automation_system`
  - `device:bms_building_management_system` (synonym for BAS in WSU usage)
  - `device:nac_network_area_controller` (NAC)
  - `device:snc_system_network_controller` (SNC — distinct from NAC)
  - `device:pec_programmable_equipment_controller`
  - `device:auc_advanced_unitary_controller`
  - `device:vykon_jace_9000` (specific JACE model — basis of design)
  - `device:variable_air_volume_controller`
- **Software**:
  - `software:niagara_4` (preferred)
  - `software:niagara_5` (referenced)
  - `software:vykon_dashboard`
- **Protocols**:
  - `protocol:bacnet_native`
  - `protocol:bacnet_mstp` (Master-Slave/Token-Passing)
  - `protocol:rs_232`
  - `protocol:rs_485`
  - `protocol:modbus_rtu`
- **CSI sections**:
  - `spec_section:25_01_00` (Operation and Maintenance of Integrated Automation)
  - `spec_section:25_01_01` (Quality Assurance)
  - `spec_section:25_01_02` (Installer and Integrator Scopes — distinguishes installer from integrator)
  - `spec_section:25_05_00` (Common Work Results for Integrated Automation)
  - `spec_section:25_05_02` (Documentation Requirements)
  - `spec_section:25_05_10` (Panel Requirements: JACE/Demark/Custom Control Panels)
  - `spec_section:25_05_20` (Conduit and Cable Installation Standards)
  - `spec_section:25_13_16_02` (RS-232/485 Communication)
  - `spec_section:25_50_02` (Basis of Design)
  - `spec_section:25_50_02_01` (Vykon JACE 9000)
  - `spec_section:25_50_02_02` (Niagara 4 and Niagara 5 Software Licenses)
  - `spec_section:25_50_03_02` (ANSI/ASHRAE and BACnet Compliance)
  - `spec_section:25_50_03_03` (BACnet and RS-485 Communication Protocols)
  - `spec_section:25_50_40` (Supervisory Controllers)
  - `spec_section:25_50_40_01` (NAC Requirements)
  - `spec_section:25_50_40_02` (SNC Requirements)
  - `spec_section:25_50_40_03` (JACE Structure)
  - `spec_section:25_50_50` (Field-level Controllers — duplicate of 25_51_10)
  - `spec_section:25_50_50_01` (PEC Requirements)
  - `spec_section:25_50_50_02` (AUC Requirements)
  - `spec_section:25_50_50_03` (VAV Controller Requirements)
  - `spec_section:25_50_60_10` (Data Upload to WSU BAS Database)
  - `spec_section:25_50_60_30` (Point Tagging and Conventions)
- **Roles**:
  - `role:bas_manager` (final decision authority)
  - `role:installing_contractor` (Section 25 01 02.10)
  - `role:integrator` (Section 25 01 02.20 — distinct from installer; usually different vendor)
- **Document control**:
  - `metadata:revision_1_0_2024_09_26`
  - `metadata:author_lloyd_brombach`
  - `metadata:approval_lloyd_brombach`
- **Compliance / Standards**:
  - `requirement:ansi_ashrae_135` (BACnet protocol)
  - `requirement:masterformat_division_25`
- **WSU-specific**:
  - `requirement:point_tagging_conventions_per_25_50_60_30` (WSU has a controlled vocabulary for point names — unusual)
  - `requirement:data_upload_to_wsu_bas_database_per_25_50_60_10`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `device:vykon_jace_9000_basis_of_design` | active | Section 25 50 02.01 — explicit basis-of-design lock-in. Vendors substituting must justify equivalence. |
| `customer_override` | `software:niagara_4_or_5_required` | active | Section 25 50 02.02 — only Tridium Niagara 4/5 acceptable. |
| `customer_override` | `requirement:bas_manager_decision_authority` | active | "BAS Manager Decisions and Escalation" — single-point authority. |
| `customer_override` | `requirement:installer_vs_integrator_scope_separation` | active | Section 25 01 02 — "Installer and Integrator Scopes" — WSU treats them as different roles. |
| `scope_inclusion` | `requirement:point_tagging_conventions` | active | Section 25 50 60.30 — WSU-specific controlled vocabulary for point names. |
| `scope_inclusion` | `requirement:data_upload_to_wsu_database` | active | All BAS data must upload to WSU BAS database. |
| `scope_inclusion` | `device:nac_snc_pec_auc_hierarchy` | active | 4-tier controller hierarchy: NAC → SNC → PEC → AUC. |
| `scope_exclusion` | `device:non_tridium_jace` | active | Strong implicit exclusion — only JACE 9000 allowed. |
| `scope_exclusion` | `software:non_niagara_4_or_5` | active | Strong implicit exclusion. |
| `meeting_decision` | `decision:released_for_review_2024_09_26` | active | Document version 1.0 release date. |
| `meeting_decision` | `decision:masterformat_alignment` | active | Re-organized to follow CSI MasterFormat. |

**Expected packet count for Wayne State**: ≥ 11 (this is a master spec, so packet count is lower — most content is requirement atoms not project decisions)

---

### University of Houston — `uh_division_25_master_spec.pdf`

#### Expected entity_keys

- `customer:university_of_houston` (alias `customer:uh`, `customer:uofh`)
- `division:building_automation` (UH building automation department)
- **Controller hierarchy** (UH-specific naming):
  - `device:b_ows_bacnet_operators_workstation_platform`
  - `device:remote_b_ows`
  - `device:portable_b_ows` (notebook computer w/ specific specs)
  - `device:nac_network_area_controller`
  - `device:eac_expandable_application_controller` (a.k.a. base unit hosting up to 8 expansion modules)
  - `device:eac_space_mounted_local_display`
  - `device:uc_unitary_controller`
  - `device:uc_atu_air_terminal_unit_with_integral_damper_operator`
  - `device:system_network_controller`
- **Specific Tridium hardware**:
  - `device:jace_3e` (must come through Tridium Richmond, VA shipping facility)
  - `device:jace_6e` (must come through Tridium Richmond, VA shipping facility)
  - `device:jace_7` (must come through Tridium Richmond, VA shipping facility)
  - `device:microsd_safe_boot` (license storage)
- **Web Server hardware specs**:
  - `device:intel_xeon_e5_2640_x64` (or better)
  - `device:ram_2gb_min_8gb_recommended_for_64bit`
  - `device:hard_drive_256gb_min`
  - `device:display_1024x768_or_greater`
  - `device:ethernet_10_100_rj_45`
  - `device:isp_t1_or_adsl_or_cable_modem`
- **Portable B-OWS specs**:
  - `device:laptop_4gb_ram_500gb_to_1tb_hdd`
  - `device:laptop_ethernet_10_100_nic`
  - `device:laptop_4_usb_2_0_ports`
  - `device:laptop_87_key_keyboard_with_touchpad_track_stick`
  - `device:laptop_microsoft_windows_7_business_or_newer`
  - `device:laptop_microsoft_office_2010_professional`
  - `device:laptop_adobe_acrobat_9_0_standard`
- **Software**:
  - `software:niagara_4_framework`
  - `software:niagara_4_workbench_embedded_toolset`
  - `software:web_browser_ie_10_or_later`
  - `software:open_nic_specifications` (Niagara Open NIC: `accept.station.in=*`, `accept.station.out=*`, `accept.wb.in=*`, `accept.wb.out=*`)
- **Protocols**:
  - `protocol:bacnet_pics_protocol_implementation_conformance_statement`
  - `protocol:bacnet_bibb_table` (BACnet Interoperability Building Block)
  - `protocol:bacnet_ip_b_ip` (Annex J)
  - `protocol:bacnet_mstp` (Master Slave/Token Passing)
  - `protocol:bacnet_ptp` (Point-to-Point)
  - `protocol:bacnet_ethernet_iso_8802_3`
  - `protocol:lontalk` (allowed via gateway)
  - `protocol:modbus` (allowed via gateway)
- **Standards**:
  - `requirement:ansi_ashrae_135_2008_bacnet`
  - `requirement:fcc_part_15_class_a_b_c_j`
  - `requirement:ul_504` (Industrial Control Equipment)
  - `requirement:ul_506` (Specialty Transformers)
  - `requirement:ul_910` (Fire and Smoke Test for Cables in Air-Handling Spaces)
  - `requirement:ul_916` (Energy Management Systems — required for all BAS controllers)
  - `requirement:ul_1449` (Transient Voltage Suppression)
  - `requirement:eia_ansi_232_e`
  - `requirement:eia_455` (Fiber Optic Test Procedures)
  - `requirement:ieee_c62_41`
  - `requirement:ieee_142` (Grounding)
  - `requirement:nema_250` (Enclosures)
  - `requirement:nema_ics_1` (Industrial Controls)
  - `requirement:nema_st_1` (Specialty Transformers)
  - `requirement:ashrae_iesna_90_1_1999`
  - `requirement:ce_61326`
  - `requirement:c_tick`
  - `requirement:cul`
  - `requirement:btl_listed` (BACnet Testing Laboratories)
  - `requirement:nist_ir_6392_annex_b`
  - `requirement:uukl_ul_864` (Fire Alarm Control Units — where required by AHJ)
- **Performance constraints** (Section 1.8 Table 1 — System Accuracy):
  - `constraint:space_temperature_accuracy_+/-0_5c_+/-1f`
  - `constraint:ducted_air_temperature_+/-1c_+/-2f`
  - `constraint:outside_air_temperature_+/-1c_+/-2f`
  - `constraint:water_temperature_+/-0_5c_+/-1f`
  - `constraint:delta_t_water_+/-0_15c_+/-0_25f`
  - `constraint:relative_humidity_+/-2pct_at_10_to_90pct`
  - `constraint:water_flow_+/-2pct_actual`
  - `constraint:air_flow_terminal_+/-10pct_actual`
  - `constraint:air_flow_measuring_station_+/-2pct_calibrated`
  - `constraint:air_pressure_ducts_+/-25pa_+/-0_1in_wg`
  - `constraint:air_pressure_space_+/-3pa_+/-0_01in_wg`
  - `constraint:water_pressure_+/-1psi`
  - `constraint:electrical_power_+/-2pct_of_range`
  - `constraint:carbon_monoxide_+/-5pct_reading`
  - `constraint:carbon_dioxide_+/-50_ppm`
  - `constraint:long_term_drift_<0_4pct_per_year`
  - `constraint:repeatability_+/-2pct_full_scale_overall_+/-5pct_in_loop`
- **Time-to-display constraints**:
  - `constraint:graphics_display_50_dynamic_points_within_10_seconds`
  - `constraint:binary_command_to_response_max_10_seconds`
  - `constraint:analog_command_to_response_max_10_seconds`
  - `constraint:value_reporting_max_15_seconds_old`
  - `constraint:alarm_to_b_ows_max_20_seconds`
  - `constraint:critical_alarm_max_5_seconds`
  - `constraint:control_loop_min_1hz`
  - `constraint:b_ows_alarm_sync_within_5_seconds`
- **Capacity constraints**:
  - `constraint:expandable_to_500_000_hard_points_no_additional_database_licensing`
  - `constraint:operating_temp_-7c_to_40c_dry_bulb`
  - `constraint:humidity_10_to_90pct_non_condensing`
  - `constraint:b_ows_operating_temp_7c_to_32c`
  - `constraint:voltage_90_to_110pct_nominal_with_orderly_shutdown_below_80pct`
- **CSI sections**:
  - `spec_section:25_50_00` (Integrated Automation Facility Controls — BACnet)
  - `spec_section:23_06_06` (Formerly — BACnet BAS DDC Guide Specification)
- **Roles**:
  - `role:control_system_contractor`
  - `role:dedicated_full_service_office_within_50_miles_of_job_site`
  - `role:single_source_responsibility_supplier`
  - `role:authorized_representative_for_5_years_of_primary_ddc_components`
- **Open NIC**:
  - `requirement:niagara_open_nic_accept_station_in_out`
  - `requirement:niagara_open_nic_accept_wb_in_out`
- **Site-specific**:
  - `metadata:ae_project_number_template_field`
  - `metadata:revision_2017_01_30`
  - `template_field:insert_project_name`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `software:niagara_4_framework_only` | active | "Only systems that utilize the Niagara 4 Framework shall satisfy the requirements of this section." |
| `customer_override` | `software:no_additional_bms_server_software` | active | "Any control vendor that shall provide additional BMS server software shall be unacceptable." |
| `customer_override` | `device:jace_3e_6e_7_via_richmond_va_shipping` | active | "any JACE 3E, 6E, or 7 hardware products used on this project shall come through the Tridium Richmond, VA shipping facility. JACE hardware products not meeting this requirement will not be allowed." |
| `customer_override` | `requirement:open_nic_statements` | active | All Niagara 4 software licenses must have the 4 NiCS strings. |
| `customer_override` | `requirement:owner_full_licensing_full_access_rights` | active | "Owner shall have full licensing and full access rights for all network management, operating system server, engineering and programming software". |
| `customer_override` | `requirement:owner_admin_login_at_first_training` | active | "Owner shall receive all Administrator level login and passwords for engineering toolset at first training session". |
| `customer_override` | `requirement:no_proprietary_or_canned_application_specific_controllers` | active | "All BAS DDC Devices at all levels shall be fully custom-programmable in the field using the standard Operators Workstation Software. No configurable, canned program application specific controllers will be permitted." |
| `customer_override` | `requirement:no_gateways_or_protocol_translators_to_bacnet` | active | Strong rule: "No Gateways, Communication Bridges, Protocol Translators or any other device that translates any proprietary or other communication protocol to the BACnet communication protocol shall be permitted". |
| `customer_override` | `requirement:btl_listed_required_for_all_controllers` | active | All BAS controllers must be BACnet Testing Laboratories listed and stamped. |
| `customer_override` | `requirement:full_service_office_within_50_miles` | active | Section 1.7 — quality-assurance constraint. |
| `customer_override` | `requirement:5_years_authorized_representative_or_manufacturer` | active | Single source responsibility. |
| `scope_inclusion` | `service:bas_mockup_session_at_contractor_facility_max_1_week` | active | Section 1.2.B — mockup at BAS contractor facility, owner+MEP+GC present, sign-off required. |
| `scope_inclusion` | `service:48hr_training_in_multiple_sessions_during_1yr_warranty` | active | Section 1.11 — 48 total hours, multiple sessions, first after final commissioning, last in last month of 1-year warranty. |
| `scope_inclusion` | `service:point_to_point_commissioning_check_out` | active | Section 1.11 — full commissioning required. |
| `scope_inclusion` | `service:95pct_demonstration_acceptance` | active | "At least 95% of the results demonstrated must perform as specified". |
| `scope_inclusion` | `service:as_built_5_sets_3_ring_binders_and_flash_media` | active | Section 1.9.F. |
| `scope_inclusion` | `service:factory_training_for_uh_building_automation_rep` | active | Section 1.11.D — UH employee gets factory training paid by contractor. |
| `scope_inclusion` | `service:bas_warranty_2yr_ddc_controllers_1yr_install` | active | DDC controllers 2 yr; install warranty 1 yr. |
| `scope_inclusion` | `service:bacnet_native_in_every_device_at_board_level` | active | Section 1.3.A.1: "The BACnet operating stack must be embedded directly in every Device at the board level". |
| `scope_inclusion` | `service:peer_to_peer_distributed_control` | active | DDC Devices on peer-to-peer bus. |
| `scope_inclusion` | `service:web_enabled_unlimited_access_20_simultaneous_clients` | active | Section 2.2.A. |
| `scope_inclusion` | `service:audit_trail_user_changes` | active | Section 2.2.K. |
| `scope_inclusion` | `service:vykon_dashboard_compatibility` (implicit) | needs_review | Niagara Framework standard — vendors should support Vykon dashboards. |
| `scope_exclusion` | `device:mercury_thermostats` | active | Section 1.8 Note 7: "No devices utilizing mercury shall be acceptable for any application". |
| `scope_exclusion` | `device:configurable_canned_application_specific_controllers` | active | Hard exclusion. |
| `scope_exclusion` | `service:control_loops_dependent_on_central_server` | active | "Each individual Device shall, to the greatest possible extent, perform its programmed sequence without reliance on the BLCN." |
| `scope_exclusion` | `device:proprietary_server_hardware_black_boxes` | active | "Proprietary server hardware or 'Black Boxes' will not be acceptable." |
| `missing_info` | `template_field:ae_project_number` | active | Header placeholder. |
| `missing_info` | `template_field:project_name` | active | Subtitle placeholder. |

**Expected packet count for UH**: ≥ 28 (master spec with very prescriptive requirements; many `customer_override` packets)

---

## Cross-artifact bundle expectations

### Expected cross-artifact edges

- **`vendor:tridium`** is the **shared vendor** across both — both customers explicitly require Niagara 4 framework and JACE controllers. Graph builder should produce a `vendor_anchor` edge for Tridium.
- **0 cross-customer `quantity_conflict` edges** — both are master specs without project-specific quantities.
- **Different controller-hierarchy naming**:
  - WSU: NAC → SNC → PEC → AUC
  - UH: B-OWS (workstation) → NAC → EAC → UC
  - The parser should resolve both to a generic 4-tier `bms_controller_hierarchy` ontology while preserving the specific naming.
- **Both reference UL 916, BACnet, ASHRAE 135** — common compliance backbone.
- **WSU has Niagara 4 OR 5; UH only Niagara 4** — version difference. Worth flagging as a `vendor_pack_compatibility` consideration.
- **Both are `master_spec` documents (not project RFPs)**; the parser should detect this and avoid generating `quantity_conflict` packets between them.

### Expected aggregate metrics

```
expected_min_atom_count: 250
expected_min_packet_count: 40
expected_min_distinct_customers: 2  # WSU, UH
expected_min_unique_vendors_referenced: 3  # Tridium, Vykon (Tridium sub-brand), Niagara
expected_min_constraint_atoms: 50  # UH alone has ~25 performance constraints
expected_min_compliance_atoms: 30  # UL/NEMA/IEEE/EIA/FCC/ASHRAE stack
expected_min_template_unsupported_receipts: 6  # both master specs have placeholders
expected_min_csi_section_atoms: 25  # 19 from WSU + 4 from UH
expected_no_quantity_conflict_edges: true  # master specs, no project-specific qty
```

## Stress-test attributes (cross-bundle)

1. **Master specs vs. project RFPs** — neither has project-specific quantities; both are templates referenced by future projects. The parser should NOT generate quantity atoms; it should generate `template_field:*` for placeholders like "Insert Project Name" and `ae_project_number`.
2. **CSI MasterFormat 25-XX** — both follow Division 25 (Integrated Automation). The parser's pack should know Division 25 codes to organize atoms hierarchically.
3. **Niagara 4 / Niagara 5 / JACE / Tridium ecosystem** — if `bms_pack` doesn't know these, it'll miss the dominant US BAS vendor stack.
4. **Different controller-hierarchy naming** — same architecture, different acronyms (NAC/SNC/PEC/AUC vs. B-OWS/NAC/EAC/UC). The parser should detect both as 4-tier BMS controller hierarchies and produce a unified anchor.
5. **WSU's "installer" vs "integrator" role separation** — UH does NOT make this distinction. Tests parser's ability to extract role-distinct scope.
6. **Tridium Richmond, VA shipping requirement** (UH) — geographic-anchored procurement constraint that's surprisingly specific. Worth detecting as a `geo_specific_procurement_constraint`.
7. **OPEN NIC statements** (UH) — 4 specific NiCS strings (`accept.station.in=*`, etc.) required on every Niagara 4 license. Tests parser's ability to extract literal-string constraints.
8. **System Accuracy Table (Section 1.8 UH)** — 14 distinct accuracy constraints, each with metric/imperial paired ranges. The parser should produce 14 `constraint:*_accuracy` atoms preserving both units.
9. **48 hours of training in multiple sessions across 1-year warranty** (UH) — temporally distributed scope. Parser should detect "first class after final commissioning, last class in last month of 1-year warranty" as a schedule constraint, not a single training event.
10. **WSU's release as Rev 1.0 (Sept 2024)** — recent revision, possibly post-Rev 0.3 draft phase. Parser should detect document revision metadata and treat it as authoritative for "current standard".
11. **No Mercury policy (UH)** — banning mercury devices is environmental compliance. Easy gap candidate for the bms_pack.
12. **`accept.station.in=*` literal strings** — these are configuration values, not human-language. The parser should preserve them verbatim and not normalize.

## Macquarie (missing artifact)

The `STRESS_BMS_SPECS/SOURCE_NOTES.md` documents that a Macquarie BAS spec was intended for the trio. Macquarie University (Sydney, AU) has a published BAS specification, but the public download was unavailable at fetch time. For completeness, expected attributes if Macquarie were available:

- Likely Australian standards-based (AS/NZS 3000 Wiring Rules, AS 1851 Maintenance, AS/NZS 1768 Lightning Protection)
- Could reference Tridium Niagara, Schneider EcoStruxure Building Operation, Honeywell Niagara, Johnson Controls Metasys
- Different metric units only (no imperial)
- Specific Australia DBYD (Dial Before You Dig) procurement language

This is a known coverage gap.

## Known difficulties & where the parser will likely fail

1. **WSU's nested CSI sections** — `25 50 02.01`, `25 50 02.02`, `25 50 60.10`, `25 50 60.30` — multi-level nesting. The parser must preserve the section hierarchy, not flatten it.
2. **UH's "(Formerly SECTION 23 06 06...)" header** — historical CSI section number reference. Parser should detect both 25 50 00 (current) and 23 06 06 (formerly) as the same section, with the former as primary.
3. **WSU's "BMS" vs "BAS"** — synonyms in WSU usage. Both terms appear; should resolve to one entity.
4. **UH's 14 accuracy lines in Table 1** with parenthetical English+SI units — `+/-0.5 deg C (+/-1 deg F)` — the parser must preserve both units and recognize they're equivalent.
5. **UH's "NACNAC's" typo** — appears multiple times in source. The parser should normalize to "NAC's" or "NACs".
6. **UH's redundant "NACNetwork Area Controllers (NAC)"** — the prefix collision is a typo. Parser should resolve to NAC.
7. **WSU's Section 25 50 50 vs 25 51 10 duplicate** — both list "Field-level Controllers" with PEC + AUC requirements. Probably an authoring typo in WSU spec. Parser should detect duplicate section numbers.
8. **Both customers' role labels** — "Controls Contractor" (UH) vs "Installing Contractor" + "Integrator" (WSU). Parser should produce role atoms for each but recognize the relationship.
9. **HVAC equipment integration list** (UH Section 1.2.D) — 4 categories of equipment to be controlled: Computer/Server Room AC Units, Utility Metering, Occupancy/Lighting Controls, VFDs. Parser should produce equipment integration atoms.
10. **"Service Tools" vs "B-OWS" vs "Workbench"** — UH distinguishes service tools (Section 2.5) from B-OWS (Section 2.4) from Niagara 4 Workbench. All are configuration/programming tools, but each has distinct purpose. Parser should preserve the distinction.
