# Gold standard — STRESS_MULTI_CAM

**Bundle**: 3 simultaneously-active public-safety/security camera RFPs from very different cities, vendors, and project shapes. The shared service line is `security_camera`, but each artifact stresses a different dimension of camera-pack vocab.

| File | Pages | City / Customer | Project shape |
|---|---|---|---|
| `milwaukee_pole_cam_RFP17341.pdf` | ~20 | City of Milwaukee Police Department, Fusion Division | Add ~50 pole cameras + ALPR + storage; preparation for 2020 Democratic National Convention. Existing Genetec Security Center 5.7 + ExacqVision 5. **Confidential 2020 marked.** |
| `chicago_housing_camera_RFP3276.pdf` | ~80 | Chicago Housing Authority, Information Technology Services | **Massive scope**: upgrade 5,288 existing + add 1,674 new + 800 at 13 new locations = ~7,762 cameras across 216+ sites. 146 servers + 146 AIO appliances. 3-year contract. All Genetec. |
| `santa_monica_video_analytics_RFP.pdf` | ~30 | Santa Monica Police Department, SMART Center | Video analytics platform on top of existing Genetec VMS. 100–400 investigative camera licenses, 50–200 real-time alert camera licenses. State of CA grant funded. |

**Service line**: `security_camera` (with networking + storage + analytics adjacency)
**Recommended domain pack**: `security_camera_pack` + cross-pack `networking_pack` for Milwaukee + `analytics` ontology for Santa Monica.

This bundle is the gold reference for testing **multi-customer corpus separation**: the parser MUST produce 3 distinct `customer:*` clusters in the cross-artifact graph and MUST NOT generate quantity_conflict edges between Milwaukee's 50 pole cameras and Chicago's 7,762 building cameras (different customers, different scopes). Cross-customer quantity comparisons are nonsense.

## Expected parser routing

| Artifact | Parser | Confidence | Why |
|---|---|---|---|
| `milwaukee_pole_cam_RFP17341.pdf` | `orbitbrief_pdf` | ≥ 0.95 | PDF + magic bytes. Has Appendix I (page 20) with map graphic — page-mode transition needed. |
| `chicago_housing_camera_RFP3276.pdf` | `orbitbrief_pdf` | ≥ 0.95 | Large PDF with embedded equipment matrix table (Article III Section 1). |
| `santa_monica_video_analytics_RFP.pdf` | `orbitbrief_pdf` | ≥ 0.95 | Standard RFP. |

## Per-artifact gold

### Milwaukee — `milwaukee_pole_cam_RFP17341.pdf`

**Service line**: `security_camera` (pole-mounted variant, with strong wireless mesh + fiber backbone adjacency)

**Setting**: Public Safety Camera Initiative for the 2020 DNC. National Special Security Event (NSSE). Department of Justice Bureau of Justice Assistance grant funded.

#### Expected entity_keys

- `customer:city_of_milwaukee_police_department` (alias `customer:mpd`)
- `division:fusion_division` (Real-Time Event Center)
- `address:200_e_wells_street_room_601_milwaukee_wi_53202` (procurement office)
- **Sites**:
  - `site:downtown_milwaukee_around_fiserv_forum`
  - `site:brady_street_business_corridor`
  - `site:milwaukee_river_east_region`
- **Existing infrastructure entities** (NOT to be inventoried as new scope, but extracted as context):
  - `device:wireless_mesh_network` (4.9 GHz, originally deployed 2007)
  - `device:90_existing_cameras_50_locations`
  - `device:point_to_point_radio_4_9ghz`
  - `device:point_to_multipoint_radio_4_9ghz`
  - `device:hd_panoramic_180deg_camera` (1 unit existing)
  - `device:box_camera_hd_single_direction` (2 units existing)
  - `device:dark_fiber_40g_ethernet_backbone_802_1aq_spb`
  - `device:fiber_collector_link_1g_10g`
  - `device:rpr_802_17` (resilient packet ring)
  - `infrastructure:eoc_data_center_800sf` (District 3)
  - `infrastructure:radio_shop_400sf`
  - `device:dell_r530_server` (Genetec Directory only, 28TB Raid 5)
  - `device:dell_per540_server` (×3, 63TB Raid 5 each)
  - `device:dell_precision_t3620_workstation` (Fusion Division)
  - `device:exacqvision_z_series` (~275 cameras; legacy)
  - `device:exacq_failover_recorder`
  - `software:genetec_security_center_5_7_sr6`
  - `software:exacqvision_5_4_3_39302`
  - `software:windows_server_2008_r2_sp1` / `windows_server_2016`
  - `software:sql_server_express_2008` / `2014`
  - `software:sql_server_standard_2019`
- **New scope devices**:
  - `device:ip_dome_camera_2mp_4_view` (qty ≥ 50, with 1 license each)
  - `device:ip_ptz_camera_2mp_outdoor` (qty ~50)
  - `device:adaptive_ir` (or "similar")
  - `software:vms_enterprise_level` (genetec_compatible)
  - `device:nvr` (turn-key with VMS pre-loaded)
  - `device:storage_for_120_day_retention`
  - `device:onvif_profile_s_camera`
- **Optional/Conditional**:
  - `device:alpr_automated_license_plate_reader` (optional add-on)
  - `software:video_analytics_search_persons_objects_clothing_colors` (optional)
- **Vendor mentions**:
  - `vendor:genetec` (sole VMS provider)
  - `vendor:dell` (server hardware)
  - `vendor:exacq_technologies`
  - `vendor:live_earth` (data visualization platform — pass-through analytics)
  - `vendor:briefcam` (video content analytics platform)
- **Compliance**:
  - `requirement:doj_bja_grant`
  - `requirement:nsse_designation` (National Special Security Event)
  - `requirement:24x7_support_dnc_aug_17_20_2020`
  - `requirement:replacement_parts_within_30_minutes_8am_12am_during_dnc`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `device:ip_dome_camera_2mp_4_view` (qty ≥ 50) | active | Section 1.3: "approximately 50 or more camera locations" with 4 directional views. |
| `scope_inclusion` | `device:ip_ptz_camera_2mp` (qty ~50) | active | Section 1.3: "PTZ cameras at preselected intersections". |
| `scope_inclusion` | `service:network_fiber_connection_to_new_cameras` | active | Project goal #3. |
| `scope_inclusion` | `service:storage_upgrade_existing_genetec` | active | Project goal #4 + must be Genetec-compatible. |
| `scope_inclusion` | `service:integration_with_genetec_security_desk` | active | Sole VMS provider; integration is mandatory. |
| `scope_inclusion` | `service:integration_with_live_earth_briefcam` | needs_review | "ability of such analytic technologies to 'pass through' to the City's Genetec and Live Earth data visualization and analytics platform" — vendor-capability question, not core scope. |
| `scope_inclusion` | `requirement:24x7_support_aug_17_20_2020_dnc` | active | Strong SLA constraint for DNC week. |
| `scope_exclusion` | `device:non_genetec_compatible` | active | "Must be compatible with our current video management system". |
| `scope_exclusion` | `device:non_onvif_profile_s` | needs_review | ONVIF Profile S is preferred — NOT strictly excluded. Soft. |
| `customer_override` | `decision:single_primary_vendor` | active | "the City of Milwaukee seeks to partner with one primary vendor who will be responsible for managing the entire system" — single-help-desk requirement. |
| `customer_override` | `pricing:30_minute_replacement_during_dnc` | active | Hard SLA: 8am–12am during DNC window must have parts within 30 minutes. 12am–8am: vendor describes capability. |
| `missing_info` | `device:storage_capacity_per_camera` | active | Storage was scaled to 120-day retention but new total not specified. |
| `missing_info` | `vendor:non_disclosure_agreement_for_network_diagram` | active | Network diagram available "to viable CSIM vendors via nondisclosure agreement process" — gated info. |
| `meeting_decision` | `decision:dnc_event_aug_17_20_2020` | active | Hard event date drives schedule. |
| `meeting_decision` | `decision:phase_1_install_jul_20_aug_3` | active | Reliability test period after install. |
| `meeting_decision` | `decision:design_install_period_jun_22_jul_20` | active | Tight 30-day design+install window. |
| `action_item` | `vendor:bonfire_portal_submission` | active | https://cityofmilwaukee.bonfirehub.com/projects/view/26827 |
| `action_item` | `vendor:hard_copy_in_addition_to_electronic` | active | "one (1) hard copy of your proposal" required at 200 E Wells St. |
| `site_access` | `site:rf_planning_required` | active | "Describe the Proposer's ability to conduct site surveys, radio frequency (RF) planning". |

**Expected packet count**: ≥ 16 for Milwaukee alone

#### Expected ontology gap candidates (Milwaukee)

- `pole_camera` (variant of camera placement)
- `national_special_security_event` / `nsse`
- `democratic_national_convention` / `dnc`
- `4_9_ghz_supportive_radio_platform` (public safety band — distinct from Wi-Fi)
- `point_to_point_ptp` / `point_to_multipoint`
- `802_1aq_spb` (Shortest Path Bridging — networking gap candidate)
- `ieee_802_17_rpr` (Resilient Packet Ring — networking gap)
- `dark_fiber`
- `fusion_division` / `real_time_event_center` / `real_time_crime_center` (RTCC)
- `sworn_officer` / `civilian_crime_analyst` (org/staffing)
- `cad_computer_aided_dispatch`
- `eoc_emergency_operations_center`
- `mobile_command_post`
- `view_shed` (camera viewing distance term)
- `sntc_8x5xnbd` (smartnet)
- `mission_critical_reliability` / `redundancy` / `data_at_rest_encryption` / `data_in_transit_encryption`
- `slavery_disclosure_affidavit` (Milwaukee-specific procurement form)

---

### Chicago Housing Authority — `chicago_housing_camera_RFP3276.pdf`

**Service line**: `security_camera` (large multi-site enterprise upgrade) with strong networking + facilities adjacency

**Setting**: 216 sites, 65,000 households, 135,000 residents. Surveillance Cameras Initiative since 2009. 3-year contract with 2 one-year options.

#### Expected entity_keys

- `customer:chicago_housing_authority` (alias `customer:cha`)
- `address:60_e_van_buren_8th_floor_chicago_il_60605` (procurement)
- `division:information_technology_services`
- **Sites**: 216 properties (per Exhibit D — names + addresses; the parser should produce one `site:*` per row)
  - Includes "senior, family, and scattered sites, headquarters, and other remote facilities"
  - 13 new locations getting first-time Genetec systems
- **Existing scope (to upgrade)**:
  - `device:streamvault_directory_server` (qty 70 to upgrade + 13 new = 83 total) → vendor `Streamvault`, model better-than-`Genetec SV-2030E-`
  - `device:streamvault_archiver_server` (qty 70 to upgrade + 13 new = 83 total)
  - `device:streamvault_aio_security_appliance` (qty 146 to upgrade + 13 new = 159 total) → model `Genetec SV-4040EX-R28-120T-12-416`
  - `device:axis_camera_p3268_lv` (qty 3277 of original 5288 + new ones)
  - `device:axis_camera_p3738_ple` (qty 2011 of original 5288)
- **New cameras**:
  - `device:axis_camera_p3268_lv` (additions: qty 1674 = 1326 in stairwells + 348 in laundry areas)
  - `device:axis_camera_p3268_lv_OR_p3738_ple` (qty ~800 at 13 new locations, depending on site survey)
- **Server hardware** for new locations (qty 13 each):
  - `device:streamvault_directory_server` × 13
  - `device:streamvault_archiver_server` × 13
  - `device:aio_security_appliance` × 13
- **Camera locations within sites**:
  - `room:stairwell` (1326 cameras new)
  - `room:laundry_area` (348 cameras new)
  - `room:hallway`
  - `room:entrance`
  - `room:exterior` (8MP minimum, optical zoom, 4K, vandal-proof, waterproof)
  - `room:interior_with_sound_recording` (5MP minimum)
- **Specs**:
  - `constraint:exterior_camera_8mp_minimum`
  - `constraint:interior_camera_5mp_minimum`
  - `constraint:4k_resolution_minimum`
  - `constraint:optical_zoom`
  - `constraint:vandal_proof`
  - `constraint:waterproof_exterior`
  - `constraint:sound_recording_interior`
  - `constraint:30_day_video_retention`
  - `constraint:motion_triggered_recording_day_night`
  - `constraint:future_expansion_capacity`
  - `constraint:non_proprietary_systems_post_contract`
  - `constraint:cybersecurity_industry_standard`
  - `constraint:encryption_video_data`
  - `constraint:network_segmentation_camera_feeds`
  - `constraint:secure_authentication`
  - `constraint:integration_with_genetec_federation`
  - `constraint:cpd_view_access` (federated to Chicago Police)
- **SLA constraints**:
  - `constraint:response_time_2_business_days`
  - `constraint:critical_resolution_4hr_response_48hr_resolution`
- **Compliance**:
  - `requirement:hud_section_3` (labor hours threshold)
  - `requirement:mbe_wbe_dbe_minimum_threshold_50001`
  - `requirement:davis_bacon_OR_hud_wage_rates`
  - `requirement:24_cfr_965_101` (HUD wage rate preemption)
  - `requirement:debarment_no_award_to_ineligible`
  - `requirement:bribery_disqualifying_5_years` (Sherman Anti-Trust, Clayton Act)
  - `requirement:economic_disclosure_statement_eds`
  - `requirement:contractor_affidavit_notarized`
  - `requirement:hud_form_5370_c` / `requirement:hud_form_5369_a`
  - `requirement:cha_supplier_portal` (https://supplier.thecha.org)
  - `requirement:e_waste_recycling_disposal_compliant`
- **Contract structure**:
  - `contract_term:3_years_base_with_2_one_year_options` (5-year max)
  - `pricing:firm_fixed_rate`
  - `phasing:33_percent_per_year`
  - `requirement:per_site_milestones`
- **Vendor mentions**:
  - `vendor:genetec` (Federation, Security Center, Streamvault)
  - `vendor:streamvault` (Genetec sub-brand)
  - `vendor:axis_communications` (camera maker)

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `device:ip_camera_replacement_5288` | active | Replace 5,288 existing cameras at EOL/EOS/out-of-service. |
| `scope_inclusion` | `device:ip_camera_addition_1674` | active | Add 1,674 new cameras across existing Genetec sites. |
| `scope_inclusion` | `device:new_genetec_systems_at_13_locations_800_cameras` | active | New Genetec at 13 locations, ~800 cameras. |
| `scope_inclusion` | `device:server_replacement_146_directory_146_archiver` | active | 146 servers replaced (page 1 says "replacing 146 servers, 146 SV-16s, and 5,288 existing cameras"). |
| `scope_inclusion` | `device:axis_p3268_lv` (qty 5288 + 1674 + ~800 = ~7762 total) | active | Multiple Axis camera SKUs. |
| `scope_inclusion` | `service:e_waste_recycling` | active | Old camera disposal compliant with federal e-waste regulations. |
| `scope_inclusion` | `service:wall_repair` | active | "Any visible alterations to walls, ceilings, or other building components must be repaired by vendor". |
| `scope_inclusion` | `service:cpd_federation_access` | active | "All cameras must be properly federated allowing view access to the Chicago Police Department". |
| `scope_inclusion` | `service:training_genetec_real_time_or_in_person` | active | Article III Section 9. |
| `scope_inclusion` | `service:risk_mitigation_plan_required` | active | Section III Section 10. |
| `scope_inclusion` | `service:formal_change_management_process` | active | Section 12. |
| `scope_inclusion` | `service:as_built_drawings_diagrams` | active | Section 13. |
| `scope_exclusion` | `device:non_genetec_compatible` | active | "all cameras must be Genetec-compatible. Camera substitutions are allowed with CHA approval". Hard exclusion of non-compatible. |
| `scope_exclusion` | `service:proprietary_systems_post_contract` | active | "Non-proprietary systems for secured access beyond the life of the contract". Vendor-lock-in excluded. |
| `customer_override` | `pricing:firm_fixed_rate_3yr_base_2_options` | active | Award structure. |
| `customer_override` | `decision:33_percent_phasing_3_years` | active | "approximately 33% of the project completed annually over a three-year period". |
| `missing_info` | `quantity:specific_camera_quantities_per_site` | active | "A detailed report with specific camera names...will be provided by CHA as part of project documentation" — not in this RFP. |
| `missing_info` | `quantity:exact_new_genetec_camera_count_per_new_location` | active | "Camera totals are approximate and are dependent on-site surveys". |
| `missing_info` | `quantity:server_count_for_new_locations` | active | 13 directory + 13 archiver + 13 AIO listed but not all storage/cabling components. |
| `meeting_decision` | `decision:pre_proposal_conference_apr_10_2025` | active | At 60 E Van Buren Conference Room 736A. Optional ("encourages all interested firms"). |
| `meeting_decision` | `decision:contract_q3_2025_award` | active | "The resulting contract from this Request for Proposal (RFP) will be issued in the third quarter of 2025". |
| `action_item` | `vendor:letter_of_intent_due_apr_23_2025` | active | Attachment B by 11:00 AM CDT. |
| `action_item` | `vendor:elevator_usage_minimized` | active | "Elevator usage must be limited to minimize resident inconvenience. Resident access to elevators must be prioritized over contractor movement." |
| `action_item` | `vendor:occupied_senior_buildings_safety` | active | All work in occupied senior resident buildings — safety + phasing required. |
| `site_access` | `site:occupied_buildings_resident_safety` | active | Multiple constraint atoms apply. |

**Expected packet count**: ≥ 22 for CHA alone

#### Expected ontology gap candidates (CHA)

- `streamvault` (Genetec sub-brand)
- `aio_security_appliance` (All-In-One)
- `sv_4040ex_r28_120t_12_416` (specific SV model)
- `genetec_federation` (vs. Genetec Security Center)
- `cpd_federation` (Chicago Police Department video sharing)
- `senior_housing_portfolio` / `family_housing_portfolio` / `scattered_sites`
- `office_of_emergency_management_and_communications` (OEMC, Chicago)
- `eol_end_of_life` / `eos_end_of_service`
- `risk_mitigation_plan` / `escalation_procedure`
- `change_management_process`
- `federated_view_access`
- `non_proprietary_post_contract` (anti-vendor-lock-in)
- `hud_section_3` (labor hours threshold)
- `mbe_wbe_dbe`
- `davis_bacon` / `prevailing_wage_hud_determined`
- `2_cfr_200_320` (federal procurement re-solicitation rule)
- `bonding` / `bid_rigging` / `clayton_act`
- `pins_automated_insurance_tracking`
- `sherman_anti_trust_act_15_usc_1`
- `25_miles_local_office_proximity_preference`

---

### Santa Monica — `santa_monica_video_analytics_RFP.pdf`

**Service line**: `security_camera` analytics overlay (no new cameras; software-only on existing Genetec)

#### Expected entity_keys

- `customer:city_of_santa_monica_police_department` (alias `customer:smpd`)
- `address:333_olympic_dr_santa_monica_ca_90401`
- `division:smart_center` (Santa Monica Analytical Real Time Center / RTCC)
- **Devices/Software** (analytics-focused):
  - `software:video_analytics_platform_va`
  - `software:investigative_module`
  - `software:real_time_alerting_module`
  - `software:license_plate_recognition_lpr` (must be from same VA manufacturer, NOT 3rd party)
  - `software:single_sign_on_authentication`
  - `software:active_directory_integration`
  - `device:server_hardware_per_va_specs`
  - `software:case_creation_management_module`
- **Existing infrastructure (NOT new scope)**:
  - `software:genetec_vms` (Santa Monica's existing VMS)
- **License blocks**:
  - `license:investigative_camera_min_100_max_400`
  - `license:realtime_alert_camera_min_50_max_200`
  - `license:user_logins_min_50`
  - `license:concurrent_users_min_10`
- **Video format support**:
  - `format:264` `format:3gp` `format:asf` `format:avi` `format:dav` `format:divx` `format:dvr` `format:flv` `format:g64` `format:g64x` `format:ge5` `format:mkv` `format:mov` `format:mp3` `format:mp4` `format:raw` `format:rt4` `format:ts` `format:wmv` `format:xba`
- **Object classifications** (VA platform must support):
  - `classification:people_man_woman_child`
  - `classification:two_wheeled_vehicles_bicycle_motorcycle`
  - `classification:other_vehicles_car_pickup_van_truck_bus_train_airplane_boat`
- **Person attributes**:
  - `attribute:upper_wear_long_short_sleeves_colors`
  - `attribute:lower_wear_long_short_colors`
  - `attribute:hat_yes_no`
  - `attribute:face_mask_yes_no`
  - `attribute:bag_no_backpack_handheld`
- **Object attributes**: color, size, speed, dwell_time, direction, proximity
- **Spatial attributes**: area, exclusion_area, path, line_crossing
- **Compliance**:
  - `requirement:state_of_california_grant_funded`
  - `requirement:living_wage_ordinance_$20_32_per_hour_fy24_25_$54200_threshold`
  - `requirement:oaks_initiative_taxpayer_protection_article_xxii`
  - `requirement:levine_act_disclosure`
  - `requirement:business_license_santa_monica_or_out_of_city`
  - `requirement:non_discrimination_debarment_non_collusion`
  - `requirement:public_records_act_disclosure`
- **SLA**:
  - `constraint:alert_latency_under_5_seconds`
  - `constraint:video_summary_60x_faster_than_realtime_review`

#### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `software:va_platform_with_investigative_and_realtime_modules` | active | Section 3.1 + 3.2. |
| `scope_inclusion` | `software:lpr_internal_same_manufacturer` | active | "LPR software component should be of the same manufacturer as the VA platform and should not be provided by a third-party manufacturer". |
| `scope_inclusion` | `software:genetec_vms_integration` | active | Direct integration with existing Genetec VMS. |
| `scope_inclusion` | `software:single_sign_on_authentication` | active | Section G. |
| `scope_inclusion` | `software:active_directory_integration` | active | Section I. |
| `scope_inclusion` | `software:object_metadata_search_classifications` | active | Section J classifications + attributes. |
| `scope_inclusion` | `service:training_in_person_or_virtual_train_the_trainer` | active | Section 5.B. |
| `scope_inclusion` | `service:warranty_maintenance_update_3_years` | active | Section 1.7 SLA. |
| `scope_exclusion` | `software:lpr_third_party_provider` | active | "should not be provided by a third-party manufacturer" — strong exclusion. |
| `scope_exclusion` | `service:direct_camera_communication_for_realtime` | active | Real-time engine should NOT directly communicate with cameras (must go through VMS) — security/performance reason. |
| `customer_override` | `vendor:5_year_minimum_experience` | active | "at least 5 years of experience in the video analytics sector". |
| `customer_override` | `vendor:5_year_minimum_install_experience` | active | "minimum of 5 years experience required installing and servicing video analytics systems". |
| `customer_override` | `vendor:4_references_2_law_enforcement` | active | At least 4 customer references; 2 of those preferably law enforcement. |
| `missing_info` | `pricing:total_amount_proposed` | active | Cost proposal hourly fee schedule for all personnel — vendor-asked. |
| `meeting_decision` | `decision:contract_term_3_years_firm_fixed_price` | active | Section 2. |
| `action_item` | `vendor:opengov_portal_inquiries` | active | Inquiries via OpenGov only. |
| `action_item` | `vendor:product_demonstration_in_person_or_remote` | active | "in-person or remote product demonstration" required as part of evaluation. |

**Expected packet count**: ≥ 16 for Santa Monica alone

#### Expected ontology gap candidates (Santa Monica)

- `video_analytics_platform` / `va_platform`
- `investigative_module` / `real_time_alerting_module`
- `condensed_video_summary` / `60x_faster_than_realtime_review`
- `bounding_boxes` / `dwell_time`
- `appearance_similarity_search`
- `object_classification_with_attributes`
- `line_crossing` / `area_dwell` / `exclusion_area` (scene analysis)
- `csv_watchlist_import` (license plate)
- `oaks_initiative_taxpayer_protection`
- `levine_act_disclosure`
- `living_wage_ordinance` (CA-specific, $20.32/hr)
- `genetec_vms_integration` (specifically — Santa Monica is a deeper integration than Milwaukee or CHA)
- `smart_center` (Santa Monica's RTCC name)
- `rtcc_real_time_crime_center`

## Cross-artifact bundle expectations

### Expected cross-artifact edges

- **0 cross-customer `quantity_conflict` edges** — Milwaukee 50 vs CHA 7,762 vs Santa Monica 100 are different customers' different scopes. No conflict.
- **3 distinct `customer:*` cluster nodes** in the graph.
- **`vendor:genetec`** is the **shared vendor** across all 3 artifacts. The graph builder should produce a `vendor_overlap` edge linking the 3 customers via `vendor:genetec`. This is a real, useful relationship — the same VMS provider serves all 3 cities.
- **Service-line consistency**: all 3 should resolve to `security_camera_pack` (Santa Monica perhaps with `analytics` sub-routing).
- **Different scope shapes**: Milwaukee is **new install** + **integration**; CHA is **mass replacement + new sites**; Santa Monica is **software overlay only**. Cross-artifact graph should preserve these distinctions.

### Expected aggregate metrics

```
expected_min_atom_count: 350
expected_min_packet_count: 54
expected_min_distinct_customers: 3   # MPD, CHA, SMPD
expected_min_distinct_sites: 240+    # 50 (Milwaukee) + 216 (CHA) + 1 (SM) = ~270
expected_min_unique_vendors_referenced: 8   # Genetec, Streamvault, Axis, Dell, Exacq, Live Earth, BriefCam, Bonfire
expected_quantity_conflict_edges_within_artifact: 0
expected_quantity_conflict_edges_cross_customer: 0
expected_min_constraint_atoms: 30
expected_min_compliance_atoms: 25
expected_min_unsupported_receipts: 5
expected_vendor_overlap_edges_via_genetec: 3+  # shared VMS provider links all 3 customers
```

## Stress-test attributes (cross-bundle)

1. **Same vendor (Genetec) across 3 customers** — tests vendor-deduplication while preserving customer-level entity separation.
2. **Massive scope variance** — 50 cameras vs 7,762 cameras vs 0 new cameras (analytics only). Tests parser's restraint (don't average, don't sum).
3. **Three different procurement portals** — Milwaukee Bonfire, CHA Supplier Portal, Santa Monica OpenGov. Each has different submission rules.
4. **Three different fundings** — DOJ BJA grant (Milwaukee), HUD-funded (CHA), CA state grant (Santa Monica). The compliance ontology differs:
   - Milwaukee: NSSE designation
   - CHA: HUD Section 3, Davis-Bacon, MBE/WBE/DBE
   - Santa Monica: Living Wage, Oaks Initiative, Levine Act
5. **DNC date pressure** (Milwaukee) — hard event date drives schedule. Cross-artifact, this is unique to Milwaukee; Santa Monica and CHA have no such date pressure.
6. **Two senior-housing safety constraint sets** — only CHA has occupied-building constraints (elevator priority, resident safety). Tests packet specificity.
7. **3 different VMS integration depths**:
   - Milwaukee: integrate new cameras with existing Genetec
   - CHA: rip-and-replace + new Genetec
   - Santa Monica: analytics overlay on existing Genetec
   The parser should distinguish "new install + integrate" from "replace existing" from "extend existing".
8. **Object-classification ontology in Santa Monica only** — pages of object classes that the security_camera_pack should NOT have but should detect as a new vocabulary domain.
9. **Bonfire portal "Confidential 2020" watermark on every page of Milwaukee** — header noise. Parser should detect and filter.
10. **One artifact (CHA) is over 2× larger than the other two combined** — tests parser's ability to maintain cross-artifact coherence when one artifact dominates the corpus.

## Known difficulties & where the parser will likely fail

1. **CHA's quantity sprawl** — 5,288 + 1,674 + 800 = 7,762 cameras across 216+ sites + 146 servers + 146 AIO appliances + 70+13 directory + 70+13 archiver. The parser must not collapse to a single "cameras" atom. Each line of the equipment matrix is its own quantity atom.
2. **Milwaukee's existing-vs-new device extraction** — the long inventory of existing equipment (Genetec servers, Dell hardware, Exacq) is *context*, not new scope. If the parser treats existing infrastructure as new scope, false positives will dominate.
3. **CHA's "replacing 146 servers, 146 SV-16s"** — the SV-16s appear once on page 4 but are NOT in the equipment matrix. This is an artifact of imprecise drafting. The parser should detect the term and flag as `missing_from_bom` or `unspecified_quantity_in_matrix`.
4. **Santa Monica's video-format list** — 20 distinct format extensions in one paragraph. The parser should NOT generate 20 separate device atoms; these are software capabilities, not devices.
5. **Compliance vs. scope distinction** — CHA pages 17–28 are nearly all compliance (HUD, DBE, debarment, insurance, evaluation criteria). Atoms emitted from these pages should be classified as `procurement_compliance` not `scope_inclusion`.
6. **Cross-customer vendor:genetec linkage** — tempting to merge 3 customers into one cluster because they share Genetec. Don't. The graph should keep them separate but add cross-customer edges through `vendor:genetec`.
7. **Genetec "Federation"** is a specific feature (multi-site cross-cluster video access), not the same as Genetec Security Center. CHA uses both terms; the parser should treat as separate concepts.
