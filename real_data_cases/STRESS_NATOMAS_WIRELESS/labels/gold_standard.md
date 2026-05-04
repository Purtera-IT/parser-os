# Gold standard — STRESS_NATOMAS_WIRELESS

**Bundle**: Natomas Unified School District RFP 25-107, Wireless Equipment, E-rate Year 28 (Year 2025)

**Service line**: `wireless` (with strong networking/E-rate adjacency)
**Recommended domain pack**: `wireless_pack` — and the parser MUST also load `e_rate` compliance vocab from `default_pack` since this is a USAC-funded procurement.

## What's actually in the bundle

| File | Pages | Type | What's in it |
|---|---|---|---|
| `NatomasYR28RFP25_107WirelessEquipment.pdf` | 25 | PDF | Single-PDF RFP. Pages 1–6 = scope/requirements. Pages 7–14 = procurement T&Cs (boilerplate). Pages 17 = list of 19 schools. Page 18–19 = cost proposal (BLANK rate cells). Pages 20–25 = compliance forms (RFP Form, Letter of Agreement, Fingerprint Cert, Non-Conflict, Insurance Acknowledgement). |

## Expected parser routing

| Artifact | Parser | Confidence | Why |
|---|---|---|---|
| `NatomasYR28RFP25_107WirelessEquipment.pdf` | `orbitbrief_pdf` | ≥ 0.95 | `.pdf` extension. Two distinct table regions (equipment list on pages 5–6, cost proposal on pages 18–19). |

## Expected entity_keys (must include)

- `customer:natomas_unified_school_district` (alias: `customer:nusd`)
- `address:1901_arena_blvd_sacramento_ca_95834`
- **Sites** (19 schools — every row from Page 17 should produce a `site:*`):
  - `site:district_office`
  - `site:american_lakes_school` (K-8) — 2800 Stonecreek Drive, Sacramento 95833
  - `site:bannon_creek_school` (K-8) — 2775 Millcreek Drive, Sacramento 95833
  - `site:discovery_high` — 3401 Fong Ranch Rd, Sacramento 95834
  - `site:h_allen_hight_elementary` — 3200 North Park Drive, Sacramento 95835
  - `site:heredia_arriaga_school` — 1800 Club Center Drive, Sacramento 95835
  - `site:heron_school` (K-8) — 5151 Banfield Drive, Sacramento 95835
  - `site:inderkum_high` — 2500 New Market Drive, Sacramento 95835
  - `site:jefferson_school` (K-8) — 2001 Pebblewood Drive, Sacramento 95833
  - `site:larry_g_meeks_academy` — 2775 Millcreek Dr, Sacramento 95833
  - `site:leroy_green_academy` — 2950 West River Dr, Sacramento 95833
  - `site:natomas_high` — 3301 Fong Ranch Rd, Sacramento 95834
  - `site:natomas_middle` — 3200 North Park Drive, Sacramento 95835
  - `site:np3_charter_high` — 3700 Del Paso Rd, Sacramento 95834
  - `site:np3_charter_elementary` — 3800 Del Paso Rd, Sacramento 95834
  - `site:np3_charter_middle` — 3700 Del Paso Rd, Sacramento 95834
  - `site:natomas_park_elementary` — 4700 Crest Drive, Sacramento 95835
  - `site:paso_verde_school` (K-8) — 3883 Del Paso Road, Sacramento 95834
  - `site:two_rivers_elementary` — 3201 West River Drive, Sacramento 95833
  - `site:witter_ranch_elementary` — 3790 Poppy Hill Way, Sacramento 95834
  - `site:any_other_greater_sacramento_area` (catchall expansion clause)
- **Devices (Cisco wireless equipment family — every BOM row → device entity)**:
  - `device:catalyst_9166i_ap` → vendor `Cisco`, part `CW9166I-B`, **qty 500 (page 5) / qty 136 (cost proposal page 18)** — **QUANTITY MISMATCH** is a critical packet
  - `device:catalyst_9166i_smartnet_8x5xnbd` → part `CON-SNT-CW911B66`, qty 500/136
  - `device:capwap_software_for_9166i` → part `SW9166-CAPWAP-K9`, qty 500/136
  - `device:ceiling_grid_clip` → part `AIR-AP-T-RAIL-R`, qty 500/136
  - `device:ap_low_profile_mounting_bracket` → part `AIR-AP-BRACKET-1`, qty 500/136
  - `device:ap_universal_mounting_bracket` → part `AIR-AP-BRACKET-2`, qty 500/136
  - `device:cisco_dna_subscription_optout` → part `CW9166I-DNA-OPTOUT`, qty 500/136
  - `device:network_pnp_connect` → part `NETWORKPNP-LIC`, qty 500/136
  - `device:single_pack_option` → part `CW9166I-SINGLE`, qty 500/136
  - `device:c9166i_over_option` → part `CW9166I-OVER`, qty 500/136
  - `device:oberon_h_plane_right_angle_surface_mount` → part `1006-CW9166`, qty 20/4 (the only line where the count differs from APs)
  - `device:cisco_dna_essential_9166_tracking` → part `CDNA-E-C9166D1`, qty 500/136
  - `device:cisco_dna_advantage_5y_term` → part `DNA-E-5Y-C9166D1`, qty 500/136
  - `device:wireless_dna_essential_term_lic` → part `AIR-DNA-E`, qty 500/136
  - `device:wireless_dna_essential_5y_term_lic` → part `AIR-DNA-E-5Y`, qty 500/136
  - `device:wireless_dna_essential_term_tracker_lic` → part `AIR-DNA-E-T`, qty 500/136
  - `device:wireless_dna_essential_5y_term_tracker_lic` → part `AIR-DNA-E-T-5Y`, qty 500/136
  - `device:wireless_dna_perpetual_network_stack_essentials` → part `AIR-DNA-NWSTACK-E`, qty 500/136
- **Compliance entities (E-rate specific — gap candidates)**:
  - `requirement:erate_eligibility_marking`
  - `requirement:erate_funding_request_number`
  - `requirement:erate_category_2_internal_connections`
  - `requirement:erate_form_470` / `requirement:erate_form_471` / `requirement:erate_form_472` / `requirement:erate_form_474` / `requirement:erate_form_486`
  - `requirement:erate_spin_service_provider_id`
  - `requirement:erate_lcp_lowest_corresponding_price` (47 CFR 54.511(b))
  - `requirement:erate_free_services_advisory`
  - `requirement:fcc_frn_registration`
  - `requirement:fcc_green_light_status` (red light disqualifies)
  - `requirement:secure_networks_act_compliance` (vendor must certify NO components from listed prohibited vendors)
  - `requirement:usac_bulk_upload_template_item_21`
  - `requirement:erate_audit_retention_10yr` (USAC requires 10-year record retention)
  - `requirement:fingerprinting_ed_code_45125_1`
  - `requirement:public_works_contractor_registration_dir` (CA Labor Code 1725.5/1771.1, post-2015-03-01)
  - `requirement:debarment_executive_order_12549`
- `phase:funding_year_2025_starts_april_1_or_jul_1`
- **Pricing structure**:
  - `pricing:erate_eligible_eligible_percentage`
  - `pricing:non_eligible_cost_allocated`
  - `pricing:lcp_lowest_corresponding_price`
  - `pricing:cat_2_eligible_purchase_apr_1_2025_to_install_summer`

## Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `device:catalyst_9166i_ap` (qty 136 vs 500) | active | **Critical:** Page 5 lists Qty 500 (per part), Page 18 cost-proposal lists Qty 136. The cost proposal is the *operative* number (vendor must price 136). Either the equipment list is generic-template + the cost proposal is project-actual, OR it's a real contradiction. **Customer-current-authored should resolve to 136 (cost proposal).** Generates `quantity_conflict` packet. |
| `scope_inclusion` | `device:catalyst_9166i_ap` (qty 136) | active | The cost proposal is the operative quantity. |
| `scope_inclusion` | `vendor:cisco_or_equivalent` | active | "or equivalent" language preserves vendor-substitution rights. |
| `scope_inclusion` | `service:installation_summer_2025` | active | Cat 2 funding requires equipment delivered post-April 1, 2025 with summer installation. |
| `scope_inclusion` | `service:detailed_billing` | active | Section "Requirements 1": "All plans proposed should include detailed billing". |
| `scope_inclusion` | `phase:start_april_1_2025` | active | Section "Requirements 2": "The Start date of this project will be April 1, 2025." |
| `scope_exclusion` | `device:secure_networks_act_prohibited_vendors` | active | "Vendor must certify that their equipment is not manufactured by, nor contains any components from, the list of vendors on 'The Secure Networks Act'." (Huawei/ZTE prohibited.) Strong exclusion. |
| `scope_exclusion` | `pricing:erate_non_eligible` | active | Non-eligible costs must be cost-allocated, not co-mingled. |
| `customer_override` | `pricing:lowest_corresponding_price` | active | LCP rule: vendor cannot charge schools more than commercial customers. **Hard pricing constraint.** |
| `customer_override` | `pricing:price_decrease_passed_through` | active | "In the event of a price decrease for service or from the manufacturer, said decrease shall be passed on to the Natomas Unified School District". Customer-favorable repricing clause. |
| `missing_info` | `device:specific_school_quantity_distribution` | needs_review | Equipment list on page 5 shows total qty 500/136 but does NOT specify per-school distribution. The 19 schools could need 26 APs each (500/19) or vary widely. Open allocation. |
| `missing_info` | `pricing:total_amount` | active | Cost proposal cells (Unit Price, Extended Cost, Sub Total, Taxes, Shipping, Grand Total) are all BLANK. |
| `missing_info` | `pricing:erate_eligibility_percentage_per_line` | active | "Erate %" column in cost proposal is blank — vendor must determine per line. |
| `missing_info` | `pricing:i_c_or_b_m_per_line` | active | "I/C B/M" column blank — vendor must classify each item as Internal Connections vs Basic Maintenance. |
| `meeting_decision` | `decision:erate_funded_contingent` | active | "The proposal and the contract negotiated implementing this proposal, are conditional and subject to full E-Rate funding by the SLD." |
| `meeting_decision` | `decision:contract_term_5yr_max` | active | Per CA Education Code 17596/81644: max 5 consecutive fiscal years. |
| `action_item` | `vendor:obtain_spin` | active | Vendors must have valid SPIN before bid. |
| `action_item` | `vendor:obtain_frn` | active | Vendors must have valid FCC FRN. |
| `action_item` | `vendor:certify_no_red_light` | active | "Any potential bidder found to be in Red Light Status will be disqualified". |
| `action_item` | `vendor:fingerprint_certification_ed_code_45125_1` | active | Required certification of criminal background checks for any employees with school access. |
| `action_item` | `vendor:bulk_upload_template_within_1_week` | active | "Within one (1) week of award, the awarded Service Provider must provide the District a bill of materials using the completed USAC 'Bulk Upload Template' (Item 21)". |
| `site_access` | `site:any_district_school` | active | School fingerprinting + criminal background requirements apply at all sites. |
| `customer_override` | `decision:proposals_withdrawal_30_day_lock` | active | Proposals locked for 30 days after submittal. |
| `meeting_decision` | `decision:bid_protest_3_business_days` | active | Protest deadline 3 BD after award notification. |

**Expected packet count**: ≥ 22

## Expected ontology gap candidates (E-rate / wireless specific)

The `wireless_pack` should know AP/SSID/2.4GHz/5GHz/6GHz/Wi-Fi 6E/802.11/CAPWAP. But these E-rate-specific terms are gaps:

- `e_rate` / `erate_year_28` / `funding_year_2025`
- `usac` (Universal Service Administrative Co.)
- `sld` (Schools and Libraries Division)
- `spin` (Service Provider Identification Number)
- `frn` (FCC Registration Number)
- `lcp` (Lowest Corresponding Price — 47 CFR 54.511(b))
- `red_light_status` / `green_light_status` (FCC delinquency flag)
- `category_1` / `category_2` / `internal_connections` / `basic_maintenance`
- `i_c` / `b_m` (column abbreviations)
- `form_470` / `form_471` / `form_472` / `form_474` / `form_486`
- `bulk_upload_template_item_21`
- `secure_networks_act` (Trade-Secret-style law that bans Huawei/ZTE/Hytera/Hikvision/Dahua components)
- `service_substitution`
- `cat_6e_w6e` (Wi-Fi 6E radio)
- `tri_band_4x4_xor` (radio configuration)
- `oberon_h_plane` / `recessed_ceiling_grid` (mounting hardware specifics)
- `dna_subscription_optout` (Cisco licensing model)
- `network_pnp` / `plug_n_play_zerotouch` (Cisco DNA)
- `nwstack` (Cisco DNA Network Stack)
- `sntc_8x5xnbd` (Smartnet 8x5 next-business-day support)

## Expected exclusion patterns

- "or equivalent" → vendor-substitution language (NOT a true exclusion; it's a flexibility clause). Should be flagged but not generated as `scope_exclusion`.
- "Manufactured by, nor contains any components from, the list of vendors on 'The Secure Networks Act'" → STRONG exclusion of prohibited vendors. Generates `scope_exclusion` with anchor `device:secure_networks_act_prohibited`.
- "Faxed or emailed RFPs will not be accepted" → procurement-mode exclusion (boilerplate)
- "RFPs received after due date and time will be returned unopened" → procurement deadline absolute
- "must be in Red Light Status will be disqualified" → vendor-eligibility exclusion
- "Failure to comply with these terms... shall constitute grounds for termination" → termination trigger (boilerplate)

## Expected constraint patterns

- "January 8, 2025 at 3:00 p.m." → submission deadline (PST implied)
- "March 2025" → 471 filing approximate
- "April 1, 2025" → project start
- "July 1, 2025" → E-rate funding year start (no invoice before this)
- "five (5) consecutive fiscal years" → contract term max
- "10 years" → audit retention period
- "30 days" → proposal lock-in period
- "$1,000,000 per occurrence" → professional general liability minimum
- "$2,000,000 aggregate" → professional malpractice aggregate
- "120 days" → USAC invoicing deadline after last service day
- "ten (10) calendar day written notice" → contract termination with cause
- "thirty (30) calendar day written notice" → contract termination without cause
- "Two and a half hours, 140 miles" → travel constraint reference (irrelevant, but in source — should be filtered)

## Stress-test attributes

- **Critical quantity contradiction in same document**: equipment list page 5 (qty 500) vs cost proposal page 18 (qty 136). The parser must:
  1. Detect both quantities exist for the same part number `CW9166I-B`
  2. Generate a `quantity_conflict` packet with `customer_override` resolution
  3. Resolve the override toward the cost-proposal number (later in doc, more authoritative for pricing)
  4. NOT silently pick one and discard the other
- **18 distinct part numbers, 17 with same qty, 1 with diff qty** — the `1006-CW9166` Oberon mount has qty 20 / 4 (different ratio). Tests parser's ability to detect that the qty mismatch is item-specific.
- **Blank cost-proposal table** — every cell in pages 18–19 (Unit Price, Extended Cost, Sub Total, Taxes, Shipping, Grand Total) is BLANK or "______". The parser should NOT generate quantity atoms for blank cells.
- **19 sites in a flat schools table** — page 17 is a 19-row table of school names + addresses + phones. Each row should produce a `site:*` entity. Tests school-name normalization (some have "K-8" suffix, some don't; "NP3" prefix; "(K-8)" parenthetical).
- **Charter and non-charter schools mixed** — NP3 Charter (3 schools) vs district-operated (16 schools). The parser should detect the charter distinction or at least preserve the names.
- **Multiple Cisco SKU prefixes** sharing `CW9166I` family — `CW9166I-B`, `CW9166I-DNA-OPTOUT`, `CW9166I-SINGLE`, `CW9166I-OVER`. Tests product-family resolution (similar to Bosch DICENTIS in Hayward).
- **E-rate compliance section is dense** — pages 7–14 are 7 pages of USAC/FCC compliance requirements. The new `e_rate` ontology should fire here.
- **Public-works contractor registration** is CA-specific — Labor Code 1725.5/1771.1 with the March 1, 2015 cutoff. Date-conditional regulatory citation.
- **"Any other location within the Greater Sacramento Area designated by the District"** is the last row of the schools table — a *catchall expansion entity*. The parser should produce a special `site:any_other` entity flagged as expansion-eligible (not a normal site).
- **Cost-proposal items deduplicated** — the equipment-list page 5 has the SKUs in one order, page 18 has them in a slightly different order with different formatting. The parser must canonicalize part numbers and not emit duplicates.

## Verification metrics

```
expected_min_atom_count: 110
expected_min_packet_count: 22
expected_min_distinct_sites: 19
expected_min_device_atoms: 18      # 18 SKUs, each → device atom
expected_min_quantity_atoms: 18    # one per SKU per quantity location (17 of which are 500/136 contradictions)
expected_quantity_conflict_edges: 17
expected_customer_override_packets: 17  # one per qty contradiction; cost proposal wins
expected_min_compliance_atoms: 15
expected_min_unsupported_receipts: 5  # blank cost-proposal cells + form fields
expected_exclusion_patterns_fired: ["Secure Networks Act", "Red Light Status", "received after due date"]
expected_constraint_patterns_fired: ["five (5) consecutive fiscal years", "10 years", "$1,000,000 per occurrence"]
expected_min_distinct_vendors_referenced: 1   # Cisco
```

## Known difficulties & where the parser will likely fail

1. **The 500-vs-136 contradiction** — if the parser only reads the *first* equipment list, it'll miss the cost proposal qty and not generate the customer_override packet. **Critical:** the parser must scan the entire document and aggregate part-number → quantity sets across all locations.
2. **"or equivalent" language** — many parsers treat "or equivalent" as scope-exclusion of vendor-specific. It's not — it's a *flexibility clause*. The pack should have a special pattern for this.
3. **Empty cost proposal cells** — common false-positive: parser reads "$_____" and emits `quantity = 0` or hallucinates a price. Tests restraint.
4. **19 school sites + 1 catchall** — tests entity extraction recall (must get all 19 + handle the catchall correctly without producing 20 normal sites).
5. **E-rate funding-year semantics** — "Year 28" is FY 2025 in E-rate calendar but **starts July 1, 2025**. The parser should normalize the funding year correctly without treating 2025 as a calendar year.
6. **Cisco DNA license SKU sprawl** — 9 different DNA-related SKUs (E, E-5Y, E-T, E-T-5Y, NWSTACK-E, etc.) all map to similar concepts. The parser should group them as license/subscription variants, not as 9 distinct device kinds.
7. **CW9166I "or equivalent" with restrictive Secure Networks Act** — the equivalent must be from a non-prohibited vendor. Tests interaction between scope_inclusion (or_equivalent) and scope_exclusion (secure_networks_act).
