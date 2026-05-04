# Gold standard — STRESS_ITAD_PAIR

**Bundle**: 2 ITAD (IT Asset Disposition) RFPs that the original fetcher targeted but **could not download** (both URLs returned 403/404 at fetch time). The case directory therefore contains only `SOURCE_NOTES.md`; `artifacts/` is empty.

This is a **documented coverage gap** — the gold standard below describes what the case *would have* contained if the artifacts were available, plus what the parser-os runtime should detect and report when it encounters an empty case.

## Intended artifacts (per `SOURCE_NOTES.md`)

| Artifact (intended) | Source (intended) | Why it would have been a stress test |
|---|---|---|
| Natrona County, WY ITAD RFP | County procurement portal (403'd) | Real customer-authored asset-disposition scope with structured asset categories + recycling certifications |
| Michigan EGLE (Department of Environment, Great Lakes, and Energy) ITAD RFP | State portal (404'd) | State-level environmental compliance overlap with ITAD (R2v3, e-Stewards, NIST SP 800-88 data sanitization) |

## Service line: `itad`

**Recommended domain pack**: `itad_pack`

ITAD is defined in `/c/Users/lilli/parser-os/.claude/worktrees/hopeful-lamarr-19879c/real_data_cases/STRESS_COVERAGE_GAPS/SOURCE_NOTES.md` as a service line with thin public corpus availability. The pack should know:

- **Asset categories**: laptops, desktops, servers, routers, switches, displays, mobile devices, hard drives (HDD/SSD), storage media (USB/optical/tape), peripherals
- **Disposition methods**: refurbish-and-resell, recycling, secure shredding, lease-return, certified destruction
- **Data sanitization standards**:
  - `requirement:nist_sp_800_88_revision_1_data_sanitization` (NIST guideline for media sanitization)
  - `requirement:dod_5220_22_m_3_pass_overwrite` (older DOD overwrite standard, often cited)
  - `requirement:hipaa_phi_disposal` (healthcare PHI)
  - `requirement:gdpr_data_disposal` (EU privacy law)
  - `requirement:ccpa_data_disposal` (CA privacy law)
- **Industry certifications**:
  - `requirement:r2v3_responsible_recycling_certification`
  - `requirement:e_stewards_certification`
  - `requirement:naid_aaa_secure_destruction_certification`
  - `requirement:iso_14001_environmental_management`
  - `requirement:iso_9001_quality_management`
  - `requirement:iso_45001_occupational_health_and_safety`
- **Asset value recovery**:
  - `service:revenue_share_with_customer`
  - `service:trade_in_credit_for_replacement`
  - `service:flat_fee_per_unit_disposal`
- **Reporting**:
  - `requirement:certificate_of_destruction_per_asset`
  - `requirement:chain_of_custody_documentation`
  - `requirement:serial_number_inventory_with_disposition_status`
  - `requirement:weight_based_recycling_report`

## Expected entity_keys (if Natrona County had been fetched)

- `customer:natrona_county_wyoming`
- `division:natrona_county_information_technology` (or similar)
- **Asset categories** (per typical county RFP):
  - `device:laptops_desktops_servers_routers_switches`
  - `device:hard_drives_solid_state_drives`
  - `device:peripherals_displays_mobile_devices`
- **Compliance specific to Wyoming**:
  - `requirement:wyoming_state_record_retention_act`
  - `requirement:wyoming_law_enforcement_data_handling_per_county_sheriff`
- **Pricing model**:
  - `pricing:revenue_share_or_flat_fee`

## Expected entity_keys (if Michigan EGLE had been fetched)

- `customer:michigan_department_of_environment_great_lakes_and_energy_egle` (alias `customer:egle`)
- **Compliance specific to Michigan EGLE**:
  - `requirement:r2v3_certification_required`
  - `requirement:e_stewards_certification_preferred`
  - `requirement:michigan_act_451_e_waste`
  - `requirement:michigan_freedom_of_information_act_foia`
- **Specialized scope** (EGLE is the state environmental agency):
  - `service:full_chain_of_custody_for_chain_of_evidence_assets`
  - `service:certificate_of_destruction_per_asset`
  - `service:annual_audit_capability_certification_holder`

## Expected packets (if both files had been fetched)

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `requirement:nist_sp_800_88_data_sanitization` | active | Federal data-sanitization standard typically required. |
| `customer_override` | `requirement:r2v3_or_e_stewards_certification` | active | Industry recycling standards typically named. |
| `customer_override` | `pricing:revenue_share_with_customer_resale` | needs_review | Many ITAD vendors offer revenue-share; customer policy varies. |
| `scope_inclusion` | `service:certificate_of_destruction_per_asset` | active | Standard ITAD deliverable. |
| `scope_inclusion` | `service:chain_of_custody_documentation` | active | Required for sensitive data. |
| `scope_inclusion` | `service:serial_number_inventory_with_disposition_status` | active | Asset-level tracking. |
| `scope_exclusion` | `requirement:non_certified_recyclers` | active | Strong implicit exclusion via certification requirement. |
| `scope_exclusion` | `device:non_authorized_storage_media_after_sanitization_failure` | active | Failed-sanitization media must be physically destroyed. |
| `missing_info` | `quantity:expected_asset_inventory_count` | active | County/state would specify approximate asset count, possibly with appendix. |

## What the parser-os runtime should do with an empty case directory

When parser-os encounters this case, the expected behavior is:

1. **Detect that `artifacts/` is empty** — the case has metadata (`SOURCE_NOTES.md`) but no files to compile.
2. **NOT crash** — gracefully report "no artifacts found, skipping compile".
3. **Read `SOURCE_NOTES.md`** as case-level metadata and note that this is a documented gap.
4. **Output a minimal envelope** with:
   - `case_id: "STRESS_ITAD_PAIR"`
   - `service_line_intended: "itad"`
   - `compile_status: "no_artifacts_found"`
   - `note: "Documented coverage gap; intended artifacts unavailable on the public web at fetch time."`
5. **Generate gap_report entries** for missing service-line vocabulary (the itad_pack should be flagged as having no real-data validation).

## Stress-test attributes (this is a documented gap, but the gold tests parser robustness)

1. **Empty artifacts directory** — case-level handling. Parser must NOT crash.
2. **Case-metadata-only mode** — `SOURCE_NOTES.md` exists but no actual scope content. Parser should produce a minimal envelope.
3. **Service-line declared but no artifacts** — itad_pack should be detected as referenced but unvalidated.
4. **Gap signals to gap_detector** — itad_pack vocabulary cannot be tested against real data. Detector should produce a `coverage_gap:itad` warning.
5. **Documents the *intent* of the bundle** — for downstream synthesis tools that may generate synthetic ITAD artifacts to fill the gap (per STRESS_COVERAGE_GAPS/SOURCE_NOTES.md).

## Recommended remediation

Per the `STRESS_COVERAGE_GAPS/SOURCE_NOTES.md`:

> **itad** — already partially covered by STRESS_ITAD_PAIR; supplement with a synthetic asset-list XLSX (OEM/serial/condition columns).

A `scripts/synthesize_gap_artifacts.py` should be created to generate:
- A synthetic asset-inventory XLSX with realistic columns: `asset_tag`, `asset_type`, `manufacturer`, `model`, `serial_number`, `acquisition_date`, `current_condition`, `data_classification`, `disposition_method`
- A synthetic ITAD RFP narrative referencing NIST SP 800-88, R2v3, e-Stewards, NAID AAA
- Realistic asset counts (50-500 mixed assets across categories)

Once synthesized, this case can be re-named or supplemented to provide gold-standard ground truth for the `itad_pack`.
