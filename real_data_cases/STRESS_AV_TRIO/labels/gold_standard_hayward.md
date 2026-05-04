# Gold standard — STRESS_AV_TRIO (Hayward portion)

This bundle has 3 artifacts: ICMA + Hayward + AMBAG. **This document covers Hayward only**; ICMA and AMBAG have separate gold sheets.

## Artifact: `hayward_boardroom_av_rfp.pdf` (~12 pages)

**Hayward Area Recreation and Park District (HARD) — RFP for Audiovisual Technology Systems, Boardroom remodel at 1099 'E' Street, Hayward, CA 94541**

- Calendar: RFP issued Feb 2, 2023; site visits Feb 13–17 (scheduled upon request); proposals due Mar 14, 2023; Notice to Proceed pending Board award Mar 22, 2023
- Contact: Monty Boyd, Senior Project Manager (boym@haywardrec.org)
- 930 SF Boardroom, hybrid Board meetings
- Bundled with full **Standard Consulting Agreement template (Attachment D)** — 14 pages of generic boilerplate

### Service line: `av` (audio_visual) with `customer_authored_BOD`

This is a **design-build contract with an explicit Basis of Design (BOD) attached as Attachment A**. The BOD includes 38+ line-items with specific manufacturer + model + quantity for every device. This is the gold reference for testing **vendor pre-specified BOD extraction**: parser-os should detect that this RFP comes with a customer-authored BOM, NOT a vendor-asked equipment list.

The Bosch DICENTIS conference system makes this distinct from ICMA (which uses generic "podium mic" / "wireless lavalier" terms). This is a **single-vendor specified system** with all proprietary part numbers.

### Expected entity_keys (must include)

- `customer:hayward_area_recreation_and_park_district`
- `site:hayward_district_office_boardroom`
- `address:1099_e_street_hayward_ca_94541`
- `room:boardroom` (930 sf, target room)
- `room:meeting_room` (existing speakers/displays to remain)
- `room:lobby` (existing displays to remain)
- **Devices** (with manufacturer:part_number resolution):
  - `device:lcd_display_98in_4k` → vendor `LG`, part `CE-98UM5JB`, qty 1
  - `device:wall_mount` → vendor `Chief`, part `CJ-LTM1U`, qty 1
  - `device:av_rack_12u` → vendor `Middle Atlantic`, part `SRSR-X-12`, qty 1
  - `device:power_conditioner` → vendor `Furman`, part `FU-PL8C`, qty 1
  - `device:hdmi_extender` → vendor `Evolution`, part `ZX-EV4K2006`, qty 4
  - `device:hdmi_cable_18in` → vendor `Legrand`, part `2D-CG29674`, qty 8
  - `device:scaler_1to4` → vendor `Evolution`, part `ZX-EVSP14SC`, qty 1
  - `device:loudspeaker_in_ceiling` → vendor `Bose`, part `FS4CE FreeSpace`, qty 4
  - `device:amplifier` → vendor `Bose` (typo: "Base"), part `IZA 2120-HZ`, qty 1
  - `device:ptz_camera` → vendor `Atlona`, part `AT-HDVS-CAM-HDMI` (also `HW3WKD`), qty 2
  - `device:camera_mounting_hardware` → vendor `Atlona`, part `CAMCL-WT`, qty 2
  - `device:cat6_cable_plenum_1000ft` → vendor `(unspecified)`, part `CAT6PL`, qty 1
  - `device:lcd_monitor_22in_dais` → vendor `Planar`, part `8Q-PLL2251MW`, qty 12
  - `device:hdmi_cable_2k` → vendor `Comprehensive`, part `RH-MHD48G3PR`, qty 14
  - `device:hdmi_splitter_1x8` → vendor `MuxLab`, part `5M-500422`, qty 2
  - **Bosch DICENTIS conference system** (proprietary multi-component system, qty 13 seats voting):
    - `device:dicentis_voting_device` → `DCNM-DVT908`, qty 13
    - `device:dicentis_long_stem_mic` → `DCNM-MICL`, qty 13
    - `device:dicentis_audio_powering_supply` → `DCNM-APS2`, qty 1
    - `device:dicentis_flush_components` → `DCNM-FEC`, `FET`, `FCOUP`, `FBD`, `FLSP`, `FMCP`, `FMICB`, `FPT`, `MICSLL` (qty 1 each)
    - `device:dicentis_network_cable_2m` → `DCNM-CB02-I`, qty 1
    - `device:dicentis_network_cable_5m` → `DCNM-CB05-I`, qty 12
    - `device:dicentis_network_cable_10m` → `DCNM-CB10-I`, qty 2
    - `device:dicentis_network_cable_25m` → `DCNM-CB25-I`, qty 3
    - `device:dicentis_server_software` → `DCNM-LSYS`, qty 1
    - `device:dicentis_meeting_prep_software` → `DCNM-LMPM`, qty 1
    - `device:dicentis_camera_control_software` → `DCNM-LCC`, qty 1
    - `device:dicentis_participation_database` → `DCNM-LPD`, qty 1
    - `device:dicentis_voting_at_seat_license` → `DCNM-LSVT`, qty 7
    - `device:dicentis_voting_prep_software` → `DCNM-LVPM`, qty 1
    - `device:dicentis_server_hardware` → `DCNM-SERVER2`, qty 1
    - `device:dicentis_software_maintenance_1yr` → `DCNM-1SMA`, qty 1
- `service:hybrid_meeting_capability`
- `service:live_streaming`
- `service:meeting_recording`
- `service:overflow_room_signal_distribution`
- `service:listening_assist_devices` (if/as required by code)
- `service:project_closeout` (testing, adjustments, training, documentation, acceptance)
- `service:vendor_warranty_transfer`
- `service:maintenance_repair_12mo`
- `service:vendor_documentation` (PDF drawings, control system program, equipment list, manuals)

### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `device:lcd_display_98in_4k` (LG CE-98UM5JB) | active | BOD line 1, qty 1 — explicit. Customer-authored. |
| `scope_inclusion` | `device:lcd_monitor_22in_dais` (Planar, 12 units total = 6 directors + 6 staff) | active | BOD line 25, qty 12. The narrative says "(6) Board Members + (6) Staff" — BOD has 12 monitors, narrative agrees. |
| `scope_inclusion` | `device:bosch_dicentis_voting_system` (13 seats) | active | BOD lines 31+, qty 13 voting devices. Narrative says "13 mics for 6 directors + 6 staff + 1 lectern". Consistent. |
| `scope_inclusion` | `device:atlona_ptz_camera` (qty 2) | active | BOD line 19, qty 2. PTZ for hybrid meetings. |
| `scope_inclusion` | `device:bose_freespace_loudspeaker` (qty 4) | active | New ceiling-mounted speakers in Boardroom (existing speakers in adjacent rooms reused). |
| `scope_inclusion` | `service:hybrid_meeting_capability` | active | Needs Statement: "ability for the public to participate in Board meetings remotely". |
| `scope_inclusion` | `service:live_streaming` | active | "ability for the public to live stream Board meetings in real time". |
| `scope_inclusion` | `service:meeting_recording` | active | "ability for the District to record Board meetings for posting on District website". |
| `scope_inclusion` | `service:overflow_signal_distribution` | active | Existing displays in Meeting Room and Lobby will continue to receive computer/video signal from clerk computer. |
| `scope_inclusion` | `device:integrated_voting_system` (5 directors) | needs_review | Narrative says "Integrated voting system for (5) Directors" but DICENTIS BOD has 7 LSVT licenses (DCNM-LSVT × 7). **Quantity mismatch — flag as quantity_conflict.** |
| `scope_exclusion` | `infrastructure:electrical_power` | active | "It is assumed that all necessary electrical power and data will be provided by the District" — exclusion of electrical scope from AV vendor. |
| `scope_exclusion` | `infrastructure:backing_and_blocking` | active | "It is assumed that all necessary backing and blocking will be provided by the District" — exclusion of carpentry/blocking. |
| `scope_inclusion` | `service:turnkey_install_cabling_terminations` | active | "AV installation, AV cabling and terminations, wall and floor plates" — included. |
| `customer_override` | `device:integrated_speech_timer` | needs_review | "Integrated speech timer" listed in BOARDROOM AV SYSTEM ITEMS but NOT in BOD — implicit gap (vendor must propose). |
| `missing_info` | `device:listening_assist_device_count` | needs_review | "Listening Assisted Devices if/as required by code" — vague, code-conditional, no count given. |
| `missing_info` | `vendor:proposal_open` | active | This is a vendor-selection RFP; named vendor "____ Associates" is a placeholder in Attachment D template. |
| `meeting_decision` | `decision:on_site_work_within_2_month_window` | active | "all on-site work by all trades be completed within a (2) month window". |
| `action_item` | `vendor:lead_time_and_install_duration` | active | "include both lead time and installation durations with your proposal". |
| `meeting_decision` | `decision:procurement_and_staging_before_schedule` | active | "the District will contract with all trades for procurement and staging of project supplies and equipment before a final schedule is determined". |
| `action_item` | `vendor:warranty_12mo_workmanship_components` | active | "12 months from date of final acceptance" warranty obligation. |

**Expected packet count**: ≥ 18

### Expected ontology gap candidates

The new `av_pack.yaml` should know LCD/PTZ/microphone/speaker terms. But these specific terms are likely gaps:
- `dicentis` (Bosch product family — proprietary)
- `freespace` (Bose product family)
- `flush_loudspeaker_panel` / `flush_microphone_button_panel` (DICENTIS-specific accessories)
- `software_maintenance_agreement` (DCNM-1SMA — a service-line concept, not a device)
- `voting_at_seat_license` (Bosch DICENTIS naming)
- `participation_database` (Bosch DICENTIS naming)
- `meeting_prep_management` (Bosch DICENTIS naming)
- `ceiling_mounted_speakers` (vs. "in-ceiling" — wording variation)
- `fop` `ZX-EV4K2006`, `ZX-EVSP14SC` (Evolution part numbers)
- `furman_pl8c` (Furman is a power conditioner specialist — possibly missing)
- `cat6pl` (custom/abbreviated part for Plenum CAT6 cable)

### Expected exclusion patterns (from `default_pack` or `av_pack`)

- "It is assumed that all necessary electrical power and data will be provided by the District" → exclusion of electrical
- "It is assumed that all necessary backing and blocking will be provided by the District" → exclusion of structural
- "Existing speakers in the Meeting Room will be reused" → exclusion of new speakers in adjacent rooms
- "Existing Lobby Display to remain" → exclusion of new lobby display
- "Existing Meeting Room Display to remain" → exclusion of new meeting room display

### Expected constraint patterns

- "(2) month window" → schedule constraint `constraint:install_duration_2mo`
- "12 months from date of final acceptance" → warranty period `constraint:warranty_12mo`
- "applicable laws, regulations, and codes" → generic compliance (low-information)
- Insurance from Attachment D: $1M Workers' Comp, $1M GL+Auto combined single limit, $1M Professional Liability, $150K SIR limit, A:VII Bests' rating → `constraint:insurance_*` cluster (boilerplate)
- "10% retention released within 60 days after completion" → contract financial constraint
- "If/as required by code" → conditional constraint (Listening Assisted Devices)

### Stress-test attributes

- **Customer-authored BOD with 38+ line items, manufacturer + part + qty** — every BOD row should produce a `block.kind = "table"` with structured projection of MFG/PART/DESCRIPTION/QTY columns. Highest-density entity extraction in the corpus.
- **Bosch DICENTIS proprietary system with 16 sub-components** — many lines share the DCNM-* prefix. Tests parser's ability to recognize a single product family from many SKUs.
- **Quantity contradiction**: 5 Directors (narrative) vs. 7 DCNM-LSVT licenses (BOD). The parser should generate a `quantity_conflict` packet between two atoms in the same artifact.
- **BOD line 18 typo: "Base IZA 2120-HZ" should be "Bose IZA 2120-HZ"** — tests vendor-name fuzzy-match. Bose does make an IZA 2120 (zone amplifier), so the parser should resolve "Base" → "Bose".
- **Page numbering injection at every page footer** — "1099 'E' Street | Hayward, CA | 510-881-6700 | 510-888-5758 fax | www.haywardrec.org" appears at the bottom of every body page. The parser should treat this as page-footer noise and not produce atoms from it.
- **CAT6 cable spec inside an AV BOD** — testing cross-pack vocabulary (cabling pack should not claim this as a `copper_cabling` job; it's incidental cable in an AV install).
- **14-page Standard Consulting Agreement appended (Attachment D)** — pages from `Page 1 of 14` through `Page 14 of 14` are pure boilerplate insurance/indemnification. Atoms emitted from these pages should be classified `boilerplate_legal` and confidence-floored.
- **Form fields with placeholder underscores** — "____ Associates", "____, 20__ for_______________." The parser should detect templates and not emit atoms.
- **Missing required equipment**: speech timer is mentioned in narrative but absent from BOD — `missing_info` candidate.

### Expected metrics

```
expected_min_atom_count: 90       # 38 BOD lines × ~2 atoms each + ~14 narrative + boilerplate floor
expected_min_packet_count: 18
expected_min_devices_in_atoms: 38
expected_min_quantity_atoms: 38   # one per BOD row (qty column)
expected_min_vendor_atoms: 12     # Bosch, LG, Chief, Middle Atlantic, Furman, Evolution, Legrand, Bose, Atlona, Planar, Comprehensive, MuxLab
expected_quantity_conflict_edges: 1  # 5 directors vs 7 voting licenses
expected_min_boilerplate_pages: 14  # consulting agreement
expected_min_unsupported_receipts: 2  # floorplan attachments mentioned but flat
expected_min_constraint_atoms: 8
```

### Known difficulties & where the parser will likely fail

1. **BOD layout is broken across pages** — each BOD row has the qty/mfg/part/description on a single visual line, but pdftotext renders them across multiple lines (the `2`, `3`, `4` between rows are line numbers from the table). The parser's table extractor must reconstruct rows, not parse line-by-line.
2. **Bosch DICENTIS proprietary product naming** is unfamiliar — without the `av_pack` having DCNM-* aliases, every line will be flagged as a gap. The pack should add DICENTIS family.
3. **Inconsistent device counts** (5 directors / 6 board members / 12 dais monitors / 7 voting licenses / 13 voting devices / 13 microphones) — there are at least 5 different "seat counts" across the doc. The parser should NOT collapse them to a single quantity; each is anchored to a different role/seat.
4. **The "Base" / "Bose" typo in BOD line 17** — vendor-name fuzzy match required.
5. **CAT6PL "1000' Plenum CAT6 Cable"** — unit is a thousand-foot reel, not 1000 individual cables. The parser should NOT generate `quantity = 1000` for cables; it's `quantity = 1` (reel) with `dimension = 1000ft`.
