# Gold standard — STRESS_VT_CAM

**Bundle**: Virginia Tech Video Surveillance System RFP #0016531 — Addendum #2 (Q&A from pre-proposal conference, dated March 16, 2011)

**Service line**: `security_camera` (with networking/access_control adjacency)

**Recommended domain pack**: `security_camera` — but this RFP also references *access control*, *fire alarms*, *intrusion detection*, *parking management*, and *enterprise networking*, so an OrbitBrief consumer should expect cross-pack vocab.

## What's actually in the bundle

| File | Pages | Type | What's in it |
|---|---|---|---|
| `RFP_0016531_Addendum2.pdf` | 16 | PDF | Q&A from pre-proposal conference (67 numbered Q/A pairs, blue-text answers from VT staff) + scanned floor plans for the Perry Street Parking Structure (P1–P5 levels) + lighting plans + a blank vendor-info form. Pages 1–6 are the addendum proper; pages 7–16 are appendix drawings. |

The addendum is small but **scope-rich**: the answers explicitly *redefine* parts of the original RFP. Three answers materially change scope (Q1/A1, Q9/A9, Q66/A66 about pricing itemization). That makes this bundle the gold reference for testing **`customer_current_authored` precedence over `quoted_old_email`** and **scope-clarification edges**.

## Expected parser routing

| Artifact | Parser | Confidence | Why |
|---|---|---|---|
| `RFP_0016531_Addendum2.pdf` | `orbitbrief_pdf` | ≥ 0.95 | `.pdf` extension + PDF magic bytes. The OrbitBrief PDF pipeline should color-extract Q/A blocks because the answers are in a different color (blue) — a strong signal for separating customer-authored from vendor-asked text. |

## Expected entity_keys (must include)

- **Sites** — `site:virginia_tech` (or `site:virginia_polytechnic_institute`), `site:perry_street_parking_deck`, `site:andrews_information_systems_bldg`, `site:national_capital_region` (mentioned in Q34)
- **Rooms** — `room:communications_closet` (Parking Services office, first floor), `room:camera_room` (Room 113 from floor plan), `room:police_office` (111), `room:parking_office` (112)
- **Devices** — `device:ip_camera`, `device:ups`, `device:fiber_optic_backhaul`, `device:nvr` (implicit from "central storage")
- **Vendor / Integration partners** — `vendor:t2_systems` (parking management software), `vendor:thyssenkrupp` (elevators), `vendor:esri` (GIS via ArcSDE), `vendor:autocad`
- **Phases** — `phase:perry_street_parking_deck`, `phase:enterprise_wide`

**Ontology gap candidates the gap detector should flag** (not all currently in `security_camera_pack`):
- `surveillance_oversight_committee` (SOC — VT-internal authority)
- `arcsde` (ESRI middleware)
- `1700_pratt_drive` (street address; should resolve to a site)
- `university_policy_5617` (retention policy reference)
- `licence_plate_detail` (camera capability variant — not currently in pack)
- `facial_recognition` (mentioned and explicitly out-of-scope — gap detector should surface as exclusion)

## Expected packet families

| Family | Anchor | Status | Why |
|---|---|---|---|
| `customer_override` | `site:perry_street_parking_deck` | needs_review | Q1/A1: VT *changes* the original RFP — Perry Street is the FIRST project, subsequent projects will be design/build. This is exactly the scope-clarification → customer_current_authored case. |
| `scope_inclusion` | `site:perry_street_parking_deck` | active | Itemized pricing for parking deck is required (Q66/A66). Includes server hardware, software, storage. |
| `scope_inclusion` | `device:ip_camera` (enterprise) | needs_review | Phased build-out from 250 → 2500 cameras (Q13). Quantity is a *range*, not a fixed scope. |
| `scope_exclusion` | `scope:facial_recognition` | active | Q46/A46: "true facial recognition would likely not be needed". Should fire `exclusion_pattern` "would not be needed". |
| `scope_exclusion` | `scope:exterior_surveillance_perry_street` | active | Q44/A44: "Currently there are no plans for surveillance of the exterior of the structure". |
| `scope_exclusion` | `scope:audio_recording` | needs_review | Q22/A22: audio "requires approval by the Surveillance Oversight Committee" — this is a *gated* exclusion (constraint, not absolute). |
| `site_access` | `site:perry_street_parking_deck` | needs_review | Q47/A47: Fiber pulled to communications closet; conduit required in garage (Q28); air-conditioned room for head-end (Q29). Multiple physical-access constraints. |
| `missing_info` | `integration:legacy_camera_makes_models` | needs_review | Q19/A19, Q20/A20, Q51/A51, Q57/A57: VT does NOT have a list of existing legacy cameras to integrate. The vendor has to ask each department/college. This is `missing_info` at the project level. |
| `missing_info` | `integration:access_control_alarms` | needs_review | Q3/A3, Q4/A4, Q48/A48, Q54/A54, Q59/A59: integration with access control / fire alarms / intrusion detection — needs-assessments deferred to deployment. |
| `missing_info` | `integration:video_analytics_vendor` | needs_review | Q52/A52: "Virginia Tech has not selected a video analytics vendor" — open vendor decision. |
| `meeting_decision` | `decision:perry_street_first_then_enterprise` | needs_review | Q1/A1, Q9/A9: phased approach decided in pre-proposal conference. |
| `meeting_decision` | `decision:storage_centralized_at_andrews` | active | Q14/A14: storage will be centralized at Andrews Information Systems Bldg. |
| `meeting_decision` | `decision:vt_will_not_provide_backup_generator` | active | Q17/A17: parking deck has back-up generator; VT will not ask vendors to provide back-up generator power for other implementations. |
| `action_item` | `customer:vt_to_identify_priorities_and_phasing` | active | Q23/A23, Q58/A58: VT and successful offeror will identify priorities/phasing. |
| `action_item` | `customer:retention_policy_litigation_hold` | needs_review | Q7/A7: vendor must describe how the system implements "litigation/investigative hold". |
| `customer_override` | `pricing:itemized_for_parking_deck` | active | Q66/A66: VT wants itemized pricing for the parking deck so they can decide whether to build or buy. |

**Expected packet count**: **≥ 14** (with the 3 contradiction/customer_override flags being the highest-priority).

## Expected contradiction edges

The gold here is testing **addendum-changes-original-RFP**. Even with only the addendum present, the parser should produce `customer_override` packets that supersede the implied original-RFP statements (e.g., the original RFP probably said "campus-wide rollout" — the addendum carves Perry Street out as Phase 1).

When the user runs this with the original RFP also present (not in this corpus snapshot), the parser MUST:
- Treat the addendum (March 16, 2011) as `customer_current_authored`
- Treat any conflicting statement in the original RFP as `quoted_old_email`-equivalent
- Generate `contradicts` edges with `edge_family = quantity_contradiction` for the 250-vs-2500 range and `customer_override` packets where the addendum scope-narrows.

## Expected exclusion patterns to fire (from `default_pack` or `security_camera`)

- "would likely not be needed" → flag as exclusion-shaped (gap candidate; not in pack)
- "no plans for surveillance" → exclusion of exterior surveillance
- "Currently there is no requirement" → integration with T2 (parking) excluded for now
- "is not currently a central provider" → ambiguous (Q19) — should be `missing_info` not `scope_exclusion`
- "VT will not ask vendors to provide back-up generator power" → vendor scope clarification (excludes generator)
- "not at this time" → soft exclusion (Q42 about central command center)

## Expected constraint patterns to fire

- "five nines (99.999%) reliability" → uptime SLA constraint (anchor: `device:storage_archival`)
- "30 day retention" → retention constraint (`constraint:retention_30day`)
- "litigation/investigative hold" → indefinite-period retention exception
- "Surveillance Oversight Committee approval" → governance constraint on audio
- "conduit requirement in garage" → physical install constraint
- "air conditioned/heated room" → environmental constraint for head-end
- "background check" — implied by VT (not explicit in addendum but standard)

## Expected vendor / authority class breakdown

- **`customer_current_authored`**: every blue-text `A1`–`A66` answer (these are John Krallman / VT IT Acquisitions speaking). **At least 50 atoms** should fall into this class.
- **`formal_rfp`**: implied references back to the original RFP (Section VI.A.10, Section III, Section V.B, etc.) — these should *not* be treated as governing on their own; they're pointers to a document not in the bundle.
- **`vendor_quote`**: zero (no vendor pricing in this bundle).
- **`meeting_note`**: the addendum is itself a meeting follow-up (pre-proposal conference Q&A) — but because it's been *codified into a written addendum*, it should be promoted to `customer_current_authored` per the lattice rule "addendum > meeting_note".

## Stress-test attributes

- **Q&A 67-pair format** — every Q is vendor-asked, every A is customer-authored. Tests parser's ability to separate question/answer roles in a single document.
- **Color-coded text** — answers are blue, questions are black. The OrbitBrief PDF pipeline's color-driven extractor should pick this up as the `customer_current_authored` signal. If the parser doesn't, that's a real gap.
- **References to absent documents** — Section III, Section V.B, Section VI.A.10, "Attachment D", "Section F", University Policy 5617, "Attached as part of this Addendum" (Q60/A60). These cannot be parsed but should generate `missing_info` flags.
- **Embedded scanned floor plans + lighting plans** (pages 7–16) — these are CAD drawings flattened into PDF images. The OrbitBrief PDF parser should treat them as low-text pages and the receipt verifier should mark them `unsupported` cleanly (not `failed`).
- **Old document (2011)** — date references like "March 31, 2011 at 3:00 PM" should normalize as past dates; if the parser is treating recent dates as more authoritative, this is a stress for time-aware authority logic.
- **Multiple integration touch-points** — access control, fire alarms, intrusion detection, video analytics, GIS, ESRI/ArcSDE, AutoCAD, T2 Systems, ThyssenKrupp. Tests the parser's ability to surface the *integration ontology* even when no one device is named explicitly.
- **No actual BOM** — this is a Q&A *about* the BOM-to-be. The parser should NOT generate `quantity_conflict` packets from this file alone; quantities here are projected (250 → 2500 range), not authoritative.

## Verification metrics for parser-os run

When this case is compiled:

```
expected_min_atom_count: 60       # Q&A pairs + appendix metadata
expected_min_packet_count: 12
expected_min_cross_artifact_edges: 0  # single artifact in this bundle
expected_max_active_packets: 5    # most should need_review (it's all clarifications)
expected_min_needs_review_packets: 8
expected_unsupported_receipts: 5+  # the 10 scanned floor-plan/lighting pages
expected_exclusion_patterns_fired: ["would not be needed", "no plans for", "not at this time"]
```

## Known difficulties & where the parser will likely fail

1. **The addendum *changes* an RFP we don't have** — the comparison logic the system needs (addendum > original) only fires when both docs are present. Standalone, this looks like a clarification-only doc, not a scope-changer. Recommendation: add the original RFP to this case before trusting `customer_override` recall metrics.
2. **Floor plans on pages 7–16 are scanned CAD images** — text extraction will return geometric labels (column letters A-H, dimensions like "304'-10\""), almost no semantic content. The parser should NOT generate atoms from these pages; if it does, atoms with sub-0.5 confidence should be auto-floored.
3. **Integration touch-points are *forward-looking*** — every "describe how your solution would integrate with X" is a *vendor capability question*, not a current scope statement. The parser must not treat these as `scope_inclusion` of the integrations themselves.
4. **The phased-build-out scope** (250 → 2500 cameras) is a *plan*, not a quantity. If the parser extracts "2500 cameras" as a `quantity` atom and pairs it cross-artifact with an actual roster of 91 cameras, that would be a false-positive `quantity_conflict`. The phasing context must be preserved.
