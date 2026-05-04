# Gold standard — STRESS_AV_TRIO (ICMA portion)

This bundle has 3 artifacts: ICMA + Hayward + AMBAG. **This document covers ICMA only**; Hayward and AMBAG have separate gold sheets.

## Artifact: `icma_av_2025_2028_rfp.pdf` (10 pages)

**ICMA RFP No. ICMAHO/Audio Visual Services/2025-2028 — Annual Conference AV**

- 10-page solicitation, dated December 5, 2023, closing February 9, 2024
- Service period: 2025 Tampa → 2026 Long Beach → 2027 Toronto → 2028 TBD (4-year contract)
- 5,000–6,000 attendees, 200 exhibiting companies per conference
- Contact: Judy Day, CMP — Senior Manager, Conferences and Events

### Service line: `av` (audio_visual)

This is a **service contract**, not a hardware install — so it's testing the parser on a *service-line* AV ontology rather than a device-count ontology.

### Expected entity_keys (must include)

- `customer:icma`
- `site:tampa_convention_center`
- `site:long_beach_convention_center`
- `site:metro_toronto_convention_centre`
- `site:headquarter_hotel` (mentioned but unnamed per year)
- **Conference dates** as scope-bounding entities — Friday → Wednesday, 5 days
- Room types: `room:executive_board_meeting`, `room:micro_certification_classroom`, `room:keynote_general_session`, `room:breakout_session`, `room:committee_meeting`, `room:product_theater`, `room:speaker_ready_room`, `room:exhibit_hall`
- **Devices** (per equipment matrix):
  - `device:lcd_projector`
  - `device:computer`
  - `device:screen_package`
  - `device:speaker_timer`
  - `device:podium_mic`
  - `device:qa_mic`
  - `device:head_table_mic`
  - `device:speaker_monitor`
  - `device:laser_pointer`
  - `device:wireless_handheld_mic`
  - `device:wireless_lavalier_mic`
  - `device:standing_mic`
  - `device:push_to_talk_mic` (28 of these for the Board of Directors meeting)
  - `device:digital_signage` (future consideration)
  - `device:audience_response_system` (future consideration)
- **Software/licenses**: `service:zoom_capability`, `service:content_capture`, `service:presentation_management_system`

### Expected packets

| Family | Anchor | Status | Why |
|---|---|---|---|
| `scope_inclusion` | `room:keynote_general_session` (5,000 → 1,200 progressively) | active | Four general sessions over four days, decreasing room set. |
| `scope_inclusion` | `room:breakout_session` (12 + 8 + 20) | active | Classifies by day. Per-day matrix from page 4. |
| `scope_inclusion` | `room:micro_certification_classroom` (8 rooms × 50 ppl, multiple time slots) | active | Saturday 8am-12pm, 8:30-12pm, 1-4:30pm; Sunday 8am-12pm. |
| `scope_inclusion` | `service:zoom_capability` | active | "Zoom capability for some meetings and breakout education sessions". |
| `scope_inclusion` | `service:content_capture` | active | Keynote + breakout content capture explicit. |
| `scope_exclusion` | `service:office_equipment_existing_provider` | active | "ICMA has a provider who provides equipment for offices as well as monitors needed" — explicitly out of scope for AV vendor. |
| `scope_exclusion` | `service:affiliate_av_at_own_expense` | active | "Affiliate events/ICW may choose to use ICMA's audio-visual supplier at their own expense" — vendor scope is bounded to ICMA-paid events. |
| `customer_override` | `pricing:transaction_or_fixed_or_loe` | active | Section 4: "transaction level, fixed-fee, level of effort rate subject to a maximum not to exceed fee" — flexible pricing. |
| `missing_info` | `pricing:labor_rates_per_city` | needs_review | Pages 7–8: blank labor-rate matrix per city × time category (Straight/OT/Double/Min Call) for vendor to fill. |
| `missing_info` | `cost:back_up_equipment_percentage` | needs_review | Q9 in submission: "What percentage of back-up equipment does the company customarily take" — open. |
| `site_access` | `site:tampa_convention_center` | active | Multiple site visits required (Section: Pre-Conference Site Inspections, "minimum of 2 pre-conference site inspections" — vendor pays). |
| `action_item` | `vendor:floor_plans_at_no_cost` | active | Q14: "Will you provide scale floor plans for audiovisual setups at no cost?" — vendor commits as part of bid. |
| `meeting_decision` | `decision:single_award_anticipated` | active | "ICMA anticipates awarding one (1) single award" stated upfront. |

### Expected constraint patterns (gap candidates)

- **5-tier event time blocks** (Friday → Saturday → Sunday → Monday → Tuesday → Wednesday) — schedule constraint patterns are not currently in the AV pack.
- **Equipment per-room matrices** (12 rooms with 8 devices each, 8 rooms with slightly different config, 1 board room with 28 push-to-talk mics) — the parser should extract these as **table rows in the structured doc**, not as flat atoms. Each row → table cell → atom should preserve room×device×count.
- **"Game changers (mini-keynotes)"** — non-standard term for mid-tier keynote. Gap candidate.
- **"Speaker Ready Room"** with "Presentation Management Systems" — terminology specific to conference AV. Gap candidate.

### Expected exclusion patterns

- "at their own expense" → ICMA scope excluded for affiliate events
- "ICMA has a provider who provides" → office equipment out-of-scope
- "Future considerations" (digital signage + audience response systems) → soft-excluded for now

### Expected vendor mentions

None named explicitly — this is a *vendor-selection* RFP, so vendor names appear only in references. The bidder companies are not yet known.

### Stress-test attributes

- **Multi-city scope (4 cities × 4 years)** — every device count needs to be replicated 4 times (or normalized per-event). The parser should handle this without emitting 4× duplicate atoms.
- **Schedule embedded in narrative text** — "Friday: 8am-5pm ICMA Executive Board Meeting (in Hotel)" etc. is a multi-line block of time-slot text. Tests whether the structured doc preserves the hierarchy (Day → Time → Room → Activity).
- **Empty rate cards (pages 7–8)** — labor rate matrix is a 4×3 grid of blank cells. Parser should NOT generate atoms for empty cells; should emit the column/row headers as a `table` block with empty rows.
- **Tabular pricing requirements without numbers** — Section 4: "transaction level, fixed-fee, level of effort rate" — these are *pricing structure options*, not pricing values. Parser must distinguish.
- **Submission-form questions (29 numbered Qs in Section 3)** — each question is *vendor-asked*, not *customer-authored*. They reveal scope but should not become `customer_current_authored` atoms; they're closer to `formal_rfp` questions.
- **Mixed device types** — single PDF mentions everything from podium mics to push-to-talk to LCD projectors to laser pointers to walkie-talkies. The new `av_pack.yaml` should match all of these; the gap detector should be quiet on AV devices but loud on conference-specific terms.

### Expected metrics

```
expected_min_atom_count: 70
expected_min_packet_count: 9
expected_min_devices_in_atoms: 14   # the AV equipment matrix
expected_min_rooms_in_atoms: 7      # boardroom, breakout, keynote, etc.
expected_min_sites_in_atoms: 3      # Tampa, Long Beach, Toronto
```
