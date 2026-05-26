# Parser-OS + OrbitBrief — Complete UI Output Catalog

Every field your UI can read, with real OPTBOT values + audit / PM
classification. Built for two front-ends:

1. **AUDIT dashboard** — "how good was this run?" (next month's project review)
2. **PM brief** — "what do I do with this deal?" (per-deal consumption)

Last verified: OPTBOT compile produces these 17 files in `<out>/`:

```
00_envelope.json              527 KB   raw parser-os evidence
10_pack_prior_state.json        7 KB   OrbitBrief pack activation
11_site_reality_state.json      4 KB   OrbitBrief site clusters
90_inspection_report.json     493 KB   per-artifact funnel
91_inspection_report.html     230 KB   browsable HTML version
PM_HANDOFF.json               147 KB   ← PRIMARY UI PAYLOAD
PM_HANDOFF.md                  53 KB   rendered markdown view
PM_HANDOFF.html                61 KB   styled HTML
SOW_DRAFT.md                   16 KB   21-section auto SOW
RFP_DRAFT.md                   10 KB   vendor RFP packets
SA_REVIEW_PACKET.md/.html      19 KB   SA technical view
PM_EXECUTIVE_SUMMARY.md/.html   5 KB   1-page exec brief
manifest.json                 0.4 KB   run audit metadata
pipeline_log.json               7 KB   per-stage telemetry
.orbitbrief_history.jsonl     0.7 KB   append-only deal corpus
```

**The UI only needs to fetch 3 files**: `PM_HANDOFF.json` (PM brief
tabs), `00_envelope.json` (audit drilldown), `manifest.json` (run
header). `PM_HANDOFF.json` already embeds `sow_draft_markdown` +
`rfp_draft_markdown` so SOW/RFP viewers don't need separate fetches.

---

# AUDIT DASHBOARD — "how good was this run?"

These are the fields a project-quality auditor needs.

## A1. Headline KPI: `PM_HANDOFF.json.parser_quality_score`

**One field tells you everything.** OPTBOT:

```json
{
  "score": 99,
  "grade": "A+",
  "components": {
    "parse_outcome_ok_pct":  100.0,   // 7/7 files parsed cleanly
    "parse_outcome_pts":      40.0,   // out of 40
    "receipt_verified_pct":   97.8,   // 132/135 atoms passed replay
    "receipt_pts":            24.4,   // out of 25
    "error_free_pts":         15,     // 0 errors
    "n_errors":                0,
    "warning_health_pts":     10.0,   // 0 warnings
    "n_warnings":              0,
    "authority_diversity_pts":10,     // 6 authority classes mixed
    "authority_classes_seen": ["approved_site_roster", "contractual_scope",
                               "customer_current_authored", "machine_extractor",
                               "meeting_note", "vendor_quote"]
  }
}
```

**UI:** giant score gauge + grade letter + 5-component breakdown chart.

## A2. Run metadata: `manifest.json`

```json
{
  "envelope_path":         "...",
  "generated_at":          "2026-05-21T00:30:13.778414Z",
  "active_packs":          ["wireless", "delivery_execution", "procurement_finance"],
  "brains_run":            [],
  "queued_for_review":     0,
  "skipped_brains_no_chat":true,
  "stage_count":           20,
  "stage_status_counts":   {"ok": 7, "skipped": 13}
}
```

**UI:** run timeline + active-packs chips.

## A3. Corpus signal: `00_envelope.json.summary` (top-level)

```json
{
  "artifact_count":              7,
  "page_count":                  20,
  "atom_count":                  135,
  "packet_count":                28,
  "entity_count":                79,
  "edge_count":                  172,
  "cross_artifact_edge_count":   84,    // ← STRONG cross-doc reconciliation signal
  "by_artifact_type":  {"pdf": 3, "docx": 2, "xlsx": 2},
  "by_atom_type":      {"scope_item": 65, "quantity": 16, "vendor_line_item": 16,
                        "entity": 15, "constraint": 11, "risk": 5,
                        "exclusion": 3, "decision": 2, "open_question": 1,
                        "assumption": 1},
  "by_authority_class":{"contractual_scope": 48, "meeting_note": 32,
                        "vendor_quote": 32, "approved_site_roster": 17,
                        "customer_current_authored": 5,
                        "machine_extractor": 1},
  "by_entity_type":    {"device": 20, "approved_site_roster": 17,
                        "money": 15, "part": 10, "stakeholder": 7,
                        "date": 6, "service": 5, "site": 5,
                        "milestone": 5, "address": 3, "part_number": 2,
                        "customer": 1},
  "by_edge_type":      {"supports": 110, "same_as": 39, "requires": 22,
                        "excludes": 1},
  "parse_outcomes":    {"ok": 7},                      // ← A6 graceful degradation signal
  "degraded_files":    []                              // ← red-flag list when non-empty
}
```

**UI:** stacked-bar donuts for each `by_*` map; KPI tiles for the
scalar counts; red-flag callout when `degraded_files` is non-empty.

## A4. Per-file health: `00_envelope.json.documents[*]`

Each document carries 10 fields. Sample OPTBOT row:

```json
{
  "artifact_id":     "art_1aca83887728bf98",
  "filename":        "07_contracting_procurement_packet.pdf",
  "artifact_type":   "pdf",                 // 19 enum values: pdf/docx/xlsx/csv/email/transcript/pptx/image/html/mbox/rtf/ics/zip/msg/odt/ods/vsdx/mpp/txt
  "sha256":          "65b82ad4d492964c...",  // change-detection hash
  "size_bytes":      5047,
  "parser_name":     "orbitbrief_pdf",
  "parser_version":  "orbitbrief_pdf_v3",
  "structured":      {...},                   // full per-doc projection
  "atom_ids":        [9 atom IDs],
  "parse_outcome":   {                          // ← A6 status column
      "status":         "ok",                  // ok / ok_empty / failed_parse / skipped_no_parser
      "atom_count":     9,
      "warning_count":  0,
      "cache_hit":      false
  }
}
```

**UI:** source-inventory table with type chip, parser badge, atom
count, status pill (✅/⚠️/❌/⏭️), size column.

## A5. Per-stage telemetry: `pipeline_log.json`

Each row = one stage of the parser-os pipeline:

```
discover_artifacts → parse_artifacts → candidate_adjudication → source_replay
→ confidence_floor → enrich_entities → entity_resolution → graph_build
→ packetize → packet_certificates → quality_gates
```

Per-stage fields: `stage`, `duration_ms`, `counts.input_count`,
`counts.output_count`, `warning_count`, `error_count`,
`status` (OK / FALLBACK / SKIPPED).

**UI:** funnel diagram showing input→output count per stage; bar
chart for stage durations; per-stage drilldown panel.

## A6. Replay verification: `90_inspection_report.json.verification`

```json
{
  "atom_total":             135,
  "counts":                 {"verified": 132, "unsupported": 1, "failed": 2},
  "verified_count":         132,
  "failed_count":           2,
  "partial_count":          0,
  "unverified_count":       0,
  "unsupported_count":      1,
  "verified_pct":           97.8,
  "failed_pct":             1.5,
  "partial_pct":            0.0,
  "health_pct":             97.8,
  "top_failed_artifacts":   [{"artifact_id": "...", "filename": "...",
                              "failed_atoms": 2, "atom_count": 40}]
}
```

**UI:** verification donut + "top failed artifacts" drilldown table.

## A7. Funnel rollup: `90_inspection_report.json.funnel`

```json
{
  "source_artifacts":         7,
  "atoms_extracted":          135,
  "entities_normalized":      79,
  "edges_built":              172,
  "packets_certified":        28,
  "active_packs":             [15 pack IDs],
  "bundled_packets_total":    23,
  "bundled_packets_per_pack": {...},
  "brain_items_per_pack":     {...},
  "brain_cited_packets":      0,    // 0 because --ollama wasn't used
  "brain_cited_atoms":        0,
  "composed_brief_items":     0,
  "atoms_to_brief_pct":       0,
  "packets_to_brief_pct":     0,
  "pack_prior_top":           "wireless",
  "pack_prior_margin":        0.83
}
```

**UI:** Sankey diagram from "atoms extracted" → "in packet" → "cited
by brain" → "in composed brief"; pack-prior chart.

## A8. Drill-down: `00_envelope.json.atoms[*]`

135 atoms on OPTBOT. Each has 12 fields. Audit drilldown table
columns:

| Field | Purpose | Sample |
|---|---|---|
| `id` | Atom ID | `atm_03774208dd92edcc` |
| `artifact_id` | Source doc | `art_8a3ff646a0735900` |
| `atom_type` | Classification | `quantity` (one of 22 types) |
| `authority_class` | Trust tier | `vendor_quote` (one of 6+ classes) |
| `confidence` | 0.0-1.0 | `0.88` |
| `text` | Display | `Quantity 1` |
| `section_path` | Breadcrumb | `["Services"]` |
| `locator` | Source link | `{"sheet": "Services", "row": 3, "columns": {...}}` |
| `verified` | Replay status | `verified` (verified/failed/partial/unsupported/unverified) |
| `entity_keys` | Tag pills | `["service:project_management_and_weekly_governance"]` |
| `structured` | Per-type fields | `{quantity_raw, uom, quantity, unit, ...}` (23 fields) |

**UI:** sortable/filterable table with confidence heat color,
authority badge, verified pill, click-through to source.

## A9. Cross-atom graph: `00_envelope.json.edges[*]`

172 edges on OPTBOT. Each has 8 fields. **The relationship signal:**

```json
{
  "id":            "edge_00a3b5c29f295fc6",
  "edge_type":     "supports",     // supports / contradicts / same_as /
                                  //  requires / excludes / refines /
                                  //  quantity_conflict
  "from_atom_id":  "atm_e7de...",
  "to_atom_id":    "atm_427b...",
  "reason":        "Cross-artifact reinforcement on money:1500000",
  "confidence":    0.78,
  "cross_artifact": true,
  "metadata":      {family, target_keys, ...}
}
```

**UI:** force-directed graph with edge-type color, click-an-atom for
neighborhood; cross-artifact filter chip.

## A10. Resolved entities: `00_envelope.json.entities[*]`

79 entities on OPTBOT. Each has 9 fields:

| Field | Sample |
|---|---|
| `id` | `ent_01705d7119adafbd` |
| `entity_type` | `device` (or: site, stakeholder, money, date, part, vendor, customer, service, milestone, address) |
| `canonical_key` | `device:rugged_logistics_tablet` |
| `canonical_name` | `rugged logistics tablet` |
| `aliases` | Multiple surface forms |
| `artifact_ids` | Which files mention it |
| `source_atom_ids` | Atoms that produced it |
| `review_status` | `auto_accepted` / `needs_review` |
| `confidence` | 0.0-1.0 |

**UI:** entity browser grouped by entity_type with tag pills.

## A11. Pack-activation prior: `10_pack_prior_state.json`

Which OrbitBrief packs the router decided to activate (each pack
= one workstream lens). OPTBOT activated 15 packs with `wireless`
at top.

**UI:** pack-prior bar chart with margin badge.

## A12. Recommended AUDIT dashboard layout

```
┌─────────────────────────────────────────────────────────────┐
│ Quality Score: A+ 99/100                                   │
│ ┌─Score Gauge──┐ ┌─Component Breakdown─────────────────┐   │
│ │      99      │ │ parse_outcome  ████████ 40/40 (100%)│   │
│ │     A+       │ │ receipt_verify ████▌    24.4/25     │   │
│ └──────────────┘ │ error_free     ███      15/15 (0)   │   │
│                  │ warning_health ██       10/10 (0)   │   │
│                  │ authority_div  ██       10/10 (6cls)│   │
│                  └─────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│ Run badge: cmp_xxx · OPTBOT · 2026-05-21T00:30:13Z         │
├─────────────────────────────────────────────────────────────┤
│ KPI tiles: 7 files · 20 pages · 135 atoms · 28 packets ·   │
│            79 entities · 172 edges · 84 cross-artifact     │
├─────────────────────────────────────────────────────────────┤
│ Atoms by Type | Atoms by Authority | Entities by Type     │
│ (donut)       | (donut)            | (donut)              │
├─────────────────────────────────────────────────────────────┤
│ Source Inventory (per-file): name · type · parser ·       │
│ atoms · status · size · sha256                            │
├─────────────────────────────────────────────────────────────┤
│ Pipeline Funnel (Sankey): atoms→packets→bundled→cited     │
├─────────────────────────────────────────────────────────────┤
│ Stage timeline: bar chart per stage                       │
├─────────────────────────────────────────────────────────────┤
│ Drift tracker: input_signature & output_signature vs prev │
│ run; if input same but output different → parser regress  │
└─────────────────────────────────────────────────────────────┘
[Drill-down: Atoms · Entities · Edges · Packets]
```

---

# PM BRIEF — "what do I do with this deal?"

Single payload: `PM_HANDOFF.json`. 54 top-level fields. Here's every
single one with its real OPTBOT value:

## P1. Header & status (5 fields)

| Field | Type | OPTBOT value | UI hint |
|---|---|---|---|
| `case_id` | str | `OPTBOT_Atlanta_Office_Refresh_Mock_Deal` | Header title |
| `status` | enum | `red` | Status traffic light |
| `status_label` | str | `Not SOW-ready: 5 blocker question(s) remain` | Header subtitle |
| `one_line_summary` | str | `OPTBOT_..._Mock_Deal: Security camera / VMS, Sites / facilities, Commercial terms, Electrical / power at airport logistics annex, atl hq; 5 blocker and 7 warning SOW question(s) need PM/SA review.` | Top banner |
| `executive_summary` | dict (3 keys) | See below | Above-the-fold callout |

`executive_summary` on OPTBOT:
```json
{
  "headline":    "OPTBOT_..._Mock_Deal: deal worth $1,847,250 across 3 confirmed site(s) covering Security camera / VMS, Sites / facilities, Commercial terms.",
  "health_line": "Status is RED: 5 blocker(s) and 7 warning(s) need PM resolution before SOW lock.",
  "next_action": "Resolve the blocker checklist below and confirm the customer clarifications email starter. Do not publish a SOW until blockers clear."
}
```

## P2. Scorecard: `metrics` (11 fields)

```json
{
  "blockers":                 5,
  "warnings":                 7,
  "info":                     1,
  "evidence_groups_certified":28,
  "evidence_items_extracted": 135,
  "missing_sow_items":        13,
  "pm_visible_fact_cards":    56,
  "sites_published":          3,
  "source_files":             7,
  "sow_validator_status":     "red",
  "top_workstream":           "Wireless / WLAN"
}
```

**UI:** KPI tile row.

## P3. Intake quality (2 fields)

`intake_completeness` (10 items, OPTBOT scores 9/10):
```
[YES] Confirmed contract value
[YES] At least one confirmed physical site
[YES] Project schedule with start + end dates
[YES] Named executive sponsor (stakeholder)
[YES] Hardware BOM or vendor quote
[YES] Risk register
[YES] Acceptance criteria definition
[NO ] Payment terms and pricing model
[YES] Out-of-scope / exclusions list
[YES] Compliance / MSA / NDA reference
```

`ocr_backend_status`:
```json
{
  "available":     [],
  "install_hints": ["Install Tesseract...", "pip install pytesseract...", "ollama pull llava..."]
}
```

**UI:** completeness checklist + progress bar; OCR-status chip with
install dropdown when empty.

## P4. Money & commercial (10 fields)

| Field | OPTBOT count | What's in it |
|---|---:|---|
| `money_mentions` | 16 | Each money value across docs |
| `reconciliation_flags` | 2 | $1.85M vs $1.5M flagged |
| `currency_mentions` | 0 | (no non-USD) |
| `currency_conversions` | 0 | (USD-only deal) |
| `tax_clauses` | 0 | |
| `margin_view` | dict (9) | See below — **$1.8M zero-margin SOW caught** |
| `engagement_model` | dict (7) | T&M / Fixed Fee / Subscription detection |
| `license_items` | 2 | Recurring software tracker |
| `eol_flags` | 0 | (none triggered) |

`margin_view` on OPTBOT (the most important finance signal):
```json
{
  "deal_total":             1847250,
  "hardware_cost_subtotal": 1015626,
  "services_subtotal":       536030,
  "other_cost_subtotal":     295594,
  "total_cost":             1847250,
  "gross_profit":                  0,
  "margin_pct":                  0.0,
  "confidence":             "high",
  "notes": ["⚠ Zero-margin SOW: deal total exactly matches computed cost. PM should add margin or confirm this is intentional (pass-through pricing)."]
}
```

**UI:** finance cards. Margin gauge with red-band <15%; reconciliation
"needs attention" panel; license tracker table.

## P5. Sites & per-site rollups (3 fields)

`sites` (3 confirmed on OPTBOT): `airport logistics annex`, `atl hq`, `atl west`.

`site_rollups` (4 items, one per site + per-alias) carrying:
- `atom_count`
- `devices` (e.g. access point, ip camera, switch)
- `money_values` (e.g. `["$18,500", "$6,125", ...]`)
- `dates` (e.g. `["2026-05-20", ...]`)
- `stakeholders` (e.g. `["Jordan Ames", "Priya Narang", ...]`)

`site_allocations` (3): per-site BOM math —
```
ATL-HQ:    52 × $995 = $51,740
ATL-WEST:  27 × $995 = $26,865
ATL-AIR:   15 × $995 = $14,925
```

**UI:** site cards + per-site coverage matrix + computed BOM rollup
table.

## P6. Stakeholders (2 fields)

`stakeholder_contacts` (6 contacts captured on OPTBOT — Jordan Ames,
Priya Narang, Elliot Tran, Camila Brooks, Noah Patel, Renee Watkins).
Each: `name`, `role`, `email`, `phone`, `source`.

`stakeholder_pagers` (3 lenses: CFO / IT / Procurement) — each with
`summary_lines`, `money_lines`, `risk_lines`, `action_lines` filtered
to that role.

**UI:** contact directory table + 3-tab one-pager browser.

## P7. Schedule (5 fields)

| Field | OPTBOT count | Renders as |
|---|---:|---|
| `schedule_phases` | 6 | Gantt + table |
| `critical_path` | 6 | Critical-path overlay |
| `critical_path_chain` | 6 | Phase chain string |
| `phase_dependencies` | 0 | (no explicit deps detected) |
| `resource_conflicts` | 0 | Owner-overlap alerts |
| `lead_time_flags` | 0 | BOM lead-time risk |

OPTBOT schedule (6 phases, all critical):
```
1. Discovery and intake    2026-05-20 → 2026-05-29   Renee Watkins
2. Design validation       2026-06-01 → 2026-06-12   Priya Narang
3. Procurement and staging 2026-06-15 → 2026-07-03   Elliot Tran
4. Site implementation     2026-07-06 → 2026-07-24   Noah Patel
5. Cutover and adoption    2026-07-27 → 2026-07-31   Jordan Ames
6. Post go-live            2026-08-03 → 2026-08-14   Renee Watkins
```

**UI:** Mermaid Gantt + critical-path highlighter + resource-conflict
red-flag table.

## P8. Risks (2 fields)

`risk_register` (5 risks on OPTBOT) — each with `risk_id`, `description`,
`likelihood`, `impact`, `mitigation`, `owner`, `sites`, `source`.

`risk_aging` — suppressed on OPTBOT because intake date = today.

**UI:** risk table sorted by L×I score; risk aging buckets (fresh /
active / stale) when timestamps land.

## P9. Compliance & legal (3 fields)

| Field | OPTBOT count | What's in it |
|---|---:|---|
| `compliance_callouts` | 3 | SOC 2 / MSA / Legal-review-required refs |
| `sla_penalties` | 0 | Liquidated damages / SLA credits |
| `change_order_triggers` | 1 | "Substitutions require written approval..." |

**UI:** compliance routing table + change-order pre-flag panel.

## P10. Scope structure (8 fields)

| Field | OPTBOT count |
|---|---:|
| `exclusions` | 3 |
| `responsibilities` | 1 (customer vs provider split) |
| `quantity_claims` | 10 |
| `quantity_contradictions` | 0 |
| `acceptance_checks` | 14 |
| `acceptance_by_site` | dict[1] |
| `domains` (workstreams) | 9 |
| `subcontractor_mentions` | 8 (Cisco / Meraki / Aruba / etc.) |

**UI:** scope tabs (in-scope / exclusions / customer-vs-provider /
acceptance) + workstream router visualization.

## P11. PM action queue (4 fields)

| Field | OPTBOT count |
|---|---:|
| `gaps` | 13 (5 blocker + 7 warning + 1 info) |
| `customer_questions` | 12 (filtered for PM-to-customer email) |
| `action_items` | 21 (consolidated from gaps + risks + phases) |
| `actions_by_week` | dict (this_week / next_week / later / no_date) |

`actions_by_week` on OPTBOT (today = 2026-05-21):
- `this_week`: 1 phase kickoff
- `next_week`: 1 phase kickoff
- `later`: 4 phase kickoffs
- `no_date`: 15 actions (blockers + warnings + risks)

**UI:** action checklist + week-bucket tabs + customer-email
clipboard-copy.

## P12. Cross-doc reconciliation (4 fields)

| Field | OPTBOT count |
|---|---:|
| `money_mentions` | 16 (per-value sources) |
| `reconciliation_flags` | 2 ($1.85M vs $1.5M caught) |
| `date_mentions` | 16 (12 are cross-doc) |
| `quantity_contradictions` | 0 (would surface "94 APs vs 92 APs" if real) |

**UI:** reconciliation queue panel with "needs PM resolution" badges.

## P13. Output deliverables embedded in PM_HANDOFF.json (3 fields)

| Field | OPTBOT size | What's in it |
|---|---:|---|
| `sow_draft_markdown` | 16,090 chars | 21-section auto-drafted SOW |
| `rfp_draft_markdown` | 9,974 chars | Categorized vendor RFP packets |
| `rfp_line_items` | 16 items | Structured BOM data backing the RFP |

**UI:** SOW / RFP viewer tabs with markdown render + "copy to
clipboard" + download buttons.

## P14. Strategic (1 field)

`comparable_deals` — historical bench match. OPTBOT corpus has 1 prior
entry (itself, from earlier run). Each row:

```json
{
  "case_id":          "OPTBOT_...",
  "closed_at":        "",
  "deal_value_usd":   1847250,
  "domains":          ["Security camera / VMS", ...],
  "sites_count":      3,
  "phase_count":      6,
  "final_margin_pct": 0.0,
  "outcome":          ""
}
```

**UI:** "similar past deals" panel; grows over time as the corpus
accretes.

## P15. Universality / source provenance (3 fields)

`source_files` (7 on OPTBOT) — each with `filename`, `artifact_type`,
`parser_name`, `evidence_items`, `status`, `status_reason`.

`facts_by_category` (8 categories on OPTBOT) — evidence cards grouped
by `sites_access`, `scope_deliverables`, `bom_procurement_pricing`,
`network_vlans_circuits`, etc.

`sa_focus` (4 SA-owned items).

**UI:** source-inventory table (audit chain) + facts browser +
SA-review-lane tabs.

## P16. Domain detection (1 field)

`domains` (9 on OPTBOT) — each row: `domain_id`, `label`,
`selected_by_router`, `active_for_sow`, `blockers`, `warnings`, `info`.

OPTBOT activated:
```
Security camera / VMS         (active, 3 blockers, 3 warnings)
Sites / facilities            (active, 1 blocker, 2 warnings)
Commercial terms              (active, 1 blocker)
Electrical / power            (active, 0 blockers, 1 warning)
Procurement / finance         (active, 0 blockers, 1 warning)
Delivery / execution planning (active)
Hardware / equipment          (active)
Wireless / WLAN               (selected, not active for SOW)
Global                        (none)
```

**UI:** workstream chips with severity badges.

---

# OPTBOT — Concrete digestibility check

PM_HANDOFF.json is **147 KB** with **54 top-level fields**. Of those:

| Group | Count |
|---|---:|
| Empty on OPTBOT (suppress in UI) | 9 fields |
| Single-value scalars (status / metrics) | 5 fields |
| Lists with 1-5 items (table-sized) | 18 fields |
| Lists with 6-20 items (still scannable) | 14 fields |
| Lists with 21+ items (need filter/search) | 3 fields (`action_items` 21, `acceptance_checks` 14, `quantity_claims` 10) |
| Dicts | 5 fields |

**Above-the-fold (first scroll):** exec summary + intake completeness
+ margin warning. PM gets the signal in ~10 seconds.

**Densest panels** (will need pagination / lazy load in UI):
- Money mentions: 16 rows
- Subcontractor mentions: 8 rows
- Source files: 7 rows + drilldowns
- Risk register: 5 rows
- Acceptance criteria: 14 rows

Everything is **JSON-serializable**, **list-or-dict shaped**, and **no
nested-string-parsing needed**. UI is pure data binding.

---

# What's READY for UI and what's MISSING

## Ready (51 of 54 fields, fully data-bound)
- All scalar fields
- All list-of-dict fields where every row is a clean record
- Both embedded markdown deliverables (SOW + RFP)
- Quality score + components

## Almost-ready (3 fields, minor polish)
- `acceptance_by_site` — keyed by site code, UI needs a per-site tab
- `actions_by_week` — keyed by week bucket, UI needs ordered tabs
- `stakeholder_pagers` — 3 fixed entries (CFO/IT/Procurement)

## Genuinely missing (not in JSON yet)
- **Per-stage durations rolled into PM brief** (currently only in
  `pipeline_log.json`) — would let the UI show "this brief took 8s
  to produce" badges
- **Drift signals** — `input_signature` / `output_signature` from
  `manifest.json` aren't in PM_HANDOFF.json yet; would enable a
  per-deal "changed vs last run" banner
- **Confidence histograms** for atoms — would help auditors see
  whether the corpus is high-confidence or borderline
- **Brain run results** when LLM stages run (`brains_run`,
  `40_brain_outputs/`) — empty on OPTBOT because `--ollama` wasn't
  used, but UI should plan for this when LLM stages fire

Each is ~30 min of work. None blocks UI development today.

---

# UI scaffolding — recommended tab structure

```
┌─ AUDIT DASHBOARD (per project) ─────────────────────────┐
│ Top: Quality Score gauge + grade letter                │
│ Tabs:                                                  │
│   • Run metadata        (manifest.json)                │
│   • Source inventory    (envelope.documents)           │
│   • Pipeline funnel     (pipeline_log + funnel)        │
│   • Atom browser        (envelope.atoms)               │
│   • Entity browser      (envelope.entities)            │
│   • Edge graph          (envelope.edges)               │
│   • Packet browser      (envelope.packets)             │
│   • Drift tracker       (signatures vs prior runs)     │
└────────────────────────────────────────────────────────┘

┌─ PM BRIEF (per project) ────────────────────────────────┐
│ Top: Executive summary + intake completeness + status  │
│ Tabs:                                                  │
│   • Overview            (metrics + workstreams)        │
│   • Sites               (sites + rollups + allocation) │
│   • Stakeholders        (contacts + 3 one-pagers)      │
│   • Money & Margin      (margin_view + mentions)       │
│   • Reconciliation      (money + quantity + date)      │
│   • Schedule            (gantt + critical path)        │
│   • Risks               (register + aging)             │
│   • Scope               (in/out + responsibilities)    │
│   • Acceptance          (checks + per-site)            │
│   • Compliance & Legal  (callouts + SLA + CO triggers) │
│   • Subcontractors      (mentions + vendor list)       │
│   • Lead-time + EOL     (BOM lifecycle alerts)         │
│   • Actions             (consolidated + by-week)       │
│   • SOW Draft           (sow_draft_markdown)           │
│   • RFP Draft           (rfp_draft_markdown)           │
│   • Customer Email      (drafted clarification email)  │
└────────────────────────────────────────────────────────┘

┌─ HISTORICAL / PORTFOLIO ────────────────────────────────┐
│   • Run history         (.orbitbrief_history.jsonl)    │
│   • Comparable deals    (per-project)                  │
│   • Portfolio rollup    (cross-project KPIs)           │
└────────────────────────────────────────────────────────┘
```

## Pulling the data

```python
# Audit dashboard
manifest  = json.load(open("manifest.json"))
envelope  = json.load(open("00_envelope.json"))
pipeline  = json.load(open("pipeline_log.json"))
inspect   = json.load(open("90_inspection_report.json"))

# PM brief — ONE file
handoff   = json.load(open("PM_HANDOFF.json"))
# handoff.sow_draft_markdown and handoff.rfp_draft_markdown
# are already embedded; no extra fetches.
```

That's the complete catalog. Every field your UI can read, every
real value it produces on OPTBOT, every audit signal, every PM
signal, every gap.
