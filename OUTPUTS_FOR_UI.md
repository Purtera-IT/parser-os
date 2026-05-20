# Parser-OS + OrbitBrief output schema — UI mapping guide

This document catalogs every field that lands on disk after a single
``compile_brief`` run on a real-deal folder. Use it to plan the
auditing-dashboard UI for next month's review phase and the
PM-facing front-end.

Each output file is listed with:
- **What it is** (what role it plays for the PM / auditor)
- **Top-level fields** (shape of the JSON / sections of the markdown)
- **Per-row / per-atom fields** when applicable
- **UI mapping notes** (how to surface it, what's "ready" vs "needs aggregation")

The two top-level deliverables are:
1. **parser-os** — `orbitbrief.input.json` envelope (the evidence corpus)
2. **OrbitBrief** — `PM_HANDOFF.json` + companion markdown / HTML files

A typical OPTBOT run drops the following files into `<out>/`:

```
00_envelope.json           # parser-os envelope (full evidence corpus)
10_pack_prior_state.json   # OrbitBrief: pack activation + prior beliefs
11_site_reality_state.json # OrbitBrief: site cluster state
20_retrieval_bundles/      # OrbitBrief: per-brain retrieval bundles
40_brain_outputs/          # OrbitBrief: per-brain LLM outputs (when --ollama)
50_validations/            # OrbitBrief: rulebook validation results
60_calibrations/           # OrbitBrief: confidence calibration outputs
70_review_queue/           # OrbitBrief: items flagged for PM review
90_inspection_report.json  # OrbitBrief: per-artifact funnel view
91_inspection_report.html  # Same, browsable HTML
PM_EXECUTIVE_SUMMARY.md    # Short PM-facing brief
PM_EXECUTIVE_SUMMARY.html
PM_HANDOFF.md              # Full 31-section PM brief
PM_HANDOFF.html
PM_HANDOFF.json            # Same data, machine-readable
RFP_DRAFT.md               # Auto-drafted vendor RFP packets
SA_REVIEW_PACKET.md        # Solution-architect technical view
SA_REVIEW_PACKET.html
SOW_DRAFT.md               # 21-section auto-drafted SOW
manifest.json              # Run audit metadata (compile_id, signatures)
pipeline_log.json          # Per-stage telemetry
.orbitbrief_history.jsonl  # Append-only corpus for comparable-deals
```

---

# 1. PARSER-OS AUDIT SIGNALS — for your auditing dashboard

These are the fields a PM / auditor needs to answer **"how good was this run?"**

## 1.1 `manifest.json` — single-record run audit

| Field | Type | UI hint | Purpose |
|---|---|---|---|
| `project_id` | str | Header tag | The project this run analyzed |
| `compile_id` | str | "Run ID" badge | Unique compile identifier (e.g. `cmp_d0cfff3a0d7e556a`) |
| `input_signature` | hex | "Inputs hash" pill | SHA256 of input artifacts — if this changes, the inputs changed |
| `output_signature` | hex | "Outputs hash" pill | SHA256 of output atoms — if THIS changes with same input, parser regressed |
| `atom_count` | int | KPI card | Total evidence atoms extracted |
| `edge_count` | int | KPI card | Cross-atom relationships discovered |
| `packet_count` | int | KPI card | Certified evidence packets emitted |
| `warning_count` | int | KPI card (orange when > 5) | Parser warnings |
| `error_count` | int | KPI card (red when > 0) | Parser errors |
| `receipt_verified` | int | Coverage % vs `atom_count` | Atoms with verified source replay |
| `receipt_failed` | int | Failures bucket | Atoms with replay failure |
| `receipt_unsupported` | int | Coverage bucket | Atoms where replay isn't supported |
| `cache_hits` / `cache_misses` | int | Perf gauge | Incremental-compile cache efficiency |

## 1.2 `00_envelope.json` (a.k.a. `orbitbrief.input.json`) — evidence corpus

This is the canonical handoff from parser-os to any downstream consumer. Top-level shape:

| Field | Type | UI hint | Purpose |
|---|---|---|---|
| `schema_version` | str | "v2" tag | Envelope contract version |
| `project_id` | str | Header | Project identifier |
| `compile_id` | str | Header | Run identifier |
| `generated_at` | ISO timestamp | "Last run" | When the envelope was produced |
| `summary` | dict | Dashboard KPI block | See **1.2.1** below |
| `documents` | list[doc] | Source-inventory table | One row per ingested artifact |
| `atoms` | list[atom] | Detail drilldown | Every evidence atom — the bedrock |
| `packets` | list[packet] | "Packets" tab | Certified evidence packets |
| `entities` | list[entity] | "Entities" tab | Cross-artifact entity records |
| `edges` | list[edge] | Graph view | Cross-atom relationships |
| `indexes` | dict | Backing data | Pre-computed lookup tables |
| `drawings` | dict | (optional) | Schematic-page detail (when present) |

### 1.2.1 `envelope.summary` — top of the auditor dashboard

| Field | Type | UI hint | Purpose |
|---|---|---|---|
| `artifact_count` | int | KPI | Files ingested |
| `page_count` | int | KPI | Total pages processed (across PDFs) |
| `atom_count` | int | KPI | Atoms produced |
| `packet_count` | int | KPI | Packets certified |
| `entity_count` | int | KPI | Entities resolved |
| `edge_count` | int | KPI | Cross-atom edges |
| `cross_artifact_edge_count` | int | "Cross-doc reconciliation" KPI | Edges spanning multiple artifacts |
| `by_artifact_type` | dict[str, int] | Stacked bar / donut | Atoms broken out by file type |
| `by_atom_type` | dict[str, int] | Stacked bar | Atoms by type (scope_item, risk, ...) |
| `by_authority_class` | dict[str, int] | Quality gauge | Distribution by authority (verified vs machine vs LLM) |
| `by_edge_type` | dict[str, int] | Stacked bar | Edges by relationship kind |
| `by_entity_type` | dict[str, int] | Stacked bar | Entities by type (site / stakeholder / device / etc.) |
| **`parse_outcomes`** | dict[str, int] | Coverage donut | Per-file outcome status (ok / ok_empty / failed_parse / skipped_no_parser) — **A6 graceful degradation signal** |
| **`degraded_files`** | list[dict] | Red-flag table | Each failed/skipped file with reason |

### 1.2.2 `envelope.documents[*]` — per-file row

| Field | Type | UI hint |
|---|---|---|
| `artifact_id` | str | Row ID |
| `filename` | str | Display name |
| `artifact_type` | enum | Type chip (pdf / docx / xlsx / pptx / image / html / mbox / rtf / ics / zip / msg / odt / ods / vsdx / mpp / email / transcript / csv / txt) |
| `sha256` | hex | Hash chip |
| `size_bytes` | int | Size column |
| `parser_name` | str | Parser badge |
| `parser_version` | str | Hover detail |
| `structured` | dict | (drilldown) Full structured projection — pages / sections / tables |
| `atom_ids` | list[str] | Count + drilldown |
| **`parse_outcome`** | dict | Status column with reason: `{"status": "ok\|ok_empty\|failed_parse\|skipped_no_parser", "atom_count": N, "warning_count": N, "reason": "..."}` |

### 1.2.3 `envelope.atoms[*]` — the evidence row

| Field | Type | UI hint |
|---|---|---|
| `id` | str | Atom ID |
| `artifact_id` | str | Link to source doc |
| `atom_type` | enum | Type chip (quantity / entity / constraint / exclusion / scope_item / customer_instruction / vendor_line_item / assumption / open_question / decision / action_item / meeting_commitment / risk / asset_record / support_entitlement / site_roster / lifecycle_status / form_option_state / project_metadata / site_survey_row / port_vlan_assignment) |
| `authority_class` | enum | Authority badge (customer_current_authored / vendor_quote / meeting_note / machine_extractor / ...) |
| `confidence` | 0..1 | Heat color |
| `text` | str | Display |
| `section_path` | list[str] | Breadcrumb |
| `locator` | dict | Source link (page / sheet / row / paragraph / etc.) |
| `verified` | enum | Replay status (verified / failed / partial / unsupported / unverified) |
| `entity_keys` | list[str] | Tag pills (`site:atl_hq`, `money:1847250`, ...) |
| `structured` | dict | Per-atom-type structured fields (canonical_cells for risk rows, etc.) |

### 1.2.4 `envelope.packets[*]` — certified evidence packets

| Field | Type | UI hint |
|---|---|---|
| `id` | str | Packet ID |
| `family` | enum | Family chip (scope_inclusion / scope_exclusion / customer_override / meeting_decision / action_item / site_access / missing_info / compliance_clause / quantity_claim / quantity_conflict / vendor_mismatch) |
| `status` | enum | Certification chip (certified / hold / dropped) |
| `governing_atom_ids` | list[str] | Linked atoms |
| `supporting_atom_ids` | list[str] | Linked atoms |
| `entity_keys` | list[str] | Tag pills |
| `risk_score` | float | Risk heat |
| `claim_hash` | hex | Dedup key |
| `policy_decisions` | list[dict] | Why-was-this-certified breakdown |

### 1.2.5 `envelope.entities[*]` — resolved entities

| Field | Type | UI hint |
|---|---|---|
| `id` | str | Entity ID |
| `entity_type` | enum | site / stakeholder / device / part / vendor / money / date / quantity / service / customer |
| `canonical_key` | str | The key (e.g. `site:atl_hq`) |
| `canonical_name` | str | Display name |
| `aliases` | list[str] | Other surface forms |
| `artifact_ids` | list[str] | Where it appears |
| `source_atom_ids` | list[str] | Atoms that mention it |
| `confidence` | float | Resolution confidence |
| `review_status` | enum | auto_accepted / needs_review |

### 1.2.6 `envelope.edges[*]` — relationships

| Field | Type | UI hint |
|---|---|---|
| `id` | str | Edge ID |
| `edge_type` | enum | supports / contradicts / same_as / requires / excludes / refines / quantity_conflict |
| `from_atom_id` / `to_atom_id` | str | Graph nodes |
| `confidence` | float | Edge weight |
| `metadata` | dict | (cross_artifact flag, family, reason) |

## 1.3 `pipeline_log.json` — per-stage telemetry (auditor's perf view)

One record per pipeline stage:

| Stage | What runs |
|---|---|
| `discover_artifacts` | Walk the project folder |
| `parse_artifacts` | Run each parser on its assigned artifact |
| `candidate_adjudication` | Resolve parser-emitted candidates |
| `source_replay` | Verify each atom's text actually appears in its source |
| `confidence_floor` | Drop atoms below confidence threshold |
| `enrich_entities` | Pull entity_keys + structured fields |
| `entity_resolution` | Alias fusion (sites + stakeholders) |
| `graph_build` | Build the cross-atom edge graph |
| `packetize` | Group atoms into evidence packets |
| `packet_certificates` | Certify packets per policy |
| `quality_gates` | Apply quality threshold checks |

Per-record fields:

| Field | UI hint |
|---|---|
| `stage` | Row label |
| `duration_ms` | Bar chart |
| `counts.input_count` / `counts.output_count` | Funnel widths |
| `warning_count` / `error_count` | Heat chips |

**Auditor insight from this:** a healthy run produces non-zero outputs at every stage, with quality_gates emitting ≤ 0 errors. A spike in any stage's duration_ms across runs flags a perf regression.

## 1.4 Parser-OS audit dashboard — recommended layout

```
┌──────────────────────────────────────────────────────────┐
│ Run badge: cmp_xxx / project_id / generated_at           │
│ Status: 🟢 OK  ·  Errors: 0  ·  Warnings: 32             │
└──────────────────────────────────────────────────────────┘

[KPI cards row]
 Files  Atoms  Packets  Entities  Edges
 7      135    28       70        172

[Coverage donuts]
 parse_outcomes: { ok: 7, failed_parse: 0, ... }
 by_authority_class: { customer_current_authored: 100, vendor_quote: 32, ... }

[Source-inventory table]
 # | File | Type | Parser | Atoms | Status | Size
 ...

[Quality timeline]
 stage_durations: bar chart per stage
 receipt_verified vs failed: stacked

[Drift tracker — compares to previous run signatures]
 input_signature: same/changed
 output_signature: same/changed → if changed with same input → parser regression
```

---

# 2. ORBITBRIEF PM_HANDOFF — for the PM-facing UI

## 2.1 PM_HANDOFF.json — single-record source of truth

Every field below is JSON-serializable and ready to feed a UI component. Field-name = key in `PM_HANDOFF.json`.

### Header / status

| Field | Type | UI hint |
|---|---|---|
| `case_id` | str | Header |
| `status` | enum | Traffic light: `red` / `yellow` / `green` |
| `status_label` | str | Status badge text |
| `one_line_summary` | str | Subheader |
| `executive_summary` | dict | Top-of-doc 3-line briefing (headline / health_line / next_action) |

### Scorecard

| Field | Type | UI hint |
|---|---|---|
| `metrics` | dict | KPI tiles: source_files, evidence_items_extracted, pm_visible_fact_cards, sites_published, blockers, warnings, top_workstream |

### Intake quality

| Field | Type | UI hint |
|---|---|---|
| `intake_completeness` | list[{item, detector_key, present}] | 10-item checklist with green/red dots |
| `ocr_backend_status` | dict[available, install_hints] | OCR backend chips + install hints |

### Universality / scope

| Field | Type | UI hint |
|---|---|---|
| `domains` | list[{domain_id, label, selected_by_router, active_for_sow, blockers, warnings, info}] | Workstream rows with counts |
| `sites` | list[{name, kind, publishable, member_evidence_count, artifact_count}] | Sites map / list |
| `source_files` | list[{filename, artifact_type, parser_name, evidence_items, status, status_reason}] | Source-inventory table |

### Stakeholders

| Field | Type | UI hint |
|---|---|---|
| `stakeholder_contacts` | list[{name, role, email, phone, source}] | Contact directory cards |
| `stakeholder_pagers` | list[{role, title, summary_lines, money_lines, risk_lines, action_lines}] | Per-stakeholder tab (CFO / IT / Procurement) |

### Money / commercial

| Field | Type | UI hint |
|---|---|---|
| `money_mentions` | list[{value, display, sources}] | Money table with source provenance |
| `reconciliation_flags` | list[{kind, label, values}] | Reconciliation queue cards |
| `currency_mentions` | list[{currency, amount, source, snippet}] | Non-USD callouts |
| `currency_conversions` | list[{currency, amount, usd_equivalent, fx_rate_used, source, snippet}] | USD-equivalent table |
| `tax_clauses` | list[{rate_pct, label, source, snippet}] | Tax handling table |
| `margin_view` | dict | Margin / profitability card (deal_total, hardware, services, other, total_cost, gross_profit, margin_pct, confidence, notes) |
| `engagement_model` | dict | Engagement-model card (detected_model, evidence, has_tm_cap, tm_cap_amount) |
| `license_items` | list[{part_number, description, quantity, unit_price, term_text, source}] | Subscription tracker table |

### Schedule

| Field | Type | UI hint |
|---|---|---|
| `schedule_phases` | list[{phase, start, end, owner, source}] | Gantt chart (Mermaid in MD) |
| `critical_path` | list[{phase, start, end, duration_days, is_critical}] | Critical-path highlighting |
| `phase_dependencies` | list[{upstream, downstream, evidence, source}] | Dependency graph |
| `critical_path_chain` | list[str] | Sequential phase chain |
| `lead_time_flags` | list[{part_number, description, quantity, lead_time_text, lead_time_days, risk_tier, source}] | Lead-time risk table |
| `resource_conflicts` | list[{owner, phases, overlap_windows}] | Resource conflict alerts |

### Risks

| Field | Type | UI hint |
|---|---|---|
| `risk_register` | list[{risk_id, description, likelihood, impact, mitigation, owner, sites, source}] | Risk register table |
| `risk_aging` | list[{risk_id, severity, days_open, aging_bucket, description}] | Risk aging buckets |

### Quality / hardware lifecycle

| Field | Type | UI hint |
|---|---|---|
| `subcontractor_mentions` | list[{name, role_hint, source, snippet}] | Subs / vendors table |
| `eol_flags` | list[{part_number, description, quantity, eol_status, replacement_hint, source}] | EOL/EOS callouts |
| `crm_detections` | list[{vendor, source, deal_fields, sample_row}] | CRM-export source chips |

### Compliance / legal

| Field | Type | UI hint |
|---|---|---|
| `compliance_callouts` | list[{framework, snippet, source, severity}] | Compliance routing table |
| `sla_penalties` | list[{kind, snippet, source}] | SLA / liq-damages alerts |
| `change_order_triggers` | list[{snippet, source, kind}] | Change-order pre-flags |

### Scope structure

| Field | Type | UI hint |
|---|---|---|
| `exclusions` | list[{text, source}] | Out-of-scope list |
| `responsibilities` | list[{party, text, source}] | Customer vs Provider split |
| `quantity_claims` | list[{target, quantity, snippet, source}] | Quantity table |
| `quantity_contradictions` | list[{target, values, files, examples}] | Quantity reconciliation alerts |
| `acceptance_checks` | list[{phase_or_step, criterion, owner, evidence_required, timing, source}] | Acceptance checklist |
| `acceptance_by_site` | dict[site, list[check]] | Per-site acceptance tabs |
| `site_rollups` | list[{site_key, site_name, atom_count, devices, money_values, dates, stakeholders}] | Per-site evidence cards |
| `site_allocations` | list[{site, device, quantity, unit_price, extended, source}] | Per-site BOM computed costs |

### PM action / output

| Field | Type | UI hint |
|---|---|---|
| `gaps` | list[{rule_id, domain_id, domain_label, label, severity, message, suggested_open_question, observed_summary}] | Gap-question queue |
| `customer_questions` | list[gap_card] | Customer-facing question list |
| `action_items` | list[{kind, label, owner, due, severity}] | Action checklist |
| `actions_by_week` | dict[bucket, list[action]] | This-week / next-week tabs |
| `sa_focus` | list[str] | SA review-lane chips |
| `facts_by_category` | dict[category, list[card]] | Evidence cards grouped by category |

### Strategic

| Field | Type | UI hint |
|---|---|---|
| `comparable_deals` | list[{case_id, closed_at, deal_value_usd, domains, sites_count, phase_count, final_margin_pct, outcome}] | "Similar past deals" panel |

### Output deliverables (auto-generated files)

| Field | Type | UI hint |
|---|---|---|
| `rfp_line_items` | list[{category, part_number, description, quantity, unit_price, lead_time, notes, source}] | Used by `RFP_DRAFT.md` |
| `date_mentions` | list[{iso, sources}] | Date cross-doc table |

## 2.2 PM_HANDOFF.md — 31 rendered sections

The markdown is the **human-readable** projection of `PM_HANDOFF.json`. Every section in the markdown corresponds to one or more JSON fields above. Use the JSON for the UI; keep the markdown for export / print / customer share.

## 2.3 SOW_DRAFT.md — 21-section auto-drafted Statement of Work

Currently markdown-only (no separate JSON). Drives off the same `PMHandoff` object. UI could either:
- Render the markdown as-is (Mermaid-compatible viewer)
- OR parse the section structure and surface as a multi-tab editor

## 2.4 RFP_DRAFT.md — categorized vendor RFP packets

Built from `rfp_line_items`. Categories (Network & Wireless, AV / Collaboration, Power & Environmentals, Endpoints / IT Devices, Structured Cabling, Security / Surveillance, Services & Labor, Miscellaneous) auto-bucket BOM rows.

## 2.5 Companion files

| File | What's in it |
|---|---|
| `90_inspection_report.json` | Per-artifact funnel view (atoms in / atoms cited / atoms in composed brief) — useful for "where does this evidence end up" debugging |
| `91_inspection_report.html` | Same, browsable HTML with click-through |
| `SA_REVIEW_PACKET.md` | Solution-architect technical view (longer evidence trail) |
| `PM_EXECUTIVE_SUMMARY.md` | Short 1-page brief for exec consumption |
| `.orbitbrief_history.jsonl` | Append-only ledger; one row per compile for the comparable-deals corpus |

---

# 3. OPTBOT example — what the PM actually sees

Latest OPTBOT compile produced these top-level numbers (from
`manifest.json` + `envelope.summary` + `PMHandoff`):

| Metric | Value |
|---|---:|
| Files ingested | 7 |
| Atoms extracted | 135 |
| Packets certified | 28 |
| Entities resolved | 70 |
| Cross-atom edges | 172 |
| Cross-artifact edges | high — reconciliation working |
| Parser errors | 0 |
| Warnings | 32 |
| `parse_outcomes.ok` | 7/7 (100%) |
| Confidence-weighted authority distribution | customer_current_authored: dominant |
| Intake completeness | 9/10 (90%) |
| Margin | 0% (zero-margin SOW flagged) |
| Sites confirmed | 3 |
| Stakeholders in directory | 7 |
| Risks tracked | 5 |
| Schedule phases | 6 |
| Cross-doc money reconciliation flags | 2 |
| Cross-doc date alignment rows | 12 |
| Compliance callouts | 2 |
| Action items | 22 |
| SOW_DRAFT.md size | 16 KB / 21 sections |

**Digestibility:** the PM opens `PM_HANDOFF.md` and reads top-to-bottom in 5 minutes. Critical signals (margin %, blockers, reconciliation flags) all appear above the fold. Everything that requires action has an owner field and (where dated) a due field.

---

# 4. What's NOT yet in the JSON (gaps for UI build)

These fields would help the UI but aren't surfaced as structured JSON today:

| Missing | What it'd enable | Effort |
|---|---|---|
| `sow_draft` (rendered markdown) embedded in PM_HANDOFF.json | Single-payload UI render of the SOW | 30 min |
| `rfp_draft` (rendered markdown) embedded in PM_HANDOFF.json | Same for RFP | 30 min |
| Aggregated `parser_quality_score` (0-100 from atoms/files/warnings/errors) | Headline quality KPI for the audit dashboard | 1 h |
| Per-file `time_to_parse_ms` from `pipeline_log.json` | Slowest-file panel | 1 h |
| `entity_coverage_pct` (% of expected entity types present) | Coverage gauge per project | 1 h |
| `change_history` (diff vs previous compile) | Re-parse drift alerts pre-rendered into the JSON | 2 h |
| Stable atom-color heat (for graph view) | Graph-visualizer styling | 1 h |
| Per-section ready/needs-review flag in SOW_DRAFT | "Skip already-OK sections" mode for the PM | 2 h |

---

# 5. Recommended UI tabs (matched to fields above)

```
┌─ AUDIT (parser-os) ─────────────────────────────────┐
│ Run badge  KPIs  Coverage donut  Source inventory   │
│ Stage timeline  Drift tracker  Atom drill-down      │
└─────────────────────────────────────────────────────┘
┌─ PM BRIEF (OrbitBrief) ─────────────────────────────┐
│ Executive summary  Intake completeness  Status      │
│ Tabs:                                               │
│   • Scope (workstreams / scope items / exclusions)  │
│   • Sites (rollup + BOM allocation)                 │
│   • Stakeholders (contacts + 3 one-pagers)          │
│   • Risks (register + aging + lead-time)            │
│   • Schedule (Gantt + critical path)                │
│   • Commercial (margin + reconciliation + currency) │
│   • Compliance & Legal (callouts + SLA + CO)        │
│   • Actions (consolidated + by-week)                │
│   • Vendor / RFP (RFP draft + subcontractor list)   │
│   • SOW draft viewer                                │
└─────────────────────────────────────────────────────┘
┌─ HISTORICAL ────────────────────────────────────────┐
│ Comparable deals  Portfolio rollup  Run history    │
└─────────────────────────────────────────────────────┘
```

Every tab maps to specific PMHandoff fields documented above; the UI
can be 100% data-driven from `PM_HANDOFF.json` + `00_envelope.json`
without any string parsing of markdown.
