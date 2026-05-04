# Parser-OS Production Gap Report

**Generated**: 2026-05-03 from running parser-os against the 12-case STRESS_* corpus and comparing actual output to gold standards.

**Cases compiled** (5 of 12 — these were enough to surface every category of gap):
- `STRESS_VT_CAM` — 1 PDF (16pp), gold expects 60+ atoms / 12+ packets
- `STRESS_NATOMAS_WIRELESS` — 1 PDF (25pp), gold expects 110+ atoms / 22+ packets / 17 quantity_conflict edges
- `STRESS_AV_TRIO` — 3 PDFs (Hayward+AMBAG+ICMA), gold expects 250+ atoms / 45+ packets
- `STRESS_XLSX_RARE` — 2 XLSX files (CalSAWS 484-row Q&A + NJEDA fee schedule), gold expects 1100+ atoms / 25+ packets
- `STRESS_ITAD_PAIR` — empty `artifacts/` (documented gap), gold expects graceful handling with `compile_status:no_artifacts_found`

## Executive scorecard (actual vs. gold)

| Case | Atoms actual / gold | Packets actual / gold | Edges actual / gold | Domain pack actual / gold | Entity_keys per atom |
|---|---|---|---|---|---|
| VT_CAM | 13 / **60+** | 5 / **12+** | 0 / 0 | `default_pack` / `security_camera_pack` | **0/13** ❌ |
| NATOMAS | 184 / 110+ ✓ | 1 / **22+** | 0 / **17 quantity_conflict** | `default_pack` / `wireless_pack` | **0/184** ❌ |
| AV_TRIO | 632 / 250+ ✓ | 8 / **45+** | 0 / multi (vendor_overlap) | `default_pack` / `av_pack` | **0/632** ❌ |
| XLSX_RARE | **0** / 1100+ | **0** / 25+ | 0 / 0 | `default_pack` / `default_pack` ✓ | **0/0** ❌ |
| ITAD_PAIR | 0 / 0 ✓ | 0 / 0 ✓ | 0 / 0 ✓ | `default_pack` / `itad_pack` | **0/0** ❌ |

✓ = passing  ❌ = failing  bold = critical shortfall

**Aggregate**: 829 atoms produced where gold expected ~1700 (49% recall). 14 packets produced where gold expected ~104 (13% recall). 0 edges produced where gold expected ~17+ (0% recall). **0% entity_resolution rate across all 5 cases.**

### Performance cliff observed in AV_TRIO

AV_TRIO (3 PDFs, 632 atoms total) took **180 seconds end-to-end**:
- `parse_artifacts`: 91 seconds (3 PDFs)
- **`graph_build`: 87 seconds** (entity-less O(n²) pass over 632 atoms = ~400K pairs)
- Other stages: <2 seconds combined

For Chicago Housing (~5000 expected atoms) this graph_build pass would extrapolate to ~30+ minutes. Production-blocking at scale.

---

## P0 — Critical production blockers

### P0.1 — Domain pack auto-routing is not happening

**Symptom**: All 4 cases used `default_pack` v2.0.0 despite each having an obvious service-line signature.

**Evidence**:
- VT_CAM (security_camera RFP) → `default_pack`
- NATOMAS (E-rate wireless RFP, says "Wireless Equipment" in filename) → `default_pack`
- XLSX_RARE → `default_pack` (acceptable here)
- ITAD_PAIR → `default_pack` (should be `itad_pack`)

**Root cause**: `python -m app.cli compile <project_dir>` doesn't auto-select a pack. The `--domain-pack` flag must be passed manually, or the system silently falls back to `default_pack`. There's no project-level config (`STRESS_*/SOURCE_NOTES.md` declares the service line, but parser-os doesn't read it).

**Impact**: Every service-line-specific exclusion pattern, constraint pattern, and entity vocabulary is missed. The Cisco CW9166I family in NATOMAS, the security_camera vocabulary in VT_CAM, etc. — all of it ignored.

**Fix priority**: P0
**Recommended fix**:
1. Add `<project>/project.yaml` or read `<project>/SOURCE_NOTES.md` for `service_line:` declaration
2. Add a routing classifier that reads filenames + first-page content and picks the highest-scoring pack
3. Emit a warning when pack selection fell back to `default_pack`

---

### P0.2 — Entity resolution emits 0 entities for every atom

**Symptom**: 100% of atoms across all 4 cases have `entity_keys: []`. Gold expected 8-50 entity_keys per atom.

**Evidence**:
- VT_CAM atoms: every row in REVIEW.md has `Entity keys: —`
- Natomas atoms with explicit content like "Part Number: CW9166I-B | Description: Catalyst 9166I AP (W6E, tri-band 4x4, XOR) w/Reg-B | Qty: 136" — `Entity keys: —`
- Natomas atom with school: "col_1: Natomas Park Elementary | NATOMAS UNIFIED SCHOOL DISTRICT: 4700 Crest Drive" — `Entity keys: —`
- The compile trace shows `entity_resolution` stage runs in 0.5-0.7ms with `output_count: 0`. This is suspicious — entity_resolution is producing zero output every time.

**Root cause**: The entity_resolution stage is broken or disconnected from atom storage. From `cli.py` trace logs: `"stage": "entity_resolution", "duration_ms": 0.69, "counts": {"input_count": 13, "output_count": 0}`. Entities aren't being written back to atoms.

**Impact**: 
- Packets default to `device:unknown` / `site:unknown` anchors (Natomas: 1 packet, anchor `device:unknown`, flag `unknown_anchor`)
- Cross-artifact graph edges can't form (need entity overlap to link artifacts)
- Quantity_conflict detection can't fire (needs same-entity matching)
- Vendor overlap edges can't form
- Site rosters can't be aggregated

**Fix priority**: P0 (blocks almost everything else)
**Recommended fix**:
1. Trace `app/core/entity_resolution.py` — find why `output_count: 0` despite `input_count: 13`
2. Check that entity_keys produced by domain_pack are written back to atoms in source_replay or after
3. Add an integration test: `assert len(atom.entity_keys) > 0` for any atom with vendor/site/device tokens

---

### P0.3 — Packet anchors land on `unknown` because entities don't resolve

**Symptom**: Packetizer fires but with `device:unknown` / `site:unknown` anchors. 1 of 1 Natomas packets, 2 of 5 VT_CAM packets, all flagged `unknown_anchor`.

**Evidence**: 
- VT_CAM `pkt_32be5ed4e4e2b1cb`: `anchor: device:unknown, flags: unknown_anchor`
- Natomas `pkt_1578d339d386cc57`: `anchor: device:unknown, flags: unknown_anchor`

**Root cause**: Direct downstream of P0.2 — packetizer reads `atom.entity_keys` and finds none, so falls back to `unknown`.

**Fix priority**: P0 (resolved by fixing P0.2)

---

### P0.4 — XLSX parser produces 0 atoms when title rows precede the header

**Symptom**: Both real-world XLSX files in STRESS_XLSX_RARE produced 0 atoms despite correct routing.

**Evidence**:
- `calsaws_qa_log.xlsx` (484-row Q&A log): routed at 0.58 confidence to `xlsx_parser_v2_1`, **0 atoms extracted**
- `njeda_fee_schedule.xlsx` (30-row fee schedule): routed at 0.82 confidence to `xlsx_parser_v2_1`, **0 atoms extracted**

**Root cause** ([app/parsers/xlsx_parser.py:786](app/parsers/xlsx_parser.py#L786)):
```python
def _parse_sheet_rows(...):
    model = _detect_header(rows)
    if model.header_idx < 0:
        return []  # ← EARLY EXIT: 0 atoms when no header found
    ...
```

When `_detect_header()` can't find a header row (because real XLSX files have title/instruction rows before the column headers), the parser bails out entirely. Every CalSAWS row 1 is "CalSAWS M&O RFP Question and Answer Log" (title), row 2 is "ID | Section | Page Number | ..." (real header).

**Impact**: ZERO support for real-world XLSX attachments — RFPs with cost-sheet templates, Q&A logs, fee schedules, vendor pricing matrices, etc. The MS ITS Managed VPN XLSX in STRESS_NET_MAINT will have the same issue. This is a category-killer bug.

**Fix priority**: P0
**Recommended fix**:
1. `_detect_header()` should scan rows 1-10 (not just row 1) for the most-table-shaped row
2. Treat rows above the header as `metadata` blocks (preserve as project-level context)
3. If no header found, fall back to extracting non-empty rows as `paragraph` atoms (current behavior buries this — only used for the `structured_doc` fallback paragraph dump)
4. Add a regression test using both XLSX_RARE files

---

### P0.5 — Quantity contradiction detection isn't firing

**Symptom**: Natomas had 17 part numbers explicitly extracted with both `Qty: 500` AND `Qty: 136` atoms — but 0 quantity_conflict edges produced.

**Evidence (atoms found in Natomas REVIEW.md)**:
- `atm_3b8439a1e8d993f7`: `Part Number: CW9166I-B | Qty: 136`
- `atm_093275403b8d7911`: `Part Number: SW9166-CAPWAP-K9 | Qty: 500`
- `atm_4cdbaa4f66693360`: `Part Number: NETWORKPNP-LIC | Qty: 500`
- `atm_14c49cdd0ef9ac1a`: `Part Number: AIR-DNA-E-5Y | Qty: 136`
- ...and so on for ~17 SKUs

The atoms are there. The edge-builder should produce 17 `quantity_conflict` edges where the same SKU has different quantities. It produced **0**.

**Root cause hypothesis**: Without entity_keys (P0.2), the graph_builder can't recognize that `atm_3b8439a1e8d993f7` and another atom with the same `CW9166I-B` part number refer to the same entity. Same SKU + different qty = contradiction edge.

**Fix priority**: P0 (cascade from P0.2)
**Recommended fix**: Confirm the contradictions logic exists in `app/core/graph_builder.py`. If it does, the fix is P0.2. If not, add a `same_part_number_different_qty → quantity_conflict` rule.

---

## P1 — Major recall and quality gaps

### P1.1 — PDF parser agglomerates Q&A pairs into single mega-atoms

**Symptom**: VT_CAM had 18+ Q&A pairs (Q8, A8, Q9, A9, Q10, A10, ... Q18) collapsed into a single atom whose anchor key is the entire 2400+ character transcript.

**Evidence** (VT_CAM packets/REVIEW.md, `pkt_bad79e4927cd5df5`):
```
anchor: missing_info:q8_storage_requirement_resolution_etc...q18_on_site_support_is_it_the_intent...
```
The anchor string is the literal transcript of Q8-Q18, normalized to lowercase, with underscores. This is unusable for any downstream consumer.

**Root cause**: PDF parser's text-block segmentation isn't splitting on Q-and-A pair boundaries. The Virginia Tech RFP has color-coded answers (blue text) which OrbitBrief PDF should pick up — that's the explicit gold expectation in `gold_standard.md`.

**Impact**: Recall drops from gold's expected 60+ atoms to actual 13 atoms. Downstream packet anchors become unanchored gibberish.

**Fix priority**: P1
**Recommended fix**:
1. PDF parser needs Q&A-aware segmentation (split on `Q\d+\.` / `A\d+\.` patterns)
2. Color-aware extraction (when answer text is in a different color from the question, treat as separate atoms with `authority_class: customer_current_authored`)
3. Cap atom raw_text length at ~600 chars; longer atoms should split

---

### P1.2 — Form-field templates emit as scope_item atoms

**Symptom**: VT_CAM had 6 of its 13 atoms be vendor-info form templates ("FULL LEGAL NAME (PRINT) (Company name as it appears with your Federal Taxpayer Number): CONTACT NAME/TITLE..."). These are blank form fields, not scope.

**Evidence (VT_CAM atoms)**:
- `atm_3a719bcf090321e6`: "FULL LEGAL NAME (PRINT)..." 
- `atm_663c90dbf3d339b1`: "FULL LEGAL NAME (PRINT)... E-MAIL ADDRESS..."
- `atm_6d086aa5977fa1b9`: "FULL LEGAL NAME (PRINT)... PURCHASE ORDER ADDRESS..."
- ...4 more
- All 6 emitted with `atom_type: scope_item, authority: contractual_scope, conf: 0.92`

**Root cause**: Form-field detector missing. PDF parser is treating multi-column form templates (extracted as table_row blocks) as scope content.

**Impact**: 
- Inflates atom count with junk
- Sends `device:unknown` packets through the pipeline
- High confidence (0.92) means these get prioritized over real scope

**Fix priority**: P1
**Recommended fix**:
1. Pattern-match form-field markers: `(PRINT)`, `(IN INK)`, `_____`, `col_\d+:`, "Name of...", "Title:", "Date:"
2. Atoms with ≥3 form-field markers in raw_text → `atom_type: form_field` (or filter entirely)
3. Add to `default_pack` filter rules

---

### P1.3 — Page footer noise emits as atoms

**Symptom**: Natomas emitted ~15 atoms that are just page footers ("RFP 25-107 Wireless Equipment November 20, 2024 Technology Services Department Page 17 of 25").

**Evidence**: Natomas atoms `atm_02a7337ce1cd844c`, `atm_0779ff870ee1a6cf`, `atm_078ef91e91e11de7`, ... all variants of page-footer text.

**Root cause**: PDF parser doesn't detect repeating page-header/page-footer patterns.

**Impact**: 
- Inflates atom count
- Confuses packetizer

**Fix priority**: P1
**Recommended fix**:
1. Detect text that appears on N≥3 pages with the same template (`Page X of Y`) → `block_kind: page_footer`
2. Filter `page_footer` from atom emission

---

### P1.4 — Single-word / fragment atoms

**Symptom**: Natomas emitted atoms with content like "Cost Proposal", "Addendums", "Cost Proposals", "Project Description", "Equipment/Service Installed".

**Evidence**: Natomas atoms `atm_0a506913ffcf0372` ("Cost Proposal"), `atm_465a8fb3c54ec529` ("Addendums"), etc.

**Root cause**: PDF parser is emitting bullet-point entries from the proposal-format checklist (which lists what should be in the proposal) as standalone atoms.

**Impact**: Low-value noise inflates atom count.

**Fix priority**: P1
**Recommended fix**:
1. Filter atoms with raw_text < 30 chars that don't contain identifiers
2. Aggregate consecutive bulleted fragments into a parent paragraph atom

---

### P1.5 — Ontology gap detector silent

**Symptom**: VT_CAM detected 0 ontology gaps, Natomas detected 1 miscategorized gap.

**Evidence**:
- VT_CAM (gold expected: license_plate_camera, facial_recognition, T2 Systems, ThyssenKrupp, ESRI, ArcSDE, surveillance_oversight_committee, etc.) → 0 vocab gaps detected
- Natomas (gold expected: Cisco CW9166I, E-rate, USAC, SPIN, FRN, LCP, Secure Networks Act, etc.) → only 1 gap detected: "Responder Service Level Agreement" miscategorized as `Site candidates`

**Root cause**: Gap detector requires entity_keys to be populated to know what's a gap (P0.2 cascade). Plus its categorization rules are too narrow.

**Impact**: The whole point of running parser-os against new RFPs is to discover new vocabulary. With 0 gaps detected, this product feature is broken.

**Fix priority**: P1 (resolved partly by P0.2; partly by improving categorization rules)

---

### P1.6 — Gold standard files / SOURCE_NOTES.md parsed as artifacts

**Symptom**: Every compile lists `labels/gold_standard.md`, `labels/gold_standard.json`, `SOURCE_NOTES.md` as parser routing inputs. These pollute the routing log and waste time.

**Evidence (VT_CAM REVIEW.md)**:
```
| `labels/gold_standard.md` | `none` | 0.00 | miss | no_parser_over_threshold |
| `SOURCE_NOTES.md` | `none` | 0.00 | miss | no_parser_over_threshold |
| `labels/gold_standard.json` | `none` | 0.00 | miss | no_parser_over_threshold |
```

Worse, in XLSX_RARE the `labels/gold_standard.json` was routed to `transcript` parser at 0.80 confidence! That's a false positive.

**Impact**: 
- Pollutes routing log
- gold_standard.json risks being parsed as transcript content (false-positive risk)
- Confuses ontology gap detector

**Fix priority**: P1
**Recommended fix**:
1. `app/core/discover_artifacts.py` should only walk `<project>/artifacts/` directory by default
2. Honor `<project>/.parserignore` glob patterns (ignore `labels/`, `*.gold_standard.*`, `SOURCE_NOTES.md`, `.orbitbrief/`)

---

### P1.7 — XLSX cell-merge mangles part-number tables

**Symptom**: Natomas cost-proposal table cells got merged across rows in extraction.

**Evidence**: `atm_185b8c149e251f0d` reads:
```
"AIR-DNA-E: AIR-DNA-E-T-5Y | Wireless Cisco DNA On-Prem Essential, Term Lic: Wireless Cisco DNA On-Prem Essential, 5Y Term, Tracker Lic | 500: 500"
```

The PDF parser fused 2 separate SKU rows into one atom with colon-separated values.

**Root cause**: Multi-row table extraction in PDF — when a table cell is empty in one row, the next row's value gets concatenated.

**Impact**: 
- Quantity values (500: 500) become unparseable
- Vendor SKU pairs (AIR-DNA-E: AIR-DNA-E-T-5Y) become unparseable
- Hurts entity extraction even more

**Fix priority**: P1
**Recommended fix**:
1. Better cell-boundary detection in the PDF table extractor
2. Detect the "X: X" pattern in cells — likely indicates two-rows-into-one-cell artifact

---

## P2 — Performance and DX issues

### P2.1 — VT_CAM 16-page PDF takes 186 seconds to parse

**Symptom**: `parse_artifacts` stage took 186,192ms for one PDF in VT_CAM (16 pages, 3.6 MB).

**Evidence**: Natomas (25-page, 350 KB, text-rich) parsed in 9.4 seconds. VT_CAM (16-page, 3.6 MB, with scanned floor plans on pages 7-16) parsed in 186 seconds — ~20x slower per page.

**Root cause hypothesis**: OCR fallback engaging on pages 7-16 (scanned CAD drawings) even though those pages have minimal extractable text (they're images with column letters/dimensions). OCR is being invoked unnecessarily.

**Impact**: At scale (1000 RFPs), a 186s/case rate = 51 hours of parse time for one batch. Production needs <10s/case median.

**Fix priority**: P2
**Recommended fix**:
1. Detect "image-dominant page with low text" → emit `unsupported` receipt without OCR
2. Make OCR opt-in (`--ocr` flag), not default
3. Add a 30s/page timeout

---

### P2.2 — Graph build is slow even on single-artifact corpora

**Symptom**: Natomas single-doc graph_build took 6.95 seconds (with 184 atoms).

**Evidence**: Trace shows `graph_build:6948.6ms` for 184 atoms. That's 38ms/atom. For Chicago Housing's expected 5000+ atoms, this would be ~3 minutes just for graph build.

**Root cause hypothesis**: O(n²) edge-checking pass. With 184 atoms, that's 33,856 atom pairs to check.

**Impact**: Doesn't scale to large RFPs.

**Fix priority**: P2
**Recommended fix**:
1. Profile graph_builder
2. Index atoms by entity_key prefix for O(n log n) edge checking
3. Cap edge candidates per pair

---

### P2.3 — Empty case (ITAD) doesn't flag itself as a documented gap

**Symptom**: ITAD compile produced 0 atoms / 0 packets / 2 warnings — same shape as a real "found nothing" compile.

**Evidence**: ITAD has only `SOURCE_NOTES.md` (declared as documented gap) and empty `artifacts/`. Compile output looks indistinguishable from a normal "no scope content found" failure.

**Impact**: Operators can't distinguish "we couldn't fetch the artifacts" from "the artifacts had no scope content".

**Fix priority**: P2
**Recommended fix**:
1. If `artifacts/` is empty, set `compile_status: "no_artifacts_found"` in output JSON
2. If `SOURCE_NOTES.md` declares this as a documented gap, set `compile_status: "documented_gap"`
3. Surface in REVIEW.md with a clear banner

---

## P3 — Production-readiness gaps

### P3.1 — No project config file for service line / pack pre-selection

Each STRESS case has `SOURCE_NOTES.md` declaring its service line. parser-os doesn't read this. Add `<project>/project.yaml`:
```yaml
service_line: security_camera
domain_pack: security_camera_pack
context_notes: |
  This RFP includes Q&A from a pre-proposal conference; treat blue-text
  answers as customer_current_authored.
```

### P3.2 — No batch compile mode

Currently each `python -m app.cli compile` is a separate Python process startup (~2-3s overhead). For 12 cases that's ~30s wasted. Need:
```bash
python -m app.cli batch-compile real_data_cases/STRESS_*/ --out-dir /tmp/stress_results/
```

### P3.3 — No gold-standard comparison tool

I had to manually compare actual output vs. gold. Need:
```bash
python -m app.cli compare \
  --gold real_data_cases/STRESS_VT_CAM/labels/gold_standard.json \
  --compiled /tmp/stress_results/STRESS_VT_CAM.json
```
Output: per-metric pass/fail (atom count, packet count, edge count, entity_keys recall, ontology gap recall, ...).

### P3.4 — No accuracy/recall metrics in compile output

The compile JSON doesn't include any quality scores. Production telemetry needs:
- `quality.atom_count_vs_gold_pct`
- `quality.entity_resolution_rate` (fraction of atoms with non-empty entity_keys)
- `quality.packet_specificity` (fraction of packets without `unknown` anchors)
- `quality.gap_detector_recall_estimate`

### P3.5 — Compile failures should fail loud, not silently fall back

XLSX parser returning `[]` on header-detection failure is silent. Production needs:
- Error: "xlsx_parser produced 0 atoms but XLSX has 484 non-empty rows; likely header-detection failure" (with path to remediate)

---

## Recommended fix sequencing (4 weeks to production)

### Week 1 — Unlock entity-driven downstream features
1. **P0.2** Fix entity_resolution stage (entity_keys writes back to atoms)
2. **P0.1** Add domain pack auto-routing from SOURCE_NOTES.md / project.yaml
3. **P1.6** Filter `labels/`, `SOURCE_NOTES.md`, `.orbitbrief/` from artifact discovery

### Week 2 — Fix XLSX + quantity contradictions
4. **P0.4** Fix `_detect_header()` to scan rows 1-10, not just row 1
5. **P0.5** Verify quantity_conflict logic in graph_builder (validates after P0.2 lands)
6. **P1.7** Better PDF table cell-boundary detection

### Week 3 — Recall + noise filtering
7. **P1.1** Q&A-aware PDF segmentation with color-coding support
8. **P1.2** Form-field detector + filter
9. **P1.3** Page-footer detection + filter
10. **P1.4** Filter single-word fragment atoms
11. **P1.5** Improve ontology gap categorization rules

### Week 4 — Performance + DX
12. **P2.1** OCR opt-in + per-page timeout
13. **P2.2** O(n log n) graph build with entity index
14. **P3.1** project.yaml schema
15. **P3.2** batch-compile mode
16. **P3.3** compare-to-gold tool
17. **P3.4** quality metrics in compile output

---

## Bonus: positive findings (parser-os already does this well)

- Receipt verification: 13/13 VT_CAM atoms verified, 184/184 Natomas. The provenance chain is solid.
- Stage tracing: trace.json output is excellent for diagnosing where time is spent.
- Cache discrimination: cache_hits/cache_misses correctly tracked.
- Per-artifact dossiers: REVIEW.md per artifact is a great review surface.
- Confidence scoring: atoms have meaningful 0.78-0.92 confidence values.
- Atom type taxonomy (`scope_item`, `vendor_line_item`, `constraint`, `assumption`, `open_question`) is good.
- Doesn't crash on empty case (ITAD).

The pipeline architecture is correct — these are tractable bug fixes, not a re-architecture.
