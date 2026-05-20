# `orbitbrief_page_os.segmentation.schematic`

Legend-first schematic intelligence for the PDF parser. Turns
construction drawings (floor plans, risers, one-lines, fire-alarm
sheets, AV plans, civil layouts) into a typed evidence graph that
a downstream reviewer can read without opening the PDF.

```
PDF page → legend_locator.locate_legend_candidates()
        → legend_parser.parse_legend()  →  ParsedLegend (with typed attributes)
        → legend_resolver.LegendResolver  →  ResolvedLegend per drawing page
        → app.parsers.schematic_atom_emitters.intersect_with_pack()
                                              → DetectionTargetSet
        → exclusion_zones.detect_exclusion_zones()  →  title block / notes / schedule regions
        → sheet_metadata.parse_sheet_metadata()  →  title-block field set
        → rooms.detect_rooms()  →  Room atoms
        → keyed_notes.detect_keyed_notes()  →  numbered notes + body callouts
        → schedules.detect_schedules()  →  ScheduleRow per equipment / door / camera
        → symbol_detector.detect_symbols()  →  list[SymbolDetection] with NMS
        → callouts.detect_callouts() + attach  →  mounting-height per detection
        → schedules.join_schedule_rows_to_detections()  →  schedule_tag per detection
        → rooms.assign_detections_to_rooms()  →  located_in_room_display per detection
        → line_runs.detect_line_runs()  →  conduit / cable / riser polylines, snapped to devices
        → app.parsers.schematic_atom_emitters.emit_*  →  atoms + derived JSONs
```

All modules in this package are deterministic, CPU-only, and have no
runtime LLM. Every emitted atom carries page + bbox + crop hash so
`app.core.source_replay` can verify it independently.

## Modules

| File | What it does |
|---|---|
| `legend_locator.py` | Layered candidate detector (text rules, header pairs, continuation hints, optional ONNX classifier). Captures rotation_deg from PyMuPDF span directions. |
| `legend_parser.py` | Turns a candidate region into `ParsedLegend` with tabular + inline row parsers and header-driven typed attributes (mounting_height, cable_count, rough_in, remarks, mfg, model, responsibility …). |
| `legend_resolver.py` | Document-level index that resolves the right legend for each drawing page. 4-priority resolution: in-page → explicit reference → drawing-index → discipline / global. Warnings carry replayable bbox provenance. |
| `symbol_detector.py` | Text-tag + glyph-template detector with PyMuPDF char-level glyph metrics, standalone-token FP suppression, and deterministic NMS. |
| `exclusion_zones.py` | Detects title block, drawing index, keyed-notes, and schedule regions so a "PTZ" inside "PTZ ROOM" or a schedule cell doesn't get counted as a detection. Rotated text auto-classified as title-block furniture. |
| `sheet_metadata.py` | Title-block field extractor: sheet number/title, project name, scale, issue date, revision, drafter, checker, approver, client. Returns None when nothing parseable. |
| `rooms.py` | Floor-plan room label detector (LOBBY 101, CONFERENCE 204, MDF, IDF, …). `assign_detections_to_rooms()` picks the nearest room within 144 pt of each detection center. |
| `keyed_notes.py` | KEYED NOTES / GENERAL NOTES block parser. Numbered rows + resolved body callout bboxes. |
| `schedules.py` | Camera / door / equipment / fixture / panel schedules. Header-driven column map. `join_schedule_rows_to_detections()` has two passes: nearby_text tag match + spatial join within 144 pt of a TAG block. |
| `callouts.py` | Mounting-height / dimension callouts (`42" AFF`, `8'-10"`, `+120"`, `CEILING`, `V.I.F.`). `attach_callouts_to_detections()` picks the nearest within 72 pt. |
| `line_runs.py` | Conduit / cable / riser polyline detector from PyMuPDF drawing primitives. Snaps endpoints to nearby detections within 36 pt. Filtered by length 18–720 pt. |
| `raster.py` | Page rasterization + deskew utilities for image-only PDFs. Fixed 200 DPI render. |
| `ocr.py` | Optional Tesseract adapter; fails closed when binary absent. `words_to_textblocks()` projects OCR words back into PDF-point TextBlocks. |
| `classifier.py` | Optional ONNX legend-block ranker; refuses mismatched model bytes. |
| `debug_overlay.py` | PIL renderer that draws legend + detection boxes on a rasterized page. Opt-in via `PARSER_OS_SCHEMATIC_OVERLAYS=1` env var. |

## Atom types emitted

| Atom | What it carries |
|---|---|
| `schematic_sheet_metadata` | Per drawing page: sheet number/title, project name, scale, date, revision, signatures. Suppressed when no field parseable. |
| `schematic_legend` | Per parsed legend: scope (page / global / continuation), entries with raw_symbol_text / normalized_label / count_column / **typed attributes** dict. |
| `schematic_room` | Per room/zone label: label, number, page, bbox. |
| `schematic_keyed_note` | Per numbered note row: number, text, callout_count. |
| `schematic_note_callout` | Per body marker resolving to a note. *(Reserved — currently included on the parent note's `callout_bboxes`.)* |
| `schematic_schedule_row` | Per schedule row: schedule_kind, tag, fields dict (mfg, model, mounting, remarks …). |
| `schematic_detection_target_set` | Per drawing page: the set of targets the parser will hunt for, intersected from legend ∩ domain-pack. |
| `schematic_symbol_detection` | Per device instance. Value carries: target_key, entity_key, modality, bbox, crop_sha256, **located_in_room_display**, **mounting_height + mounting_height_source**, **responsibility**, **legend_remarks**, **schedule_tag** + schedule_kind + schedule_fields, **legend_entry_id**. |
| `schematic_line_run` | Per conduit/cable polyline. Value carries: polyline, endpoints, length_pt, from_detection_id, to_detection_id. |
| `quantity` (schematic role) | Per-page detected count vs. legend-declared count. Cross-artifact rollups via `parent_entity_keys` reach BOM/RFP atoms. |
| `schematic_warning` | Structured failure-mode signals (see warning types below). |

## Warning types

| Warning type | When it fires |
|---|---|
| `missing_legend` | Drawing page has no resolvable legend (in-page, explicit, drawing-index, discipline, or global). |
| `weak_legend` | Reserved for soft legend-quality issues (low candidate score, partial parse). |
| `legend_gap` | Pack declares a load-bearing target the legend doesn't mention. Deduped per (legend_id, target_key) — fires once, not per drawing page. |
| `legend_orphan` | Legend entry exists but produced zero detections on the drawing body. |
| `unknown_symbol` | Repeated short ALL-CAPS token in the body that isn't in the legend. Excludes sheet numbers, grid bubbles, keyed-note integers, and conventional drawing abbreviations. |
| `ambiguous_legend_reference` | Two or more legends tie at the same resolution priority. |
| `unresolved_legend_reference` | `see sheet X` reference points to a sheet that doesn't exist or has no parsed legend. |
| `ocr_unavailable` | Page has no text layer AND Tesseract is not installed. |
| `ocr_recovered` | Page had no text layer but OCR ran successfully — atom carries the recovered word/block count. |
| `low_ocr_confidence` | Reserved for per-word OCR confidence below the threshold. |
| `schematic_quantity_contradiction` | Same-sheet detected count differs from legend-declared count by ≥ 1. |
| `prepass_failure` | The schematic pre-pass raised inside `parse_artifact`. Atom carries failure message + truncated traceback. |
| `weak_declared_count_provenance` | Legend row declared a count but had no replayable symbol bbox — the declared-count atom was refused; this warning surfaces the count without laundering it through fake provenance. |

## Derived files written next to the source PDF

Path is `<stem>.derived/`:

- `schematic_legends.json` — every parsed legend with entries + attributes
- `schematic_targets.json` — per-page detection target sets with the resolution rationale
- `schematic_detections.json` — every detection with bbox + crop hash + enrichment fields
- `overlays/page_NNNN.png` — debug overlay images (opt-in via `PARSER_OS_SCHEMATIC_OVERLAYS=1`)
- `schematic_overlays.json` — manifest of overlay sidecars

## Project-manager-visible enrichment

A single detection atom shows:

```json
{
  "target_key": "ptz_camera",
  "entity_key": "device:ptz_camera",
  "page": 1,
  "sheet_number": "E1.01",
  "modality": "text_tag",
  "bbox": [...],
  "crop_sha256": "...",
  "located_in_room_display": "LOBBY 101",
  "mounting_height": "120\" AFF",
  "mounting_height_source": "legend_column",
  "responsibility": "NIC",
  "legend_remarks": "NIC LENS",
  "schedule_tag": "C-101",
  "schedule_kind": "camera",
  "schedule_fields": {"mfg": "Axis", "model": "P3245-LV", "mounting": "120\" AFF"}
}
```

Entity keys include the subtype (`device:ptz_camera`), the
parent rollups (`device:ip_camera`, `device:camera`), the room
(`room:room_…`), the schedule tag (`schedule_tag:C-101`), and the
responsibility marker (`responsibility:nic`) so cross-artifact
graph edges find the join automatically.

## Determinism + provenance contracts

The parser layer guarantees:

1. **Re-running on the same PDF yields byte-identical atoms.** Sort
   order, IDs, crop hashes are all stable.
2. **Every schematic atom has a `SourceRef.locator` carrying `page`,
   `bbox`, `bbox_units="pdf_points"`, and `crop_sha256` where a real
   region exists.** `app.core.source_replay._verify_pdf_bbox_crop`
   re-renders and hash-verifies on demand.
3. **No runtime LLM.** Optional ONNX classifier is a frozen, hash-
   pinned static asset.
4. **Cross-artifact intelligence via `parent_entity_keys` rollups.**
   16 domain packs carry rollups so schematic subtype counts join
   to broader BOM line items.

## Testing

Local development uses a smoke runner because pytest crashes on the
Windows dev environment (see `pyproject.toml` note):

```bash
python scripts/_schematic_smoke.py \
  tests.test_schematic_contracts \
  tests.test_pdf_bbox_replay \
  tests.test_domain_detection_targets \
  tests.test_schematic_legend_parser \
  tests.test_schematic_legend_resolver \
  tests.test_orbitbrief_pdf_schematic_legend_first \
  tests.test_schematic_symbol_detector \
  tests.test_orbitbrief_pdf_schematic_detections \
  tests.test_schematic_quantity_conflicts \
  tests.test_schematic_raster_fallback \
  tests.test_gold_compare_schematic \
  tests.test_schematic_debug_overlays \
  tests.test_schematic_boss_review_fixes \
  tests.test_schematic_round3_fixes \
  tests.test_schematic_round4_fixes \
  tests.test_schematic_richer_atoms \
  tests.test_schematic_schedules \
  tests.test_orbitbrief_envelope_drawings \
  tests.test_schematic_line_runs \
  tests.test_schematic_real_gold \
  tests.test_schematic_pm_enrichment \
  tests.test_schematic_real_world_stress
```

CI on Linux runs the full pytest grid.

The synthetic gold corpus lives under `real_data_cases/SCHEMATIC_*/`
and is rebuildable via:

```bash
python scripts/_build_schematic_gold_corpus.py
```
