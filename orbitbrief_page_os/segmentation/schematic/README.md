# `orbitbrief_page_os.segmentation.schematic`

Legend-first schematic intelligence for the PDF parser.

```
PDF page → legend_locator.locate_legend_candidates()
        → legend_parser.parse_legend()  →  ParsedLegend
        → legend_resolver.LegendResolver  →  ResolvedLegend per drawing page
        → app.parsers.schematic_atom_emitters.intersect_with_pack()
                                              → DetectionTargetSet
        → symbol_detector.detect_symbols()  →  list[SymbolDetection]
        → app.parsers.schematic_atom_emitters.emit_*  →  atoms + derived JSONs
```

All modules in this package are deterministic, CPU-only, and have no
runtime LLM. Every emitted atom carries page + bbox + crop hash so
`app.core.source_replay` can verify it independently.

## Modules

| File | What it does |
|---|---|
| `legend_locator.py` | Layered candidate detector (text rules, header pairs, continuation hints, optional ONNX classifier). |
| `legend_parser.py` | Turns a candidate region into `ParsedLegend` with tabular + inline row parsers. |
| `legend_resolver.py` | Document-level index that resolves the right legend for each drawing page. |
| `symbol_detector.py` | Text-tag + glyph-template detector with deterministic NMS. |
| `raster.py` | Page rasterization + deskew utilities for image-only PDFs. |
| `ocr.py` | Optional Tesseract adapter; fails closed when binary absent. |
| `classifier.py` | Optional ONNX legend-block ranker; refuses mismatched model bytes. |
| `debug_overlay.py` | PIL renderer that draws legend + detection boxes on a rasterized page. |

## Derived files written by the PDF parser

For every PDF whose schematic pre-pass fires, the parser drops three
JSON sidecars next to the source artifact in `<stem>.derived/`:

- `schematic_legends.json` — every parsed legend, its entries, and
  the legend's source bbox.
- `schematic_targets.json` — per-page detection target sets and the
  resolution rationale.
- `schematic_detections.json` — every `SymbolDetection`, with bbox
  and crop hash so a reviewer can replay-verify by hand.

Plus, when overlays are enabled (`debug_overlay.render_overlay`),
one PNG per drawing page lands under `<stem>.derived/overlays/`.

## Determinism + provenance contracts

The parser layer guarantees:

1. **Re-running on the same PDF yields byte-identical atoms.**
   Sort order, IDs, crop hashes are all stable.
2. **Every schematic atom has a `SourceRef.locator` carrying
   `page`, `bbox`, `bbox_units="pdf_points"`, and `crop_sha256`.**
   `app.core.source_replay._verify_pdf_bbox_crop` reads exactly
   those fields.
3. **No runtime LLM.** Optional ONNX classifier is a frozen, hash-
   pinned static asset; loading a mismatched file refuses to use it.

If you change a heuristic, run:

```bash
python scripts/_schematic_smoke.py \
  tests.test_schematic_contracts \
  tests.test_pdf_bbox_replay \
  tests.test_schematic_legend_parser \
  tests.test_schematic_legend_resolver \
  tests.test_orbitbrief_pdf_schematic_legend_first \
  tests.test_schematic_symbol_detector \
  tests.test_orbitbrief_pdf_schematic_detections \
  tests.test_schematic_quantity_conflicts \
  tests.test_schematic_raster_fallback \
  tests.test_gold_compare_schematic \
  tests.test_schematic_debug_overlays
```
