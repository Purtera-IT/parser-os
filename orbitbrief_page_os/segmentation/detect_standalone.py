"""Compatibility CLI/API for Parser OS visible-box overlays.

The public flags and imports are preserved.  Internally this file delegates to
`core.pipeline`, where new passes are registered instead of extending the core
detector file.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from .core.config import Cfg
from .core.models import Rect, VisibleBox, VisibleBoxResult
from .core.pipeline import build_pipeline, detect, render_overlay
from .overlay_layers import OverlayLayer, parse_layers_arg


def _box_to_dict(b: VisibleBox) -> dict[str, Any]:
    out: dict[str, Any] = {
        "box_id": b.box_id,
        "rect": [b.rect.x0, b.rect.y0, b.rect.x1, b.rect.y1],
        "area_pt2": b.area_pt2,
        "fill_ratio": b.fill_ratio,
        "nested_depth": b.nested_depth,
        "is_outer_wrapper": b.is_outer_wrapper,
        "parent_box_id": b.parent_box_id,
        "color": b.color,
        "px_bbox": list(b.px_bbox),
        "children_count": b.children_count,
        "synthetic": b.synthetic,
    }
    # Surface the marker attributes that downstream passes attach to a
    # box via ``object.__setattr__``.  These drive overlay rendering
    # (yellow footer / red sub-header / green & purple bullet bands)
    # and are required for the color-driven structured extractor to
    # tell what role each box should play.
    for marker in (
        "cover_footer_band",
        "subhdr_red_band",
        "subbullet_green_band",
        "subbullet_purple_band",
        "is_subheader",
    ):
        if getattr(b, marker, False):
            out[marker] = True
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--out", default="overlay_standalone.png", help="Output PNG path")
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--no-labels", action="store_true", help="Compatibility flag; labels are off by default in v2 QA overlays")
    ap.add_argument("--labels", action="store_true", help="Draw debug id/depth/child-count label pills")
    ap.add_argument(
        "--layers", default="all", choices=("all", "blue"),
        help=("Overlay passes: 'all' (default) or 'blue' = BLUE_FAMILY only "
              "(wrappers + title bands + *_body - no orange/cyan/green/purple)."),
    )
    ap.add_argument(
        "--json-out", default=None,
        help="Optional structured detection JSON. Added without changing existing flags.",
    )
    ap.add_argument(
        "--extraction-out",
        default=None,
        help="After --json-out, also write <base>.extraction.json and <base>.extraction.md "
        "text artifacts (use same path as --json-out, e.g. out.json → out.extraction.md).",
    )
    ap.add_argument(
        "--structured-out",
        default=None,
        help="After --json-out, also write a clean color-driven structured JSON "
        "to this path (e.g. compiled_artifacts/page.structured.json).",
    )
    ap.add_argument(
        "--pipeline-plan", action="store_true",
        help="Print the ordered pipeline stages before running detection.",
    )
    ap.add_argument(
        "--geometry-only", action="store_true",
        help="Disable optional text-section/mini-table synthesis for fast raster-only hard-case checks.",
    )
    args = ap.parse_args(argv)
    if args.extraction_out and not args.json_out:
        ap.error("--extraction-out requires --json-out")
    if args.structured_out and not args.json_out:
        ap.error("--structured-out requires --json-out")

    cfg = Cfg(render_scale=args.scale)
    if args.geometry_only:
        cfg = replace(cfg, detect_text_sections=False, detect_mini_tables=False)
    if args.pipeline_plan:
        print(json.dumps(build_pipeline().pass_table(), indent=2))

    result, rgb = detect(args.pdf, page_index=args.page, cfg=cfg)

    print(f"\n=== page {args.page} ===")
    s = result.debug_stats
    print(
        f"  image: {s['W']}x{s['H']}  |  merged: {s['merged']}  |  "
        f"validated: {s['validated']}  |  after_nms: {s['after_nms']}"
    )
    blue_n = sum(1 for b in result.boxes if b.color == "BLUE")
    orange_n = sum(1 for b in result.boxes if b.color == "ORANGE")
    print(f"  BLUE={blue_n}  ORANGE={orange_n}  total={len(result.boxes)}")
    print("  rejections:", {k: v for k, v in s["rejects"].items() if v > 0})

    for b in result.boxes:
        print(
            f"  [{b.color:6s}] {b.box_id:5s} depth={b.nested_depth} "
            f"ch={b.children_count:3d}  "
            f"px=({b.px_bbox[0]},{b.px_bbox[1]},{b.px_bbox[2]},{b.px_bbox[3]})  "
            f"parent={b.parent_box_id}"
        )

    layers = parse_layers_arg(args.layers)
    out = render_overlay(rgb, result, args.out, draw_labels=(args.labels and not args.no_labels), layers=layers)
    print(f"\nSaved -> {out}  (layers={args.layers!r})")

    if args.json_out:
        payload = {
            "pdf": str(args.pdf),
            "page": args.page,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "debug_stats": result.debug_stats,
            "boxes": [_box_to_dict(b) for b in result.boxes],
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"JSON -> {args.json_out}")
        if args.extraction_out:
            from .extract_overlay_text import extract_from_overlay_json, write_extraction_artifacts

            doc = extract_from_overlay_json(payload, pdf_path=args.pdf)
            paths = write_extraction_artifacts(args.extraction_out, doc)
            for k, v in paths.items():
                print(f"Extraction {k} -> {v}")
        if args.structured_out:
            from .structured_extract import extract_structured, write_structured

            structured_doc = extract_structured(payload, pdf_path=args.pdf)
            written = write_structured(args.structured_out, structured_doc)
            print(f"Structured -> {written}")
    return 0


__all__ = [
    "Cfg",
    "Rect",
    "VisibleBox",
    "VisibleBoxResult",
    "OverlayLayer",
    "parse_layers_arg",
    "build_pipeline",
    "detect",
    "render_overlay",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
