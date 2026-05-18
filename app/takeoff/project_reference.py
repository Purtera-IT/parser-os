"""Build a Project Reference document from a PDF's intro pages.

Every drawing set has the same opening pattern: the first 1-3 pages
define WHAT the rest of the set is talking about — project specs,
the legend that maps symbols to device classes, and a component
schedule that pins down vendors / part numbers / quantities. Pages 0,
1, and 2 (T0.00 / T0.01 / T0.02 on a Cooper Carry T-set; equivalent
prefixes on other firms) are reference material. Every page after
them USES that reference.

This module produces a single :class:`ProjectReference` JSON document
that captures the reference layer in one place. The rest of the
takeoff pipeline (legend-aware device detection, keynote interpretation,
quote line generation) reads it to answer questions like:

  * What does the ``WN`` symbol mean on a plan?  → reference.legend
  * What's the spec for "structured cabling"?     → reference.spec
  * What's the project's preferred CAT6 vendor?   → reference.schedule

Two outputs:
  * ``project_reference.json``           — full structural data
  * ``project_reference.readable.json``  — clean LLM-consumable view

Universal across firms: classification is by page_type (set by
``sheet_classifier``), not by hardcoded sheet number. Drop a Cooper
Carry / NTI / Newcomb & Boyd / etc. drawing set and the reference
builder finds whichever pages were typed as ``spec`` / ``legend`` /
``component_schedule``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.takeoff.legend_extract import (
    extract_legend,
    legend_doc_to_readable,
    crop_symbol_icons,
    _resolve_symbol_column_index,
)
from app.takeoff.sheet_classifier import classify_sheet


SCHEMA_VERSION = "purtera.lowvoltage.project_reference.v1"
READABLE_SCHEMA_VERSION = "purtera.lowvoltage.project_reference.readable.v1"


def _classify_intro_pages(pdf_path: Path, scan_first_n: int = 6) -> dict[str, list[int]]:
    """Walk the first ``scan_first_n`` pages and bucket them by page_type.

    Returns a dict like ``{"spec": [0], "legend": [1], "component_schedule": [2]}``.
    Pages whose ``page_type`` is anything else are ignored — the
    Project Reference only cares about intro / reference pages.
    """
    import fitz

    buckets: dict[str, list[int]] = {}
    pdf_doc = fitz.open(str(pdf_path))
    try:
        n_pages = min(scan_first_n, pdf_doc.page_count)
        for page_index in range(n_pages):
            page = pdf_doc[page_index]
            try:
                page_text = page.get_text("text") or ""
            except Exception:
                page_text = ""
            sheet = classify_sheet(page_index, page_text)
            # Stop scanning once we hit a real plan page — the reference
            # layer ends at the first plan / typical / equipment_room sheet.
            if sheet.page_type in {"floor_plan", "typical_plan", "equipment_room"}:
                break
            if sheet.page_type in {"spec", "legend", "component_schedule"}:
                buckets.setdefault(sheet.page_type, []).append(page_index)
    finally:
        pdf_doc.close()
    return buckets


def _extract_spec_page(pdf_path: Path, page_index: int) -> dict[str, Any]:
    """Read a spec page using the existing structured_extract pipeline.

    Returns a small dict that mirrors what ``structured.json`` holds for
    that page, but extracts the most useful bits (top-level section
    headings + their paragraph texts) for direct consumption.
    """
    try:
        from orbitbrief_page_os.segmentation.core.pipeline import detect
        from orbitbrief_page_os.segmentation.structured_extract import extract_structured
    except Exception as exc:  # pragma: no cover - env-specific
        return {"page_index": page_index, "skipped_reason": f"missing_dependency: {exc!r}"}

    try:
        result, _rgb = detect(str(pdf_path), page_index=page_index)
        # extract_structured expects the same overlay payload that
        # OrbitBriefPdfParser builds — minimal shim here.
        payload = {
            "pdf": str(pdf_path),
            "page": page_index,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "debug_stats": result.debug_stats,
            "boxes": [
                {
                    "box_id": b.box_id,
                    "rect": [b.rect.x0, b.rect.y0, b.rect.x1, b.rect.y1],
                    "px_bbox": list(b.px_bbox),
                    "color": b.color,
                    "nested_depth": b.nested_depth,
                    "parent_box_id": b.parent_box_id,
                    "children_count": b.children_count,
                }
                for b in result.boxes
            ],
        }
        structured = extract_structured(payload, pdf_path=str(pdf_path))
    except Exception as exc:  # pragma: no cover - extraction can fail
        return {"page_index": page_index, "skipped_reason": f"extract_failed: {exc!r}"}

    # Flatten the hierarchical sections into a readable list of
    # heading-grouped paragraphs.
    flat_sections: list[dict[str, Any]] = []
    def walk(secs):
        for s in secs:
            heading = (s.get("heading") or "").strip()
            blocks = s.get("blocks") or []
            paras = [
                (b.get("text") or "").strip()
                for b in blocks
                if b.get("kind") == "paragraph" and (b.get("text") or "").strip()
            ]
            bullets = []
            for b in blocks:
                if b.get("kind") == "bullet_list":
                    for item in (b.get("items") or []):
                        text = (item.get("text") or "").strip() if isinstance(item, dict) else str(item).strip()
                        if text:
                            bullets.append(text)
            if heading or paras or bullets:
                flat_sections.append({
                    "heading": heading,
                    "paragraphs": paras,
                    "bullets": bullets,
                })
            walk(s.get("subsections") or [])
    walk(structured.get("sections") or [])

    return {
        "page_index": page_index,
        "sections": flat_sections,
        "section_count": len(flat_sections),
    }


def build_project_reference(
    *,
    pdf_path: Path,
    icons_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the Project Reference document.

    When ``icons_dir`` is provided, the legend page's symbol icons are
    cropped to PNG files under that directory and the icon map gets
    included in the reference so consumers can render them alongside.
    """
    started = time.perf_counter()
    pdf_path = Path(pdf_path)
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_pdf": str(pdf_path),
        "intro_pages": {},
        "spec":     None,
        "legend":   None,
        "schedule": None,
        "summary": {},
        "elapsed_seconds": 0.0,
    }

    buckets = _classify_intro_pages(pdf_path)
    out["intro_pages"] = buckets

    # SPEC page(s) — typically T0.00. We support 0 or 1 spec page for now;
    # if there are multiple, we read them all and store as a list.
    spec_records: list[dict[str, Any]] = []
    for page_index in buckets.get("spec", []) or []:
        spec_records.append(_extract_spec_page(pdf_path, page_index))
    if spec_records:
        out["spec"] = spec_records if len(spec_records) > 1 else spec_records[0]

    # LEGEND page(s) — typically T0.01.
    legend_records: list[dict[str, Any]] = []
    for page_index in buckets.get("legend", []) or []:
        leg = extract_legend(pdf_path=pdf_path, page_index=page_index)
        # Optional icon crop.
        if icons_dir is not None:
            icons_dir = Path(icons_dir)
            icons_dir.mkdir(parents=True, exist_ok=True)
            icon_map = crop_symbol_icons(
                pdf_path=pdf_path,
                doc=leg,
                icons_dir=icons_dir,
                zoom=4.0,
            )
            leg["icon_map"] = {f"{k[0]}/{k[1]}": v for k, v in icon_map.items()}
        legend_records.append(leg)
    if legend_records:
        out["legend"] = legend_records if len(legend_records) > 1 else legend_records[0]

    # COMPONENT SCHEDULE page(s) — typically T0.02. Same table-extraction
    # primitive — the schedule page has the same tabular structure as
    # the legend page, just with different columns (part number, vendor,
    # description, etc.). Reusing extract_legend gets us the structure
    # for free without writing a new parser.
    schedule_records: list[dict[str, Any]] = []
    for page_index in buckets.get("component_schedule", []) or []:
        sched = extract_legend(pdf_path=pdf_path, page_index=page_index)
        schedule_records.append(sched)
    if schedule_records:
        out["schedule"] = schedule_records if len(schedule_records) > 1 else schedule_records[0]

    # Summary stats.
    def _legend_rows(rec: Any) -> int:
        if isinstance(rec, list):
            return sum(_legend_rows(r) for r in rec)
        if not isinstance(rec, dict):
            return 0
        return sum(
            len(s.get("rows") or [])
            for t in (rec.get("tables") or [])
            for s in (t.get("sections") or [])
        )
    def _spec_paragraphs(rec: Any) -> int:
        if isinstance(rec, list):
            return sum(_spec_paragraphs(r) for r in rec)
        if not isinstance(rec, dict):
            return 0
        return sum(len(s.get("paragraphs") or []) for s in (rec.get("sections") or []))

    out["summary"] = {
        "spec_pages":     len(spec_records),
        "legend_pages":   len(legend_records),
        "schedule_pages": len(schedule_records),
        "spec_paragraphs":    _spec_paragraphs(out.get("spec")),
        "legend_rows_total":  _legend_rows(out.get("legend")),
        "schedule_rows_total": _legend_rows(out.get("schedule")),
    }
    out["elapsed_seconds"] = time.perf_counter() - started
    return out


def project_reference_to_readable(doc: dict[str, Any]) -> dict[str, Any]:
    """Project the full reference doc into a clean LLM/human-readable view.

    Drops bbox / box_id / pixel-coord noise. What remains:

    * ``spec``: list of {heading, paragraphs[], bullets[]}
    * ``legend``: list of {title, columns, rows[ {col_name: value, ...} ]}
                  exactly matching ``legend.readable.json``'s shape so
                  downstream consumers can use the same parser
    * ``schedule``: same shape as ``legend``
    """
    out: dict[str, Any] = {
        "schema_version": READABLE_SCHEMA_VERSION,
        "source": Path(doc.get("source_pdf", "")).name or "(unknown)",
        "intro_pages": doc.get("intro_pages", {}),
        "spec": None,
        "legend": None,
        "schedule": None,
        "summary": doc.get("summary", {}),
    }

    def _readable_spec(rec: Any) -> Any:
        if rec is None:
            return None
        if isinstance(rec, list):
            return [_readable_spec(r) for r in rec]
        return {
            "page_index": rec.get("page_index"),
            "sections": [
                {
                    "heading": s.get("heading"),
                    "paragraphs": s.get("paragraphs") or [],
                    "bullets": s.get("bullets") or [],
                }
                for s in (rec.get("sections") or [])
                if (s.get("heading") or s.get("paragraphs") or s.get("bullets"))
            ],
        }

    def _readable_legend(rec: Any) -> Any:
        if rec is None:
            return None
        if isinstance(rec, list):
            return [_readable_legend(r) for r in rec]
        readable = legend_doc_to_readable(rec)
        # The legend_extract.readable form already strips bboxes.
        return readable

    out["spec"]     = _readable_spec(doc.get("spec"))
    out["legend"]   = _readable_legend(doc.get("legend"))
    out["schedule"] = _readable_legend(doc.get("schedule"))
    return out


def write_project_reference(
    *,
    pdf_path: Path,
    out_dir: Path,
    crop_icons: bool = True,
) -> dict[str, Path]:
    """Build the reference and write all output files.

    Writes:
      * ``out_dir/project_reference.json``           full structural
      * ``out_dir/project_reference.readable.json``  LLM-friendly
      * ``out_dir/legend_icons/*.png``               cropped symbol icons (when crop_icons=True)

    Returns a dict of output paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    icons_dir = out_dir / "legend_icons" if crop_icons else None
    doc = build_project_reference(pdf_path=pdf_path, icons_dir=icons_dir)

    full_path = out_dir / "project_reference.json"
    readable_path = out_dir / "project_reference.readable.json"

    full_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    readable_path.write_text(
        json.dumps(project_reference_to_readable(doc), indent=2),
        encoding="utf-8",
    )

    paths: dict[str, Path] = {
        "full":     full_path,
        "readable": readable_path,
    }
    if icons_dir is not None:
        paths["icons_dir"] = icons_dir
    return paths


__all__ = [
    "SCHEMA_VERSION",
    "READABLE_SCHEMA_VERSION",
    "build_project_reference",
    "project_reference_to_readable",
    "write_project_reference",
]
