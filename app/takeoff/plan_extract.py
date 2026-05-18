"""Build a structured PlanExtract for a single floor-plan page.

This module ties together everything the takeoff pipeline has learned
about a project so that a single plan page produces a self-contained,
human/LLM-readable JSON document. The contract:

    PlanExtract = sheet metadata
                + schematic region (the actual drawing area)
                + keyed-notes table (extracted with the legend's
                  table-parsing primitive, not regex)
                + zone (homerun) notes
                + device list — each device enriched with the FULL
                  legend row data looked up from the ProjectReference

The whole point is "the parser ACTUALLY KNOWS what each symbol means
on this project". Instead of every consumer looking up the YAML
defaults, downstream code reads ``plan_extract.devices[*].legend_row``
to get cable counts, terminations, mounting heights, power, remarks —
the same fields the project's own legend table declares.

The page is interpreted via ``ProjectReference`` (built once per PDF
from pages 0-2). When no reference is supplied, the per-device
``legend_row`` field stays empty and the rest of the extract still
works.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.takeoff.keynotes import parse_keynote_table_spatial
from app.takeoff.legend_extract import _resolve_symbol_column_index
from app.takeoff.multipliers import floor_label_for_title, multiplier_for_title
from app.takeoff.pdf_native import extract_page_words
from app.takeoff.plan_regions import (
    default_excluded_regions,
    default_plan_viewport,
)
from app.takeoff.sheet_classifier import classify_sheet
from app.takeoff.zones import parse_zones


SCHEMA_VERSION = "purtera.lowvoltage.plan_extract.v1"
READABLE_SCHEMA_VERSION = "purtera.lowvoltage.plan_extract.readable.v1"


# ───────────────────────── Schematic region ─────────────────────────


def _find_notes_block_bboxes(words: list, plan_viewport: dict | None) -> list[tuple[float, float, float, float]]:
    """Find bboxes of KEYED-NOTES / CABLE-ZONING-NOTES blocks on the page.

    Each match returns the bbox of the header text. The caller can
    expand that bbox downward to estimate the full notes-block region
    when subtracting from the schematic area.
    """
    targets = ("KEYED NOTES", "KEY NOTES", "CABLE ZONING NOTES", "GENERAL NOTES")
    out: list[tuple[float, float, float, float]] = []
    if not words:
        return out
    # Build a per-line scan: group words into rough lines by y, then
    # join text and look for header phrases.
    by_y: dict[int, list] = {}
    for w in words:
        key = int((w.y0 + w.y1) / 2 // 6)  # 6 pt bucket
        by_y.setdefault(key, []).append(w)
    for line_words in by_y.values():
        line_words.sort(key=lambda w: w.x0)
        line_text = " ".join((w.text or "").strip() for w in line_words).upper()
        for hdr in targets:
            if hdr in line_text:
                xs = [w.x0 for w in line_words] + [w.x1 for w in line_words]
                ys = [w.y0 for w in line_words] + [w.y1 for w in line_words]
                out.append((min(xs), min(ys), max(xs), max(ys)))
                break
    return out


def identify_schematic_region(
    *,
    pdf_path: Path,
    page_index: int,
) -> dict[str, Any]:
    """Compute the schematic-drawing region for a plan page.

    Returns:

        {
          "plan_viewport_pt":    [x0, y0, x1, y1],
          "excluded_regions_pt": [[x0, y0, x1, y1], ...],
          "notes_block_bboxes_pt": [...],  # KEYED NOTES / similar header bboxes
        }

    The schematic region is what's INSIDE ``plan_viewport_pt`` and OUTSIDE
    ``excluded_regions_pt + notes_block_bboxes_pt``. We surface the
    components rather than a single polygon because callers can choose
    how strict they want to be (e.g. only reject devices whose center
    falls inside a notes block, vs. requiring full containment).
    """
    import fitz

    pdf_doc = fitz.open(str(pdf_path))
    try:
        page = pdf_doc[page_index]
        pv = default_plan_viewport(page)
        ex = default_excluded_regions(page) or []
        words = extract_page_words(page)
    finally:
        pdf_doc.close()

    notes_bboxes = _find_notes_block_bboxes(words, pv)
    return {
        "plan_viewport_pt": [pv.x0, pv.y0, pv.x1, pv.y1] if pv is not None else None,
        "excluded_regions_pt": [[r.x0, r.y0, r.x1, r.y1] for r in ex],
        "notes_block_bboxes_pt": [list(b) for b in notes_bboxes],
    }


# ──────────────────────── Keyed-notes extraction ────────────────────


def extract_plan_keyed_notes_table(
    *,
    pdf_path: Path,
    page_index: int,
) -> list[dict[str, str]]:
    """Extract the keyed-notes table on a plan page using spatial pairing.

    Reuses :func:`parse_keynote_table_spatial` (the same primitive the
    overlay's keynote-callout matcher uses), so the parser sees keynotes
    the same way on the legend and on plan pages.

    Returns a list of ``{"number": str, "text": str}`` records.
    """
    import fitz

    pdf_doc = fitz.open(str(pdf_path))
    try:
        page = pdf_doc[page_index]
        page_text = ""
        try:
            page_text = page.get_text("text") or ""
        except Exception:
            page_text = ""
        words = extract_page_words(page)
    finally:
        pdf_doc.close()

    table = parse_keynote_table_spatial(
        page_index=page_index,
        page_text=page_text,
        page_words=words,
    )
    return [
        {"number": num, "text": text}
        for num, text in sorted(
            table.notes.items(),
            key=lambda kv: int(kv[0]) if kv[0].isdigit() else 9999,
        )
    ]


# ───────────────────────── Reference lookup ─────────────────────────


def _iter_legend_rows(project_reference: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Yield every legend row from the reference doc with its column data.

    Each yielded entry is a dict that includes ``_symbol`` (the SYMBOL
    column's text content) plus all other column→value pairs.
    """
    if not project_reference:
        return []
    legend = project_reference.get("legend")
    if legend is None:
        return []
    legend_list = legend if isinstance(legend, list) else [legend]
    out: list[dict[str, Any]] = []
    for leg in legend_list:
        for table in leg.get("tables", []) or []:
            for section in table.get("sections", []) or []:
                cols = section.get("column_headers") or []
                sym_idx = _resolve_symbol_column_index(cols)
                if sym_idx is None:
                    continue
                sym_col_name = (cols[sym_idx].get("text") or "SYMBOL").strip()
                for row in section.get("rows") or []:
                    cells_by_header = row.get("cells_by_header") or {}
                    sym_text = (cells_by_header.get(sym_col_name) or "").strip()
                    if not sym_text:
                        continue
                    entry = dict(cells_by_header)
                    entry["_symbol"] = sym_text
                    entry["_section_title"] = section.get("title")
                    out.append(entry)
    return out


def find_legend_row_for_symbol(
    *, raw_symbol: str, project_reference: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Look up the first legend row whose SYMBOL column matches ``raw_symbol``.

    Match is case-insensitive on exact text equality. Returns ``None`` if
    nothing matches or no reference was supplied.
    """
    if not project_reference:
        return None
    target = raw_symbol.strip().upper()
    for row in _iter_legend_rows(project_reference):
        sym = (row.get("_symbol") or "").strip().upper()
        if sym == target:
            return row
    return None


# ──────────────────────── PlanExtract builder ───────────────────────


def build_plan_extract(
    *,
    pdf_path: Path,
    page_index: int,
    project_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a single PlanExtract document for one page.

    The page is run through:
      * sheet_classifier   → page_type, in_scope, floor labels
      * multipliers        → levels_represented + multiplier
      * identify_schematic_region → drawing area + excluded regions
      * extract_plan_keyed_notes_table → numbered keyed notes
      * parse_zones        → homerun zone notes
      * build_low_voltage_takeoff (low-level, per-page) for devices
      * find_legend_row_for_symbol → enriches each device with the
        FULL legend row from the project reference

    Returns the JSON document.
    """
    started = time.perf_counter()
    pdf_path = Path(pdf_path)
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_pdf": str(pdf_path),
        "page_index": page_index,
        "sheet": {},
        "schematic_region": {},
        "keyed_notes": [],
        "zone_notes": [],
        "devices": [],
        "summary": {},
        "elapsed_seconds": 0.0,
    }

    import fitz

    pdf_doc = fitz.open(str(pdf_path))
    try:
        page = pdf_doc[page_index]
        page_text = ""
        try:
            page_text = page.get_text("text") or ""
        except Exception:
            page_text = ""
    finally:
        pdf_doc.close()

    sheet = classify_sheet(page_index, page_text)
    levels, mult = multiplier_for_title(sheet.sheet_name or "")
    out["sheet"] = {
        "page_index": page_index,
        "sheet_number": sheet.sheet_number,
        "sheet_name": sheet.sheet_name,
        "page_type": sheet.page_type,
        "in_scope": sheet.in_scope,
        "scope_reason": sheet.scope_reason,
        "floor_label": floor_label_for_title(sheet.sheet_name or ""),
        "levels_represented": levels,
        "multiplier": mult,
    }

    out["schematic_region"] = identify_schematic_region(
        pdf_path=pdf_path, page_index=page_index
    )
    out["keyed_notes"] = extract_plan_keyed_notes_table(
        pdf_path=pdf_path, page_index=page_index
    )

    zones = parse_zones(page_text) if sheet.in_scope else []
    out["zone_notes"] = [
        {
            "raw_text": z.raw_text,
            "target": z.target,
            "target_level": z.target_level,
            "levels": z.levels,
            "applies_to_all_levels": z.applies_to_all_levels,
        }
        for z in zones
    ]

    # Devices — only for in-scope plan pages. We delegate to the
    # existing per-page takeoff slice rather than building a custom
    # one-page detector, so the device counts on this extract match
    # exactly what ``build_low_voltage_takeoff`` produces on the
    # whole PDF.
    devices_out: list[dict[str, Any]] = []
    if sheet.in_scope and sheet.page_type in {"floor_plan", "typical_plan", "equipment_room"}:
        from app.takeoff.pipeline import build_low_voltage_takeoff
        try:
            full_takeoff = build_low_voltage_takeoff(pdf_path)
            page_devices = [d for d in full_takeoff.devices if d.page_index == page_index]
        except Exception:
            page_devices = []
        for d in page_devices:
            legend_row = find_legend_row_for_symbol(
                raw_symbol=d.raw_symbol, project_reference=project_reference
            )
            devices_out.append({
                "id": d.id,
                "symbol": d.raw_symbol,
                "normalized_class": d.normalized_class,
                "system": d.system,
                "bbox_pt": [d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1],
                "floor_label": d.floor_label,
                "levels_represented": list(d.levels_represented),
                "multiplier": d.multiplier,
                "room_guess": d.room_guess,
                "keynote": d.keynote,
                "keynote_text": d.keynote_text,
                "home_run_to": d.home_run_to,
                "home_run_level": d.home_run_level,
                "zone_notes": list(d.zone_notes),
                "review_flags": list(d.review_flags),
                "confidence": d.confidence,
                "legend_row": legend_row,
            })
    out["devices"] = devices_out

    # Summary stats.
    by_class = Counter(d["normalized_class"] for d in devices_out if d.get("normalized_class"))
    by_symbol = Counter(d["symbol"] for d in devices_out if d.get("symbol"))
    out["summary"] = {
        "device_count_total": len(devices_out),
        "by_normalized_class": dict(by_class),
        "by_symbol": dict(by_symbol),
        "keyed_notes_count": len(out["keyed_notes"]),
        "zone_notes_count": len(out["zone_notes"]),
        "reference_legend_rows_available": len(_iter_legend_rows(project_reference)),
        "devices_with_legend_row_lookup_hit": sum(1 for d in devices_out if d.get("legend_row")),
    }
    out["elapsed_seconds"] = time.perf_counter() - started
    return out


def plan_extract_to_readable(doc: dict[str, Any]) -> dict[str, Any]:
    """Project a PlanExtract into the clean readable view (LLM-friendly).

    Drops bbox / id noise. Each device's ``legend_row`` is kept as-is
    because it's already the project's own legend data — that's exactly
    what we want a consumer to read.
    """
    sheet = doc.get("sheet") or {}
    devs_in = doc.get("devices") or []
    devs_out: list[dict[str, Any]] = []
    for d in devs_in:
        entry: dict[str, Any] = {
            "symbol": d.get("symbol"),
            "normalized_class": d.get("normalized_class"),
            "system": d.get("system"),
            "room_guess": d.get("room_guess"),
            "home_run_to": d.get("home_run_to"),
            "keynote": d.get("keynote"),
            "keynote_text": d.get("keynote_text"),
            "multiplier": d.get("multiplier"),
            "review_flags": d.get("review_flags") or [],
        }
        legend_row = d.get("legend_row")
        if legend_row:
            # Drop the synthetic helper fields so the LLM sees the same
            # column → value dict the project's legend table provides.
            clean = {k: v for k, v in legend_row.items() if not k.startswith("_")}
            section_title = legend_row.get("_section_title")
            if section_title:
                entry["legend_section"] = section_title
            entry["legend_row"] = clean
        devs_out.append(entry)
    return {
        "schema_version": READABLE_SCHEMA_VERSION,
        "source": Path(doc.get("source_pdf", "")).name or "(unknown)",
        "page_index": doc.get("page_index"),
        "sheet": sheet,
        "keyed_notes": doc.get("keyed_notes") or [],
        "zone_notes": doc.get("zone_notes") or [],
        "devices": devs_out,
        "summary": doc.get("summary") or {},
    }


def write_plan_extract(
    *,
    pdf_path: Path,
    page_index: int,
    out_dir: Path,
    project_reference: dict[str, Any] | None = None,
    filename_stem: str | None = None,
) -> dict[str, Path]:
    """Build and write a PlanExtract for one page.

    Writes ``{stem}.json`` (full) and ``{stem}.readable.json`` (clean).
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = build_plan_extract(
        pdf_path=pdf_path,
        page_index=page_index,
        project_reference=project_reference,
    )
    stem = filename_stem or f"page_{page_index:02d}"
    full_path = out_dir / f"{stem}.json"
    readable_path = out_dir / f"{stem}.readable.json"
    full_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    readable_path.write_text(
        json.dumps(plan_extract_to_readable(doc), indent=2),
        encoding="utf-8",
    )
    return {"full": full_path, "readable": readable_path}


__all__ = [
    "SCHEMA_VERSION",
    "READABLE_SCHEMA_VERSION",
    "build_plan_extract",
    "extract_plan_keyed_notes_table",
    "find_legend_row_for_symbol",
    "identify_schematic_region",
    "plan_extract_to_readable",
    "write_plan_extract",
]
