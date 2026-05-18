"""Extract a structured legend JSON from a legend-style page.

The OrbitBriefPdfParser already produces a generic ``structured.json``
(``orbitbrief.pdf.structured.v1``) for every PDF page. That schema is
hierarchical sections/blocks — great for prose pages, but for a legend
sheet what you actually want is the *legend table* itself: per-table,
per-section title + column headers + body rows where each row's cells
are keyed by their column header.

This module produces that view as a sidecar derived file
(``legend.json``) using the same segmentation pipeline + geometric
rules the QA legend overlay uses:

1. Run :func:`orbitbrief_page_os.segmentation.core.pipeline.detect`
   to get the segmentation result (BLUE wrappers + ORANGE cells).
2. For each depth-1 BLUE wrapper that has children, group its ORANGE
   cells into rows by y-baseline.
3. Walk rows: a TITLE row is a single cell spanning ≥60% of the
   wrapper's width AND ≤95 px tall (single line of text). The row
   immediately after a title with ≥2 cells is the COLUMN HEADER row.
4. Every subsequent multi-cell row up to the next title is a BODY ROW.
   Body cells are paired 1:1 with column headers by index (with each
   body row carrying both a positional ``cells`` array AND a
   ``cells_by_header`` dict for convenient consumption).
5. Cell text is extracted via PyMuPDF native words whose center falls
   inside the cell's PDF-pt bbox — same primitive the rest of the
   takeoff layer uses.

Deterministic: same PDF + same page_index → byte-identical legend.json
output (modulo formatting). The function never raises into the parse
path — returns a summary dict with ``skipped_reason`` set on failure.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.takeoff.pdf_native import extract_page_words


SCHEMA_VERSION = "purtera.lowvoltage.legend.v1"

# Same constants the QA legend overlay uses — keep them in lockstep so
# what you see in the overlay is exactly what gets extracted here.
SAME_ROW_TOL_PX = 18
TITLE_WIDTH_FRAC = 0.60
MAX_TITLE_HEIGHT_PX = 95


def _text_inside_pt_bbox(words: list, x0: float, y0: float, x1: float, y1: float) -> str:
    """Concatenate the native PDF words whose centers fall inside the rect."""
    hits: list[tuple[float, float, str]] = []
    for w in words:
        cx = (w.x0 + w.x1) / 2.0
        cy = (w.y0 + w.y1) / 2.0
        if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1:
            hits.append((cy, cx, w.text))
    hits.sort()
    joined = " ".join(t for _, _, t in hits).strip()
    return " ".join(joined.split())


def _group_rows(cells: list, tol_px: float = SAME_ROW_TOL_PX) -> list[tuple[float, list]]:
    """Sort cells by y-baseline and group into rows (within tol_px)."""
    cells_sorted = sorted(cells, key=lambda b: b.px_bbox[1])
    rows: list[tuple[float, list]] = []
    cur: list = []
    cur_y: float | None = None
    for b in cells_sorted:
        y0 = b.px_bbox[1]
        if cur_y is None:
            cur = [b]
            cur_y = y0
        elif abs(y0 - cur_y) <= tol_px:
            cur.append(b)
        else:
            rows.append((cur_y, cur))
            cur = [b]
            cur_y = y0
    if cur:
        rows.append((cur_y, cur))  # type: ignore[arg-type]
    # Within each row, sort left-to-right by x0.
    return [(y, sorted(r, key=lambda b: b.px_bbox[0])) for y, r in rows]


def extract_legend(
    *,
    pdf_path: Path,
    page_index: int,
) -> dict[str, Any]:
    """Produce a structured legend document for one page.

    Returns the document dict (caller writes it to disk). When segmentation
    or PyMuPDF is unavailable, returns ``{"schema_version": ..., "skipped_reason": ...}``.
    """
    started = time.perf_counter()
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_pdf": str(pdf_path),
        "page_index": page_index,
        "page_size_pt": None,
        "image_size_px": None,
        "tables": [],
        "summary": {},
        "elapsed_seconds": 0.0,
    }

    try:
        import fitz
        from orbitbrief_page_os.segmentation.core.pipeline import detect
    except Exception as exc:  # pragma: no cover - env-specific
        out["skipped_reason"] = f"missing_dependency: {exc!r}"
        out["elapsed_seconds"] = time.perf_counter() - started
        return out

    try:
        result, _rgb = detect(str(pdf_path), page_index=page_index)
    except Exception as exc:  # pragma: no cover - segmentation can fail
        out["skipped_reason"] = f"segmentation_failed: {exc!r}"
        out["elapsed_seconds"] = time.perf_counter() - started
        return out

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        page_w_pt = page.rect.width
        page_h_pt = page.rect.height
        words = extract_page_words(page)
    finally:
        doc.close()

    sx = result.image_width / page_w_pt if page_w_pt else 1.0
    sy = result.image_height / page_h_pt if page_h_pt else 1.0
    out["page_size_pt"] = [page_w_pt, page_h_pt]
    out["image_size_px"] = [result.image_width, result.image_height]

    def px_to_pt(px_bbox: tuple[float, float, float, float]) -> list[float]:
        x0, y0, x1, y1 = px_bbox
        return [x0 / sx, y0 / sy, x1 / sx, y1 / sy]

    blue_d1 = [
        b for b in result.boxes
        if b.color == "BLUE" and b.nested_depth == 1 and (b.children_count or 0) >= 1
    ]

    tables: list[dict[str, Any]] = []
    section_count = 0
    column_header_count = 0
    body_row_count = 0

    for table in blue_d1:
        t_x0_px, t_y0_px, t_x1_px, t_y1_px = table.px_bbox
        t_width_px = t_x1_px - t_x0_px

        # Cells fully contained in the wrapper bbox.
        inside = []
        for b in result.boxes:
            if b.color != "ORANGE":
                continue
            cx0, cy0, cx1, cy1 = b.px_bbox
            if cx0 < t_x0_px - 2 or cx1 > t_x1_px + 2:
                continue
            if cy0 < t_y0_px - 2 or cy1 > t_y1_px + 2:
                continue
            inside.append(b)
        if not inside:
            continue
        rows = _group_rows(inside)

        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        # Track which row-indices are titles / column-header rows so we
        # don't re-add them as body rows.
        consumed_row_indices: set[int] = set()
        # Within a wrapper, the FIRST section establishes the column
        # headers. Subsequent sections in the same wrapper inherit those
        # headers and the row below their title becomes a body row, not
        # a header row. This matches the Marriott / Cooper Carry style
        # where INTRUSION DETECTION declares the SYMBOL/DESCRIPTION/...
        # columns once and ACCESS CONTROL + CCTV reuse them.
        wrapper_column_headers: list[dict[str, Any]] = []
        wrapper_headers_inherited: bool = False  # True if current section is using inherited headers

        for ri, (_row_y_px, row_cells) in enumerate(rows):
            if len(row_cells) == 1:
                cell = row_cells[0]
                cw = cell.px_bbox[2] - cell.px_bbox[0]
                ch = cell.px_bbox[3] - cell.px_bbox[1]
                is_title = (
                    cw / t_width_px >= TITLE_WIDTH_FRAC
                    and ch <= MAX_TITLE_HEIGHT_PX
                )
                if is_title:
                    title_bbox_pt = px_to_pt(cell.px_bbox)
                    title_text = _text_inside_pt_bbox(words, *title_bbox_pt)
                    column_headers: list[dict[str, Any]] = []
                    inherited = False
                    if not wrapper_column_headers:
                        # First title in this wrapper — try to capture
                        # column headers from the next row.
                        if ri + 1 < len(rows):
                            _, next_cells = rows[ri + 1]
                            if len(next_cells) >= 2:
                                for nc in next_cells:
                                    bb = px_to_pt(nc.px_bbox)
                                    column_headers.append({
                                        "box_id": nc.box_id,
                                        "bbox_pt": bb,
                                        "text": _text_inside_pt_bbox(words, *bb),
                                    })
                                consumed_row_indices.add(ri + 1)
                                wrapper_column_headers = column_headers
                    else:
                        # Subsequent title — inherit headers, don't
                        # consume the next row (it's the first body row).
                        column_headers = wrapper_column_headers
                        inherited = True
                    current = {
                        "title": title_text,
                        "title_box_id": cell.box_id,
                        "title_bbox_pt": title_bbox_pt,
                        "column_headers": column_headers,
                        "column_headers_inherited": inherited,
                        "rows": [],
                    }
                    sections.append(current)
                    consumed_row_indices.add(ri)
                    continue

            # Body row — append to current section (or open an
            # anonymous section if no title has been seen yet).
            if ri in consumed_row_indices:
                continue
            if current is None:
                # Top of wrapper without a title — first multi-cell row
                # acts as column headers, rest are body.
                if len(row_cells) >= 2:
                    column_headers = []
                    for nc in row_cells:
                        bb = px_to_pt(nc.px_bbox)
                        column_headers.append({
                            "box_id": nc.box_id,
                            "bbox_pt": bb,
                            "text": _text_inside_pt_bbox(words, *bb),
                        })
                    current = {
                        "title": None,
                        "title_box_id": None,
                        "title_bbox_pt": None,
                        "column_headers": column_headers,
                        "rows": [],
                    }
                    sections.append(current)
                    consumed_row_indices.add(ri)
                    continue
                # Single-cell row at the very top that isn't a title —
                # treat as an anonymous standalone row.
                current = {
                    "title": None, "title_box_id": None,
                    "title_bbox_pt": None,
                    "column_headers": [], "rows": [],
                }
                sections.append(current)

            # Append the row to current.
            cells_out = []
            for c in row_cells:
                bb = px_to_pt(c.px_bbox)
                cells_out.append({
                    "box_id": c.box_id,
                    "bbox_pt": bb,
                    "text": _text_inside_pt_bbox(words, *bb),
                })
            cells_by_header = {}
            if current.get("column_headers"):
                hdrs = current["column_headers"]
                for idx, cell_entry in enumerate(cells_out):
                    if idx < len(hdrs):
                        key = hdrs[idx].get("text") or f"_col{idx}"
                    else:
                        key = f"_col{idx}"
                    cells_by_header[key] = cell_entry["text"]
            current["rows"].append({
                "cells": cells_out,
                "cells_by_header": cells_by_header,
            })
            body_row_count += 1

        section_count += len(sections)
        column_header_count += sum(len(s["column_headers"]) for s in sections)

        tables.append({
            "box_id": table.box_id,
            "bbox_pt": px_to_pt(table.px_bbox),
            "sections": sections,
        })

    out["tables"] = tables
    out["summary"] = {
        "tables_detected": len(tables),
        "sections_total": section_count,
        "column_header_cells_total": column_header_count,
        "body_rows_total": body_row_count,
    }
    out["elapsed_seconds"] = time.perf_counter() - started
    return out


def write_legend_extract(
    *,
    pdf_path: Path,
    page_index: int,
    out_path: Path,
) -> dict[str, Any]:
    """Run :func:`extract_legend` and write the JSON to ``out_path``.

    Returns the same document dict. ``out_path`` is created (with parent
    dirs) if missing.
    """
    import json

    doc = extract_legend(pdf_path=pdf_path, page_index=page_index)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def legend_doc_to_markdown(doc: dict[str, Any]) -> str:
    """Render the legend document as human/LLM-readable markdown.

    Output format is one ``## Table N: <title>`` section per detected
    section, followed by a Markdown table with each body row's cells
    keyed to the column headers. Inherited column headers are noted
    in italics so the reader knows where they came from. Empty
    sections render with a small ``(no body rows)`` note.
    """
    lines: list[str] = []
    src = Path(doc.get("source_pdf", "")).name or "(unknown source)"
    lines.append(f"# Legend Page Extract — page index {doc.get('page_index')}")
    lines.append("")
    lines.append(f"- **schema**: `{doc.get('schema_version', SCHEMA_VERSION)}`")
    lines.append(f"- **source**: `{src}`")
    summary = doc.get("summary", {}) or {}
    if summary:
        lines.append(
            f"- **detected**: {summary.get('tables_detected', 0)} tables · "
            f"{summary.get('sections_total', 0)} sections · "
            f"{summary.get('column_header_cells_total', 0)} header cells · "
            f"{summary.get('body_rows_total', 0)} body rows"
        )
    lines.append("")

    def _esc(text: str) -> str:
        # Markdown table-cell escaping: pipes / newlines break tables.
        if not text:
            return ""
        return text.replace("|", "\\|").replace("\n", " ").strip()

    section_idx = 0
    for ti, table in enumerate(doc.get("tables", []) or []):
        for si, section in enumerate(table.get("sections", []) or []):
            section_idx += 1
            title = section.get("title") or "(untitled section)"
            cols = section.get("column_headers") or []
            rows = section.get("rows") or []
            inherited = section.get("column_headers_inherited", False)
            lines.append(f"## {section_idx}. {title}")
            lines.append("")
            if cols:
                hdr_note = " *(inherited from preceding section)*" if inherited else ""
                lines.append(
                    f"_columns: {len(cols)} · body rows: {len(rows)}_{hdr_note}"
                )
                lines.append("")
                header_texts = [_esc(c.get("text") or f"col{i+1}") for i, c in enumerate(cols)]
                lines.append("| " + " | ".join(header_texts) + " |")
                lines.append("|" + "|".join(["---"] * len(cols)) + "|")
                for row in rows:
                    by_hdr = row.get("cells_by_header") or {}
                    if by_hdr:
                        # Use header-keyed cells when available.
                        row_vals = []
                        for h_text in header_texts:
                            # The raw key in cells_by_header is the
                            # (un-escaped) column header text.
                            raw_key = h_text.replace("\\|", "|")
                            row_vals.append(_esc(by_hdr.get(raw_key, "")))
                    else:
                        # Fallback: positional cells.
                        cells = row.get("cells") or []
                        row_vals = [_esc(c.get("text", "")) for c in cells]
                    # Pad/truncate to header count for valid markdown.
                    while len(row_vals) < len(cols):
                        row_vals.append("")
                    row_vals = row_vals[: len(cols)]
                    lines.append("| " + " | ".join(row_vals) + " |")
                lines.append("")
            else:
                lines.append(f"_no columns detected · {len(rows)} body row(s)_")
                lines.append("")
                for row in rows:
                    cells = row.get("cells") or []
                    txt = " · ".join(_esc(c.get("text", "")) for c in cells if c.get("text"))
                    if txt:
                        lines.append(f"- {txt}")
                if rows:
                    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_legend_markdown(
    *,
    doc: dict[str, Any],
    out_path: Path,
) -> Path:
    """Write the markdown projection of ``doc`` to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(legend_doc_to_markdown(doc), encoding="utf-8")
    return out_path


__all__ = [
    "SCHEMA_VERSION",
    "extract_legend",
    "legend_doc_to_markdown",
    "write_legend_extract",
    "write_legend_markdown",
]
