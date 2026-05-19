"""Deterministic legend parser.

Given a ``LegendCandidate`` and the page's ``TextBlock`` stream, build
a ``ParsedLegend`` whose ``entries`` enumerate the legend rows.

The parser handles two row shapes:

1. **Tabular rows.** Two columns (symbol / description) and optional
   third column (count/qty). Rows are clustered by y-band so a row
   that wraps into two physical text lines still resolves to a
   single entry.
2. **Inline rows.** ``WN - WIRELESS NODE``, ``CR = CARD READER``,
   ``CR: CARD READER``, etc. Used when the legend is a vertical list
   inside a single narrow column.

Glyph crops are out of scope for PR3 — those land in PR6 (symbol
detector) where the OpenCV crop pipeline already exists. The parser
records the symbol bbox so a future pass can crop later.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from app.core.normalizers import normalize_text
from app.parsers.schematic_models import (
    LegendScope,
    ParsedLegend,
    ParsedLegendEntry,
)
from orbitbrief_page_os.segmentation.schematic.legend_locator import (
    LegendCandidate,
    TextBlock,
)

# Inline ``SYMBOL - DESCRIPTION`` / ``SYMBOL = DESCRIPTION`` /
# ``SYMBOL: DESCRIPTION`` patterns. The symbol must be short and made
# of caps/digits/punctuation (no lowercase) to avoid eating prose
# bullets like "Note - the contractor shall..."
_INLINE_RE = re.compile(
    r"^\s*(?P<symbol>[A-Z0-9][A-Z0-9\-/_.+]{0,7})\s*(?:[-=:–—])\s*(?P<label>[A-Z0-9].{2,80})\s*$"
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _bbox_contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    return (
        inner[0] >= outer[0] - 1.0
        and inner[1] >= outer[1] - 1.0
        and inner[2] <= outer[2] + 1.0
        and inner[3] <= outer[3] + 1.0
    )


def _blocks_in_region(
    region: tuple[float, float, float, float],
    blocks: Sequence[TextBlock],
) -> list[TextBlock]:
    return [b for b in blocks if _bbox_contains(region, b.bbox)]


def _cluster_rows(blocks: Sequence[TextBlock], y_tol: float = 4.0) -> list[list[TextBlock]]:
    """Cluster blocks into rows by y-center."""

    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    rows: list[list[TextBlock]] = []
    for blk in sorted_blocks:
        y_center = (blk.bbox[1] + blk.bbox[3]) / 2.0
        placed = False
        for row in rows:
            ref = (row[0].bbox[1] + row[0].bbox[3]) / 2.0
            if abs(y_center - ref) <= y_tol:
                row.append(blk)
                placed = True
                break
        if not placed:
            rows.append([blk])
    for row in rows:
        row.sort(key=lambda b: b.bbox[0])
    return rows


def _row_looks_like_header(row: Sequence[TextBlock]) -> bool:
    tokens = " | ".join(_norm(b.text) for b in row)
    head_keys = ("symbol", "description", "meaning", "name", "definition", "count", "qty", "remarks", "abbreviation")
    return any(k in tokens for k in head_keys) and len(tokens) <= 80


def _looks_like_symbol_token(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 8:
        return False
    # All-caps / digits / common punctuation. Reject lowercase prose.
    if any(ch.islower() for ch in t):
        return False
    if not re.match(r"^[A-Z0-9][A-Z0-9\-/_.+]*$", t):
        return False
    return True


def _parse_count_token(text: str) -> float | None:
    t = text.strip()
    if not t:
        return None
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\b", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _entry_from_tabular_row(
    page_index: int,
    row: Sequence[TextBlock],
) -> ParsedLegendEntry | None:
    if len(row) < 2:
        return None
    cells = list(row)
    symbol_cell = cells[0]
    description_cell = cells[1]
    count_cell = cells[2] if len(cells) >= 3 else None
    if not _looks_like_symbol_token(symbol_cell.text):
        return None
    label_text = description_cell.text.strip()
    if len(label_text) < 3:
        return None
    count_val = _parse_count_token(count_cell.text) if count_cell else None
    notes_tail: list[str] = []
    for extra in cells[3:]:
        t = extra.text.strip()
        if t:
            notes_tail.append(t)
    return ParsedLegendEntry.make(
        page_index=page_index,
        label_text=label_text,
        normalized_label=normalize_text(label_text),
        raw_symbol_text=symbol_cell.text.strip(),
        normalized_symbol_text=symbol_cell.text.strip().lower(),
        symbol_bbox_pdf=symbol_cell.bbox,
        count_column=count_val,
        notes=tuple(notes_tail),
        source_ref_locator={
            "page": page_index,
            "row_y": round((symbol_cell.bbox[1] + symbol_cell.bbox[3]) / 2.0, 2),
        },
        confidence=0.85,
    )


def _entry_from_inline_text(
    page_index: int,
    blk: TextBlock,
) -> ParsedLegendEntry | None:
    m = _INLINE_RE.match(blk.text)
    if not m:
        return None
    symbol = m.group("symbol").strip()
    label = m.group("label").strip()
    if not _looks_like_symbol_token(symbol):
        return None
    return ParsedLegendEntry.make(
        page_index=page_index,
        label_text=label,
        normalized_label=normalize_text(label),
        raw_symbol_text=symbol,
        normalized_symbol_text=symbol.lower(),
        symbol_bbox_pdf=blk.bbox,
        source_ref_locator={
            "page": page_index,
            "row_y": round((blk.bbox[1] + blk.bbox[3]) / 2.0, 2),
        },
        confidence=0.7,
    )


def parse_legend(
    *,
    candidate: LegendCandidate,
    page_blocks: Sequence[TextBlock],
    sheet_number: str | None = None,
    scope: LegendScope | None = None,
) -> ParsedLegend | None:
    """Parse a candidate region into a ``ParsedLegend``.

    Returns ``None`` only if zero entries could be extracted from the
    region. A weak header with no parseable rows is not a legend.
    Caller (PR4 resolver) decides what to do when this is None
    (typically: drop the candidate, emit a ``weak_legend`` warning).
    """

    page_index = candidate.page_index
    region_blocks = _blocks_in_region(candidate.bbox, page_blocks)
    if not region_blocks:
        return None

    entries: list[ParsedLegendEntry] = []
    rows = _cluster_rows(region_blocks)

    # First pass — tabular rows.
    used_block_ids: set[int] = set()
    for row in rows:
        if _row_looks_like_header(row):
            for b in row:
                used_block_ids.add(id(b))
            continue
        entry = _entry_from_tabular_row(page_index, row)
        if entry is not None:
            entries.append(entry)
            for b in row:
                used_block_ids.add(id(b))

    # Second pass — inline rows for whatever was not consumed tabular-side.
    for blk in region_blocks:
        if id(blk) in used_block_ids:
            continue
        entry = _entry_from_inline_text(page_index, blk)
        if entry is not None:
            entries.append(entry)

    if not entries:
        return None

    # Deterministic ordering: by symbol token, then by row.
    entries.sort(
        key=lambda e: (
            (e.normalized_symbol_text or ""),
            (e.normalized_label or ""),
        )
    )

    resolved_scope: LegendScope = scope or ("global" if "symbols & legends" in (candidate.header_text or "") else "page")

    legend_title = candidate.header_text.upper() if candidate.header_text else None

    return ParsedLegend.make(
        page_index=page_index,
        sheet_number=sheet_number,
        title=legend_title,
        scope=resolved_scope,
        entries=tuple(entries),
        source_ref_locator={
            "page": page_index,
            "bbox": list(candidate.bbox),
            "bbox_units": "pdf_points",
            "layer": candidate.layer,
        },
        confidence=max(0.0, min(1.0, candidate.score + 0.10 * (len(entries) > 0))),
    )
