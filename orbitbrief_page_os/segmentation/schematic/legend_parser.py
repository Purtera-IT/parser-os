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
    """Detect a column-header row (SYMBOL / DESCRIPTION / COUNT / ...).

    Heuristics: at least 2 cells, every cell short (<= 30 chars), and
    at least one cell text matches a canonical header keyword. The
    short-cell requirement rejects banner lines like
    "SHEET T0.01 - SYMBOL LEGEND" which contain a header keyword
    inside prose.
    """
    if len(row) < 2:
        return False
    # Header keywords. Intentionally excludes ambiguous tokens like
    # ``aff`` (which would match a data cell like ``48" AFF``) and
    # ``nic`` (which is a status marker, not a header label).
    head_keys = (
        "symbol",
        "description",
        "meaning",
        "name",
        "definition",
        "count",
        "qty",
        "remarks",
        "abbreviation",
        "mounting",
        "cable",
        "rough",
        "power",
        "termination",
        "manufacturer",
        "mfg",
        "model",
        "by others",
        "responsibility",
        "color",
        "size",
        # Common abbreviations the construction trade uses on legends.
        "mtg",
        "mnt",
        "ht",
        "cbl",
        "cnt",
        "ct",
        "pwr",
        "rem",
        "rgh",
        "term",
        "term.",
        "vendor",
        "brand",
        "scope",
        "resp",
        "part",
        "dim",
    )
    cells = [_norm(b.text) for b in row]
    # Every header cell should be short — real header rows use 1-3
    # word column labels, not full sentences.
    if any(len(c) > 30 for c in cells):
        return False
    # Data rows often have one or two cells that happen to start
    # with a keyword (``MFG NVR-X``, ``MOUNTING DETAIL``).  A real
    # header row has many such cells.  Require a majority of cells
    # to either equal a head keyword or be a header-shaped phrase.
    def _cell_is_header_like(c: str) -> bool:
        if not c:
            return False
        # Reject cells that contain digits (real header labels are
        # textual: "COUNT", not "12").
        if any(ch.isdigit() for ch in c):
            return False
        return any(c == k or c.startswith(k + " ") or c.endswith(" " + k) for k in head_keys)

    header_like = sum(1 for c in cells if _cell_is_header_like(c))
    if header_like < 2:
        return False
    # At least half the cells (rounded up) should look header-like.
    threshold = (len(cells) + 1) // 2
    return header_like >= threshold


# Maps a header cell's normalized text to the canonical attribute key
# we store on ParsedLegendEntry.attributes. Order matters — the most
# specific header is tried first so "cable count" wins over "cable".
# Patterns are matched as either substrings or whole-token aliases
# (a token like ``mtg`` matches a 1-3 word cell where it appears as a
# standalone token, but not as part of a longer word like ``image``).
_HEADER_ATTRIBUTE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Cable columns
    ("cable count", "cable_count"),
    ("cbl count", "cable_count"),
    ("cbl cnt", "cable_count"),
    ("cbl ct", "cable_count"),
    ("cable cnt", "cable_count"),
    ("cable ct", "cable_count"),
    ("cable description", "cable_description"),
    ("cbl desc", "cable_description"),
    ("cable desc", "cable_description"),
    ("cable type", "cable_type"),
    ("cable", "cable"),
    ("cbl", "cable"),
    ("strand count", "strand_count"),
    ("strands", "strand_count"),
    # Mounting / height
    ("mounting height", "mounting_height"),
    ("mtg height", "mounting_height"),
    ("mtg ht", "mounting_height"),
    ("mtg. ht", "mounting_height"),
    ("mnt height", "mounting_height"),
    ("mnt ht", "mounting_height"),
    ("ht aff", "mounting_height"),
    ("mounting", "mounting"),
    ("mtg", "mounting"),
    ("mnt", "mounting"),
    # Rough-in / install detail
    ("rough-in", "rough_in"),
    ("rough in", "rough_in"),
    ("rgh in", "rough_in"),
    ("rgh-in", "rough_in"),
    # Power requirement
    ("power requirement", "power_requirement"),
    ("power req", "power_requirement"),
    ("pwr req", "power_requirement"),
    ("power", "power_requirement"),
    ("pwr", "power_requirement"),
    # Terminations
    ("work area termination", "termination_work_area"),
    ("wa termination", "termination_work_area"),
    ("wa term", "termination_work_area"),
    ("closet termination", "termination_closet"),
    ("tr termination", "termination_closet"),
    ("idf termination", "termination_closet"),
    ("idf term", "termination_closet"),
    ("termination", "termination"),
    ("term.", "termination"),
    # Vendor info
    ("manufacturer", "mfg"),
    ("mfg", "mfg"),
    ("mfgr", "mfg"),
    ("vendor", "mfg"),
    ("brand", "mfg"),
    ("model", "model"),
    ("part number", "part_number"),
    ("part no", "part_number"),
    ("part #", "part_number"),
    # Responsibility / scope
    ("by others", "responsibility"),
    ("not in contract", "responsibility"),
    ("responsibility", "responsibility"),
    ("resp", "responsibility"),
    ("scope", "responsibility"),
    # Remarks / notes
    ("remarks", "remarks"),
    ("notes", "remarks"),
    ("rem", "remarks"),
    ("comments", "remarks"),
    # Misc
    ("color", "color"),
    ("size", "size"),
    ("dimensions", "size"),
    ("dim.", "size"),
)


def _classify_header_cell(text: str) -> str | None:
    """Return the canonical attribute key for a header cell, or None.

    ``count`` / ``qty`` headers are NOT returned here; they're handled
    separately because count is a special-cased numeric column on
    ``ParsedLegendEntry``.
    """
    n = _norm(text)
    if not n:
        return None
    # The first two columns are always symbol + description.
    if n in {"symbol", "abbr", "abbreviation", "tag"}:
        return "__symbol__"
    if n in {"description", "name", "meaning", "definition", "device"}:
        return "__label__"
    # ``__count__`` reserves the quantity column only. ``CABLE COUNT``
    # is a separate attribute column ("cable count" the cable wire
    # count, not a device count). Match attribute patterns first so
    # cable count, port count, strand count don't accidentally claim
    # the quantity slot.
    for pattern, key in _HEADER_ATTRIBUTE_PATTERNS:
        if pattern in n:
            return key
    if n in {"count", "qty", "quantity"} or n.startswith("qty ") or n.endswith(" qty"):
        return "__count__"
    return None


def _build_column_map(
    header_row: Sequence[TextBlock],
) -> list[tuple[float, float, str | None]]:
    """Return ``[(x0, x1, attribute_key), ...]`` for the parsed header.

    Used to assign each data-row cell to a typed attribute by checking
    which header span x-range the data cell falls under.
    """
    columns: list[tuple[float, float, str | None]] = []
    sorted_header = sorted(header_row, key=lambda b: b.bbox[0])
    for i, blk in enumerate(sorted_header):
        key = _classify_header_cell(blk.text)
        x0 = blk.bbox[0]
        x1 = (
            sorted_header[i + 1].bbox[0]
            if i + 1 < len(sorted_header)
            else max(blk.bbox[2], blk.bbox[0] + 60.0)
        )
        columns.append((x0, x1, key))
    return columns


def _row_to_column_cells(
    row: Sequence[TextBlock],
    columns: list[tuple[float, float, str | None]],
) -> dict[str | None, str]:
    """Bucket a data-row's cells into header columns by x-overlap."""
    bucketed: dict[str | None, list[str]] = {}
    for blk in row:
        cell_center = (blk.bbox[0] + blk.bbox[2]) / 2.0
        target_key: str | None = None
        for x0, x1, key in columns:
            if x0 - 1.0 <= cell_center <= x1 + 1.0:
                target_key = key
                break
        bucketed.setdefault(target_key, []).append(blk.text.strip())
    return {k: " ".join(parts).strip() for k, parts in bucketed.items() if parts}


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
    column_map: list[tuple[float, float, str | None]] | None = None,
) -> ParsedLegendEntry | None:
    if len(row) < 2:
        return None
    cells = list(row)

    # When we have a header-derived column map, route every cell to
    # a typed slot.  Otherwise fall back to positional parsing
    # (symbol, description, count, then notes).
    if column_map:
        bucketed = _row_to_column_cells(row, column_map)
        symbol_text = (bucketed.get("__symbol__") or "").strip()
        label_text = (bucketed.get("__label__") or "").strip()
        count_text = (bucketed.get("__count__") or "").strip()
        if not symbol_text or not _looks_like_symbol_token(symbol_text):
            return None
        if len(label_text) < 3:
            return None
        count_val = _parse_count_token(count_text) if count_text else None
        attributes: dict[str, str] = {}
        notes_tail: list[str] = []
        for key, value in bucketed.items():
            if not value:
                continue
            if key in (None, "__symbol__", "__label__", "__count__"):
                if key is None:
                    notes_tail.append(value)
                continue
            attributes[str(key)] = value
        # Find the cell whose text matched the symbol to anchor a bbox.
        symbol_bbox = cells[0].bbox
        for c in cells:
            if c.text.strip() == symbol_text:
                symbol_bbox = c.bbox
                break
        return ParsedLegendEntry.make(
            page_index=page_index,
            label_text=label_text,
            normalized_label=normalize_text(label_text),
            raw_symbol_text=symbol_text,
            normalized_symbol_text=symbol_text.lower(),
            symbol_bbox_pdf=symbol_bbox,
            count_column=count_val,
            notes=tuple(notes_tail),
            attributes=attributes,
            source_ref_locator={
                "page": page_index,
                "row_y": round((symbol_bbox[1] + symbol_bbox[3]) / 2.0, 2),
            },
            confidence=0.9,
        )

    # Positional fallback.
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

    # First pass — tabular rows. Capture the header row (if any) so we
    # can map data rows into typed attribute columns. Multi-column
    # construction legends carry mounting_height, cable_count,
    # rough_in, remarks, etc.; without a column map we'd lose them.
    used_block_ids: set[int] = set()
    column_map: list[tuple[float, float, str | None]] = []
    header_seen = False
    for row in rows:
        if _row_looks_like_header(row):
            if not header_seen:
                column_map = _build_column_map(row)
                header_seen = True
            for b in row:
                used_block_ids.add(id(b))
            continue
        entry = _entry_from_tabular_row(page_index, row, column_map=column_map)
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
