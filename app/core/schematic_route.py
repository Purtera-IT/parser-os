"""Per-sheet routing: deterministic text extraction vs vision.

Measured gap: a VLM single-pass on a dense SCHEDULE got 31%, but the same vector
sheet's TEXT LAYER holds 100% of the rows (1,000+ lines) exactly. So sending a
text-bearing schedule/table to a vision model is the wrong tool — slower, costlier,
and worse. Route it to deterministic text extraction.

Conversely, a scanned/raster drawing or an icon-only schematic has no usable text
layer -> it genuinely NEEDS the vision model. This router sends each sheet to the
right path:

  needs_vision(page) is True  -> scanned/raster or image-dominant -> VISION
  needs_vision(page) is False -> rich vector text layer          -> DETERMINISTIC

This is what makes "never full-time VLM" real: most CD sheets are vector text and
route deterministically (local, free, exact); the vision model is reserved for the
sheets that actually need pixels.
"""
from __future__ import annotations

# A page with at least this many non-trivial text lines has a usable text layer.
_MIN_TEXT_LINES = 25
# Below this char count the page is effectively scanned/blank -> vision.
_MIN_TEXT_CHARS = 400


def _text_lines(page) -> list[str]:
    try:
        txt = page.get_text() or ""
    except Exception:
        return []
    return [ln.strip() for ln in txt.splitlines() if len(ln.strip()) > 2]


def text_layer_strength(page) -> tuple[int, int]:
    """(non-trivial line count, total char count) of the page's text layer."""
    lines = _text_lines(page)
    return len(lines), sum(len(l) for l in lines)


def needs_vision(page) -> bool:
    """True iff the sheet lacks a usable text layer (scanned/raster/image-only) and
    therefore needs the vision model. False for vector text-bearing sheets, which
    route to deterministic extraction."""
    n_lines, n_chars = text_layer_strength(page)
    return not (n_lines >= _MIN_TEXT_LINES or n_chars >= _MIN_TEXT_CHARS)


def extract_text_rows(page) -> list[str]:
    """Deterministic row extraction from the text layer — exact, local, free. This
    is the right path for schedules/tables/notes on vector PDFs (~100% vs the VLM's
    31% single-pass). Returns the non-trivial text lines (rows)."""
    return _text_lines(page)


def route_sheet(page) -> dict:
    """Decide the path for one sheet and, for the deterministic path, return the
    extracted rows so the caller never needlessly invokes the VLM."""
    n_lines, n_chars = text_layer_strength(page)
    if needs_vision(page):
        return {"path": "vision", "reason": f"thin text layer ({n_lines} lines / {n_chars} chars) -> scanned/icon",
                "rows": []}
    rows = extract_text_rows(page)
    return {"path": "deterministic", "reason": f"vector text layer ({n_lines} lines) -> exact extraction, no VLM",
            "rows": rows}
