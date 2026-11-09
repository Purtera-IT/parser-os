"""Form grids: a fill-in form drawn as a ruled table, read row by row.

A Word questionnaire exported to PDF (and most Acrobat forms) is one big
ruled table whose cells are either a printed LABEL ("SITE NAME", "Is a
Freight Elevator Available?") set bold or on a shaded band, or a VALUE:
typed text, a blank, or a row of checkbox options ("☐YES ☐NO"). Read as an
ordinary table it went wrong three ways:

* ``find_tables`` takes every shading-band edge as a wall, so one cell set
  on two text lines comes back as two rows: "ALTERNATE" / "SITE CONTACT",
  "SITE ADDRESS" / "(Pickup Location)", a "STATE:" label alone on a row
  of its own beside the row that holds its value;
* the first form row became the column names, so every later row was keyed
  by the first field ("CLIENT NAME: Does the site require a pre-call...");
* unnamed columns came out as ``col_N`` placeholders in the atom text.

This module repairs the rows (:func:`merge_continuation_rows`) and reads a
form grid as the label -> value pairs each row holds
(:func:`read_form_grid`). It also maps symbol-font box / tick glyphs to
☐ / ☑ by the font they are set in (:func:`page_glyph_map`). Everything here
is structural -- rules, fills, weight, glyph fonts -- never a form's wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any

# ── glyphs ─────────────────────────────────────────────────────────────

#: Symbol fonts whose private-use glyphs are boxes and ticks.
_BOX_FONT_RE = re.compile(r"wingding|webding|symbol|dingbat|marlett", re.I)
_TICK_FONT_RE = re.compile(r"zapf|dingbat", re.I)

#: Wingdings code points (as PUA U+F0xx or as their Latin-1 byte) that draw
#: an empty box, and those that draw a ticked / crossed box.
_WINGDINGS_EMPTY = {0xA8, 0x6F, 0x71, 0x72, 0xA1}
_WINGDINGS_CHECKED = {0xFE, 0xFD, 0x78}

UNCHECKED_BOX = "☐"  # ☐
CHECKED_BOX = "☑"  # ☑
_UNICODE_BOXES = "☐□❑❒"
_UNICODE_CHECKED = "☑☒"


def _symbol_code(ch: str) -> int | None:
    """The font byte a symbol-font glyph stands for: U+F0xx (how PDF text
    extraction surfaces Wingdings) or the bare Latin-1 byte."""
    o = ord(ch)
    if 0xF000 <= o <= 0xF0FF:
        return o - 0xF000
    if o <= 0xFF:
        return o
    return None


def glyph_kind(ch: str, font: str) -> str | None:
    """"box" / "checked" when ``ch`` set in ``font`` draws a checkbox, else None."""
    if ch in _UNICODE_BOXES:
        return "box"
    if ch in _UNICODE_CHECKED:
        return "checked"
    if _BOX_FONT_RE.search(font or "") and not _TICK_FONT_RE.search(font or ""):
        code = _symbol_code(ch)
        if code in _WINGDINGS_EMPTY:
            return "box"
        if code in _WINGDINGS_CHECKED:
            return "checked"
    return None


def _is_tick_char(ch: str, font: str) -> bool:
    """A ZapfDingbats glyph that draws a tick: a checkbox widget's appearance
    ("4" / "3" in the standard encoding, a control code when the producer
    re-encoded the font)."""
    if not _TICK_FONT_RE.search(font or "") or not ch.strip():
        return False
    o = ord(ch)
    return ch in "34" or o < 0x20 or 0xF000 <= o <= 0xF0FF or ch in "✓✔"


@dataclass
class Glyph:
    kind: str  # "box" | "checked" | "tick"
    bbox: tuple[float, float, float, float]
    char: str
    font: str


def _rawdict_chars(page: Any) -> list[tuple[str, tuple[float, float, float, float], str, int, float]]:
    """Every character on the page: (char, bbox, font, flags, size)."""
    out: list[tuple[str, tuple[float, float, float, float], str, int, float]] = []
    try:
        data = page.get_text("rawdict") or {}
    except Exception:
        return out
    for blk in data.get("blocks", []) or []:
        for ln in blk.get("lines", []) or []:
            for sp in ln.get("spans", []) or []:
                font = str(sp.get("font") or "")
                flags = int(sp.get("flags") or 0)
                size = float(sp.get("size") or 0.0)
                for ch in sp.get("chars", []) or []:
                    c = str(ch.get("c") or "")
                    if not c:
                        continue
                    out.append((c, tuple(float(v) for v in ch.get("bbox")), font, flags, size))
    return out


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float = 0.0) -> bool:
    return (min(a[2], b[2]) + pad > max(a[0], b[0])
            and min(a[3], b[3]) + pad > max(a[1], b[1]))


def page_glyphs(page: Any) -> list[Glyph]:
    """Box, checked-box and tick glyphs on the page, by the font they use."""
    chars = _rawdict_chars(page)
    out: list[Glyph] = []
    ticks: list[Glyph] = []
    for c, bbox, font, _flags, _size in chars:
        k = glyph_kind(c, font)
        if k:
            out.append(Glyph(k, bbox, c, font))
        elif _is_tick_char(c, font):
            ticks.append(Glyph("tick", bbox, c, font))
    return out + ticks


def page_glyph_map(page: Any) -> dict[str, str]:
    """Private-use / control characters this page sets in a symbol font,
    mapped to ☐ (empty box) or ☑ (checked box / tick).

    Only characters no text font uses are mapped page-wide (a Wingdings box
    comes out of text extraction as U+F0A8, a ZapfDingbats tick as a control
    code), so a plain letter set in a symbol font never rewrites the same
    letter elsewhere on the page.
    """
    out: dict[str, str] = {}
    glyphs = page_glyphs(page)
    boxes = [g for g in glyphs if g.kind in ("box", "checked")]
    for g in glyphs:
        o = ord(g.char)
        page_wide = (0xF000 <= o <= 0xF0FF) or o < 0x20 or g.char == "\u00a8"
        if not page_wide:
            continue
        if g.kind == "box":
            out[g.char] = UNCHECKED_BOX
        elif g.kind == "checked":
            out[g.char] = CHECKED_BOX
        elif g.kind == "tick":
            # A tick drawn over a box is the box's checked state.
            if g.char in "34":
                continue
            if any(_overlap(g.bbox, b.bbox, 2.0) for b in boxes) or o < 0x20:
                out[g.char] = CHECKED_BOX
    return out


#: A tick extracted right beside the box it is drawn over (no space between).
_BOX_THEN_TICK = re.compile(rf"{UNCHECKED_BOX}{CHECKED_BOX}|{CHECKED_BOX}{UNCHECKED_BOX}")


def apply_glyph_map(text: Any, gmap: dict[str, str]) -> Any:
    """``text`` with the page's symbol glyphs mapped; a tick read beside the
    box it is drawn over collapses into one ☑. Non-strings pass through."""
    if not gmap or not isinstance(text, str) or not text:
        return text
    out = text.translate({ord(k): v for k, v in gmap.items()})
    return _BOX_THEN_TICK.sub(CHECKED_BOX, out)


def has_box_glyph(text: str) -> bool:
    return bool(re.search(f"[{_UNICODE_BOXES}{_UNICODE_CHECKED}]", text or ""))


# ── page geometry: rules and fills ─────────────────────────────────────

def _lum(color: Any) -> float:
    try:
        vals = [float(c) for c in color]
    except Exception:
        return 1.0
    if not vals:
        return 1.0
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 4:  # CMYK
        c, m, y, k = vals
        return 1.0 - min(1.0, 0.3 * c + 0.59 * m + 0.11 * y + k)
    r, g, b = vals[:3]
    return 0.3 * r + 0.59 * g + 0.11 * b


@dataclass
class PageGeometry:
    """Horizontal rules (y, x0, x1) and filled rectangles on a page."""

    hrules: list[tuple[float, float, float]] = field(default_factory=list)
    fills: list[tuple[tuple[float, float, float, float], tuple[float, ...]]] = field(default_factory=list)


def page_geometry(page: Any) -> PageGeometry:
    """The page's visible horizontal rules and its filled (non-white) areas.

    A rule is a stroked horizontal line, a stroked rectangle's top or bottom
    edge, or a filled rectangle at most 2pt tall that is darker than near
    white (Word draws table borders as thin filled rectangles)."""
    geo = PageGeometry()
    try:
        drawings = page.get_drawings() or []
    except Exception:
        return geo
    for d in drawings:
        fill = d.get("fill")
        stroke = d.get("color")
        width = d.get("width")
        stroked = stroke is not None and _lum(stroke) < 0.95 and (width is None or float(width) > 0)
        for it in d.get("items", []) or []:
            op = it[0]
            if op == "l":
                p1, p2 = it[1], it[2]
                if stroked and abs(float(p1.y) - float(p2.y)) <= 1.0 and abs(float(p1.x) - float(p2.x)) > 2.0:
                    geo.hrules.append(((float(p1.y) + float(p2.y)) / 2.0,
                                       min(float(p1.x), float(p2.x)), max(float(p1.x), float(p2.x))))
            elif op == "re":
                r = it[1]
                x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                if x1 - x0 < 0.1 and y1 - y0 < 0.1:
                    continue
                if fill is not None and y1 - y0 <= 2.0 and x1 - x0 > 2.0:
                    if _lum(fill) < 0.95:
                        geo.hrules.append(((y0 + y1) / 2.0, x0, x1))
                    continue
                if fill is not None and y1 - y0 > 2.0 and x1 - x0 > 2.0 and _lum(fill) < 0.97:
                    try:
                        key = tuple(round(float(c), 2) for c in fill)
                    except Exception:
                        key = ()
                    geo.fills.append(((x0, y0, x1, y1), key))
                if stroked and x1 - x0 > 2.0:
                    geo.hrules.append((y0, x0, x1))
                    geo.hrules.append((y1, x0, x1))
    return geo


def _ruled_between(geo: PageGeometry, y_lo: float, y_hi: float, x0: float, x1: float) -> bool:
    """A rule runs somewhere in ``[y_lo, y_hi]`` under at least half of
    ``[x0, x1]`` (pieces of one rule at the same height add up)."""
    width = max(x1 - x0, 1.0)
    by_y: dict[float, float] = {}
    for ry, rx0, rx1 in geo.hrules:
        if y_lo <= ry <= y_hi:
            key = round(ry, 0)
            by_y[key] = by_y.get(key, 0.0) + max(0.0, min(x1, rx1) - max(x0, rx0))
    return any(v >= 0.5 * width for v in by_y.values())


def _gap_between(spans: list["_Span"], above: Any, cell: Any) -> tuple[float, float]:
    """The band between the last text line of the cell above and the first
    text line of this cell (falls back to the shared edge +-2.5pt)."""
    edge = float(cell[1])
    up = _spans_in(spans, above)
    down = _spans_in(spans, cell)
    lo = max((s.bbox[3] for s in up), default=edge - 2.5)
    hi = min((s.bbox[1] for s in down), default=edge + 2.5)
    lo, hi = min(lo, edge - 0.5), max(hi, edge + 0.5)
    # Glyph boxes overhang their line; keep the band to the two lines' gap.
    return lo - 0.5, hi + 0.5


def _fill_at(geo: PageGeometry, x: float, y: float) -> list[int]:
    return [i for i, (r, _k) in enumerate(geo.fills)
            if r[0] - 0.5 <= x <= r[2] + 0.5 and r[1] - 0.5 <= y <= r[3] + 0.5]


def _one_band(geo: PageGeometry, x: float, y: float) -> bool:
    """The points just above and just below ``y`` at ``x`` sit on the same
    filled band, or both on the bare page: nothing visible separates them."""
    above, below = set(_fill_at(geo, x, y - 2.0)), set(_fill_at(geo, x, y + 2.0))
    if not above and not below:
        return True
    return bool(above & below)


def is_shaded(geo: PageGeometry, bbox: tuple[float, float, float, float]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return bool(_fill_at(geo, cx, cy))


# ── table proxy: rows edited, geometry kept in step ───────────────────

class _Row:
    def __init__(self, cells: list[Any]):
        self.cells = cells


class TableView:
    """A ``find_tables`` table with its rows edited: ``rows[i].cells`` stays
    index-aligned with the edited text rows, the way the table readers in
    ``_shared`` expect."""

    def __init__(self, bbox: Any, geo_rows: list[list[Any]]):
        self.bbox = bbox
        self.rows = [_Row(list(g)) for g in geo_rows]


def _geo_rows(table: Any) -> list[list[Any]] | None:
    try:
        return [list(getattr(r, "cells", []) or []) for r in (getattr(table, "rows", []) or [])]
    except Exception:
        return None


def _text(cell: Any) -> str:
    return " ".join(str(cell or "").split())


# ── text styles inside a cell ──────────────────────────────────────────

@dataclass
class _Span:
    bbox: tuple[float, float, float, float]
    text: str
    font: str
    bold: bool
    size: float


def page_spans(page: Any) -> list[_Span]:
    out: list[_Span] = []
    try:
        data = page.get_text("dict") or {}
    except Exception:
        return out
    for blk in data.get("blocks", []) or []:
        for ln in blk.get("lines", []) or []:
            for sp in ln.get("spans", []) or []:
                t = str(sp.get("text") or "")
                if not t.strip():
                    continue
                font = str(sp.get("font") or "")
                bold = bool(int(sp.get("flags") or 0) & 16) or bool(
                    re.search(r"bold|black|heavy|semibold|demi", font, re.I))
                out.append(_Span(tuple(float(v) for v in sp.get("bbox")), t, font, bold,
                                 float(sp.get("size") or 0.0)))
    return out


def _spans_in(spans: list[_Span], bbox: Any) -> list[_Span]:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return [s for s in spans
            if x0 - 1 <= (s.bbox[0] + s.bbox[2]) / 2.0 <= x1 + 1
            and y0 - 1 <= (s.bbox[1] + s.bbox[3]) / 2.0 <= y1 + 1]


def _text_spans(spans: list[_Span]) -> list[_Span]:
    """Spans of real text: symbol-font glyphs (boxes, ticks) left out."""
    return [s for s in spans if not (_BOX_FONT_RE.search(s.font) or _TICK_FONT_RE.search(s.font))]


_CAPTION_RE = re.compile(r"^\(.{1,200}\)\s*:?$")


# ── 1. continuation merge ──────────────────────────────────────────────

def merge_continuation_rows(
    page: Any, table: Any, rows: list[list[Any]],
    *, geometry: PageGeometry | None = None, spans: list[_Span] | None = None,
    gmap: dict[str, str] | None = None,
) -> tuple[list[list[Any]], Any]:
    """Put a cell's text that ``find_tables`` split into a sub-row back into
    the row above it, column by column.

    A sub-row is a continuation of the row above when nothing visible
    separates any of its cells from the cell above in the same column: no
    horizontal rule runs between their text lines, and the two sit on one
    filled band (or both on the bare page). That is how a label wrapped over
    two lines ("ALTERNATE" / "SITE CONTACT"), a value set on the line under
    its row, and a "(...)" caption under its label come back from a Word
    form. A grid whose rows really are ruled apart is left as it is.

    The text has a say too: a cell whose line above ends a sentence or a
    question ("...Available?") starts a new item unless it is a "(...)"
    caption, and a row is merged whole or not at all.

    Plain-page (unshaded) cells merge only in a table that draws at least one
    interior rule, so a rule-less grid is never folded into one row. A small,
    regular-weight "(...)" caption directly under a bold label merges with it
    on the same evidence even then.

    First, a cell drawn tall across several sub-rows that holds one line per
    sub-row (a Word cell with two questions and two "☐YES ☐NO" lines) gives
    each line to the sub-row it sits beside (:func:`split_tall_cells`).

    Returns ``(rows, table_view)``: edited rows plus a table whose
    ``rows[i].cells`` geometry is in step with them. When nothing changes the
    input comes back as given (the original table object).
    """
    geo_rows = _geo_rows(table)
    if not rows or geo_rows is None or len(geo_rows) != len(rows) or len(rows) < 2:
        return rows, table
    ncols = max(len(r) for r in rows)
    if any(len(r) != ncols for r in rows) or any(len(g) != ncols for g in geo_rows):
        return rows, table
    geometry = geometry if geometry is not None else page_geometry(page)
    try:
        tx0, ty0, tx1, ty1 = (float(v) for v in table.bbox)
    except Exception:
        return rows, table
    interior_rules = any(
        ty0 + 2.0 < y < ty1 - 2.0 and min(tx1, rx1) - max(tx0, rx0) > 4.0
        for y, rx0, rx1 in geometry.hrules
    )
    spans = spans if spans is not None else page_spans(page)

    text_rows = [list(r) for r in rows]
    geo = [[(tuple(float(v) for v in c) if c is not None else None) for c in g] for g in geo_rows]
    anchored = split_tall_cells(page, text_rows, geo, gmap=gmap)
    changed = bool(anchored)
    for i in range(1, len(text_rows)):
        moves: list[tuple[int, int]] = []  # (column, row it merges into)
        ok = True
        for j in range(ncols):
            cell = geo[i][j]
            t = _text(text_rows[i][j])
            if cell is None or not t:
                continue
            if (i, j) in anchored:
                ok = False
                break
            k = next((r for r in range(i - 1, -1, -1) if geo[r][j] is not None), None)
            if k is None:
                ok = False
                break
            above = geo[k][j]
            # The cell above must sit over this one (same left edge, this
            # one no wider) and end where this one starts.
            if (abs(above[0] - cell[0]) > 2.0 or cell[2] > above[2] + 2.0
                    or cell[1] - above[3] > 2.0):
                ok = False
                break
            y_lo, y_hi = _gap_between(spans, above, cell)
            if _ruled_between(geometry, y_lo, y_hi, cell[0], cell[2]):
                ok = False
                break
            xm = (cell[0] + cell[2]) / 2.0
            if not _one_band(geometry, xm, cell[1]):
                ok = False
                break
            caption = _is_caption_under_label(spans, above, cell, t)
            shaded_band = bool(set(_fill_at(geometry, xm, cell[1] - 2.0))
                               & set(_fill_at(geometry, xm, cell[1] + 2.0)))
            if not (interior_rules or shaded_band or caption):
                ok = False
                break
            prev = _text(text_rows[k][j])
            if prev and _ENDS_ITEM.search(prev) and not _CAPTION_RE.match(t):
                ok = False
                break
            moves.append((j, k))
        if not ok or not moves:
            continue
        for j, k in moves:
            cell, above = geo[i][j], geo[k][j]
            prev = _text(text_rows[k][j])
            t = _text(text_rows[i][j])
            text_rows[k][j] = f"{prev} {t}".strip() if prev else t
            geo[k][j] = (above[0], above[1], max(above[2], cell[2]), max(above[3], cell[3]))
            text_rows[i][j] = None
            geo[i][j] = None
        changed = True
    if not changed:
        return rows, table
    keep = [i for i in range(len(text_rows)) if any(_text(c) for c in text_rows[i]) or i == 0]
    text_rows = [text_rows[i] for i in keep]
    geo = [geo[i] for i in keep]
    return text_rows, TableView(table.bbox, geo)


#: A line that closes an item: a question, a sentence, a "Label:" prompt.
_ENDS_ITEM = re.compile(r"[?.!:]\s*$")


def _starts_own_item(rows: list[list[Any]], geo: list[list[Any]], r: int, skip: int) -> bool:
    """Row ``r`` holds text outside column ``skip`` that opens an item of its
    own: the nearest text above it in that column is empty or ends one."""
    found = False
    for c in range(len(geo[r])):
        if c == skip or geo[r][c] is None or not _text(rows[r][c]):
            continue
        found = True
        k = next((q for q in range(r - 1, -1, -1) if geo[q][c] is not None), None)
        if k is None:
            continue
        prev = _text(rows[k][c])
        if prev and not _ENDS_ITEM.search(prev):
            return False
    return found


def _row_top(g: list[Any]) -> float | None:
    tops = [c[1] for c in g if c is not None]
    return min(tops) if tops else None


def split_tall_cells(
    page: Any, rows: list[list[Any]], geo: list[list[Any]], *, gmap: dict[str, str] | None = None,
) -> set[tuple[int, int]]:
    """Give each line of a cell drawn down across several sub-rows to the
    sub-row it sits beside, when every one of those sub-rows has text of its
    own in another column (its question) and the lines land in two or more
    of them. A sub-row whose own text only continues the line above it (a
    label wrapped onto a second line) is not a sub-row of its own here.
    Edits ``rows`` / ``geo`` in place; returns the (row, column) of every
    cell it filled, which the merge then keeps apart."""
    if gmap is None:
        gmap = page_glyph_map(page)
    out: set[tuple[int, int]] = set()
    if len(rows) < 2:
        return out
    try:
        words = page.get_text("words") or []
    except Exception:
        return out
    tops = [_row_top(g) for g in geo]
    ncols = len(geo[0]) if geo else 0
    for i in range(len(rows)):
        for j in range(ncols):
            cell = geo[i][j]
            if cell is None or not _text(rows[i][j]):
                continue
            x0, y0, x1, y1 = cell
            below = [r for r in range(i + 1, len(rows))
                     if tops[r] is not None and y0 + 1.0 < tops[r] < y1 - 1.0 and geo[r][j] is None]
            if not below:
                continue
            band_rows = [i] + below
            bounds = [(tops[r] if r != i else y0) for r in band_rows] + [y1]
            inside = [w for w in words
                      if x0 - 1 <= (w[0] + w[2]) / 2.0 <= x1 + 1
                      and y0 - 1 <= (w[1] + w[3]) / 2.0 <= y1 + 1]
            if not inside:
                continue
            if not any(_text(rows[i][c]) for c in range(ncols) if c != j):
                continue
            # A line goes to the sub-row whose own item it sits beside: a
            # sub-row that is empty elsewhere, or only wraps the line above
            # it, belongs to the item above.
            owner: dict[int, int] = {}
            prev_r = i
            for r in band_rows:
                owner[r] = r if (r == i or _starts_own_item(rows, geo, r, j)) else owner[prev_r]
                prev_r = r
            placed: dict[int, list[Any]] = {}
            for w in inside:
                cy = (float(w[1]) + float(w[3])) / 2.0
                b = next((n for n in range(len(band_rows)) if bounds[n] - 0.5 <= cy < bounds[n + 1] + 0.5), None)
                if b is None:
                    placed = {}
                    break
                placed.setdefault(owner[band_rows[b]], []).append(w)
            if len(placed) < 2:
                continue
            owners = sorted(set(owner.values()), key=band_rows.index)
            for o in owners:
                ws = sorted(placed.get(o, []), key=lambda w: (round(float(w[3])), float(w[0])))
                txt = apply_glyph_map(" ".join(str(w[4]) for w in ws), gmap)
                members = [band_rows.index(r) for r in band_rows if owner[r] == o]
                top, bottom = bounds[members[0]], bounds[members[-1] + 1]
                if o == i:
                    rows[i][j] = txt
                    geo[i][j] = (x0, y0, x1, bottom)
                elif txt:
                    rows[o][j] = txt
                    geo[o][j] = (x0, top, x1, bottom)
                    out.add((o, j))
    return out


def _is_caption_under_label(spans: list[_Span], above: Any, cell: Any, text: str) -> bool:
    if not _CAPTION_RE.match(text):
        return False
    lab = _text_spans(_spans_in(spans, above))
    cap = _text_spans(_spans_in(spans, cell))
    if not lab or not cap:
        return False
    return (all(s.bold for s in lab) and not any(s.bold for s in cap)
            and max(s.size for s in cap) <= max(s.size for s in lab) + 0.2)


# ── 2. form-grid reading ───────────────────────────────────────────────

@dataclass
class FormCell:
    text: str
    bbox: tuple[float, float, float, float]
    kind: str  # "label" | "value" | "options"
    lines: list[str] = field(default_factory=list)


_GLYPHS_ONLY = re.compile(f"^[\\s{_UNICODE_BOXES}{_UNICODE_CHECKED}\uf0a8\uf06f\uf0fe\uf0fd]+$")


def snap_ticks(chars: list[tuple[str, tuple[float, float, float, float], str, int, float]],
               bbox: Any) -> str | None:
    """The cell's text re-read from its characters with each tick drawn over
    a box turned into that box's ☑ -- a checkbox widget's appearance tick is
    drawn after the page text, so plain extraction leaves it stray at the end
    of the row ("☐Work ☐Home ☐Cell ☑"). None when no tick sits on a box."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    inside = [c for c in chars
              if x0 - 1 <= (c[1][0] + c[1][2]) / 2.0 <= x1 + 1
              and y0 - 1 <= (c[1][1] + c[1][3]) / 2.0 <= y1 + 1]
    boxes = [n for n, c in enumerate(inside) if glyph_kind(c[0], c[2])]
    ticks = [c for c in inside if _is_tick_char(c[0], c[2])]
    if not boxes or not ticks:
        return None
    checked = {n for n in boxes if any(_overlap(inside[n][1], t[1], 1.0) for t in ticks)}
    if not checked:
        return None
    keep = [(n, c) for n, c in enumerate(inside) if not _is_tick_char(c[0], c[2])]
    keep.sort(key=lambda nc: ((nc[1][1][1] + nc[1][1][3]) / 2.0, nc[1][1][0]))
    lines: list[list[tuple[int, Any]]] = []
    for n, c in keep:
        cy = (c[1][1] + c[1][3]) / 2.0
        if lines and abs(cy - lines[-1][0][2]) <= 3.0:
            lines[-1].append((n, c, lines[-1][0][2]))
        else:
            lines.append([(n, c, cy)])
    out: list[str] = []
    for ln in lines:
        ln.sort(key=lambda t: t[1][1][0])
        buf: list[str] = []
        prev_x1: float | None = None
        for n, c, _cy in ln:
            k = glyph_kind(c[0], c[2])
            # Words the PDF sets apart by position, not by a space character.
            if prev_x1 is not None and buf and buf[-1] != " " and c[0] != " " \
                    and c[1][0] - prev_x1 > max(1.0, 0.2 * float(c[4] or 0)):
                buf.append(" ")
            if k:
                if buf and buf[-1] != " ":
                    buf.append(" ")
                buf.append(CHECKED_BOX if (n in checked or k == "checked") else UNCHECKED_BOX)
            else:
                buf.append(c[0])
            prev_x1 = c[1][2]
        t = " ".join("".join(buf).split())
        if t:
            out.append(t)
    return "\n".join(out) or None


def classify_cells(
    page: Any, table: Any, rows: list[list[Any]],
    *, geometry: PageGeometry | None = None, spans: list[_Span] | None = None,
) -> list[list[FormCell]] | None:
    """Each row's non-empty cells, left to right, typed by style: a LABEL is
    set bold or on a shaded band (or ends with ":"), an OPTIONS cell holds a
    checkbox glyph, anything else is a VALUE. None when the geometry does not
    line up with the rows."""
    geo_rows = _geo_rows(table)
    if geo_rows is None or len(geo_rows) != len(rows):
        return None
    geometry = geometry if geometry is not None else page_geometry(page)
    spans = spans if spans is not None else page_spans(page)
    chars: list[Any] | None = None
    raw: list[list[Any]] = []
    for r, g in zip(rows, geo_rows):
        cells: list[Any] = []
        for j, c in enumerate(r):
            t = _text(c)
            if not t or j >= len(g) or g[j] is None:
                continue
            bbox = tuple(float(v) for v in g[j])
            inside = _spans_in(spans, bbox)
            txt = _text_spans(inside)
            boxed = has_box_glyph(t) or any(
                glyph_kind(ch, s.font) for s in inside for ch in s.text)
            bold = bool(txt) and all(s.bold for s in txt)
            raw_text = str(c)
            if boxed:
                if chars is None:
                    chars = _rawdict_chars(page)
                snapped = snap_ticks(chars, bbox)
                if snapped:
                    raw_text, t = snapped, _text(snapped)
            lines = [_text(x) for x in raw_text.split("\n") if _text(x)]
            # A box drawn in a cell of its own, its option label in the next
            # cell ("☐" | "YES"): one option.
            if cells and _GLYPHS_ONLY.match(cells[-1][0]) and not boxed and len(t) <= 40:
                pt, pb = cells[-1][0], cells[-1][1]
                ub = (min(pb[0], bbox[0]), min(pb[1], bbox[1]), max(pb[2], bbox[2]), max(pb[3], bbox[3]))
                cells[-1] = (f"{pt} {t}", ub, False, cells[-1][3], True, [f"{pt} {t}"], cells[-1][6])
                continue
            face = (txt[0].font, round(txt[0].size, 1)) if txt else None
            cells.append((t, bbox, bold, is_shaded(geometry, bbox), boxed, lines, face))
        raw.append(cells)
    every = [c for row in raw for c in row]
    if not every:
        return []
    # Shading marks a label only when the shaded cell is also set in another
    # face or size than the bare cells beside it (a PO header's grey label
    # column); a grid shaded all over, or a shaded cell in the value face,
    # says nothing by its shading.
    plain_faces = {c[6] for c in every if not c[3] and not c[4] and c[6] is not None}
    shading_tells = not all(c[3] for c in every)
    out: list[list[FormCell]] = []
    for row in raw:
        typed: list[FormCell] = []
        for t, bbox, bold, shaded, boxed, lines, face in row:
            shaded_label = shaded and shading_tells and face is not None and face not in plain_faces
            if boxed:
                kind = "options"
            elif bold or shaded_label or (t.endswith(":") and len(t) <= 60):
                kind = "label"
            else:
                kind = "value"
            typed.append(FormCell(t, bbox, kind, lines))
        out.append(typed)
    return out


def is_form_grid(cells: list[list[FormCell]]) -> bool:
    """Rows of label -> value pairs, not a header row over data rows.

    It takes two or more rows that open with a label, and at least one row
    where a label is followed by a value or by options. Row 0 is a header
    (and the grid is not a form) when it is the only row with labels, or
    when it is all labels -- two or more, no value -- over rows that only
    label their first cell."""
    nonempty = [r for r in cells if r]
    if len(nonempty) < 2:
        return False
    label_rows = [r for r in nonempty if r[0].kind == "label"]
    if len(label_rows) < 2:
        return False
    paired = any(
        any(a.kind == "label" and b.kind in ("value", "options") for a, b in zip(r, r[1:]))
        for r in nonempty
    )
    if not paired:
        return False
    head = nonempty[0]
    head_labels = [c for c in head if c.kind == "label"]
    if len(head_labels) >= 2 and len(head_labels) == len(head):
        if all(not any(c.kind == "label" for c in r[1:]) for r in nonempty[1:]):
            return False
    return True


def read_form_row(row: list[FormCell]) -> tuple[str, list[tuple[str, str]]]:
    """One form row as text ("SITE ADDRESS: 1 Main St | CITY: Springfield")
    and its (label, value) pairs. A label opens a field and takes the values
    up to the next label; a label with no value stands alone; values before
    any label stand alone. Nothing carries over from another row."""
    paired = _paired_questions(row)
    if paired:
        return " | ".join(f"{q} {a}" for q, a in paired), paired
    segs: list[str] = []
    fields: list[tuple[str, str]] = []
    label: str | None = None
    values: list[str] = []

    def flush() -> None:
        nonlocal label, values
        v = " ".join(values).strip()
        if label is not None:
            lab = label.strip()
            if v:
                segs.append(f"{lab} {v}" if lab.endswith(":") else f"{lab}: {v}")
            else:
                segs.append(lab)
            fields.append((lab.rstrip(":").strip(), v))
        elif v:
            segs.append(v)
            fields.append(("", v))
        label, values = None, []

    for c in row:
        if c.kind == "label":
            flush()
            label = c.text
        else:
            values.append(c.text)
    flush()
    return " | ".join(segs), fields


_QUESTION_SPLIT = re.compile(r"(?<=\?)\s+(?=[A-Z])")


def _paired_questions(row: list[FormCell]) -> list[tuple[str, str]] | None:
    """A cell holding several questions ("Is there X? If so, may we use
    it?") beside an options cell holding one line of boxes per question:
    each question takes its own line of options."""
    if len(row) != 2 or row[0].kind != "label" or row[1].kind != "options":
        return None
    qs = [q.strip() for q in _QUESTION_SPLIT.split(row[0].text) if q.strip()]
    opts = [ln for ln in row[1].lines if has_box_glyph(ln)]
    if len(qs) < 2 or len(qs) != len(opts) or not all(q.endswith("?") for q in qs):
        return None
    return list(zip(qs, opts))


def read_form_grid(
    page: Any, table: Any, rows: list[list[Any]],
    *, geometry: PageGeometry | None = None, spans: list[_Span] | None = None,
) -> list[dict[str, Any]] | None:
    """A form grid as one record per row: ``{"text", "fields"}``; None when
    the grid is not a form (see :func:`is_form_grid`)."""
    cells = classify_cells(page, table, rows, geometry=geometry, spans=spans)
    if not cells or not is_form_grid(cells):
        return None
    out: list[dict[str, Any]] = []
    header: list[FormCell] | None = None
    header_rec: dict[str, Any] | None = None
    header_used = False
    for row in cells:
        if not row:
            continue
        if _is_sub_header(row):
            if header_rec is not None and not header_used:
                out.append(header_rec)
            header, header_used = row, False
            header_rec = {"text": " | ".join(c.text for c in row), "fields": [(c.text, "") for c in row]}
            continue
        if header is not None and not _fits_header(row, header):
            if not header_used and header_rec is not None:
                out.append(header_rec)
            header, header_rec = None, None
        if header is not None:
            text, fields = _read_under_header(row, header)
            header_used = True
        else:
            text, fields = read_form_row(row)
        if text:
            out.append({"text": text, "fields": fields})
    if header_rec is not None and not header_used:
        out.append(header_rec)
    return out or None


def _is_sub_header(row: list[FormCell]) -> bool:
    """A row of two or more bare labels ("ITEM | QUANTITY | DETAILS"): the
    column names of the rows under it, not fields of its own."""
    return (len(row) >= 2 and all(c.kind == "label" for c in row)
            and not any(c.text.rstrip().endswith((":", "?")) for c in row))


def _fits_header(row: list[FormCell], header: list[FormCell]) -> bool:
    """A row read under the sub-header above it: each of its label cells
    (row names, "PC-Desktop") sits inside one header column and asks
    nothing ("...?", "Details:") -- a wider label or a prompt opens a new
    part of the form."""
    for c in row:
        if c.kind != "label":
            continue
        if c.text.rstrip().endswith((":", "?")):
            return False
        cx = (c.bbox[0] + c.bbox[2]) / 2.0
        home = [h for h in header if h.bbox[0] - 2 <= cx <= h.bbox[2] + 2]
        if not home or (c.bbox[2] - c.bbox[0]) > 1.4 * (home[0].bbox[2] - home[0].bbox[0]) + 4:
            return False
    return True


def _read_under_header(row: list[FormCell], header: list[FormCell]) -> tuple[str, list[tuple[str, str]]]:
    """Each value under the header cell it overlaps most by x."""
    fields: list[tuple[str, str]] = []
    for c in row:
        best = max(header, key=lambda h: min(c.bbox[2], h.bbox[2]) - max(c.bbox[0], h.bbox[0]))
        if min(c.bbox[2], best.bbox[2]) - max(c.bbox[0], best.bbox[0]) <= 0:
            fields.append(("", c.text))
        else:
            fields.append((best.text, c.text))
    text = " | ".join(f"{h}: {v}" if h else v for h, v in fields)
    return text, fields


# ── nested tables ──────────────────────────────────────────────────────

def nested_table_indexes(tables: list[Any]) -> set[int]:
    """Tables drawn inside another found table's cell (an option grid inside
    a form cell): the outer table already reads their text."""
    boxes = []
    for t in tables:
        try:
            boxes.append(tuple(float(v) for v in t.bbox))
        except Exception:
            boxes.append(None)
    out: set[int] = set()
    for i, a in enumerate(boxes):
        if a is None:
            continue
        area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
        for j, b in enumerate(boxes):
            if i == j or b is None:
                continue
            barea = (b[2] - b[0]) * (b[3] - b[1])
            if barea <= area * 1.5:
                continue
            inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            if inter >= 0.9 * area:
                out.add(i)
                break
    return out
