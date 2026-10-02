"""Per-cell source fragments for atoms whose text the parser rebuilt.

A table row read into one atom is joined with " | " ("SUBTOTAL | $4,309.20"),
and a header row over its value row becomes "QUOTE #: PSNV676 | QUOTE DATE:
1/14/2026". Neither string is anywhere in the document: the cells are, but
apart (a label column beside a value column, a header row over a value row).
A source pane that looks for the atom's text finds nothing and the labeler
sees an un-highlighted page.

This module names each piece of such an atom where the document has it:
``locator["cell_fragments"] = [{"text", "char_start", "char_end", ...}]``,
offsets into the text the caller supplied (a PDF page's text, a sheet's cell
grid, a document's paragraphs), plus whatever the caller knows about the spot
(a PDF bbox, a sheet row/column). A viewer marks every fragment as one match.

Purely additive provenance: atoms whose text IS in the source verbatim get
nothing, grouping and atom text are never changed, and atom ids (hashed when
the atom was made) do not move.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

#: Separators the parsers put between cells they glued into one atom.
_SEGMENT_SPLIT = re.compile(r"\s+\|\s+|\s*\|\s*|\s+(?:→|->|=>)\s+")
#: "Header: value" / "Header = value" inside one segment.
_PAIR_SPLIT = re.compile(r"\s*:\s+|\s+=\s+|(?<=\S)=(?=\S)")
#: A fragment this short is only trusted beside the others (a "$0.00" or a
#: "1" occurs everywhere on a quote).
MIN_FRAGMENT = 1
#: At most this many fragments per atom (a 40-column row is still one row).
MAX_FRAGMENTS = 40


def _fold_char(ch: str) -> str:
    if ch in "‘’‚‛′ʼ´`":
        return "'"
    if ch in "“”„‟″«»":
        return '"'
    if "‐" <= ch <= "―" or ch == "−":
        return "-"
    if ch in "​‌‍⁠﻿­":
        return ""
    low = ch.lower()
    return low if len(low) == 1 else ch


def fold_with_map(text: str) -> tuple[str, list[int]]:
    """``text`` lower-cased with every whitespace and invisible character
    dropped, and for each kept character its index in ``text``."""
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text or ""):
        if ch.isspace():
            continue
        f = _fold_char(ch)
        if not f:
            continue
        out.append(f)
        idx.append(i)
    return "".join(out), idx


def fold(text: str) -> str:
    return fold_with_map(text)[0]


def in_source(text: str, source: str) -> bool:
    """The atom's text is in the source as it stands (whitespace aside)."""
    n = fold(text)
    return bool(n) and n in fold(source)


def pieces_of(text: str) -> list[list[str]]:
    """The cells of a rebuilt atom, in order. Each entry is the alternatives
    for one cell: the segment itself, then its header and value when it is a
    "Header: value" pair whose two halves sit apart in the source."""
    out: list[list[str]] = []
    for seg in _SEGMENT_SPLIT.split(text or ""):
        seg = seg.strip()
        if not seg:
            continue
        parts = [p.strip() for p in _PAIR_SPLIT.split(seg, maxsplit=1) if p.strip()]
        out.append([seg] + (parts if len(parts) > 1 else []))
    return out


def _occurrences(hay: str, needle: str) -> list[int]:
    found: list[int] = []
    if not needle:
        return found
    at = hay.find(needle)
    while at >= 0 and len(found) < 200:
        found.append(at)
        at = hay.find(needle, at + 1)
    return found


def locate_fragments(
    text: str,
    source: str,
    *,
    distance: Callable[[int, int], float] | None = None,
    taken: set[tuple[int, int]] | None = None,
    after: int | None = None,
) -> list[dict[str, Any]] | None:
    """Where each cell of ``text`` sits in ``source``, or ``None`` when the
    atom is not a rebuilt one (its text is in the source verbatim, or it has
    one piece) or too little of it can be placed.

    Fragments are ``{"text", "char_start", "char_end"}`` with end-exclusive
    offsets into ``source``, in the atom's own order. A cell found more than
    once is taken where it sits nearest the atom's rarest cell (the anchor),
    so "SHIPPING | $0.00" marks the "$0.00" beside SHIPPING, not the first
    "$0.00" on the page. ``distance(a, b)`` (source offsets) overrides the
    default reading-order distance -- a PDF passes geometry.

    ``taken`` holds the source spans (start, end) earlier atoms of the same
    source already claimed; a cell found more than once prefers a copy no
    earlier atom marked (the header row's "GRAND TOTAL" belongs to the header
    atom, the totals box's to the totals row). Updated in place.

    ``after`` is where the previous atom of the same source ended (a source
    offset): atoms come in reading order, so the anchor prefers a copy at or
    after it.
    """
    if not text or not source:
        return None
    hay, idx = fold_with_map(source)
    if not hay:
        return None
    whole = fold(text)
    if not whole or whole in hay:
        return None
    cells = pieces_of(text)
    if not cells or (len(cells) == 1 and len(cells[0]) == 1):
        return None

    # Each placeable piece: (display text, folded, occurrences).
    wanted: list[tuple[str, str, list[int]]] = []
    total = 0
    covered = 0
    for alts in cells:
        seg = alts[0]
        total += len(fold(seg))
        fs = fold(seg)
        occ = _occurrences(hay, fs)
        if occ:
            wanted.append((seg, fs, occ))
            covered += len(fs)
            continue
        for part in alts[1:]:
            fp = fold(part)
            if len(fp) < MIN_FRAGMENT:
                continue
            o = _occurrences(hay, fp)
            if o:
                wanted.append((part, fp, o))
                covered += len(fp)
    if not wanted or not total or covered / total < 0.5:
        return None
    wanted = wanted[:MAX_FRAGMENTS]

    # Reading order: a cell after the anchor (a value after its label) is
    # nearer than the same distance before it.
    dist = distance or (lambda a, b: (a - b) if a >= b else 3 * (b - a))
    claimed = taken if taken is not None else set()

    def overlaps(o: int, n: int, spans: Iterable[tuple[int, int]]) -> bool:
        s0, e0 = idx[o], idx[o + n - 1] + 1
        return any(s0 < e and e0 > s for s, e in spans)

    def free(fs: str, occ: list[int]) -> list[int]:
        return [o for o in occ if not overlaps(o, len(fs), claimed)] or occ

    # The anchor: the cell found fewest times, the longest of those; of its
    # copies, the one the other cells sit closest to.
    anchor_i = min(range(len(wanted)), key=lambda k: (len(free(wanted[k][1], wanted[k][2])), -len(wanted[k][1])))
    others = [w for k, w in enumerate(wanted) if k != anchor_i]

    def cost(o: int) -> float:
        return sum(min(dist(p, o) for p in free(fs, occ)[:200]) for _, fs, occ in others)

    a_fs, a_occ = wanted[anchor_i][1], free(wanted[anchor_i][1], wanted[anchor_i][2])
    if after is not None:
        a_occ = [o for o in a_occ if idx[o] >= after] or a_occ
    anchor = min(a_occ[:50], key=lambda o: (cost(o), o)) if len(a_occ) > 1 and others else a_occ[0]
    out: list[dict[str, Any]] = []
    mine: list[tuple[int, int]] = []
    for k, (shown, fs, occ) in enumerate(wanted):
        if k == anchor_i:
            at = anchor
        else:
            cand = [o for o in free(fs, occ) if not overlaps(o, len(fs), mine)] or free(fs, occ)
            at = min(cand, key=lambda o: (dist(o, anchor), o))
        start = idx[at]
        end = idx[at + len(fs) - 1] + 1
        mine.append((start, end))
        out.append({"text": source[start:end], "char_start": start, "char_end": end})
    claimed.update(mine)
    return out


# ── PDF ─────────────────────────────────────────────────────────────────────


def pdf_page_text(page: Any) -> tuple[str, list[tuple[float, float, float, float] | None]]:
    """The page's characters in MuPDF order (lines separated by newlines) and
    each character's box, so a fragment's offsets give its bbox."""
    chars: list[str] = []
    boxes: list[tuple[float, float, float, float] | None] = []
    raw = page.get_text("rawdict")
    for block in raw.get("blocks") or []:
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                for ch in span.get("chars") or []:
                    c = ch.get("c") or ""
                    if not c:
                        continue
                    chars.append(c)
                    b = ch.get("bbox")
                    boxes.append(tuple(float(v) for v in b) if b else None)  # type: ignore[arg-type]
            chars.append("\n")
            boxes.append(None)
    return "".join(chars), boxes


def _union(boxes: Iterable[tuple[float, float, float, float] | None]) -> list[float] | None:
    bs = [b for b in boxes if b]
    if not bs:
        return None
    return [
        round(min(b[0] for b in bs), 2),
        round(min(b[1] for b in bs), 2),
        round(max(b[2] for b in bs), 2),
        round(max(b[3] for b in bs), 2),
    ]


def pdf_fragments(
    text: str,
    page_text: str,
    boxes: list,
    page_index: int,
    taken: set[tuple[int, int]] | None = None,
    after: int | None = None,
) -> list[dict[str, Any]] | None:
    def center(o: int) -> tuple[float, float] | None:
        b = boxes[o] if 0 <= o < len(boxes) else None
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) if b else None

    hay_idx = fold_with_map(page_text)[1]

    def distance(a: int, b: int) -> float:
        # Folded offsets -> source offsets -> character centres. Rows sit
        # closer than columns matter: a cell's own row beats the same words
        # one row down.
        ca = center(hay_idx[a]) if a < len(hay_idx) else None
        cb = center(hay_idx[b]) if b < len(hay_idx) else None
        if not ca or not cb:
            return float(abs(a - b))
        return abs(ca[0] - cb[0]) + 3.0 * abs(ca[1] - cb[1])

    frags = locate_fragments(text, page_text, distance=distance, taken=taken, after=after)
    if not frags:
        return None
    hay = fold_with_map(page_text)[0]
    pos = {src: k for k, src in enumerate(hay_idx)}
    for f in frags:
        f["page"] = page_index
        # Which copy of these words on the page this is (0 = the first, in
        # the page's own text order, whitespace aside): a viewer that cannot
        # use the bbox can still pick the same "$0.00" of several.
        at = pos.get(f["char_start"])
        if at is not None:
            f["nth"] = len(_occurrences(hay[:at], fold(f["text"])))
        bb = _union(boxes[f["char_start"]:f["char_end"]])
        if bb:
            f["bbox"] = bb
    return frags


def stamp_pdf_cell_fragments(atoms: list[Any], pdf_path: Any) -> list[Any]:
    """Give every PDF atom whose text is not on its page the page spots of
    its cells. Atoms with no page, or whose text is on the page, are left
    exactly as they are."""
    try:
        import fitz  # type: ignore
    except Exception:  # pragma: no cover
        return atoms
    pages: dict[int, tuple[str, list, set]] = {}
    out: list[Any] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return atoms
    try:
        for a in atoms:
            out.append(_stamp_one_pdf(a, doc, pages))
    finally:
        doc.close()
    return out


def _stamp_one_pdf(a: Any, doc: Any, pages: dict) -> Any:
    refs = list(getattr(a, "source_refs", None) or [])
    if not refs:
        return a
    loc = refs[0].locator if isinstance(refs[0].locator, dict) else {}
    if loc.get("cell_fragments"):
        return a
    pg = loc.get("page")
    if not isinstance(pg, int) or pg < 0 or pg >= len(doc):
        return a
    text = getattr(a, "raw_text", "") or ""
    if " | " not in text and ":" not in text and "=" not in text and "→" not in text:
        return a
    if pg not in pages:
        try:
            pages[pg] = (*pdf_page_text(doc[pg]), set())
        except Exception:
            pages[pg] = ("", [], set())
    page_text, boxes, taken = pages[pg]
    after = max((e for _, e in taken), default=None)
    frags = pdf_fragments(text, page_text, boxes, pg, taken, after) if page_text else None
    if not frags:
        return a
    return with_fragments(a, frags)


def with_fragments(a: Any, frags: list[dict[str, Any]]) -> Any:
    """``a`` with ``frags`` on its primary locator (a copy; ids unchanged)."""
    refs = list(a.source_refs)
    loc = dict(refs[0].locator or {})
    loc["cell_fragments"] = frags
    refs[0] = refs[0].model_copy(update={"locator": loc})
    return a.model_copy(update={"source_refs": refs})


# ── grids (xlsx / csv / docx tables) ────────────────────────────────────────


def grid_fragments(text: str, cells: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]] | None:
    """Fragments for a row atom read off a grid: ``cells`` are the row's
    cells in order as ``(text, where)`` -- ``where`` is merged into each
    fragment (``{"row": 4, "col": 2}``, ``{"sheet": ...}``)."""
    joined = "\n".join(c for c, _ in cells)
    frags = locate_fragments(text, joined)
    if not frags:
        return None
    starts: list[int] = []
    pos = 0
    for c, _ in cells:
        starts.append(pos)
        pos += len(c) + 1
    for f in frags:
        k = max(i for i, s in enumerate(starts) if s <= f["char_start"])
        # The offsets index this function's own join; the cell is the address.
        del f["char_start"], f["char_end"]
        f.update(cells[k][1])
    return frags


def stamp_grid_cell_fragments(
    atoms: list[Any],
    cells_for: Callable[[dict[str, Any]], list[tuple[str, dict[str, Any]]] | None],
    direct_for: Callable[[dict[str, Any]], list[tuple[str, dict[str, Any]]] | None] | None = None,
) -> list[Any]:
    """Give every row atom of a grid source (a docx table, a sheet) the cells
    its text was read from. ``cells_for(locator)`` returns the row's cells
    as ``(text, where)`` -- or ``None`` when the locator names no row. An
    atom whose text sits whole inside one cell gets nothing.

    ``direct_for(locator)`` names cells the atom was read from outright (a
    sheet atom's ``columns``): those are its fragments whatever its text
    says ("Quantity 4 Wireless access point" is the parser's sentence)."""
    out: list[Any] = []
    for a in atoms:
        try:
            refs = list(getattr(a, "source_refs", None) or [])
            loc = refs[0].locator if refs and isinstance(refs[0].locator, dict) else None
            text = getattr(a, "raw_text", "") or ""
            if not loc or loc.get("cell_fragments") or not text.strip():
                out.append(a)
                continue
            direct = direct_for(loc) if direct_for else None
            if direct and not any(fold(text) and fold(text) in fold(c) for c, _ in direct):
                out.append(with_fragments(a, [{"text": c, **w} for c, w in direct[:MAX_FRAGMENTS]]))
                continue
            cells = cells_for(loc)
            if not cells:
                out.append(a)
                continue
            whole = fold(text)
            if any(whole and whole in fold(c) for c, _ in cells):
                out.append(a)
                continue
            frags = grid_fragments(text, cells)
            out.append(with_fragments(a, frags) if frags else a)
        except Exception:  # provenance never costs an atom
            out.append(a)
    return out


def cell_text(value: Any) -> str:
    """A cell value as the parsers print it (2000.0 -> "2000")."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _candidate(a: Any) -> dict[str, Any] | None:
    refs = getattr(a, "source_refs", None) or []
    loc = refs[0].locator if refs and isinstance(refs[0].locator, dict) else None
    if not loc or loc.get("cell_fragments"):
        return None
    return loc


#: Sheets: rows read per workbook at most (a 15,000-row export is not a
#: table a labeler clicks through row by row).
MAX_SHEET_ROWS = 20000


def stamp_xlsx_cell_fragments(atoms: list[Any], path: Any) -> list[Any]:
    """Sheets: the row's cells, or the columns the atom names
    (``locator["columns"]``, a field -> column letter map). Only the rows
    some atom names are read, in one streaming pass per sheet."""
    want: dict[str, set[int]] = {}
    for a in atoms:
        loc = _candidate(a)
        if not loc:
            continue
        sheet, r = loc.get("sheet"), loc.get("row")
        if isinstance(sheet, str) and isinstance(r, int) and not isinstance(r, bool) and r >= 1:
            want.setdefault(sheet, set()).add(r)
    if not want:
        return atoms
    try:
        import openpyxl  # type: ignore
        from openpyxl.utils import column_index_from_string, get_column_letter  # type: ignore

        wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    except Exception:
        return atoms
    rows: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = {}
    try:
        for sheet, needed in want.items():
            if sheet not in wb.sheetnames:
                continue
            last = min(max(needed), MAX_SHEET_ROWS)
            for r, values in enumerate(wb[sheet].iter_rows(max_row=last, values_only=True), start=1):
                if r in needed:
                    rows[(sheet, r)] = [
                        (cell_text(v), {"sheet": sheet, "row": r, "col": get_column_letter(c)})
                        for c, v in enumerate(values[:400], start=1)
                        if cell_text(v)
                    ]
    except Exception:
        return atoms
    finally:
        try:
            wb.close()
        except Exception:
            pass

    def cells_for(loc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]] | None:
        return rows.get((loc.get("sheet"), loc.get("row"))) or None  # type: ignore[arg-type]

    def direct_for(loc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]] | None:
        cells = cells_for(loc)
        cols = loc.get("columns")
        if not cells or not isinstance(cols, dict) or not cols:
            return None
        letters = {v.upper() for v in cols.values() if isinstance(v, str) and v.isalpha()}
        named = [c for c in cells if c[1]["col"] in letters]
        return sorted(named, key=lambda c: column_index_from_string(c[1]["col"])) or None

    return stamp_grid_cell_fragments(atoms, cells_for, direct_for)


def stamp_cell_fragments(atoms: list[Any], path: Any) -> list[Any]:
    """The fragments pass for a source file of any kind this module reads.
    Atoms that already carry fragments are left alone, so it is safe to run
    after a parser that stamped its own."""
    suffix = str(getattr(path, "suffix", "") or "").lower()
    if len(atoms) > MAX_SHEET_ROWS:
        return atoms
    if suffix in (".xlsx", ".xlsm"):
        return stamp_xlsx_cell_fragments(atoms, path)
    return atoms


def stamp_docx_cell_fragments(atoms: list[Any], tables: list[Any]) -> list[Any]:
    """Word tables: the row's cells (merged cells once), with the table's
    first row ahead of them for "Header: value" atoms."""
    cache: dict[tuple[int, int], list[tuple[str, dict[str, Any]]]] = {}

    def one_row(ti: int, r: int) -> list[tuple[str, dict[str, Any]]]:
        key = (ti, r)
        if key not in cache:
            got: list[tuple[str, dict[str, Any]]] = []
            try:
                seen: set[int] = set()
                for ci, c in enumerate(tables[ti].rows[r].cells):
                    if id(c._tc) in seen:
                        continue
                    seen.add(id(c._tc))
                    t = (c.text or "").strip()
                    if t:
                        got.append((t, {"table_index": ti, "row": r, "col": ci}))
            except Exception:
                got = []
            cache[key] = got
        return cache[key]

    def cells_for(loc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]] | None:
        ti, r = loc.get("table_index"), loc.get("row")
        if not isinstance(ti, int) or not isinstance(r, int) or not (0 <= ti < len(tables)):
            return None
        if isinstance(ti, bool) or isinstance(r, bool):
            return None
        own = one_row(ti, r)
        if not own:
            return None
        return (one_row(ti, 0) if r > 0 else []) + own

    return stamp_grid_cell_fragments(atoms, cells_for)
