"""Recover a site roster that was pasted as a TABLE into a HubSpot note.

WHY THIS EXISTS

A PM pastes a site table into a HubSpot note. HubSpot stores it as an HTML
``<table>``; the ingest converts that to the ``.txt`` parser-os receives.

Until Platform-infra's ``html-to-text`` change, that conversion turned every
tag into a space and collapsed all whitespace, so the table arrived as a single
line. Deal 010310 is the case that surfaced it -- a five-column, three-row site
table reached parser-os as one 448-byte line, and the deal published FOUR sites:
one real site split across two rows, all four named from city+province+postal
fragments, and the site actually named "Malport" -- the only name that is not
also a city, so the only one that cannot be re-derived from geography -- lost.

The extractor was never the problem. ``site_roster_extractor`` already maps
"site name", "address", "city", "province" and "postal code" to canonical
roster fields; fed the columns it returns exactly the right rows. It was simply
unreachable: the roster path ran only from the xlsx and PDF parsers, never from
a note.

This module closes that gap. It finds a delimited table in the note body and
hands it to the SAME gate and extractor the spreadsheet path uses, so a roster
is read the same way no matter which door it came in through.

Structure is the only signal used -- a run of lines that share a delimiter and
a field count. No filename patterns and no word lists: the header synonyms in
``site_roster_extractor`` remain the single place that knows what a roster
column is called.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

#: Field separator emitted by the HTML->text converter for ``</td>`` / ``</th>``.
#: A tab cannot occur in a HubSpot note body by accident: the converter collapses
#: every literal tab typed in prose back to a space precisely so that a surviving
#: tab means "column break" and nothing else.
_DELIM = "\t"

#: A table needs a header and at least this many data rows. Two rows sharing a
#: shape is a coincidence a single stray line can manufacture; three is a table.
#: The gate in ``looks_like_site_roster`` still has the final say.
_MIN_DATA_ROWS = 2

#: A single-column "table" is a list, and a run of lines each holding one field
#: is just prose that happens to contain a tab.
_MIN_COLUMNS = 2


def find_delimited_table(
    lines: Sequence[str],
) -> tuple[list[str], list[list[str]], int] | None:
    """Return ``(columns, rows, header_index)`` for the first delimited table
    in ``lines``, or None.

    A table is the longest run of consecutive lines that all split into the same
    number of fields on the delimiter. The first line of the run is the header.
    """
    best: tuple[list[str], list[list[str]], int] | None = None
    i = 0
    n = len(lines)
    while i < n:
        first = str(lines[i] or "")
        if _DELIM not in first:
            i += 1
            continue
        width = len(first.split(_DELIM))
        if width < _MIN_COLUMNS:
            i += 1
            continue
        j = i + 1
        while j < n and len(str(lines[j] or "").split(_DELIM)) == width:
            j += 1
        # j is now one past the last line sharing this shape.
        if (j - i - 1) >= _MIN_DATA_ROWS:
            columns = [c.strip() for c in first.split(_DELIM)]
            rows = [
                [c.strip() for c in str(lines[k] or "").split(_DELIM)]
                for k in range(i + 1, j)
            ]
            if best is None or len(rows) > len(best[1]):
                best = (columns, rows, i)
        i = max(j, i + 1)
    return best


#: A table cell is a value, not a sentence. Used only to tell where the table
#: ENDS when the markup gave no row boundaries -- the prose that follows a
#: roster is the first thing that stops being cell-shaped. Shape, not
#: vocabulary: no word is special, only length is.
_MAX_CELL_WORDS = 12
_MAX_CELL_CHARS = 80


def _looks_like_cell(value: str) -> bool:
    """Could this line be one cell of a table row?"""
    v = str(value or "").strip()
    if not v:
        return True  # a blank cell is still a cell
    if len(v) > _MAX_CELL_CHARS:
        return False
    return len(v.split()) <= _MAX_CELL_WORDS


#: A column header is a LABEL. "Address" is; "The customer has two locations,
#: addresses below, where they want assistance..." is not -- but the synonym
#: table matches long synonyms as substrings, so that sentence maps to
#: street_address and, unguarded, becomes a header that shifts every column by
#: one. Shape decides what may be a header; the synonym table decides what it
#: means.
_MAX_HEADER_WORDS = 5
_MAX_HEADER_CHARS = 40

#: How far past the recognised headers to look for further columns. A roster
#: carries columns nobody named canonically -- "APs", "Users", "Notes" -- and
#: stopping at the last RECOGNISED header would cut the row short and shift
#: every value left.
_EXTRA_COLUMN_SEARCH = 7

#: A column whose values disagree in shape is not a column. Half is a low bar
#: deliberately: a real roster usually scores 1.0, and the wrong width usually
#: scores near 0, so this only has to separate those two.
_MIN_COLUMN_COHERENCE = 0.5


def _header_shaped(value: str) -> bool:
    """Could this line be a column LABEL rather than a sentence?"""
    v = str(value or "").strip()
    return bool(v) and len(v) <= _MAX_HEADER_CHARS and len(v.split()) <= _MAX_HEADER_WORDS


def _shape(value: str) -> str:
    """A coarse class for a cell, used only to test whether a column agrees
    with itself. Never used to decide what a value MEANS."""
    v = str(value or "").strip()
    if not v:
        return "empty"
    if re.fullmatch(r"\d+", v):
        return "int"
    if re.fullmatch(r"\d{5}(?:-\d{4})?", v):
        return "zip5"
    if re.fullmatch(r"[A-Za-z]\d[A-Za-z]\s*\d[A-Za-z]\d", v):
        return "postal_ca"
    if re.fullmatch(r"[A-Za-z]{2}", v):
        return "code2"
    if re.search(r"\d", v):
        return "alnum"
    return "alpha"


def _column_coherence(rows: list[list[str]], width: int) -> float:
    """Fraction of columns whose every value shares one shape.

    This is what tells the right column count from the wrong one. At the wrong
    width the rows are cut mid-record, so a column holds a street address in
    one row and a postcode in the next; at the right width each column is one
    kind of thing all the way down.
    """
    if not rows or width <= 0:
        return 0.0
    agreed = 0
    for c in range(width):
        if len({_shape(r[c]) for r in rows}) == 1:
            agreed += 1
    return agreed / width


def find_stacked_table(
    lines: Sequence[str],
) -> tuple[list[str], list[list[str]], int] | None:
    """Return ``(columns, rows, header_index)`` for a table whose every cell
    landed on its OWN line, or None.

    HubSpot does not always store a pasted table as ``<table><tr><td>``. When
    each cell is wrapped in a block element instead, the markup carries no
    distinction between "end of cell" and "end of row" -- both are just block
    ends -- so the HTML->text step cannot recover rows no matter how careful it
    is. The information is genuinely absent from the source.

    It survives in the HEADER and in the SHAPE OF THE COLUMNS. Recognised
    headers give a lower bound on the width; the true width is the one at which
    every column agrees with itself all the way down. Deal 010310 arrived as
    five headers then fifteen cells; deal 02557291 as a paragraph, then five
    headers -- one of them ("APs") not a roster field at all -- then ten.
    """
    try:
        from app.parsers.site_roster_extractor import map_columns_to_fields
    except Exception:  # pragma: no cover
        return None

    cells = [str(l or "").strip() for l in lines]
    n = len(cells)
    best: tuple[tuple[float, int], list[str], list[list[str]], int] | None = None

    for start in range(n):
        # The recognised run: consecutive header-SHAPED lines that each map to a
        # distinct roster field. Header-shaped is checked first, so a paragraph
        # mentioning "address" can never open a table.
        recognised: list[str] = []
        for cell in cells[start:]:
            if not _header_shaped(cell):
                break
            candidate = recognised + [cell]
            try:
                mapped = map_columns_to_fields(candidate)
            except Exception:  # pragma: no cover
                break
            if len(mapped) != len(candidate):
                break
            recognised = candidate
        if len(recognised) < _MIN_COLUMNS:
            continue

        # The real table may be wider than the part we recognise.
        for width in range(len(recognised), len(recognised) + _EXTRA_COLUMN_SEARCH + 1):
            if start + width > n:
                break
            header = cells[start : start + width]
            if not all(_header_shaped(h) for h in header):
                break
            body = cells[start + width :]
            rows: list[list[str]] = []
            i = 0
            while i + width <= len(body):
                group = body[i : i + width]
                # A table is contiguous: the first group that stops looking like
                # cells is the prose after it, not a row with odd values.
                if not all(_looks_like_cell(c) for c in group):
                    break
                rows.append(group)
                i += width
            if len(rows) < _MIN_DATA_ROWS:
                continue
            score = (_column_coherence(rows, width), len(rows))
            if best is None or score > best[0]:
                best = (score, header, rows, start)

    if best is None or best[0][0] < _MIN_COLUMN_COHERENCE:
        return None
    _score, header, rows, at = best
    return header, rows, at


def site_roster_from_note_lines(
    lines: Sequence[str],
    *,
    surrounding_text: str = "",
) -> tuple[list[Any], list[str], list[list[str]]]:
    """``(roster_rows, columns, rows)`` for a site roster in the note body.

    Returns empty lists when the note holds no delimited table, or when the
    table is not a site roster by ``looks_like_site_roster``'s own judgment.
    """
    # A real <table> gives tab-delimited rows; block-wrapped cells give one cell
    # per line. Both are the same table.
    found = find_delimited_table(lines) or find_stacked_table(lines)
    if not found:
        return [], [], []
    columns, rows, _header_index = found
    try:
        from app.parsers.site_roster_extractor import extract_site_roster
    except Exception:  # pragma: no cover - extractor must never break a parse
        return [], [], []
    try:
        roster_rows = extract_site_roster(
            columns=columns, rows=rows, surrounding_text=surrounding_text
        )
    except Exception:  # pragma: no cover
        return [], [], []
    if not roster_rows:
        return [], columns, rows
    return list(roster_rows), columns, rows


def site_entity_key(site_row: Any) -> str:
    """``site:<slug>`` for a roster row, or "" when it names nothing.

    The slug comes from the roster's own site_id / facility name / address --
    the identity the PM wrote down -- never from geography assembled after the
    fact, which is what produced "mississauga on l4t" and "torbram rd
    mississauga on l4t" as two different sites for one row.
    """
    for candidate in (
        getattr(site_row, "site_id", "") or "",
        getattr(site_row, "facility_name", "") or "",
        getattr(site_row, "street_address", "") or "",
    ):
        slug = re.sub(r"[^a-z0-9]+", "_", str(candidate).lower()).strip("_")
        if slug:
            return f"site:{slug}"
    return ""
