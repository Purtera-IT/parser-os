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

    It survives in exactly one place: the HEADER. The count of leading lines
    that name a roster column IS the column count, and the cells that follow are
    that table in row-major order. So read the header run, then re-fold.

    Deal 010310 arrived exactly this way -- 'Site Name' / 'Address' / 'City' /
    'Province' / 'Postal Code' on five lines, then fifteen cells.
    """
    try:
        from app.parsers.site_roster_extractor import map_columns_to_fields
    except Exception:  # pragma: no cover
        return None

    # The header is the longest run of leading lines that EVERY map to a
    # distinct roster field. The first line that does not is the first datum.
    header: list[str] = []
    for line in lines:
        candidate = header + [str(line or "").strip()]
        try:
            mapped = map_columns_to_fields(candidate)
        except Exception:  # pragma: no cover
            break
        if len(mapped) != len(candidate):
            break
        header = candidate
    width = len(header)
    if width < _MIN_COLUMNS:
        return None

    body = [str(l or "").strip() for l in lines[width:]]
    rows: list[list[str]] = []
    i = 0
    while i + width <= len(body):
        group = body[i : i + width]
        # A table is contiguous: the first group that stops looking like cells
        # is the prose after it, not a row with odd values.
        if not all(_looks_like_cell(c) for c in group):
            break
        rows.append(group)
        i += width
    if len(rows) < _MIN_DATA_ROWS:
        return None
    return header, rows, 0


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
