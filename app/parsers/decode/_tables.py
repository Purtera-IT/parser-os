"""Repairing what the PDF table extractor got wrong.

This is decoding, not interpretation: it decides what a cell SAYS, never
what it means. It lived in orbitbrief_pdf.py because that is where the
extractor was called, which also made decode/pdf.py import back into a
10,072-line module for three pure functions.

``fitz.find_tables()`` re-reads each cell and can emit its glyphs out of
order, or drop an underscore outright. ``page.get_text(clip=cell)`` reads
the same rectangle correctly but is not a blanket replacement -- a clip can
slice a glyph off the edge, and a loose bbox can pull in a neighbour. So a
substitution happens only under a guard a repair can satisfy and a
corruption cannot.
"""

from __future__ import annotations

from typing import Any


def _same_glyphs(a: str, b: str) -> bool:
    """Same characters, different order — the transposition bug and nothing else.

    A repair can reorder glyphs; it can never add, drop or change one. So an
    anagram is safe to accept: whatever the clip returned, it is built from
    exactly the characters ``extract()`` already had.
    """
    return sorted(a.replace(" ", "")) == sorted(b.replace(" ", ""))


def _same_but_for_underscores(a: str, b: str) -> bool:
    """Identical once underscores and spaces are removed.

    Measured across the stored corpus, the table extractor mishandles the
    underscore in two ways. Usually it relocates it to the end of the cell and
    leaves a space behind ("quantity_conflict" -> "quantity conflict _"), which
    ``_same_glyphs`` already catches. Sometimes it drops the character outright:

        extract()  'Send to russell r@aps.edu'
        clip       'Send to russell_r@aps.edu'

    That is not an anagram, so the anagram guard let it through — and the result
    is not a cosmetic blemish, it is an email address that no longer resolves.
    Snake_case part numbers and file names fail the same way.

    Accepting the clip here is still safe. The two strings agree on every
    character that is not an underscore or a space, so the substitution cannot
    introduce a different word; it can only restore a separator the extractor
    lost. The guard stays tight against the real hazards: a sliced glyph or a
    neighbouring cell bleeding in both change non-underscore characters and are
    still rejected.
    """
    strip = str.maketrans("", "", "_ ")
    return a.translate(strip) == b.translate(strip)


def _table_rows_repaired(page: Any, table: Any) -> list[list[Any]]:
    """``table.extract()`` with transposed glyphs repaired from the page text.

    PyMuPDF's table extractor re-reads each cell and can emit its glyphs out of
    order: the Xtra Lease install spec came back with "Initail document",
    "order of executoin" and "add lifgtate operatoin" where the page itself
    says Initial / execution / liftgate. 9 of 54 cells (17%) were corrupted,
    and nothing downstream can tell -- the words are plausible, just wrong, and
    they flow into the SOW.

    ``page.get_text(clip=cell)`` reads the same rectangle correctly, but it is
    not a blanket replacement: a clip boundary can slice a glyph off the edge
    (one cell here lost its leading "f", giving "ication:" for "fication:"), and
    a loose bbox can pull in a neighbour. So substitute only under a guard that
    a repair can satisfy and a corruption cannot -- see ``_same_glyphs`` and
    ``_same_but_for_underscores``.
    """
    import fitz  # type: ignore[import-not-found]

    rows = table.extract()
    try:
        cell_rows = list(getattr(table, "rows", []) or [])
    except Exception:
        return rows
    for ri, row in enumerate(cell_rows):
        if ri >= len(rows):
            break
        for ci, cell in enumerate(getattr(row, "cells", []) or []):
            if cell is None or ci >= len(rows[ri]):
                continue
            original = rows[ri][ci]
            if not original:
                continue
            a = " ".join(str(original).split())
            try:
                b = " ".join((page.get_text("text", clip=fitz.Rect(cell)) or "").split())
            except Exception:
                continue
            if not b or a == b:
                continue
            if _same_glyphs(a, b) or _same_but_for_underscores(a, b):
                rows[ri][ci] = b
    return rows

