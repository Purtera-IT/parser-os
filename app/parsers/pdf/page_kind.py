"""Is this PDF page a drawing, or a page with a table on it?

Both table finders ask, so the test lives where both can import it without a
cycle: `orbitbrief_pdf` already imports from `pdf.tables`.
"""
from __future__ import annotations

from typing import Any


#: A floor plan's walls are vector paths, and there are thousands of them
#: against almost no text. A ruled table is the other way round: a few dozen
#: rules and a page of words. Measured on 010180's SP-6 sheet: 8,582 paths to
#: 1,014 characters. The margin either side of this is enormous.
_DRAWING_MIN_PATHS = 1000
_DRAWING_PATHS_PER_CHAR = 2.0


def _page_is_a_drawing(page: Any) -> bool:
    """Is this page a drawing rather than a page with a table on it?

    ``find_tables(strategy="lines")`` finds tables by looking for ruled lines,
    and a floor plan is nothing BUT ruled lines. On 010180's SP-6 sheet it
    found five "tables" in the walls, and every room name that crossed a wall
    was split across two cells: RECEPTION became ``RECEPT`` | ``ION``,
    EXECUTIVE OFFICE became ``EXECUTIV OFFICE``, and the row rendered as
    ``RECEPT: ION | col_4: COPY | col_5: PRINT/ COPY``. Eighteen of the
    drawing's forty atoms were wreckage of this kind, and a labeler had to
    reject every one.

    Nothing is lost by skipping them. The page's text layer carries every room
    name whole -- RECEPTION, BOARD ROOM, PRINT/COPY, MECH. RM -- and the room
    SCHEDULE with its counts ("EXECUTIVE OFFICE 2", "IT CLOSET 1",
    "4-6 PERSON HUDDLE ROOM 2"), which is better evidence than the grid was
    ever going to give. Vision reads the sheet as well.

    Kept as a ratio rather than a flag so it needs no per-file configuration,
    and deliberately far from the boundary: a ruled rate card runs around 0.1
    paths per character.
    """
    try:
        paths = len(page.get_drawings())
        chars = len(page.get_text() or "")
    except Exception:
        return False
    return paths >= _DRAWING_MIN_PATHS and paths >= _DRAWING_PATHS_PER_CHAR * max(chars, 1)




def table_cuts_words(page: Any, table: Any) -> bool:
    """Does this found "table" draw its cell walls through words?

    A real table's rules run between its words. A page whose ruled lines are
    a diagram's boxes or arrows -- an install guide's figure beside its steps,
    a floor plan too small for :func:`_page_is_a_drawing` -- yields a grid
    whose walls cross the text, and the extraction shreds words mid-word
    ("RECEPT | ION", "Loos | en the turnbuckles", 010246 page 5). The text
    layer knows where each word is: when two or more words (and at least one
    in twenty inside the table) straddle a cell's vertical edge, the grid is
    not a table and its region is left to the layout reader.
    """
    try:
        cells = [c for c in (getattr(table, "cells", None) or []) if c]
        if not cells:
            return False
        tb = getattr(table, "bbox", None)
        if not tb:
            return False
        tx0, ty0, tx1, ty1 = (float(v) for v in tb)
        cells = [tuple(float(v) for v in c) for c in cells]
        if not any(tx0 + 1.0 < e < tx1 - 1.0 for c in cells for e in (c[0], c[2])):
            return False

        def _walls_at(y: float) -> list[float]:
            # Only the walls of the cells on the word's own line can cut it: a
            # merged cell ("ESTIMATED TOTAL FEES" across four columns) spans the
            # walls of the rows above it, and those walls do not run through it
            # (010087's fee table was thrown out as a figure on that alone).
            return [e for c in cells if c[1] - 0.5 <= y <= c[3] + 0.5
                    for e in (c[0], c[2]) if tx0 + 1.0 < e < tx1 - 1.0]
        words = [w for w in (page.get_text("words") or [])
                 if tx0 - 1 <= (w[0] + w[2]) / 2.0 <= tx1 + 1 and ty0 - 1 <= (w[1] + w[3]) / 2.0 <= ty1 + 1]
        if not words:
            return False
        cut = 0
        for w in words:
            x0, x1 = float(w[0]), float(w[2])
            if len(str(w[4]).strip()) < 2:
                continue
            if any(x0 + 1.5 < e < x1 - 1.5 for e in _walls_at((float(w[1]) + float(w[3])) / 2.0)):
                cut += 1
        return cut >= 2 and cut * 20 >= len(words)
    except Exception:
        return False
