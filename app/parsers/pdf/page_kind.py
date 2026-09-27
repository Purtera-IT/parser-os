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


