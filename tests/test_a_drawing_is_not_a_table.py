"""A floor plan is nothing but ruled lines, so a table finder sees a grid.

010180's SP-6 sheet: 8,582 vector paths against 1,014 characters of text. Every
table detector on the page found "tables" in the walls, and every room name that
crossed a wall was split across two cells --

    RECEPTION        -> "RECEPT: ION | col_4: COPY | col_5: PRINT/ COPY"
    WOMEN'S RESTROOM -> "WOM RESTR: EN'S ROOM | col_5: EXEC OF"
    EXECUTIVE OFFICE -> "EXECUTIV OFFICE"
    Commercial Leasing -> "as: Com", "as: mm", "as: cial Leasi", "Co: mm"

-- a row claiming that RECEPT is a field and ION is its value. Measured on the
page: 26 of 43 atoms came out of these grids and every one was wreckage, while
all 12 `note` blocks were clean. A labeler had to reject 22 atoms by hand.

Nothing is lost by leaving the walls alone. The page's own text layer carries
every room name whole and the room SCHEDULE with its counts --

    12 PERSON BOARD ROOM / EXECUTIVE OFFICE 2 / PRIVATE OFFICE 2 / IT CLOSET 1
    PHONE ROOM 1 / ADA RESTROOM 1 / 4-6 PERSON HUDDLE ROOM 2 / PRINT/COPY 1
    5'-0" WORKSTATIONS 106 / Floor 12 | Suite 1200 | 12,154 RSF

-- which is better evidence than the grid was ever going to give, and vision
reads the sheet as well.

The measure is a ratio so it needs no per-file configuration, and it sits a
long way from the boundary: a ruled rate card runs about 0.1 paths per
character, a floor plan 8.5.
"""
from __future__ import annotations

import pytest

from app.parsers.pdf.page_kind import (_DRAWING_MIN_PATHS,
                                       _DRAWING_PATHS_PER_CHAR,
                                       _page_is_a_drawing)


class _Page:
    """Only what the test asks of a page."""

    def __init__(self, paths: int, chars: int):
        self._paths = paths
        self._chars = chars

    def get_drawings(self):
        return [None] * self._paths

    def get_text(self):
        return "x" * self._chars


#: (paths, chars, is_a_drawing, what it is)
CASES = [
    (8582, 1014, True, "010180's SP-6 floor plan, measured"),
    (12000, 300, True, "a denser drawing"),
    (2100, 900, True, "a sparse plan"),
    (300, 3000, False, "a ruled rate card"),
    (40, 6000, False, "a prose page with one table"),
    (0, 4000, False, "plain prose"),
    (5000, 4000, False, "a busy page that is still mostly words"),
    (900, 10, False, "below the path floor, whatever the ratio"),
]


@pytest.mark.parametrize("paths,chars,expected,what", CASES)
def test_the_ratio_separates_walls_from_tables(paths, chars, expected, what):
    assert _page_is_a_drawing(_Page(paths, chars)) is expected, what


def test_the_measured_page_clears_the_threshold_by_a_mile():
    """Not a near miss: the sheet is four times the path floor and four times
    the ratio. A rate card is two orders of magnitude the other side."""
    assert 8582 >= 4 * _DRAWING_MIN_PATHS
    assert 8582 / 1014 >= 4 * _DRAWING_PATHS_PER_CHAR
    assert 300 / 3000 < _DRAWING_PATHS_PER_CHAR / 10


def test_a_page_that_cannot_be_measured_is_not_a_drawing():
    """Fails open: an unreadable page keeps its tables. Losing a rate card to a
    raised exception would be worse than carrying some wreckage."""

    class _Broken:
        def get_drawings(self):
            raise RuntimeError("no")

        def get_text(self):
            return ""

    assert _page_is_a_drawing(_Broken()) is False
