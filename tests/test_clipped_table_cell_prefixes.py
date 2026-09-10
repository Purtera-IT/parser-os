"""A table bbox narrower than its text silently eats the first letters.

`find_tables()` reports the bounds of the RULING, and a cell's text can start
left of it. Extraction clips to the cell, so the opening characters are dropped
-- and what survives still reads like words, which is the dangerous part.

The Anova install guide (table bbox x0=81.0, text line x0=50.8) lost the start
of every row in the per-tank checklist:

    "IMPORTANT INFORMATION - ..."   ->  "TANT INFORMATION - ..."
    "UTM device serial number"      ->  "evice serial number"
    "Tank style and dimensions"     ->  "tyle and dimensions"
    "Product in the tank"           ->  "t in the tank"
    "Current stick level of product" -> "t stick level of product"

Those are the data points an installer must collect per tank. Every one of them
reached the atoms wrong.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.pdf._shared import _restore_clipped_prefixes


class _Cell(tuple):
    pass


class _Row:
    def __init__(self, cells):
        self.cells = cells


class _Page:
    """A page with one text line, positioned by the caller."""

    def __init__(self, text, x0, y0, y1):
        self._d = {
            "blocks": [
                {"lines": [{"bbox": [x0, y0, x0 + 400.0, y1],
                            "spans": [{"text": text}]}]}
            ]
        }

    def get_text(self, _kind):
        return self._d


def test_the_clipped_opening_is_given_back():
    page = _Page("IMPORTANT INFORMATION - Please have the following", 50.8, 493.4, 504.8)
    rows = [["TANT INFORMATION - Please have the following"]]
    cell_rows = [_Row([(81.0, 491.1, 576.2, 510.0)])]
    _restore_clipped_prefixes(page, cell_rows, rows)
    assert rows[0][0] == "IMPORTANT INFORMATION - Please have the following"


def test_a_cell_that_was_never_clipped_is_left_alone():
    page = _Page("Device serial number", 100.0, 493.4, 504.8)
    rows = [["Device serial number"]]
    cell_rows = [_Row([(81.0, 491.1, 576.2, 510.0)])]
    _restore_clipped_prefixes(page, cell_rows, rows)
    assert rows[0][0] == "Device serial number"


def test_only_a_strict_suffix_is_repaired():
    """The guard: a suffix is provably a left-truncation. Anything else could
    be a different line and must not be substituted."""
    page = _Page("Something else entirely", 50.0, 493.4, 504.8)
    rows = [["evice serial number"]]
    cell_rows = [_Row([(81.0, 491.1, 576.2, 510.0)])]
    _restore_clipped_prefixes(page, cell_rows, rows)
    assert rows[0][0] == "evice serial number", "must not borrow an unrelated line"


def test_a_line_starting_right_of_the_cell_is_not_used():
    page = _Page("XX evice serial number", 200.0, 493.4, 504.8)
    rows = [["evice serial number"]]
    cell_rows = [_Row([(81.0, 491.1, 576.2, 510.0)])]
    _restore_clipped_prefixes(page, cell_rows, rows)
    assert rows[0][0] == "evice serial number"


def test_a_line_in_a_different_band_is_not_used():
    page = _Page("Device serial number", 50.0, 100.0, 112.0)  # far above the cell
    rows = [["evice serial number"]]
    cell_rows = [_Row([(81.0, 491.1, 576.2, 510.0)])]
    _restore_clipped_prefixes(page, cell_rows, rows)
    assert rows[0][0] == "evice serial number"


def test_an_empty_cell_is_untouched():
    page = _Page("Device serial number", 50.0, 493.4, 504.8)
    rows = [[""]]
    cell_rows = [_Row([(81.0, 491.1, 576.2, 510.0)])]
    _restore_clipped_prefixes(page, cell_rows, rows)
    assert rows[0][0] == ""


def test_the_repair_only_ever_adds_back_a_prefix():
    """It can never add, drop or change a character beyond restoring the
    opening -- the same standard the transposition repair holds itself to."""
    page = _Page("Current stick level of product", 50.0, 493.4, 504.8)
    rows = [["t stick level of product"]]
    cell_rows = [_Row([(81.0, 491.1, 576.2, 510.0)])]
    _restore_clipped_prefixes(page, cell_rows, rows)
    restored = rows[0][0]
    assert restored.endswith("t stick level of product")
    assert restored == "Current stick level of product"
