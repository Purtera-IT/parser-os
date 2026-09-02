"""A merged table header must not silently delete cells.

python-docx expands a merged cell into one entry per grid column, repeating its
text. A SOW header cell spanning four columns therefore yields
["Requestor Information"] * 4, and `dict(zip(header, cells))` overwrites three of
every four values with the fourth.

Measured on the dev corpus 2026-09-02: 204 of 418 docx (49%) across 99 deals hold
at least one such table, and 3,836 cell values were being dropped. On deal 010215
that was 141 cells per site SOW -- including "Address Line 1 | 601 Gurley St" --
so ten per-site scope documents produced three atoms each and no site address.
Sites had to be scraped from email prose instead, arriving merged
("601 gurley street 1205 south main street") and truncated ("1123" -> "123").

The loss was invisible: nothing warned, and the atom still carried the full text
in raw_text, so the document looked parsed.
"""

from __future__ import annotations

from app.parsers.docx_parser import _cells_by_column


def test_a_merged_header_no_longer_eats_cells():
    header = ["Requestor Information"] * 4
    cells = ["Address Line 1", "601 Gurley St", "Address Line 2", ""]
    out = _cells_by_column(header, cells)
    assert len(out) == 4, "every cell must survive"
    assert "601 Gurley St" in out.values(), "the address itself must be present"


def test_the_old_behaviour_would_have_lost_three_of_four():
    # Pinning what the bug did, so nobody reintroduces dict(zip(...)) as a tidy-up.
    header = ["Requestor Information"] * 4
    cells = ["Address Line 1", "601 Gurley St", "Address Line 2", ""]
    assert len(dict(zip(header, cells))) == 1
    assert len(_cells_by_column(header, cells)) == 4


def test_distinct_headers_are_untouched():
    # The common case must keep its readable keys.
    out = _cells_by_column(["Qty", "Item", "Price"], ["2", "Clock", "$305"])
    assert out == {"Qty": "2", "Item": "Clock", "Price": "$305"}


def test_a_repeat_keeps_the_first_under_its_own_name():
    # The first occurrence stays addressable as the plain column name, so existing
    # readers of well-formed tables see no change.
    out = _cells_by_column(["Site", "Site"], ["Johnakin", "Marion HS"])
    assert out["Site"] == "Johnakin"
    assert "Marion HS" in out.values()


def test_no_header_falls_back_to_positional():
    assert _cells_by_column([], ["a", "b"]) == {"col_0": "a", "col_1": "b"}


def test_more_cells_than_headers_still_keeps_them_all():
    # A ragged row must not lose its tail.
    out = _cells_by_column(["A"], ["1", "2", "3"])
    assert len(out) == 3
    assert set(out.values()) == {"1", "2", "3"}


def test_blank_header_names_do_not_collapse_together():
    out = _cells_by_column(["", "", ""], ["x", "y", "z"])
    assert len(out) == 3
