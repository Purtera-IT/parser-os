"""The collision is not a docx bug — it is a table bug, in core.

`dict(zip(columns, row))` drops a cell whenever two columns share a name, which
a merged header guarantees. It appeared in the docx parser AND in
table_schema_registry, which is fed `_columns` by the docx, xlsx and quote
parsers alike — so fixing only the docx parser would have left the same loss
standing for every other format.

Measured on the dev corpus 2026-09-02:
  docx  204 / 418 files (49%) with a colliding table, 3,836 cells lost
  xlsx   78 / 120 sampled files (65%) with duplicate header names
"""

from __future__ import annotations

from app.core.table_cells import cells_by_column


def test_a_merged_header_keeps_every_cell():
    cols = ["Requestor Information"] * 4
    row = ["Address Line 1", "601 Gurley St", "Address Line 2", ""]
    out = cells_by_column(cols, row)
    assert len(out) == 4
    assert "601 Gurley St" in out.values()


def test_this_is_what_the_zip_was_doing():
    cols = ["Requestor Information"] * 4
    row = ["Address Line 1", "601 Gurley St", "Address Line 2", ""]
    assert len(dict(zip(cols, row))) == 1, "one survivor out of four"
    assert len(cells_by_column(cols, row)) == 4


def test_well_formed_tables_are_byte_identical():
    # The common case must not change, or every downstream reader keyed on
    # column names breaks at once.
    assert cells_by_column(["Qty", "Item", "Price"], ["2", "Clock", "$305"]) == {
        "Qty": "2", "Item": "Clock", "Price": "$305"
    }


def test_the_first_occurrence_keeps_its_plain_name():
    out = cells_by_column(["Site", "Site"], ["A", "B"])
    assert out["Site"] == "A"
    assert "B" in out.values()


def test_a_row_longer_than_its_header_keeps_its_tail():
    out = cells_by_column(["A"], ["1", "2", "3"])
    assert len(out) == 3


def test_no_columns_falls_back_to_positional():
    assert cells_by_column([], ["a", "b"]) == {"col_0": "a", "col_1": "b"}
    assert cells_by_column(None, ["a"]) == {"col_0": "a"}


def test_blank_column_names_do_not_collapse():
    assert len(cells_by_column(["", "", ""], ["x", "y", "z"])) == 3
