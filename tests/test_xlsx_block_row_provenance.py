"""A locator's `row` must be the WORKSHEET row.

Block detection slices a sheet by blank-row bands and blank-column groups, and
the `row` written into an xlsx_block_row_v1 locator was a running counter over
those blocks -- reading order, not a location. Source replay reads absolute
worksheet rows, so on deal 010215 every such atom cited a row exactly two off
and the Deal Kit's priced service lines could not be verified at all.
"""
import pytest

openpyxl = pytest.importorskip("openpyxl")

from app.parsers.xlsx_blocks import sheet_blocks


class TestSheetBlocksCarryRowOrigin:
    def test_a_table_reports_the_worksheet_row_of_each_data_row(self):
        rows = [
            ["Price", "QTY", "NB Price"],   # sheet row 1 (header)
            ["Technician Dispatch", "", 75],  # 2
            ["Time Clock Installation", "", 125],  # 3
        ]
        table = [b for b in sheet_blocks(rows) if b["kind"] == "table"][0]
        assert table["row_indices"] == [2, 3]

    def test_a_blank_row_does_not_shift_the_numbering(self):
        # The bug: a counter skips blanks, a sheet row does not.
        rows = [
            ["Price", "QTY"],          # 1
            ["Tech Dispatch", 75],     # 2
            [],                        # 3 blank -> band boundary
            ["Task", "Hours"],         # 4
            ["Review install", 1],     # 5
        ]
        tables = [b for b in sheet_blocks(rows) if b["kind"] == "table"]
        assert tables[0]["row_indices"] == [2]
        assert tables[1]["row_indices"] == [5]

    def test_leading_blank_rows_are_counted(self):
        rows = [[], [], ["Price", "QTY"], ["Tech Dispatch", 75]]
        table = [b for b in sheet_blocks(rows) if b["kind"] == "table"][0]
        assert table["row_indices"] == [4]

    def test_indices_line_up_one_for_one_with_rows(self):
        rows = [["Price", "QTY"]] + [[f"Item {i}", i] for i in range(1, 6)]
        table = [b for b in sheet_blocks(rows) if b["kind"] == "table"][0]
        assert len(table["row_indices"]) == len(table["rows"])

    def test_a_column_split_keeps_each_row_origin(self):
        # _col_split rebuilds rows to slice columns; without carrying the origin
        # across that, the block loses its only link back to the sheet.
        rows = [
            ["Price", "QTY", "", "Task", "Hours"],
            ["Tech Dispatch", 75, "", "Review install", 1],
            ["Time Clock", 125, "", "Site arrival", 2],
        ]
        for table in [b for b in sheet_blocks(rows) if b["kind"] == "table"]:
            assert table["row_indices"] == [2, 3], table["header"]


class TestKeyvalBlocksCarryRowOrigin:
    """Key-value boxes need the same treatment as tables.

    Only the table branch was fixed first, which left 19 xlsx_block_keyval_v1
    atoms on deal 010215 still citing a running counter.
    """

    def test_a_keyval_box_reports_each_pair_s_worksheet_row(self):
        rows = [
            ["Overall Deal Kit Summary"],   # 1 (title)
            ["Total Labor Revenue", "1000"],  # 2
            ["Total Labor Cost", "800"],      # 3
        ]
        kv = [b for b in sheet_blocks(rows) if b["kind"] == "keyval"][0]
        assert kv["pair_rows"] == [2, 3]

    def test_pairs_keep_their_two_tuple_shape(self):
        # Every existing consumer unpacks (key, value); the row travels beside
        # them rather than inside them.
        rows = [["Summary"], ["Total Labor Revenue", "1000"], ["Total Labor Cost", "800"]]
        kv = [b for b in sheet_blocks(rows) if b["kind"] == "keyval"][0]
        assert all(len(p) == 2 for p in kv["pairs"])

    def test_a_titleless_box_does_not_raise(self):
        # _synth_keyval_title unpacked two values and raised once pairs carried
        # three. The exception was swallowed upstream, so a sheet produced ZERO
        # block atoms and merely looked empty -- the worst kind of failure.
        rows = [
            ["Expected internal cost target, low", "100"],
            ["Expected internal cost target, high", "200"],
        ]
        blocks = sheet_blocks(rows)  # must not raise
        assert blocks
