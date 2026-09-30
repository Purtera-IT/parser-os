"""A sign-off must not discard the table printed under it.

`in_signature` latches on `_SIGNOFF_RE` and never unlatches, so everything
after a sign-off is dropped. Mail clients put the signature ABOVE quoted
content and above tables pasted beneath it, so "after the sign-off" is not the
same as "part of the signature".

Live 010334 (deal d883788b): "Best regards" sits at line 6 of the extracted
body and a switch BOM starts at line 9. All 23 rows of it -- every site,
closet, model and quantity -- were discarded as sign-off chrome. The parser
emitted 5 atoms from that mail and none carried a part number.

The rows reached NEITHER the atoms NOR the suppressed ledger, so no
content-loss audit could see them go: the ledger records what a stage dropped,
and nothing recorded this.

The fix is narrow on purpose. Reconstructing the grid per site is NOT
attempted: the flattened rows are ragged (3 to 9 cells, headers and site names
appearing mid-row where cells were merged), and any algorithm assigning a
switch to a site from that would be confidently wrong some of the time. The row
is kept verbatim, which is a statement a person can act on, and nothing is
invented.
"""

from __future__ import annotations

from app.parsers.email_parser import _is_data_table_row


class TestWhatCountsAsATable:
    def test_a_bom_row_is_a_table_row(self):
        assert _is_data_table_row("MDF | C9200L-48P-4X-E | 7")
        assert _is_data_table_row("NFN235 | C9200L-24P-4X-E | 1 | CW9172I-CFG | 3")

    def test_the_header_row_counts_too(self):
        # It carries no values and is still part of the table: it is what makes
        # the rows under it readable.
        assert _is_data_table_row("Closet | Model | Amount")
        assert _is_data_table_row("ALBANY | Boise | QRS")

    def test_a_signature_laid_out_in_a_table_is_still_a_signature(self):
        # Mail clients lay contact blocks out in <table>, and a phone number is
        # digits -- so "has a number in a cell" alone would re-open the latch
        # on exactly the lines it exists to close.
        assert not _is_data_table_row("Mobile: 832-560-1300 | Email: hanhle@cdw.com")
        assert not _is_data_table_row("Cell: 555-1234 | Office: 555-9999")
        assert not _is_data_table_row("Hanh Le | Professional Services Manager | hanhle@cdw.com")

    def test_an_address_cell_is_not_data(self):
        assert not _is_data_table_row("trent | t@purtera-it.com")

    def test_prose_and_stubs_are_not_tables(self):
        assert not _is_data_table_row("Below is the site list for TEL.")
        assert not _is_data_table_row("one | two")
        assert not _is_data_table_row("")
        assert not _is_data_table_row("   ")


class TestTheTableSurvivesTheSignOff:
    """End to end through EmailParser, on the shape that lost the BOM."""

    def _mail(self, tmp_path):
        body = "\r\n".join([
            "<html><body>",
            "<p>Trent,</p>",
            "<p>Below is the site list for TEL.</p>",
            "<p>Best regards</p>",
            "<p>Hanh Le<br>Professional Services Manager<br>"
            "Mobile: 832-560-1300 &nbsp;|&nbsp; Email: hanhle@cdw.com</p>",
            "<table>",
            "<tr><td>ALBANY</td><td>Boise</td></tr>",
            "<tr><td>Closet</td><td>Model</td><td>Amount</td></tr>",
            "<tr><td>MDF</td><td>C9200L-48P-4X-E</td><td>7</td></tr>",
            "<tr><td>SEMLAB</td><td>C9200L-24P-4X-A</td><td>2</td></tr>",
            "</table>",
            "</body></html>",
        ])
        raw = (
            "From: Hanh Le <hanhle@cdw.com>\r\n"
            "To: t@purtera-it.com\r\n"
            "Subject: TEL site list\r\n"
            "Date: Wed, 16 Sep 2026 12:28:00 -0500\r\n"
            "MIME-Version: 1.0\r\n"
            'Content-Type: text/html; charset="utf-8"\r\n'
            "\r\n" + body
        )
        p = tmp_path / "010334-hs-email-bom.eml"
        p.write_text(raw, encoding="utf-8")
        return p

    def _atom_texts(self, path):
        from app.parsers.email_parser import EmailParser

        res = EmailParser().parse(path)
        atoms = getattr(res, "atoms", res)
        return [str(getattr(a, "raw_text", "") or getattr(a, "text", "") or "") for a in atoms]

    def test_the_part_numbers_reach_the_atoms(self, tmp_path):
        texts = self._atom_texts(self._mail(tmp_path))
        joined = "\n".join(texts)
        assert "C9200L-48P-4X-E" in joined, "the BOM was swallowed by the sign-off again"
        assert "C9200L-24P-4X-A" in joined

    def test_the_quantity_comes_with_it(self, tmp_path):
        # A model number without its count is half a fact. The row is kept
        # whole precisely so the two cannot be separated.
        texts = self._atom_texts(self._mail(tmp_path))
        row = next((t for t in texts if "C9200L-48P-4X-E" in t), "")
        assert "7" in row, f"quantity lost from the row: {row!r}"

    def test_the_prose_before_the_sign_off_is_untouched(self, tmp_path):
        texts = self._atom_texts(self._mail(tmp_path))
        assert any("site list for TEL" in t for t in texts)

    def test_the_signature_contact_line_is_still_dropped(self, tmp_path):
        # The latch must keep doing its job; only a data table re-opens it.
        texts = self._atom_texts(self._mail(tmp_path))
        assert not any(t.strip().startswith("Mobile:") for t in texts)
