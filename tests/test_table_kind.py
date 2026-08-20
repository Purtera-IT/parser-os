"""One question, one place, one correctable answer.

These pin the properties that make judge_table a readout rather than a wrapper:
every format reaches it through the same signature, it can abstain, and its
answer carries enough to disagree with usefully.
"""
from __future__ import annotations

from app.interpret import TableJudgment, judge_table
from app.interpret.table_kind import BOM, RATE_CARD, SITE_ROSTER, UNKNOWN


class TestOneQuestionManyFormats:
    def test_a_pdf_roster_is_a_roster(self):
        # The header Document Intelligence recovered from APS_fiber_Attachment_B,
        # the 159-site table fitz reported as 13 cells.
        j = judge_table(
            columns=["Site No.", "Administrative Site", "Street", "City", "Zip",
                     "Lat, Long"],
            rows=[["1", "Albuquerque GigaPop", "505 Marquette NW", "Alb.",
                   "87102", "35.5, -106.3"]],
        )
        assert j.kind == SITE_ROSTER
        assert j.decided_by

    def test_a_spreadsheet_reaches_the_same_front_door(self):
        j = judge_table(sheet_name="SELL RATES - Do not Edit",
                        rows=[["Role", "Rate"], ["Tech", "95"]])
        assert j.kind == RATE_CARD
        # Same signature, same return type as the PDF path above -- which is
        # the whole point: one thing to correct, not two.
        assert isinstance(j, TableJudgment)

    def test_a_bom_is_not_a_roster(self):
        j = judge_table(
            columns=["Part Number", "Description", "Qty", "Unit Price"],
            rows=[["77-H135", "Wiring", "192", "$4.00"]],
        )
        assert j.kind == BOM


class TestAbstention:
    def test_no_signal_abstains_rather_than_guessing(self):
        j = judge_table(columns=["a", "b"], rows=[["1", "2"]])
        assert j.kind == UNKNOWN
        assert j.abstained is True

    def test_abstaining_is_distinct_from_deciding_unknown(self):
        """A component that cannot say "I don't know" cannot be promoted."""
        confident_unknown = TableJudgment(kind=UNKNOWN, confidence=0.9)
        assert confident_unknown.abstained is False
        assert TableJudgment().abstained is True

    def test_empty_input_does_not_raise(self):
        assert judge_table().abstained is True
        assert judge_table(columns=[], rows=[]).abstained is True


class TestCorrectability:
    def test_the_answer_carries_a_correction_target(self):
        j = judge_table(
            columns=["Site No.", "Administrative Site", "Street", "City"],
            rows=[["1", "GigaPop", "505 Marquette NW", "Alb."]],
        )
        t = j.as_correction_target()
        assert t["relation"] == "table_kind"
        assert t["label"] == SITE_ROSTER
        # Records WHICH implementation decided, so a correction knows what it
        # is correcting and a later head can be compared against it.
        assert t["provenance"]["decided_by"]

    def test_the_target_shape_does_not_depend_on_the_source_format(self):
        pdf = judge_table(
            columns=["Site No.", "Administrative Site", "Street"],
            rows=[["1", "GigaPop", "505 Marquette NW"]],
        ).as_correction_target()
        sheet = judge_table(sheet_name="Sites",
                            rows=[["Site", "Addr"], ["A", "1 St"]]).as_correction_target()
        assert set(pdf) == set(sheet)
        assert pdf["relation"] == sheet["relation"] == "table_kind"
