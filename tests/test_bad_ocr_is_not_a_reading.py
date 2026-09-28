"""An OCR pass over a drawing we already parsed is not a second reading.

Live 010180 carried SP-6 as both .dwg and .pdf. The drawing gave the program
summary, every layer and every coordinate. The PDF gave twenty-six OCR
fragments: "JAN", "ADA RR", "IT", "P: 203.246.1900", "NOTHING BEATS 72 YEARS
OF STABILITY". None of that reaches a deal kit or a SOW.

Demoting them was the first attempt and it was not enough -- a demoted row
still takes a line in the labeling pane. They are withheld now, and one atom
asks the PM for anything the drawing did not give.
"""
from app.core.drawing_pairs import export_rows_to_withhold, withhold_export_rows

DWG = "07.22.26_FEIL ORGANIZATION_7 PENN PLAZA_12 FL_SP-6.dwg"
PDF = "07.21.26_FEIL ORGANIZATION_7 PENN PLAZA_12 FL_SP-6.pdf"


class _Ref:
    def __init__(self, filename):
        self.filename = filename


class _Atom:
    def __init__(self, filename, text, kind=None):
        self.raw_text = self.text = text
        self.source_refs = [_Ref(filename)]
        self.artifact_id = "art:" + filename
        self.review_flags = []
        self.value = {"kind": kind} if kind else {}
        self.atom_type = "site_room_mix"


def test_the_export_rows_are_selected_for_withholding():
    atoms = [
        _Atom(DWG, "106 workstations drawn on the plan"),
        _Atom(DWG, "828 linear feet of new partition"),
        _Atom(PDF, "JAN"),
        _Atom(PDF, "ADA RR"),
        _Atom(PDF, "NOTHING BEATS 72 YEARS OF STABILITY"),
    ]
    by_sheet = export_rows_to_withhold(atoms)
    assert list(by_sheet) == [DWG]
    assert len(by_sheet[DWG]) == 3


def test_the_pm_is_asked_rather_than_left_with_nothing():
    """A sheet whose rows were all withheld would otherwise be absent, and
    absent is indistinguishable from never-looked-at."""
    rows = [_Atom(PDF, "JAN"), _Atom(PDF, "ADA RR")]
    ask = withhold_export_rows(rows, DWG, rows)
    assert "describe it here" in ask
    assert "2 fragments" in ask
    assert "'JAN'" in ask          # it shows what it withheld
    assert DWG in ask              # and which sheet it stands for


def test_an_unread_drawing_withholds_nothing():
    """When the .dwg would not convert, the export is the ONLY reading of the
    sheet -- and withholding it would leave the deal with no account of the
    drawing at all."""
    atoms = [
        _Atom(DWG, "[CAD drawing awaiting conversion]", kind="cad_marker"),
        _Atom(PDF, "JAN"),
    ]
    assert export_rows_to_withhold(atoms) == {}


def test_a_pdf_that_is_not_a_drawing_export_is_untouched():
    atoms = [
        _Atom(DWG, "106 workstations drawn"),
        _Atom(DWG, "828 linear feet"),
        _Atom("Scope of work.pdf", "Cat 6A drops, two per workstation"),
    ]
    assert export_rows_to_withhold(atoms) == {}
