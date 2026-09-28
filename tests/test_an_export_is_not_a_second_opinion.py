"""A sheet that arrives as both DWG and PDF is one sheet, read twice.

Live 010180: SP-6 arrived both ways. The DWG gave 62 atoms including the
program summary and every coordinate; the PDF gave twelve room words off an
OCR pass. Both sat in the workspace looking equally authoritative.
"""
from app.core.drawing_pairs import (
    SUPERSEDED_FLAG, demote_export_duplicates, same_sheet, sheet_code,
)

DWG = "07.22.26_FEIL ORGANIZATION_7 PENN PLAZA_12 FL_SP-6.dwg"
PDF = "FEIL ORGANIZATION_7 PENN PLAZA_12 FL_SP-6.pdf"


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


def test_the_sheet_code_is_read_from_the_end():
    """'7 PENN PLAZA' has the same shape as a sheet code. The code sits last."""
    assert sheet_code(DWG) == "SP6"
    assert sheet_code(PDF) == "SP6"
    assert sheet_code("A-101 First Floor.dwg") == "A101"


def test_one_sheet_in_two_formats_pairs():
    assert same_sheet(PDF, DWG) is True


def test_two_sheets_of_one_set_do_not_pair():
    """SP-6 and SP-7 are different drawings of the same building. Pairing them
    would delete a whole sheet's reading."""
    assert same_sheet("FEIL_7 PENN PLAZA_12 FL_SP-7.pdf", DWG) is False


def test_another_deals_sp6_does_not_pair():
    """Every architect issues an SP-6. The code alone is not identity."""
    assert same_sheet("MBRANY_GREAT NECK_SP-6.pdf", DWG) is False


def test_the_export_is_demoted_when_the_drawing_was_read():
    atoms = [
        _Atom(DWG, "5'-0\" WORKSTATIONS 106"),
        _Atom(DWG, "EXECUTIVE OFFICE 2"),
        _Atom(PDF, "WOMEN'S RESTROOM"),
        _Atom(PDF, "JAN"),
    ]
    demote_export_duplicates(atoms)
    assert [a.review_flags for a in atoms] == [[], [], [SUPERSEDED_FLAG], [SUPERSEDED_FLAG]]
    # The words are still there. Only the claim is withdrawn.
    assert atoms[2].raw_text == "WOMEN'S RESTROOM"
    assert atoms[2].atom_type == "deal_metadata"


def test_nothing_is_deleted():
    atoms = [_Atom(DWG, "IT CLOSET 1"), _Atom(DWG, "PANTRY 1"), _Atom(PDF, "JAN")]
    assert len(demote_export_duplicates(atoms)) == 3


def test_an_unread_drawing_supersedes_nothing():
    """The DWG that produced only its 'awaiting conversion' marker has not been
    read. Then the PDF is the ONLY account of the sheet, and demoting it would
    leave the deal with no reading of the drawing at all."""
    atoms = [
        _Atom(DWG, "[CAD drawing awaiting conversion]", kind="cad_marker"),
        _Atom(PDF, "WOMEN'S RESTROOM"),
    ]
    demote_export_duplicates(atoms)
    assert atoms[1].review_flags == []


def test_a_pdf_with_no_drawing_is_untouched():
    atoms = [_Atom("Scope of work.pdf", "Cat 6A drops, two per workstation")]
    demote_export_duplicates(atoms)
    assert atoms[0].review_flags == []
