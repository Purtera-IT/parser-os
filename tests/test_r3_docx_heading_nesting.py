"""R3 (010087): sibling headings replace each other; heading atoms carry
their own section; the document title never nests under another heading.

010087's SOW put PURTERA RESPONSIBILITIES and CUSTOMER RESPONSIBILITIES under
OUT OF SCOPE (the heading stack only popped on a strictly higher level, and an
all-caps bold line / colon list-intro was always one level deeper than the
style heading before it), so PMO duties were typed as exclusions.
"""

from __future__ import annotations

import pytest
from docx import Document

from app.parsers.docx_parser import DocxParser

PMO = "Develop schedule and coordinate resources with the customer team"
BILL = "Complete billing tasks for the project at closeout"
CUST = "Provide site access during normal business hours"
EXCL = "Cabling of new drops is not included in this project"


def _bold_line(doc, text):
    p = doc.add_paragraph()
    p.add_run(text).bold = True
    return p


def _atoms(path):
    out = DocxParser().parse_artifact("p", "a", path)
    return out if isinstance(out, list) else out.atoms


def _by_text(atoms, needle):
    hits = [a for a in atoms if needle.lower() in (a.raw_text or "").lower()]
    assert hits, f"{needle!r} not emitted: {[a.raw_text for a in atoms]}"
    return hits


def _path(a):
    return list(a.source_refs[0].locator.get("section_path") or [])


def _type(a):
    return getattr(a.atom_type, "value", a.atom_type)


def _build(path, variant: str) -> None:
    doc = Document()
    doc.add_heading("OUT OF SCOPE", level=1)
    doc.add_paragraph(EXCL, style="List Bullet")
    if variant == "bold_caps":
        _bold_line(doc, "PURTERA RESPONSIBILITIES")
        doc.add_paragraph(PMO, style="List Bullet")
        doc.add_paragraph(BILL, style="List Bullet")
        _bold_line(doc, "CUSTOMER RESPONSIBILITIES")
        doc.add_paragraph(CUST, style="List Bullet")
    elif variant == "caps_intro":
        doc.add_paragraph("PURTERA RESPONSIBILITIES:")
        doc.add_paragraph(PMO, style="List Bullet")
        doc.add_paragraph(BILL, style="List Bullet")
        doc.add_paragraph("CUSTOMER RESPONSIBILITIES:")
        doc.add_paragraph(CUST, style="List Bullet")
    elif variant == "caps_plain":
        doc.add_paragraph("PURTERA RESPONSIBILITIES")
        doc.add_paragraph(PMO + ".")
        doc.add_paragraph(BILL + ".")
        doc.add_paragraph("CUSTOMER RESPONSIBILITIES")
        doc.add_paragraph(CUST + ".")
    elif variant == "intro_siblings":
        # Bold caps exclusions heading on Normal, then colon intros.
        doc.add_paragraph("Purtera Responsibilities:")
        doc.add_paragraph(PMO, style="List Bullet")
        doc.add_paragraph(BILL, style="List Bullet")
        doc.add_paragraph("Customer Responsibilities:")
        doc.add_paragraph(CUST, style="List Bullet")
    doc.save(path)


@pytest.mark.parametrize("variant", ["bold_caps", "caps_intro", "caps_plain"])
def test_sibling_caps_headings_do_not_nest_under_out_of_scope(tmp_path, variant) -> None:
    path = tmp_path / "SOW.docx"
    _build(path, variant)
    atoms = _atoms(path)
    for needle, section in ((PMO, "PURTERA RESPONSIBILITIES"), (BILL, "PURTERA RESPONSIBILITIES"),
                            (CUST, "CUSTOMER RESPONSIBILITIES")):
        for a in _by_text(atoms, needle[:30]):
            p = _path(a)
            assert p and p[-1] == section, (variant, needle, p)
            assert "OUT OF SCOPE" not in p, (variant, needle, p)
            assert _type(a) != "exclusion", (variant, needle, _type(a))
    # The real exclusions section still types its own content.
    ex = _by_text(atoms, EXCL[:30])
    assert any(_type(a) == "exclusion" for a in ex)
    assert all(_path(a) == ["OUT OF SCOPE"] for a in ex)


def test_title_case_colon_intros_are_siblings(tmp_path) -> None:
    path = tmp_path / "SOW.docx"
    _build(path, "intro_siblings")
    atoms = _atoms(path)
    for a in _by_text(atoms, CUST[:30]):
        assert "Purtera Responsibilities" not in _path(a), _path(a)


def test_heading_atoms_carry_their_own_section(tmp_path) -> None:
    path = tmp_path / "SOW.docx"
    doc = Document()
    doc.add_heading("INTRODUCTION", level=1)
    doc.add_paragraph("PurTera will provide structured cabling services to the customer.")
    doc.add_heading("DELIVERABLES", level=1)
    doc.add_heading("Network Design", level=2)
    doc.add_paragraph("PurTera will deliver an as-built network diagram to the customer.")
    doc.add_heading("OUT OF SCOPE", level=1)
    doc.add_paragraph(EXCL, style="List Bullet")
    doc.save(path)
    atoms = _atoms(path)
    heads = {a.raw_text: a for a in atoms if (a.value or {}).get("structure")}
    assert _path(heads["INTRODUCTION"]) == ["INTRODUCTION"]
    assert _path(heads["DELIVERABLES"]) == ["DELIVERABLES"]
    assert _path(heads["Network Design"]) == ["DELIVERABLES", "Network Design"]
    assert _path(heads["OUT OF SCOPE"]) == ["OUT OF SCOPE"]
    # Still the reject-able chatter atom #268 made it, not an exclusion.
    assert _type(heads["OUT OF SCOPE"]) == "deal_metadata"
    assert heads["OUT OF SCOPE"].value.get("rejected_by") == "section_heading"


def test_document_title_does_not_nest_under_earlier_heading(tmp_path) -> None:
    path = tmp_path / "SOW.docx"
    doc = Document()
    doc.add_heading("SOW LOCATION", level=1)
    t = doc.add_table(rows=1, cols=2)
    t.cell(0, 0).text = "Site"
    t.cell(0, 1).text = "601 Gurley St, Marion SC 29571"
    _bold_line(doc, "STATEMENT OF WORK (SOW)")
    doc.add_heading("INTRODUCTION", level=1)
    doc.add_paragraph("PurTera will provide structured cabling services to the customer.")
    doc.save(path)
    atoms = _atoms(path)
    title = _by_text(atoms, "STATEMENT OF WORK (SOW)")[0]
    assert "SOW LOCATION" not in _path(title), _path(title)
    assert _path(title) == ["STATEMENT OF WORK (SOW)"]
    for a in _by_text(atoms, "structured cabling services"):
        assert "SOW LOCATION" not in _path(a), _path(a)


def test_party_responsibilities_subheading_ends_exclusion_typing(tmp_path) -> None:
    """Even where the author's own outline nests it (Heading 2 under Heading
    1), a "<party> Responsibilities" section is not part of OUT OF SCOPE."""
    path = tmp_path / "SOW.docx"
    doc = Document()
    doc.add_heading("Out of Scope", level=1)
    doc.add_paragraph(EXCL, style="List Bullet")
    doc.add_heading("PurTera Responsibilities", level=2)
    doc.add_paragraph(PMO, style="List Bullet")
    doc.save(path)
    atoms = _atoms(path)
    for a in _by_text(atoms, PMO[:30]):
        assert _type(a) != "exclusion", (_path(a), _type(a))
