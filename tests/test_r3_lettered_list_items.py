"""A typed lettered list item is a list item, however short (010003).

Deal 010003's SOW lists the PM's duties "a." ... "f." as typed letters, not
Word numbering. "a." to "e." have five words or more and passed the prose
gate; "f. Complete billing tasks" has four words and no digit, so it fell to
the gate and came out as chatter -- the one duty missing from the scope. A
typed enumerator is the same structural fact as w:numPr: every item fails
open. The clause under "Invoicing Procedures" stays its own atom too.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document

from app.parsers.docx_parser import DocxParser

ITEMS = [
    "a. Conduct a remote kickoff meeting with Customer",
    "b. Develop schedule for installation activities and share with Customer",
    "c. Provide weekly status reports",
    "d. Manage the escalation process",
    "e. Track the order",
    "f. Complete billing tasks",
    "(g) Close project",
    "8) Archive files",
]
INVOICE = ("Timesheets for the prior week must be submitted by Monday at 8 a.m. Eastern Time; "
           "CDW will invoice the Customer monthly.")


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _atoms(tmp_path: Path):
    d = Document()
    d.add_heading("Project Management", 1)
    d.add_paragraph("CDW will provide project management for the duration of the project. The CDW PM will:")
    for it in ITEMS:
        d.add_paragraph(it)
    d.add_heading("Invoicing Procedures", 1)
    d.add_paragraph(INVOICE)
    d.add_paragraph("e.g. weekly")
    path = tmp_path / "SOW.docx"
    d.save(path)
    out = DocxParser().parse_artifact("p", "a", path)
    return out if isinstance(out, list) else out.atoms


def test_every_typed_list_item_is_content_not_chatter(tmp_path: Path) -> None:
    by = {a.raw_text: a for a in _atoms(tmp_path)}
    for it in ITEMS:
        a = by.get(it)
        assert a is not None, (it, list(by))
        assert "chatter" not in a.review_flags, (it, a.review_flags, a.value)
        assert a.atom_type.value != "deal_metadata", it
    inv = by.get(INVOICE)
    assert inv is not None and "chatter" not in inv.review_flags
    assert inv.source_refs[0].locator["section_path"][-1] == "Invoicing Procedures"


def test_an_abbreviation_is_not_an_enumerator(tmp_path: Path) -> None:
    by = {a.raw_text: a for a in _atoms(tmp_path)}
    assert "chatter" in by["e.g. weekly"].review_flags
