"""The first paragraph under a SOW heading keeps its atom (CDW template, 010003).

A signed SOW's PAYMENT TERMS clause opened with a multi-line paragraph that
mentions "... field notes per date ...". The meeting-summary repair read the
word "notes" as a trailing "Notes" header with no bullets under it and dropped
the whole paragraph as header-only chrome, so only the second paragraph under
the heading had an atom. A meeting word inside a sentence is not a header.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.orbitbrief_pdf import (  # noqa: E402
    OrbitBriefPdfParser,
    _split_glued_meeting_summary_paragraph,
)

FIRST = [
    "Every Friday by noon, Vendor will enter the hours worked during the current week into the",
    "Client's scheduling portal, as instructed by Client, together with signed work orders that include field notes per",
    "visit, and complete parts receipts, and at least once per quarter, Vendor shall bill",
    "Client for all charges incurred since the last bill, with appropriate backup attached.",
]
SECOND = "Paying a bill does not mean that Client accepts the work delivered under this agreement."


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _sow(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 60
    page.insert_text((36, y), "BILLING TERMS", fontsize=8.5, fontname="hebo")
    y += 11
    for line in FIRST:
        page.insert_text((36, y), line, fontsize=8.5)
        y += 11
    y += 6
    page.insert_text((36, y), SECOND, fontsize=8.5)
    doc.save(str(path))
    doc.close()


def test_first_paragraph_after_heading_is_an_atom(tmp_path):
    pdf = tmp_path / "Signed SOW.pdf"
    _sow(pdf)
    out = OrbitBriefPdfParser().parse(pdf)
    atoms = list(getattr(out, "atoms", out))
    texts = [a.raw_text for a in atoms]
    first = [a for a in atoms if a.raw_text.startswith("Every Friday by noon")]
    assert first, texts
    assert "field notes per visit" in first[0].raw_text
    assert (first[0].source_refs[0].locator or {}).get("section_path")[-1] == "BILLING TERMS"
    assert any(t.startswith("Paying a bill") for t in texts), texts
    # The heading stays structure, never its own atom.
    assert "BILLING TERMS" not in texts


def test_meeting_word_inside_a_sentence_is_not_a_trailing_header():
    prose = " ".join(FIRST)
    assert _split_glued_meeting_summary_paragraph(prose) is None
    # A real trailing header with nothing under it still carries.
    blocks, trailing = _split_glued_meeting_summary_paragraph(
        "Action Items I Dana to send the floor plan I Lee to confirm access Notes"
    )
    assert trailing == "Notes" and blocks
