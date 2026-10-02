"""A fee sentence that frames the next paragraph keeps its own atom (CDW SOW, 010003).

The signed SOW's SERVICES FEES section opens with a one-line paragraph that
states the fixed fee, then a paragraph about how it is invoiced. The fee line
read as a framing lead-in, so it was lifted onto the next paragraph as lead_in
and never emitted: the dollar figure had no atom. The docx parser already
keeps a lead-in that states a figure as content; the PDF path now does too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: E402

FEE = "Installation fees for this order are fixed and include $2,450.00 for all labor outlined below."
NEXT = [
    "Each invoice will equal the share of the fee assigned to the milestones completed in that period, as listed",
    "in the schedule attached to this order, and no invoice will be issued for a milestone the Client has not yet",
    "accepted in writing.",
]
LATER = [
    "Vendor will send invoices to the address on the purchase order and will reference the order number on",
    "every invoice it submits under this agreement, together with the completion sign-off for each milestone",
    "billed on that invoice.",
]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _sow(path: Path) -> None:
    # Geometry mirrors the real page: an underlined all-caps heading, the fee on
    # one line 7pt below it, then wrapped paragraphs with a 9pt paragraph gap.
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((54, 268), "INSTALLATION FEES", fontsize=14, fontname="hebo")
    page.draw_line((54, 270), (190, 270))
    y = 285.6
    page.insert_text((54, y), FEE, fontsize=10)
    y += 19.3
    for block in (NEXT, LATER):
        for line in block:
            page.insert_text((54, y), line, fontsize=10)
            y += 13.2
        y += 6.1
    doc.save(str(path))
    doc.close()


def test_fee_lead_in_keeps_its_atom(tmp_path):
    pdf = tmp_path / "Signed SOW.pdf"
    _sow(pdf)
    atoms = list(getattr(OrbitBriefPdfParser().parse(pdf), "atoms", []))
    texts = [a.raw_text for a in atoms]
    assert FEE in texts, texts
    nxt = next(a for a in atoms if a.raw_text.startswith("Each invoice will equal"))
    # It still frames the paragraph it introduces.
    assert FEE in ((nxt.source_refs[0].locator or {}).get("lead_in") or [])
    assert "INSTALLATION FEES" not in texts
