"""A numbered list item stays one list atom under its real section (CDW SOW, 010003).

The signed SOW's assumptions list carries on from the previous page. Item 7
("7. After a Change Order that requires an amended PO is executed, ...") read
as a run-on numbered clause, so "After a Change" became a heading and the
section of every block after it until the next real heading. Items 1-6 took
the page's header id ("SOW 198950") as their section, because that all-caps
line read as a heading and hid the heading carried over from the page before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: E402

ITEMS = [
    ["Client will give the crew access to each room during normal office hours."],
    ["Client will provide power at every location where a screen is to be hung."],
    ["Before a Site Survey that confirms the wall type is completed, Vendor will not order any mounting",
     "hardware for the project. If hardware was ordered earlier Vendor may bill for it."],
    ["All parties will agree on an install date before the crew is booked."],
]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _sow(path: Path) -> None:
    doc = fitz.open()
    p0 = doc.new_page(width=612, height=792)
    y = 60
    p0.insert_text((54, y), "PROJECT OVERVIEW", fontsize=14, fontname="hebo")
    y += 24
    for j in range(40):
        p0.insert_text((54, y), f"Vendor will hang and connect the screens in the client office, task {j}.", fontsize=10)
        y += 13.2
    p0.insert_text((54, 700), "SITE ASSUMPTIONS", fontsize=14, fontname="hebo")
    # Next page: the header id, then the list carrying on (marker at x 72, text at x 90).
    p = doc.new_page(width=612, height=792)
    p.insert_text((400, 30), "SOW 554210", fontsize=8)
    top = 63.5
    for n, lines in enumerate(ITEMS, 1):
        p.insert_text((72, top), f"{n}.", fontsize=10)
        for line in lines:
            p.insert_text((90, top), line, fontsize=10)
            top += 11.5
    p.insert_text((54, top + 30), "OUT OF SCOPE", fontsize=14, fontname="hebo")
    p.insert_text((54, top + 50), "Any work not listed in this SOW is out of scope.", fontsize=10)
    doc.save(str(path))
    doc.close()


def _path(a) -> list[str]:
    return list((a.source_refs[0].locator or {}).get("section_path") or [])


def test_numbered_item_is_one_atom_under_its_section(tmp_path):
    pdf = tmp_path / "Signed SOW.pdf"
    _sow(pdf)
    atoms = list(getattr(OrbitBriefPdfParser().parse(pdf), "atoms", []))
    item = [a for a in atoms if "Site Survey that confirms" in a.raw_text]
    assert len(item) == 1, [a.raw_text for a in atoms]
    assert item[0].raw_text.startswith("Before a Site Survey")
    assert "Vendor may bill for it" in item[0].raw_text
    for a in atoms:
        p = _path(a)
        assert "Before a Site" not in p and "SOW 554210" not in p, (a.raw_text, p)
    first = next(a for a in atoms if a.raw_text.startswith("Client will give the crew"))
    assert _path(first)[-1] == "SITE ASSUMPTIONS"
    assert _path(item[0])[-1] == "SITE ASSUMPTIONS"
