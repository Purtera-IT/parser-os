"""A PDF list goes on across a page break at its own level (010003).

A SOW's scope sets a colon lead-in ("The vendor will do the following:")
over a run of "•" items (marker x 72, text x 90, 18.2pt apart). The page
ends inside that list, a two-line 8pt footer sits under it, and the next
page opens with one more "•" item at the same x, nothing above it. That
item started a new list: bullet_path [0], and its section_path lost the
lead-in subsection the list sat under on the page before. It is the next
sibling of the last item above the break, under the same section.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"

LEAD = "The vendor will do the following:"
ITEMS = [
    "Send two field engineers to the store",
    "Unpack and check the six wall clocks",
    "Fix each clock to the bracket on the wall",
    "Set each clock to the local time zone",
    "Test the battery backup on every clock",
    "Label each clock with its asset tag",
    "Record each serial number on the job sheet",
    "Show the store manager how to set the alarm",
    "Remove the packing from the store",
]
CARRIED = "Visits happen during normal store opening hours"
PROSE = "This work order sets out the work the vendor will carry out for the client at the store named"


def _text(page, x: float, top: float, s: str, *, font: str = SERIF, size: float = 10.0) -> None:
    page.insert_text((x, top + 0.891 * size), s, fontfile=font,
                     fontname="b" if font == BOLD else "s", fontsize=size)


def _bullet(page, top: float, s: str) -> None:
    _text(page, 72.0, top + 0.1, "•")
    _text(page, 90.0, top, s)


def _footer(page, n: int) -> None:
    _text(page, 57.3, 727.9, "Internal and Confidential", size=8.0)
    _text(page, 294.2, 727.9, f"Page {n}", size=8.0)
    _text(page, 478.0, 727.9, "Example Vendor Inc", size=8.0)
    _text(page, 57.6, 743.1, "WO 4471", size=8.0)


def _pdf(tmp: Path) -> Path:
    doc = fitz.open()
    p = doc.new_page(width=612.28, height=790.87)
    _text(p, 54.0, 60.0, "WORK ORDER", font=BOLD, size=14)
    for k in range(14):
        _text(p, 54.0, 85.0 + 13.8 * k, PROSE if k < 13 else "in the order form.")
    _text(p, 54.0, 470.0, "SCOPE OF WORK", font=BOLD, size=14)
    _text(p, 54.0, 497.6, "Subject to the other terms of this order, the vendor will do the following work:")
    _text(p, 54.0, 516.8, LEAD)
    for k, s in enumerate(ITEMS):
        _bullet(p, 536.7 + 18.2 * k, s)
    _footer(p, 1)
    q = doc.new_page(width=612.28, height=790.87)
    _bullet(q, 56.2, CARRIED)
    _text(q, 54.0, 90.0, "ASSUMPTIONS", font=BOLD, size=14)
    _text(q, 54.0, 115.0, "The client gives the vendor access to the store during the visit.")
    _footer(q, 2)
    out = tmp / "wo.pdf"
    doc.save(str(out))
    return out


def test_item_after_page_break_is_the_next_sibling_under_the_same_lead(tmp_path: Path) -> None:
    atoms = OrbitBriefPdfParser().parse_artifact("p", "art", _pdf(tmp_path)).atoms
    got = {
        a.raw_text: a.source_refs[0].locator
        for a in atoms
        if a.source_refs[0].locator.get("block_kind") == "bullet_list"
    }
    last, carried = got[ITEMS[-1]], got[CARRIED]
    assert [got[s]["bullet_path"] for s in ITEMS] == [[k] for k in range(len(ITEMS))]
    assert carried["page"] == last["page"] + 1
    assert carried["bullet_path"] == [len(ITEMS)]
    assert carried["bullet_depth"] == 1
    assert last["section_path"][-1] == LEAD.rstrip(":")
    assert carried["section_path"] == last["section_path"]
    assert carried["lead_in"] == last["lead_in"]
