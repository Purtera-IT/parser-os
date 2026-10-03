"""A PDF sub-list keeps its nesting across a page break (010353, 010003).

A SOW sets a "•" item (marker x 90, text x 108) over "o" sub-items (marker
x 126, text x 144). The last sub-item on the page wraps onto the next page
(its tail line has no marker), a page footer sits between, and the next
page opens with one more "o" sub-item at the same x. Each page nested its
own bullets, so that sub-item started a new list: bullet_path [0], depth 1,
without its parent. It is the parent's fourth child, and the "•" item after
it stays where it was numbered before.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

TIERS = [
    "Day shift: 7:00 AM to 4:00 PM (16:00) site time at the base fee.",
    "Night shift: 4:00 PM (16:00) to 7:00 AM: 40% uplift on the base fee.",
    "Rest days: 12:00 AM Sunday to 7:00 AM Tuesday: 40% uplift on the base fee for local sites,",
]
TAIL = "and 90% uplift on the base fee for remote sites."
LAST = "Public holidays: 90% uplift on the base fee."
NEXT = "The client names one onsite contact before the visit."


def _text(page, x: float, top: float, s: str, *, font: str = SERIF, size: float = 9) -> None:
    page.insert_text((x, top + size * 0.78), s, fontfile=font, fontname="b" if font == BOLD else "s", fontsize=size)


def _sub(page, top: float, s: str) -> None:
    page.insert_text((126.1, top + 8.8), "o", fontfile=MONO, fontname="m", fontsize=11)
    _text(page, 144.1, top + 0.6, s)


def _pdf(tmp: Path, lead: tuple[str, str]) -> Path:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    _text(p, 72.1, 613.0, "TERMS OF SERVICE", font=BOLD, size=11)
    p.insert_text((90.1, 634.5 + 8.8), "•", fontfile=SERIF, fontname="s", fontsize=11)
    _text(p, 108.1, 635.0, lead[0])
    _text(p, 108.1, 650.0, lead[1])
    for top, s in zip((664.4, 679.4, 694.4), TIERS):
        _sub(p, top, s)
    _text(p, 72.1, 744.0, "WWW.EXAMPLE-VENDOR.COM", font=BOLD, size=12)
    q = doc.new_page(width=612, height=792)
    _text(q, 144.1, 105.3, TAIL)
    _sub(q, 119.6, LAST)
    q.insert_text((90.1, 140.0 + 8.8), "•", fontfile=SERIF, fontname="s", fontsize=11)
    _text(q, 108.1, 140.5, NEXT)
    _text(q, 72.1, 170.0, "PAYMENT TERMS", font=BOLD, size=11)
    _text(q, 72.1, 190.0, "Invoices fall due thirty days after the invoice date.")
    _text(q, 72.1, 744.0, "WWW.EXAMPLE-VENDOR.COM", font=BOLD, size=12)
    out = tmp / "sow.pdf"
    doc.save(str(out))
    return out


def _bullets(pdf: Path) -> dict[str, dict]:
    atoms = OrbitBriefPdfParser().parse_artifact("p", "art", pdf).atoms
    return {
        a.raw_text: {**a.source_refs[0].locator, "atom_type": a.atom_type}
        for a in atoms
        if a.source_refs[0].locator.get("block_kind") == "bullet_list"
    }


LEAD = ("Work is done in normal day-shift hours at the base fee and is billed,", "unless this SOW says otherwise.")


def test_sub_item_after_page_break_stays_under_its_parent(tmp_path: Path) -> None:
    got = _bullets(_pdf(tmp_path, LEAD))
    lead = " ".join(LEAD)
    assert got[lead]["bullet_path"] == [0]
    joined = TIERS[2] + " " + TAIL
    assert [got[t]["bullet_path"] for t in (TIERS[0], TIERS[1], joined)] == [[0, 0], [0, 1], [0, 2]]
    last = got[LAST]
    assert last["page"] == got[joined]["page"] + 1
    assert last["bullet_path"] == [0, 3]
    assert last["bullet_depth"] == 2
    # Same parent context as its siblings above the break.
    assert last["lead_in"] == got[TIERS[0]]["lead_in"]
    assert last["section_path"] == got[TIERS[0]]["section_path"]
    assert last["atom_type"] == got[TIERS[0]]["atom_type"]


def test_colon_parent_heads_the_carried_sub_item(tmp_path: Path) -> None:
    lead = ("Work is billed at the base fee with these uplifts by time of the work and the site", "location as follows:")
    got = _bullets(_pdf(tmp_path, lead))
    last = got[LAST]
    assert last["bullet_path"] == [0, 3]
    assert last["section_path"] == got[TIERS[0]]["section_path"]
    assert last["section_path"][-1] == " ".join(lead).rstrip(":")


def test_top_level_item_after_the_carry_keeps_its_number(tmp_path: Path) -> None:
    got = _bullets(_pdf(tmp_path, LEAD))
    nxt = got[NEXT]
    assert nxt["bullet_depth"] == 1
    # Its index on its own page, as before: the carry never renumbers the top level.
    assert nxt["bullet_path"] == [1]
    assert nxt["lead_in"] == got[" ".join(LEAD)]["lead_in"]
