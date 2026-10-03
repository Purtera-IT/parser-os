"""A signed PDF's section runs on across a page break (010087).

An e-signed SOW prints a "Docusign Envelope ID: ..." stamp at the top of every
page, and the page builder sets that stamp apart as the page's own leading
section. The passes that continue the previous page looked only at that first
section, found nothing but the stamp, and did nothing: the bullets that
continue a section on the next page came out with an empty section_path, a
sentence the break cut stayed two atoms, and a sub-item stayed out of its list.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
STAMP = "Docusign Envelope ID: 0A1B2C3D-4E5F-6071-8293-A4B5C6D7E8F9"
CARRIED = [
    "A standard working day is eight (8) hours long.",
    "The client provides parking for the technicians at each site.",
]


def _t(page, x: float, top: float, s: str, *, font: str = SERIF, size: float = 9) -> None:
    page.insert_text((x, top + size * 0.78), s, fontfile=font, fontname="b" if font == BOLD else "s", fontsize=size)


def _bullet(page, top: float, s: str) -> None:
    page.insert_text((90.1, top + 8.3), "•", fontfile=SERIF, fontname="s", fontsize=11)
    _t(page, 108.1, top + 0.5, s)


def _sub(page, top: float, s: str) -> None:
    page.insert_text((126.1, top + 8.8), "o", fontfile=MONO, fontname="m", fontsize=11)
    _t(page, 144.1, top + 0.6, s)


def _band(page) -> None:
    _t(page, 20, 10, STAMP, size=8)
    _t(page, 72.1, 744.0, "WWW.EXAMPLE-VENDOR.COM", font=BOLD, size=12)


def _pdf(tmp: Path) -> Path:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    _band(p)
    _t(p, 72.1, 80, "SCOPE OF WORK", font=BOLD, size=11)
    for k in range(8):
        _bullet(p, 100 + 15 * k, f"The technician completes scope task {k + 1} at each listed site.")
    _t(p, 72.1, 613.0, "ASSUMPTIONS", font=BOLD, size=11)
    _bullet(p, 634.5, "Work is billed by the time of day when it happens and by the location")
    _t(p, 108.1, 650.0, "of the site, unless this SOW says otherwise.")
    _sub(p, 664.4, "Day: base fee for every hour worked on weekdays.")
    _sub(p, 679.4, "Evening: uplift on the base fee for every hour worked after hours.")
    _sub(p, 694.4, "Rest days: uplift on the base fee for every hour worked at local sites,")
    q = doc.new_page(width=612, height=792)
    _band(q)
    _t(q, 144.1, 105.3, "and a double fee at remote sites.")
    _sub(q, 119.6, "Holidays: double fee for every hour.")
    for k, s in enumerate(CARRIED):
        _bullet(q, 140 + 15 * k, s)
    _t(q, 72.1, 200, "PAYMENT TERMS", font=BOLD, size=11)
    _t(q, 72.1, 220, "Invoices fall due thirty days after the invoice date.")
    out = tmp / "sow.pdf"
    doc.save(str(out))
    return out


def _atoms(pdf: Path) -> dict[str, dict]:
    atoms = OrbitBriefPdfParser().parse_artifact("p", "art", pdf).atoms
    return {a.raw_text: a.source_refs[0].locator for a in atoms if a.source_refs}


def test_bullets_after_the_stamp_keep_their_section(tmp_path: Path) -> None:
    got = _atoms(_pdf(tmp_path))
    for s in CARRIED:
        assert got[s]["page"] == 1
        assert got[s]["block_kind"] == "bullet_list"
        assert got[s]["section_path"] and got[s]["section_path"][-1] == "ASSUMPTIONS"
    assert got["Invoices fall due thirty days after the invoice date."]["section_path"][-1] == "PAYMENT TERMS"


def test_sentence_and_sub_list_continue_past_the_stamp(tmp_path: Path) -> None:
    got = _atoms(_pdf(tmp_path))
    joined = "Rest days: uplift on the base fee for every hour worked at local sites, and a double fee at remote sites."
    assert joined in got
    assert "and a double fee at remote sites." not in got
    holidays = got["Holidays: double fee for every hour."]
    assert holidays["bullet_path"] == [0, 3]
    assert holidays["section_path"] == got[joined]["section_path"]
