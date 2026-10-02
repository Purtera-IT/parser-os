"""A Word "o" sub-bullet set in a monospace face is a list item (010353).

Word sets a second-level list marker as the letter "o" in Courier New
(LiberationMono after a re-export): 11pt at one indent, the 9pt body-face
item text a tab to its right. The text layer reads "o Business Hours: ...",
so the letter was taken for prose and every tier of a premium-rate list
glued into one paragraph. Each "o" item is its own bullet.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
from app.parsers.pdf.layout_text import layout_page_text

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

TIERS = [
    "Weekday Shift: 7:00 AM to 4:00 PM (16:00) local time at the posted rate.",
    "Evening Shift: 4:00 PM to 7:00 AM Monday through Friday: 30% premium over the posted rate.",
    "Sunday Shift: 12:00 AM Sunday to 7:00 AM Monday: 40% premium over the posted rate.",
]


def _pdf(tmp: Path) -> Path:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((72.1, 300 + 8.6), "ASSUMPTIONS", fontfile=SERIF_BOLD, fontname="lsb", fontsize=11)
    p.insert_text((72.1, 330 + 7.0), "Crew time is billed at the posted rate with premiums as set out below.",
                  fontfile=SERIF, fontname="ls", fontsize=9)
    for i, tier in enumerate(TIERS):
        top = 360.0 + 14.3 * i  # the real list's pitch; "o" at 126.1, text at 144.1
        p.insert_text((126.1, top + 8.8), "o", fontfile=MONO, fontname="lm", fontsize=11)
        p.insert_text((144.1, top + 7.6), tier, fontfile=SERIF, fontname="ls", fontsize=9)
    p.insert_text((72.1, 430 + 7.0), "Customer will confirm the onsite contact before the visit.",
                  fontfile=SERIF, fontname="ls", fontsize=9)
    out = tmp / "sow.pdf"
    doc.save(str(out))
    return out


def test_each_o_sub_bullet_is_its_own_bullet_atom(tmp_path: Path) -> None:
    out = OrbitBriefPdfParser().parse_artifact("p", "art", _pdf(tmp_path))
    by_text = {a.raw_text: a for a in out.atoms}
    for tier in TIERS:
        assert tier in by_text, sorted(by_text)
        assert by_text[tier].source_refs[0].locator.get("block_kind") == "bullet_list"
    assert not any((t or "").startswith("o ") for t in by_text), sorted(by_text)


def test_a_letter_o_in_the_body_face_stays_text(tmp_path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((72, 300), "o Canada is sung before the match.", fontfile=SERIF, fontname="ls", fontsize=9)
    with fitz.open(str(_save(doc, tmp_path))) as d:
        assert "o Canada is sung" in (layout_page_text(d[0]) or "")


def _save(doc, tmp: Path) -> Path:
    out = tmp / "plain.pdf"
    doc.save(str(out))
    return out
