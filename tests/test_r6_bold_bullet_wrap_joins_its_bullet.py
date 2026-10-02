"""A bullet's wrap line, set apart by a layout gap, joins that bullet (010353).

A Word-exported SOW sets each bullet as an 11pt U+F0B7 glyph at one indent and
bold 9pt text at a hanging indent, the wrap line under the text at the same
indent. The text layer opens a blank line before the wrap. The blank rule let
the wrap through as a tail, but the bullet-continuation rule then measured the
wrap against that blank line, so the wrap became its own paragraph; two bullets
wrapping onto the same words gave two identical paragraphs, and dedup dropped
one. The wrap is measured against the bullet's line and joins it.
"""
from __future__ import annotations

from pathlib import Path

import fitz
from fontTools.ttLib import TTFont

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

TAIL = "Plan changes listed above."
# (glyph top, first line, wrap top) at the real page's pitch and indents.
BULLETS = [
    (359.8, "Moving the visit inside one working day before it is due - full fee for the visit crew, travel and/or", 375.3),
    (389.8, "Moving the visit inside two working days before it is due - half the fee for the visit crew, travel and/or", 405.3),
]


def _glyph_font(tmp: Path) -> str:
    font = TTFont(DEJAVU)
    for table in font["cmap"].tables:
        if table.isUnicode():
            table.cmap[0xF0B7] = table.cmap[0x2022]
    out = tmp / "pua_bullet.ttf"
    font.save(str(out))
    return str(out)


def _pdf(tmp: Path) -> Path:
    glyph = _glyph_font(tmp)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 300), "ASSUMPTIONS", fontfile=BOLD, fontname="lb", fontsize=11)
    for i, ln in enumerate([
        "The customer will give the crew access to the yard in normal working hours for the whole visit and the",
        "crew will check in with the yard contact when they arrive. Visits run to the agreed plan set out below.",
    ]):
        page.insert_text((72, 320 + 12.7 * i), ln, fontfile=SERIF, fontname="ls", fontsize=9)
    for gtop, first, wtop in BULLETS:
        page.insert_text((90.1, gtop + 9.6), "", fontfile=glyph, fontname="pb", fontsize=11)
        page.insert_text((108.1, gtop + 0.5 + 7.0), first, fontfile=BOLD, fontname="lb", fontsize=9)
        page.insert_text((108.1, wtop + 7.0), TAIL, fontfile=BOLD, fontname="lb", fontsize=9)
    page.insert_text((72, 440), "PRICING & PAYMENT TERMS", fontfile=BOLD, fontname="lb", fontsize=11)
    page.insert_text((72, 460), "The fee is fixed for the visits listed in this statement of work.",
                     fontfile=SERIF, fontname="ls", fontsize=9)
    out = tmp / "sow.pdf"
    doc.save(str(out))
    return out


def test_each_bullet_keeps_its_own_wrap_line(tmp_path: Path) -> None:
    out = OrbitBriefPdfParser().parse_artifact("p", "art", _pdf(tmp_path))
    texts = [a.raw_text for a in out.atoms]
    assert TAIL not in texts, texts
    for _, first, _ in BULLETS:
        whole = f"{first} {TAIL}"
        hits = [a for a in out.atoms if a.raw_text == whole]
        assert len(hits) == 1, texts
        assert hits[0].source_refs[0].locator.get("block_kind") == "bullet_list"
