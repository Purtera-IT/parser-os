"""A bullet whose text reads as a photo request stays one bullet (010353).

A Word-exported SOW sets each scope bullet as a 12pt U+F0B7 glyph at one
indent and 9pt text at a hanging indent, the wrap line under the text. The
taller glyph opens a layout break before the wrap line. One bullet asked for
completion photographs; the photo-request branch of the prose splitter took
that line before the bullet rule, so the glyph stayed on the text as a
paragraph and the wrapped tail became its own atom. A list item is its own
unit already: the bullet rule wins, and the tail joins it like its neighbours.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from fontTools.ttLib import TTFont

import app.parsers.orbitbrief_pdf as ob
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

INTRO = (
    "The provider will send a qualified technician to mount one weather station and one battery pack on a "
    "pole at the north yard of the depot. The customer will supply the ladder needed for the work and will "
    "have a contact onsite for the whole visit. The customer asked for the work to be done by the end of May."
)
# (glyph top, [(line top, text)]), the real page's pitch and indents.
BULLETS = [
    (550.8, [(551.5, "Adjust the station angle as asked by the site contact during the walkthrough window.")]),
    (573.5, [(574.2, "Capture clear completion photographs showing the mounted station, battery pack, pole, wiring, and overall"),
             (586.9, "tidy work zone.")]),
    (607.9, [(608.6, "Send the completion report to the site contact, call the site contact before leaving to confirm the readings"),
             (621.2, "and the angle, clear any debris left by the technician, and log any open item through the change process.")]),
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
    import textwrap

    glyph = _glyph_font(tmp)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 300), "PROJECT OVERVIEW", fontfile=SERIF, fontname="ls", fontsize=11)
    for i, ln in enumerate(textwrap.wrap(INTRO, 120)):
        page.insert_text((72, 320 + 12.7 * i), ln, fontfile=SERIF, fontname="ls", fontsize=9)
    page.insert_text((72, 520), "SCOPE OF WORK", fontfile=SERIF, fontname="ls", fontsize=11)
    for gtop, lines in BULLETS:
        page.insert_text((90.1, gtop + 9.6), "", fontfile=glyph, fontname="pb", fontsize=12)
        for top, text in lines:
            page.insert_text((108.1, top + 7.0), text, fontfile=SERIF, fontname="ls", fontsize=9)
    page.insert_text((72, 660), "DELIVERABLES", fontfile=SERIF, fontname="ls", fontsize=11)
    page.insert_text((72, 680), "Completion photographs and the readings are delivered to the customer.",
                     fontfile=SERIF, fontname="ls", fontsize=9)
    out = tmp / "sow.pdf"
    doc.save(str(out))
    return out


@pytest.fixture
def photo_rule_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    # Offline the photo-request rule falls back to a keyword net that misses
    # "photographs"; the embedding rule the live compile runs fires on it.
    monkeypatch.setattr(ob, "_is_photo_request", lambda t: "photographs showing" in (t or ""))


def test_a_photo_request_bullet_keeps_its_wrapped_tail(tmp_path: Path, photo_rule_fires: None) -> None:
    out = OrbitBriefPdfParser().parse_artifact("p", "art", _pdf(tmp_path))
    texts = {a.raw_text: a for a in out.atoms}
    whole = ("Capture clear completion photographs showing the mounted station, battery pack, pole, wiring, "
             "and overall tidy work zone.")
    assert whole in texts, sorted(texts)
    assert texts[whole].source_refs[0].locator.get("block_kind") == "bullet_list"
    assert "tidy work zone." not in texts
    assert not any((t or "").startswith("") for t in texts)
