"""A sentence cut by a page break joins its own line, not the page footer.

A Word-exported SOW prints a bold 12pt website footer in the bottom band of
every page. A rate bullet ran to the foot of one page, stopping on a comma,
and finished on the next page with a lowercase line at the text indent. The
footer was the last block on the page, so the cross-page join glued the
continuation onto the footer ("<footer> and 100% increase ...") with the
previous page's locator, and the bullet stayed cut. The footer is skipped:
the continuation joins the paragraph or bullet item it finishes.
"""
from __future__ import annotations

from pathlib import Path

import fitz
from fontTools.ttLib import TTFont

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FOOTER = "WWW.EXAMPLE-PROVIDER.COM"


def _glyph_font(tmp: Path) -> str:
    font = TTFont(DEJAVU)
    for table in font["cmap"].tables:
        if table.isUnicode():
            table.cmap[0xF0B7] = table.cmap[0x2022]
    out = tmp / "pua_bullet.ttf"
    font.save(str(out))
    return str(out)


def _text(page, x: float, top: float, s: str, *, size: float = 9, bold: bool = False) -> None:
    page.insert_text((x, top + 0.78 * size), s, fontfile=SERIF_BOLD if bold else SERIF,
                     fontname="lsb" if bold else "ls", fontsize=size)


def _footer(page) -> None:
    # Bottom band: top 744, bold 12pt, at the left margin.
    _text(page, 72.1, 744.0, FOOTER, size=12, bold=True)


def _pdf(tmp: Path, *, as_bullets: bool) -> Path:
    glyph = _glyph_font(tmp)
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    _text(p, 72.1, 600, "ASSUMPTIONS", size=11, bold=True)
    rows = [
        (650.0, "Crew time is billed at the posted rate during normal working hours on weekdays."),
        (665.0, "Night work is billed at a premium over the posted rate on any weekday."),
        (695.0, "Sunday work is billed at a 40% premium over the posted rate for local crews,"),
    ]
    for top, s in rows:
        if as_bullets:
            p.insert_text((90.1, top - 0.5 + 8.8), "", fontfile=glyph, fontname="pb", fontsize=11)
            _text(p, 108.1, top, s)
        else:
            _text(p, 72.1, top, s)
    _footer(p)
    p = doc.new_page(width=612, height=792)
    _text(p, 108.1 if as_bullets else 72.1, 105.3, "and a 90% premium over the posted rate for travelling crews.")
    _text(p, 72.1, 140, "PAYMENT TERMS", size=11, bold=True)
    _text(p, 72.1, 160, "Invoices are payable within thirty days of the invoice date.")
    _footer(p)
    out = tmp / "sow.pdf"
    doc.save(str(out))
    return out


WHOLE = ("Sunday work is billed at a 40% premium over the posted rate for local crews, "
         "and a 90% premium over the posted rate for travelling crews.")


def _atoms(tmp: Path, *, as_bullets: bool):
    return OrbitBriefPdfParser().parse_artifact("p", "art", _pdf(tmp, as_bullets=as_bullets)).atoms


def _check(atoms) -> None:
    texts = [a.raw_text or "" for a in atoms]
    assert any(t.endswith(WHOLE) for t in texts), texts
    assert not any(t.startswith(FOOTER + " ") for t in texts), texts
    assert not any(t.startswith("and a 90% premium") for t in texts), texts


def test_page_footer_skipped_when_a_bullet_continues_on_the_next_page(tmp_path: Path) -> None:
    atoms = _atoms(tmp_path, as_bullets=True)
    _check(atoms)
    hit = next(a for a in atoms if (a.raw_text or "") == WHOLE)
    loc = hit.source_refs[0].locator
    assert loc.get("block_kind") == "bullet_list"
    assert loc.get("page") == 0


def test_page_footer_skipped_when_a_paragraph_continues_on_the_next_page(tmp_path: Path) -> None:
    _check(_atoms(tmp_path, as_bullets=False))
