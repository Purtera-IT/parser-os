"""A PDF page footer that runs across the pages is page chrome (010003).

A signed SOW prints a two-line 8pt footer at the bottom of every page:
"Internal and Confidential | Page N | Example Vendor Inc" with "WO 4471"
under it. The page builder joins the two lines into one footer band, kept
as a chatter atom. The page number differs per page, so the verbatim repeat
check never matched it, and the band stayed an atom on every page. On the
first page it also sat between a lead-in paragraph and the list it framed,
and took that lead-in as its own.

Now each copy goes to the chrome ledger with reason ``page_footer``; the
lead-in stays as an atom of its own; a footer-shaped line printed on one page
only stays an atom.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from app.core import orbitbrief_envelope as env
from app.core.compiler import compile_project

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"

PAGES = 4
LEAD = "Subject to the other terms of this order, the vendor will do the following work:"
SUB_LEAD = "The vendor will do the following:"
ITEMS = ["Send two field engineers to the store", "Fix each clock to the bracket on the wall"]
ONE_OFF = "Confidential | Page 9 | Appendix B"


def _text(page, x: float, top: float, s: str, *, font: str = SERIF, size: float = 10.0) -> None:
    page.insert_text((x, top + 0.891 * size), s, fontfile=font,
                     fontname="b" if font == BOLD else "s", fontsize=size)


def _footer(page, n: int) -> None:
    _text(page, 57.3, 727.9, "Internal and Confidential", size=8.0)
    _text(page, 294.2, 727.9, f"Page {n}", size=8.0)
    _text(page, 478.0, 727.9, "Example Vendor Inc", size=8.0)
    _text(page, 57.6, 743.1, "WO 4471", size=8.0)


def _pdf(path: Path) -> None:
    doc = fitz.open()
    for n in range(1, PAGES + 1):
        p = doc.new_page(width=612.28, height=790.87)
        _text(p, 54.0, 60.0, f"PART {n} SCOPE", font=BOLD, size=14)
        for k in range(14):
            _text(p, 54.0, 85.0 + 13.8 * k,
                  f"This part sets out the work the vendor does on visit number {n} at the store named in"
                  if k < 13 else "the order form.")
        if n == 1:
            _text(p, 54.0, 497.6, LEAD)
            _text(p, 54.0, 516.8, SUB_LEAD)
            for k, s in enumerate(ITEMS):
                _text(p, 72.0, 536.8 + 18.2 * k, "•")
                _text(p, 90.0, 536.7 + 18.2 * k, s)
        if n == 3:
            _text(p, 54.0, 300.0, ONE_OFF)
        _footer(p, n)
    doc.save(str(path))


def test_running_footer_goes_to_chrome_and_body_stays(tmp_path: Path, monkeypatch) -> None:
    deal = tmp_path / "deal"
    deal.mkdir()
    _pdf(deal / "work order.pdf")
    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")
    result = compile_project(deal, project_id="p", allow_errors=True, use_cache=False)

    kept = [a.raw_text or "" for a in result.atoms]
    assert not [t for t in kept if "Example Vendor Inc" in t or "WO 4471" in t], kept

    chrome = [r for r in env._suppressed_chrome_for_review(result) if "Example Vendor Inc" in r["text"]]
    assert len(chrome) == PAGES, chrome
    assert {r["reason"] for r in chrome} == {"page_footer"}

    # The lead-in the footer used to take is an atom of its own, and the list
    # and the body text stay.
    assert LEAD in kept
    for s in ITEMS:
        assert s in kept
    for n in range(1, PAGES + 1):
        assert any(f"visit number {n}" in t for t in kept)
    # Footer-shaped, but printed on one page only: not a running band.
    assert ONE_OFF in kept
