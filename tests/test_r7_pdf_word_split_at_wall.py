"""A table wall never cuts a word in two.

Shape of a live signed SOW (DocuSign of a Word table; names here are
synthetic): a revision table drawn only with filled rects -- each cell an
outer band plus an inset padding band -- so every band edge is a grid wall and
each column comes back as margin | text | margin. The name in the shaded v.1
row starts on the inset's edge, its first glyph fell in the margin sliver, and
the row read "SOW VERSION: v.1 | col_1: O | QUOTED BY: skar Lindqvist | ...".
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: E402

BLUE = (0.0, 0.439, 0.753)
GRAY = (0.949, 0.949, 0.949)
BORDER = (0.749, 0.749, 0.749)
OUTER = [(72.50, 185.11), (185.59, 298.44), (298.92, 392.54), (393.02, 524.83)]
INNER = [(77.78, 180.07), (190.87, 293.16), (303.96, 387.50), (398.30, 519.55)]


def _banded_revision_table(path: Path, name_x: float) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), "STATEMENT OF WORK", fontsize=14, fontname="hebo")
    page.insert_text((72, 105), "This statement of work describes the services to be delivered.", fontsize=10)

    def fill(x0, y0, x1, y1, c):
        page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=None, fill=c, width=0)

    for o, i in zip(OUTER, INNER):
        fill(o[0], 128.69, o[1], 151.49, BLUE)
        fill(i[0], 133.73, i[1], 146.45, BLUE)
        fill(o[0], 151.97, o[1], 172.61, GRAY)   # only the v.1 row is shaded,
        fill(i[0], 151.97, i[1], 172.61, GRAY)   # with an inset padding band
    for x in (72.02, 185.11, 298.44, 392.54, 524.83):
        for a, b in ((128.69, 151.49), (151.97, 172.61), (173.09, 193.73)):
            fill(x, a, x + 0.48, b, BORDER)
    for y in (128.21, 151.49, 172.61, 193.73):
        fill(72.02, y, 525.31, y + 0.48, BORDER)
    header = [("SOW VERSION", 77.78), ("QUOTED BY", 190.87), ("DATE", 303.96), ("REVISION HISTORY", 398.30)]
    for t, x in header:
        page.insert_text((x, 144.2), t, fontsize=11.04, fontname="hebo", color=(1, 1, 1))
    for t, x in (("v.1", 77.78), ("Oskar Lindqvist", name_x), ("07/09/2026", 303.96), ("First", 398.30)):
        page.insert_text((x, 162.5), t, fontsize=11.04, fontname="helv")
    for t, x in (("v.2", 77.78), ("Dana Whitfield", 190.87), ("07/16/2026", 303.96), ("Second", 398.30)):
        page.insert_text((x, 183.6), t, fontsize=11.04, fontname="helv")
    page.insert_text((72, 230), "The services described above will be performed on site.", fontsize=10)
    doc.save(str(path))


def _ruled_revision_table(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), "STATEMENT OF WORK", fontsize=14, fontname="hebo")
    xs, ys = [72, 172, 302, 402, 540], [140, 160, 180, 200]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]), width=0.5)
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), width=0.5)
    rows = [["SOW VERSION", "QUOTED BY", "DATE", "REVISION HISTORY"],
            ["v.1", "Oskar Lindqvist", "07/09/2026", "First"],
            ["v.2", "Dana Whitfield", "07/16/2026", "Second"]]
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            # the v.1 name's first letter sits left of the vertical rule
            x = xs[c] - 8 if (r, c) == (1, 1) else xs[c] + 4
            page.insert_text((x, ys[r] + 14), t, fontsize=10, fontname="hebo" if r == 0 else "helv")
    page.insert_text((72, 240), "The services described above will be performed on site.", fontsize=10)
    doc.save(str(path))


def _rows(path: Path) -> list[str]:
    out = OrbitBriefPdfParser().parse_artifact("p", "art_signed", path)
    return [a.raw_text for a in out.atoms if "v.1" in a.raw_text or "v.2" in a.raw_text]


@pytest.mark.parametrize("name_x", [190.87, 185.97])
def test_banded_grid_keeps_the_name_whole(tmp_path, name_x):
    pdf = tmp_path / "signed.pdf"
    # 185.97: most of the "O" lies in the margin sliver left of the inset band.
    _banded_revision_table(pdf, name_x)
    rows = _rows(pdf)
    assert "SOW VERSION: v.1 | QUOTED BY: Oskar Lindqvist | DATE: 07/09/2026 | REVISION HISTORY: First" in rows
    assert "SOW VERSION: v.2 | QUOTED BY: Dana Whitfield | DATE: 07/16/2026 | REVISION HISTORY: Second" in rows
    assert not any("col_" in r or "skar Lindqvist" in r.replace("Oskar", "") for r in rows)


def test_ruled_grid_keeps_the_name_whole(tmp_path):
    pdf = tmp_path / "ruled.pdf"
    _ruled_revision_table(pdf)
    rows = _rows(pdf)
    assert "SOW VERSION: v.1 | QUOTED BY: Oskar Lindqvist | DATE: 07/09/2026 | REVISION HISTORY: First" in rows


def test_two_words_meeting_at_a_wall_are_not_joined():
    from app.parsers.pdf._shared import _rejoin_words_split_at_walls

    class _Row:
        def __init__(self, cells):
            self.cells = cells

    class _Page:
        def get_text(self, kind):
            return [(10.0, 0.0, 30.0, 10.0, "v.1", 0, 0, 0), (52.0, 0.0, 80.0, 10.0, "Oskar", 0, 0, 1)]

    rows = [["v.1", "Oskar"]]
    _rejoin_words_split_at_walls(_Page(), [_Row([(0, 0, 50, 12), (50, 0, 100, 12)])], rows)
    assert rows == [["v.1", "Oskar"]]
