"""A page is not one column, and a block is a box.

Page lines were flattened across EVERY block and sorted by (y, x). That reads a
page as though it were a single column, so two text boxes standing side by side
interleave line by line.

M980 Copper Rack Elevations page 1 carries a legend box beside a port-count box,
offset by a constant 7.2pt against a 14.4pt line height. Sorting by y alone
produced:

    Items in RED are new and need be
    installed. Items in BLACK should
    MDF - port count 47                 <- the other box
    already be in the new MDF. The
    12 data drops = 36 Cat6             <- the other box

welding "12 data drops = 36 Cat6" -- a quote input -- into the middle of a
sentence about what the colours mean. The Anova install guide sheared "STEP 1"
into "P 1" the same way.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")


LEGEND = [
    "Items in RED are new and need be",
    "installed. Items in BLACK should",
    "already be in the new MDF. The",
    "items in GREEN need to be moved",
    "from the old MDF rack to the new",
    "MDF rack.",
]
PORTS = [
    "MDF - port count 47",
    "12 data drops = 36 Cat6",
    "4 APs = 8 Cat6A",
    "3 Camera locations = 3 Cat6",
]


def _two_boxes(path):
    """The M980 shape: two independent flows, constant half-line offset."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 45.5
    for line in PORTS:
        page.insert_text((34, y), line, fontsize=11)
        y += 14.4
    y = 23.9  # 7.2pt higher — its own leading, not a shared baseline
    for line in LEGEND:
        page.insert_text((387, y), line, fontsize=11)
        y += 14.4
    doc.save(str(path))
    doc.close()


def _page_text(path):
    from pathlib import Path

    from app.parsers.orbitbrief_pdf import _page_prose_excluding_tables

    # The function only engages on pages that HAVE a table; pass one that sits
    # in an empty corner so nothing is actually excluded.
    far_away = [fitz.Rect(0, 770, 8, 780)]
    return _page_prose_excluding_tables(Path(str(path)), 0, far_away) or ""


def test_each_box_stays_whole(tmp_path):
    p = tmp_path / "two_boxes.pdf"
    _two_boxes(p)
    text = _page_text(p)
    assert "\n".join(LEGEND) in text, "the legend was split by the other box"
    assert "\n".join(PORTS) in text, "the port-count box was split by the legend"


def test_a_quote_input_is_never_welded_into_another_sentence(tmp_path):
    """The failure in one assertion: a countable fact must survive intact."""
    p = tmp_path / "two_boxes.pdf"
    _two_boxes(p)
    for line in text_lines(_page_text(p)):
        if "data drops" in line:
            assert line.strip() == "12 data drops = 36 Cat6"
            break
    else:
        pytest.fail("the drop count did not survive as its own line")


def text_lines(text):
    return [l for l in text.split("\n") if l.strip()]


def test_a_single_column_page_is_unchanged(tmp_path):
    """The common case must read exactly as before."""
    p = tmp_path / "one_column.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 72
    for line in ("First paragraph line one.", "First paragraph line two.",
                 "Second paragraph begins here.", "And it continues."):
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    doc.save(str(p))
    doc.close()
    assert text_lines(_page_text(p)) == [
        "First paragraph line one.",
        "First paragraph line two.",
        "Second paragraph begins here.",
        "And it continues.",
    ]


def test_columns_read_in_column_order_not_zigzag(tmp_path):
    """Two real columns of prose read one column then the other."""
    p = tmp_path / "columns.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 72
    for line in ("Left column line A.", "Left column line B."):
        page.insert_text((60, y), line, fontsize=11)
        y += 16
    y = 72
    for line in ("Right column line A.", "Right column line B."):
        page.insert_text((340, y), line, fontsize=11)
        y += 16
    doc.save(str(p))
    doc.close()
    lines = text_lines(_page_text(p))
    assert lines.index("Left column line B.") < lines.index("Right column line A."), lines
