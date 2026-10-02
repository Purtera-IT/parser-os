"""A PDF list's sub-bullets nest under the bullet they are indented beneath (010003).

A Word-exported SOW sets its top-level bullets as a "•" at x 72 with text at
x 90, and a second-level item as a Courier "o" at x 108 with text at x 126,
each glyph sitting a fraction of a point below its text line. The reader's
text has no indentation, so every item came out at the top level: the
sub-item under "Mount the panel:" got the next top-level index instead of
being that item's child. A "•" after the sub-item returns to the top level.

The same geometry keeps a list item's second line with its item: a numbered
item whose first sentence ends a line, and whose next sentence starts at the
item's hanging indent, read the second sentence as a new paragraph.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

# (top, marker, text): marker "o" is a second-level item.
ITEMS = [
    (536.8, "•", "Send one field technician to the office"),
    (555.0, "•", "Unpack and check the three meeting room panels"),
    (573.2, "•", "Mount the panel:"),
    (591.5, "o", "Brackets are already fixed to the wall and ready for the panels"),
    (608.9, "•", "Tighten and square the panel so it hangs safely"),
    (627.1, "•", "Route the cables along the surface only"),
    (645.4, "•", "Tidy the room when the work is done"),
]


def _list_pdf(tmp: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612.3, height=790.9)
    page.insert_text((54, 479), "PROJECT SCOPE", fontfile=BOLD, fontname="sb", fontsize=12)
    page.insert_text((54, 525.8), "The provider will do the following:", fontfile=SERIF, fontname="s", fontsize=10)
    for top, marker, text in ITEMS:
        if marker == "o":
            page.insert_text((108, top + 8.3), "o", fontfile=MONO, fontname="m", fontsize=10)
            page.insert_text((126, top + 8.2), text, fontfile=SERIF, fontname="s", fontsize=10)
        else:
            page.insert_text((72, top + 9.1), "•", fontfile=SERIF, fontname="s", fontsize=10)
            page.insert_text((90, top + 9), text, fontfile=SERIF, fontname="s", fontsize=10)
    out = tmp / "sow.pdf"
    doc.save(str(out))
    return out


def _bullets(pdf: Path) -> dict[str, dict]:
    atoms = OrbitBriefPdfParser().parse_artifact("p", "art", pdf).atoms
    return {
        a.raw_text: a.source_refs[0].locator
        for a in atoms
        if a.source_refs[0].locator.get("block_kind") == "bullet_list"
    }


def test_sub_bullet_nests_under_its_colon_parent(tmp_path: Path) -> None:
    got = _bullets(_list_pdf(tmp_path))
    parent = got["Mount the panel:"]
    child = got[ITEMS[3][2]]
    assert parent["bullet_path"] == [2]
    assert child["bullet_path"] == [2, 0]
    assert child["bullet_depth"] == 2
    assert child["section_path"] == parent["section_path"] + ["Mount the panel"]


def test_next_top_level_bullet_pops_back(tmp_path: Path) -> None:
    got = _bullets(_list_pdf(tmp_path))
    tops = [t for _, m, t in ITEMS if m == "•"]
    assert [got[t]["bullet_path"] for t in tops] == [[0], [1], [2], [3], [4], [5]]
    base = got[tops[0]]["section_path"]
    assert all(got[t]["section_path"] == base for t in tops)


def _numbered_pdf(tmp: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612.3, height=790.9)
    page.insert_text((54, 90), "PROJECT ASSUMPTIONS", fontfile=BOLD, fontname="sb", fontsize=12)
    items = {
        1: ["The provider will work during the hours this SOW sets out for each visit."],
        2: ["Both parties will agree a delivery date before any equipment is shipped to the site.",
            "Typically, the provider needs five to ten business days of notice to book a technician."],
        3: ["The customer will give the crew access to the room for the whole visit."],
    }
    top = 113.0
    for n, lines in items.items():
        page.insert_text((72, top + 8), f"{n}.", fontfile=SERIF, fontname="s", fontsize=10)
        for ln in lines:
            page.insert_text((90, top + 8), ln, fontfile=SERIF, fontname="s", fontsize=10)
            top += 11.5
    out = tmp / "assumptions.pdf"
    doc.save(str(out))
    return out


def test_second_sentence_at_hanging_indent_stays_in_its_item(tmp_path: Path) -> None:
    atoms = OrbitBriefPdfParser().parse_artifact("p", "art", _numbered_pdf(tmp_path)).atoms
    by_text = {a.raw_text: a.source_refs[0].locator for a in atoms}
    first = by_text["Both parties will agree a delivery date before any equipment is shipped to the site."]
    second = by_text["Typically, the provider needs five to ten business days of notice to book a technician."]
    assert second.get("block_kind") == "bullet_list"
    assert first["bullet_path"] == second["bullet_path"] == [1]
    assert (first["sentence_index"], second["sentence_index"]) == (0, 1)
    assert second["sentence_count"] == 2
    third = by_text["The customer will give the crew access to the room for the whole visit."]
    assert third["bullet_path"] == [2]
