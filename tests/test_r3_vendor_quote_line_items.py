"""A vendor quote's line item is one atom with its quantity in the QTY field (010003).

CDW's quote sets each item as an anchor line (description, QTY, CDW#, unit
and extended price) with tail lines under the description only: its wrap,
"Mfg. Part#: QM55C", "Contract: Standard Pricing". The whitespace-column
reader let the right-aligned quantity drift (the QTY "4" had no atom and the
CDW# landed under QTY); the text reader glued each item's tail onto the next
item ("... Standard Pricing Samsung QM75C ..."). Quantity and display size
drive the tech count, so each item must be whole, with its fields.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

ITEMS = [
    (["Samsung QM55C QMC Series - 55\" LED-backlit LCD display - 4K - for", "digital signage"],
     "QM55C", "4", "7506871", "$689.99", "$2,759.96"),
    (["Samsung QM75C QMC Series - 75\" LED-backlit LCD display - 4K - for", "digital signage"],
     "QM75C", "2", "7506872", "$1,077.30", "$2,154.60"),
    (["Chief Fusion Large Fixed Wall Display Mount"], "LSA1U", "6", "1813461", "$129.99", "$779.94"),
]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _quote(path: Path, right_aligned: bool, ruled: bool) -> None:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((140, 30), "Hardware   Software   Services   IT Solutions   Brands   Research Hub", fontsize=8)
    p.insert_text((36, 70), "QUOTE CONFIRMATION", fontsize=18, fontname="hebo")
    y = 150
    heads = {"ITEM": 36, "QTY": 330, "CDW#": 380, "UNIT PRICE": 450, "EXT. PRICE": 530}
    if ruled:
        p.draw_rect(fitz.Rect(30, y - 11, 582, y + 4), color=(0.85, 0.85, 0.85), fill=(0.85, 0.85, 0.85))
    for h, x in heads.items():
        p.insert_text((x, y), h, fontsize=8, fontname="hebo")
    y += 20

    def put(x, t, right):
        w = fitz.get_text_length(t, fontsize=8) if right else 0
        p.insert_text((x - w, y), t, fontsize=8)

    for desc, mfg, qty, cdw, unit, ext in ITEMS:
        for i, line in enumerate(desc):
            p.insert_text((36, y + 11 * i), line, fontsize=8, fontname="hebo")
        yy = y + 11 * len(desc)
        p.insert_text((36, yy), f"Mfg. Part#: {mfg}", fontsize=7)
        p.insert_text((36, yy + 10), "Contract: Standard Pricing", fontsize=7)
        if right_aligned:
            put(352, qty, True); put(380, cdw, False); put(500, unit, True); put(578, ext, True)
        else:
            put(330, qty, False); put(380, cdw, False); put(450, unit, False); put(530, ext, False)
        y = yy + 28
        if ruled:
            p.draw_line((30, y - 6), (582, y - 6), color=(0.7, 0.7, 0.7))
    doc.save(str(path))


@pytest.mark.parametrize("right_aligned,ruled", [(False, False), (True, False), (True, True)])
def test_each_quote_item_is_one_line_item_with_fields(tmp_path: Path, right_aligned: bool, ruled: bool) -> None:
    path = tmp_path / "CDW Quote.pdf"
    _quote(path, right_aligned, ruled)
    out = OrbitBriefPdfParser().parse_artifact("p", "a", path)
    atoms = out if isinstance(out, list) else out.atoms
    texts = [a.raw_text for a in atoms]
    for desc, mfg, qty, cdw, unit, ext in ITEMS:
        hits = [a for a in atoms if cdw in a.raw_text]
        assert len(hits) == 1, (cdw, texts)
        cells = (hits[0].value or {}).get("cells") or {}
        assert cells.get("QTY") == qty, (cdw, cells)
        assert cells.get("CDW#") == cdw, (cdw, cells)
        assert cells.get("Mfg. Part#") == mfg, (cdw, cells)
        assert cells.get("EXT. PRICE") == ext, (cdw, cells)
        assert " ".join(desc) in cells.get("ITEM", ""), (cdw, cells)
    for t in texts:
        assert sum(1 for it in ITEMS if it[3] in t) <= 1, t   # never two items in one atom
        if any(it[3] in t for it in ITEMS):
            assert "Research Hub" not in t, t
