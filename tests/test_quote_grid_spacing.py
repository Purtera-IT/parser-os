"""A CDW order with items set far apart, a wrap that reads like a figure, and
items carried onto a second page stays one line item per item (010043,
PVDP685.pdf: the grid ended after its first item and the rest fell back to
text, each item's tail glued onto the next)."""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

HEADS = {"ITEM": 47, "QTY": 351, "CDW#": 393, "UNIT PRICE": 450, "EXT. PRICE": 518}
PAGE1 = [
    (["Verkada CF83-E - network surveillance camera - fisheye - with 30 days of", "st"],
     "CF83-512E-HW", "1", "8091753", "$1,325.27", "$1,325.27"),
    (["Verkada Dome Series CD53-E - network surveillance camera - dome - with", "60 d"],
     "CD53-512E-HW", "2", "8317148", "$1,398.93", "$2,797.86"),
    (["Verkada camera junction box - circular"], "ACC-MNT-CJBOX-1", "2", "8120609", "$80.30", "$160.60"),
]
PAGE2 = [(["Verkada Video Security Cloud License Capacity Increase - camera license"],
          "LIC-CAM-3Y-CAP", "3", "8262069", "$437.93", "$1,313.79")]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _items(p, y, items):
    for desc, mfg, qty, cdw, unit, ext in items:
        for i, line in enumerate(desc):
            p.insert_text((47, y + 8 * i), line, fontsize=7)
        for x, t in ((357, qty), (390, cdw), (461, unit), (525, ext)):
            p.insert_text((x, y), t, fontsize=7)
        yy = y + 8 * len(desc) + 4
        p.insert_text((47, yy), f"Mfg. Part#: {mfg}", fontsize=7)
        p.insert_text((47, yy + 11), "Contract: Standard Pricing", fontsize=7)
        y = yy + 44          # CDW's spacing: ~33pt of white between items
    return y


def _order(path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((47, 470), "ORDER DETAILS", fontsize=7)
    for h, x in HEADS.items():
        p.insert_text((x, 490), h, fontsize=7)
    _items(p, 506, PAGE1)
    q = doc.new_page(width=612, height=792)
    q.insert_text((47, 40), "ORDER DETAILS (CONT.)", fontsize=7)
    y = _items(q, 55, PAGE2)
    q.insert_text((414, y + 20), "SUBTOTAL", fontsize=7)
    q.insert_text((520, y + 20), "$5,597.52", fontsize=7)
    doc.save(str(path))


def test_spaced_items_and_continued_page_are_whole(tmp_path: Path) -> None:
    path = tmp_path / "PVDP685.pdf"
    _order(path)
    atoms = OrbitBriefPdfParser().parse_artifact("p", "a", path).atoms
    for desc, mfg, qty, cdw, unit, ext in PAGE1 + PAGE2:
        hits = [a for a in atoms if cdw in a.raw_text]
        assert len(hits) == 1, (cdw, [a.raw_text for a in hits])
        cells = (hits[0].value or {}).get("cells") or {}
        assert cells.get("QTY") == qty and cells.get("EXT. PRICE") == ext, (cdw, cells)
        assert cells.get("Mfg. Part#") == mfg, (cdw, cells)
        assert " ".join(desc) in cells.get("ITEM", ""), (cdw, cells)
    for a in atoms:   # no item's tail glued onto another item
        assert sum(1 for it in PAGE1 + PAGE2 if it[3] in a.raw_text) <= 1, a.raw_text
