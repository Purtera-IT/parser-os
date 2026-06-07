"""Tests for the universal symbol-crop harvester. Synthesizes a vector PDF page
(so region proposals fire) and checks clean symbol crops come out — the path that
fixed dense vector sheets yielding zero crops."""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("PIL.Image")

from app.core.schematic_crop_harvester import harvest_page, harvest_pdf


def _vector_page_pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # scatter symbol-sized vector glyphs (circles + small rects) across the page
    for gx in range(60, 560, 70):
        for gy in range(60, 740, 70):
            page.draw_circle((gx, gy), 10, color=(0, 0, 0), width=1.2)
            page.draw_rect(fitz.Rect(gx + 20, gy - 8, gx + 36, gy + 8), color=(0, 0, 0), width=1.0)
    p = tmp_path / "vec.pdf"
    doc.save(str(p))
    doc.close()
    return str(p)


def test_vector_page_yields_clean_crops(tmp_path):
    path = _vector_page_pdf(tmp_path)
    doc = fitz.open(path)
    crops = harvest_page(doc[0], 0)
    assert len(crops) > 0
    # each crop is (bbox_pdf, png_bytes)
    bbox, png = crops[0]
    assert len(bbox) == 4
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG


def test_harvest_pdf_returns_png_list(tmp_path):
    path = _vector_page_pdf(tmp_path)
    crops = harvest_pdf(path, max_pages=1, max_crops=500)
    assert isinstance(crops, list)
    assert len(crops) > 0
    assert all(c[:8] == b"\x89PNG\r\n\x1a\n" for c in crops[:5])


def test_max_crops_is_respected(tmp_path):
    path = _vector_page_pdf(tmp_path)
    crops = harvest_pdf(path, max_pages=1, max_crops=5)
    assert len(crops) <= 5
