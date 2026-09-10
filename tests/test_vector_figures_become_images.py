"""A diagram drawn in vector operations is a picture too.

``page.get_images()`` only sees embedded raster XObjects, so a figure DRAWN
with vector operations is invisible to the whole image pipeline -- not skipped
by a gate, never known to exist.

The Anova install guide is the case: page 1 carries 4 embedded images (all tiny
icons) and 1,657 drawing operations; page 2 carries 5 and 3,373. The components
diagram (Sensor Interface Box / HDP Sensor / UTM), both venting options with
the flapper-valve part number, and every step illustration are vector line art.

Across the dev corpus, 46 of 305 PDFs (15%) over 31 deals are vector-heavy, and
17 have NO raster the vision pass can see at all -- floor plans, rack
elevations, low-voltage layouts, where the drawing IS the content.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.pdf.images import _cluster_rects, _pdf_vector_region_markers


def _markers(path, tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_IMAGE_DIR", str(tmp_path / "imgs"))
    return _pdf_vector_region_markers(
        path=path, project_id="p", artifact_id="a", parser_version="v",
    )


def _drawn_page(path, shapes):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for (x0, y0, x1, y1) in shapes:
        page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=1)
    doc.save(str(path))
    doc.close()


def _figure(x0, y0, n=14, step=6):
    """A cluster of strokes: what a real drawing looks like."""
    return [(x0 + i * step, y0 + i * step, x0 + 90 + i * step, y0 + 70 + i * step)
            for i in range(n)]


def test_a_drawn_figure_becomes_a_marker(tmp_path, monkeypatch):
    p = tmp_path / "drawn.pdf"
    _drawn_page(p, _figure(80, 80))
    out = _markers(p, tmp_path, monkeypatch)
    assert len(out) == 1
    v = out[0].value
    assert v["region_ref"].startswith("page0/vector")
    assert v["saved_path"], "the figure must be rasterised to a file"
    assert v["size_bytes"] > 3000, "must clear the vision pass's minimum"


def test_two_separated_figures_stay_separate(tmp_path, monkeypatch):
    p = tmp_path / "two.pdf"
    _drawn_page(p, _figure(60, 60) + _figure(60, 500))
    out = _markers(p, tmp_path, monkeypatch)
    assert len(out) == 2


def test_a_page_with_no_drawings_yields_nothing(tmp_path, monkeypatch):
    p = tmp_path / "prose.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Just words on a page.", fontsize=11)
    doc.save(str(p))
    doc.close()
    assert _markers(p, tmp_path, monkeypatch) == []


def test_a_lone_rectangle_is_a_border_not_a_figure(tmp_path, monkeypatch):
    """One rectangle is a box around something. A figure is made of strokes."""
    p = tmp_path / "border.pdf"
    _drawn_page(p, [(60, 60, 550, 700)])
    assert _markers(p, tmp_path, monkeypatch) == []


def test_a_tiny_cluster_is_an_icon_not_a_figure(tmp_path, monkeypatch):
    p = tmp_path / "icon.pdf"
    _drawn_page(p, [(80 + i, 80 + i, 100 + i, 100 + i) for i in range(12)])
    assert _markers(p, tmp_path, monkeypatch) == []


def test_the_page_background_is_never_a_figure(tmp_path, monkeypatch):
    p = tmp_path / "bg.pdf"
    _drawn_page(p, [(1, 1, 611, 791)] + _figure(80, 80))
    out = _markers(p, tmp_path, monkeypatch)
    assert len(out) == 1, "the full-page rect must not become its own figure"


def test_an_unreadable_pdf_never_breaks_the_parse(tmp_path, monkeypatch):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"not a pdf at all")
    assert _markers(p, tmp_path, monkeypatch) == []


def test_clustering_unions_what_sits_together():
    boxes = _cluster_rects([(0, 0, 10, 10), (11, 0, 20, 10), (200, 200, 210, 210)], 6.0)
    assert len(boxes) == 2


def test_clustering_keeps_apart_what_is_far_apart():
    boxes = _cluster_rects([(0, 0, 10, 10), (400, 400, 410, 410)], 6.0)
    assert len(boxes) == 2
