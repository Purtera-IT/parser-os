# -*- coding: utf-8 -*-
"""A linked drawing is read by Document Intelligence once, not once per compile.

``read_lines_with_polygons`` had no cache while the plain-text chain did, so
every re-parse of a deal with a drawing paid for the same picture again.
"""
from __future__ import annotations

import pytest

from app.core import doc_intel_ocr, ocr_cache

LINES = [{"content": "CAT6 DROP", "polygon": [0.0, 0.0, 9.0, 0.0, 9.0, 3.0, 0.0, 3.0], "words": []}]


@pytest.fixture()
def cache_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_OCR_CACHE_DB", str(tmp_path / "ocr.db"))
    monkeypatch.delenv("SOWSMITH_OCR_CACHE_DISABLE", raising=False)
    monkeypatch.setattr(doc_intel_ocr, "doc_intel_available", lambda: True)
    ocr_cache.reset_cache()
    yield
    ocr_cache.reset_cache()


def _stub(monkeypatch, result, calls):
    def fake(image_bytes):
        calls.append(1)
        return result
    monkeypatch.setattr(doc_intel_ocr, "_read_lines_with_polygons_uncached", fake)


def test_second_compile_reuses_the_first_read(cache_db, monkeypatch):
    calls: list[int] = []
    _stub(monkeypatch, LINES, calls)
    assert doc_intel_ocr.read_lines_with_polygons(b"drawing") == LINES
    ocr_cache.reset_cache()                      # a new container
    assert doc_intel_ocr.read_lines_with_polygons(b"drawing") == LINES
    assert len(calls) == 1, "paid Document Intelligence twice for one drawing"


def test_a_miss_is_not_frozen_in(cache_db, monkeypatch):
    calls: list[int] = []
    _stub(monkeypatch, [], calls)
    assert doc_intel_ocr.read_lines_with_polygons(b"drawing") == []
    assert doc_intel_ocr.read_lines_with_polygons(b"drawing") == []
    assert len(calls) == 2


def test_does_not_collide_with_the_plain_text_read(cache_db, monkeypatch):
    """Same bytes, two different readings: the polygon read must not answer a text lookup."""
    calls: list[int] = []
    _stub(monkeypatch, LINES, calls)
    doc_intel_ocr.read_lines_with_polygons(b"drawing")
    assert ocr_cache.get_cache().get(b"drawing") is None
