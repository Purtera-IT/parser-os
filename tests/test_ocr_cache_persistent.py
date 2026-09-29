# -*- coding: utf-8 -*-
"""The same picture must read back the same text, across processes.

OCR text becomes the atom's text, and atom text is three quarters of
``label_key``. The in-process dict in ``_ocr_chain`` dies with the container
(minReplicas 0, 300s cooldown), so before this every cold start re-read every
image -- against a hosted service whose version moves, or a vision LLM, which
is not reproducible even at ``temperature: 0``. A gold label should not depend
on which afternoon the page was read on.
"""
from __future__ import annotations

import pytest

import app.parsers._ocr_chain as chain
from app.core import ocr_cache


@pytest.fixture()
def cache_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_OCR_CACHE_DB", str(tmp_path / "ocr.db"))
    monkeypatch.delenv("SOWSMITH_OCR_CACHE_DISABLE", raising=False)
    ocr_cache.reset_cache()
    yield tmp_path / "ocr.db"
    ocr_cache.reset_cache()


def _stub(chain_module, monkeypatch, text: str, counter: list[int]):
    def fake(image_bytes, notes):
        counter.append(1)
        return {"text": text, "backend": "stub", "confidence": 0.9, "notes": []}
    monkeypatch.setattr(chain_module, "_ocr_image_bytes_uncached", fake)


def test_a_second_process_reuses_the_first_reading(cache_db, monkeypatch):
    image = b"\x89PNG-pretend-this-is-a-scan"
    calls: list[int] = []
    _stub(chain, monkeypatch, "INVOICE TOTAL 4,182.00", calls)

    first = chain._ocr_image_bytes(image, [])
    assert first["text"] == "INVOICE TOTAL 4,182.00"
    assert len(calls) == 1

    # A new container: the process dict is gone, the sqlite file is not.
    chain._OCR_CACHE.clear()
    ocr_cache.reset_cache()

    second = chain._ocr_image_bytes(image, [])
    assert second["text"] == first["text"], "the same image read back differently"
    assert second["backend"] == first["backend"]
    assert len(calls) == 1, "re-OCR'd an image it had already read"


def test_a_miss_is_never_frozen_in(cache_db, monkeypatch):
    """"Nothing fired" is a statement about the day, not about the picture."""
    image = b"an image nothing could read"
    calls: list[int] = []
    _stub(chain, monkeypatch, "", calls)          # outage: no backend fired

    assert chain._ocr_image_bytes(image, [])["text"] == ""
    chain._OCR_CACHE.clear()
    ocr_cache.reset_cache()

    # The backend recovers; the empty answer must NOT have been persisted.
    _stub(chain, monkeypatch, "RECOVERED TEXT", calls)
    assert chain._ocr_image_bytes(image, [])["text"] == "RECOVERED TEXT"


def test_different_images_do_not_collide(cache_db, monkeypatch):
    calls: list[int] = []
    _stub(chain, monkeypatch, "first", calls)
    assert chain._ocr_image_bytes(b"image-one", [])["text"] == "first"
    _stub(chain, monkeypatch, "second", calls)
    assert chain._ocr_image_bytes(b"image-two", [])["text"] == "second"


def test_the_version_bump_invalidates(cache_db, monkeypatch):
    """Upgrading the chain is an explicit re-read, not a silent drift."""
    image = b"same bytes either way"
    calls: list[int] = []
    _stub(chain, monkeypatch, "old engine", calls)
    assert chain._ocr_image_bytes(image, [])["text"] == "old engine"

    chain._OCR_CACHE.clear()
    monkeypatch.setattr(ocr_cache, "OCR_CACHE_VERSION", "v2")
    _stub(chain, monkeypatch, "new engine", calls)
    assert chain._ocr_image_bytes(image, [])["text"] == "new engine"


def test_a_broken_cache_never_breaks_a_parse(cache_db, monkeypatch):
    class Boom:
        def get(self, b):
            raise RuntimeError("sqlite is unhappy")

        def put(self, b, r):
            raise RuntimeError("sqlite is unhappy")

    monkeypatch.setattr("app.core.ocr_cache.get_cache", lambda: Boom())
    calls: list[int] = []
    _stub(chain, monkeypatch, "still parsed", calls)
    chain._OCR_CACHE.clear()
    assert chain._ocr_image_bytes(b"whatever", [])["text"] == "still parsed"


def test_disabled_means_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_OCR_CACHE_DISABLE", "1")
    monkeypatch.setenv("SOWSMITH_OCR_CACHE_DB", str(tmp_path / "ocr.db"))
    ocr_cache.reset_cache()
    try:
        assert ocr_cache.get_cache() is None
    finally:
        ocr_cache.reset_cache()
