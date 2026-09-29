# -*- coding: utf-8 -*-
"""A read survives the container, or the determinism work stops at its edge.

`fetch_ml.py` DOWNLOADS /tmp/ml at boot and nothing ever uploaded the caches
back. So the OCR cache only held whatever was last put in blob by hand, every
cold start re-read every picture through a billed service, and the same deal
got a fresh chance to read differently on Tuesday than it did on Monday.

The mirror is per-batch and immutable for the same reason the training-row
mirror is: up to three workers compile at once, and uploading a whole SQLite
file is last-write-wins -- one worker silently erasing another's reads.

Re-merging is safe because the cache is CONTENT-ADDRESSED: the key is
sha256(version || image bytes), so the same key always carries the same text.
"""
from __future__ import annotations

import pytest

from app.core import cache_blob, ocr_cache


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_OCR_CACHE_DB", str(tmp_path / "ocr.db"))
    monkeypatch.delenv("SOWSMITH_OCR_CACHE_DISABLE", raising=False)
    ocr_cache.reset_cache()
    yield ocr_cache.get_cache()
    ocr_cache.reset_cache()


def test_rows_survive_the_json_round_trip(cache) -> None:
    cache.put(b"a-scanned-page", {"text": "INVOICE TOTAL 4,182.00",
                                  "backend": "azure_doc_intel", "confidence": 0.94})
    rows = cache.export_rows()
    assert len(rows) == 1

    text = cache_blob.rows_to_jsonl(rows)
    back = cache_blob.jsonl_to_rows(text)
    assert back == rows, "a mirrored row came back different from the one sent"


def test_a_restore_puts_the_read_back(cache, tmp_path, monkeypatch) -> None:
    image = b"a-scanned-page"
    cache.put(image, {"text": "INVOICE TOTAL 4,182.00", "backend": "stub", "confidence": 0.9})
    rows = cache.export_rows()

    # A new container: fresh file, nothing in it.
    monkeypatch.setenv("SOWSMITH_OCR_CACHE_DB", str(tmp_path / "fresh.db"))
    ocr_cache.reset_cache()
    fresh = ocr_cache.get_cache()
    assert fresh.get(image) is None

    for row in rows:
        fresh.put_raw(**row)
    got = fresh.get(image)
    assert got is not None, "the restored container could not find a read it had mirrored"
    assert got["text"] == "INVOICE TOTAL 4,182.00"
    assert got["backend"] == "stub"


def test_restoring_twice_changes_nothing(cache) -> None:
    """Content-addressed, so a re-merge is a no-op rather than a conflict."""
    cache.put(b"img", {"text": "hello", "backend": "stub", "confidence": 0.5})
    rows = cache.export_rows()
    for _ in range(3):
        for row in rows:
            cache.put_raw(**row)
    assert cache.count() == 1


def test_an_empty_read_is_never_mirrored(cache) -> None:
    """"Nothing fired" is a statement about the day, not the picture. It is
    never stored, so it must never be restored either."""
    cache.put(b"unreadable", {"text": "", "backend": "", "confidence": 0.0})
    assert cache.export_rows() == []
    cache.put_raw(key="deadbeef", text="   ")
    assert cache.count() == 0


def test_the_mirror_is_gated_and_silent(monkeypatch) -> None:
    """Off without the gate, and a failure never reaches the compile."""
    monkeypatch.delenv("SOWSMITH_FEEDBACK_BLOB", raising=False)
    assert cache_blob.upload_rows(cache_blob.OCR, [{"key": "k", "text": "t"}]) is False
    assert cache_blob.sync_ocr_into_cache() == 0

    monkeypatch.setenv("SOWSMITH_FEEDBACK_BLOB", "1")
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    # Gate on, storage unconfigured: still silent, still False.
    assert cache_blob.upload_rows(cache_blob.OCR, [{"key": "k", "text": "t"}]) is False


def test_a_batch_name_is_unique_per_upload(monkeypatch) -> None:
    """Three workers at once must not be able to overwrite each other."""
    seen: list[str] = []

    class FakeCC:
        def upload_blob(self, name, data, overwrite):
            seen.append(name)

    monkeypatch.setattr(cache_blob, "_container_client", lambda: FakeCC())
    for _ in range(5):
        cache_blob.upload_rows(cache_blob.OCR, [{"key": "k", "text": "t"}])
    assert len(seen) == 5
    assert len(set(seen)) == 5, "two batches shared a blob name"
    assert all(n.startswith("_cache_rows/ocr/") for n in seen)
