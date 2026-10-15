"""A mirrored cache batch is merged once, not on every compile (2026-10-15:
1.7 GB of embedding batches re-downloaded before every compile's first stage)."""
from types import SimpleNamespace

from app.core import cache_blob


class _CC:
    def __init__(self, blobs):
        self.blobs = blobs          # name -> (etag, text)
        self.downloads = []

    def list_blobs(self, name_starts_with=""):
        return [SimpleNamespace(name=n, etag=e) for n, (e, _t) in self.blobs.items() if n.startswith(name_starts_with)]

    def download_blob(self, name):
        self.downloads.append(name)
        return SimpleNamespace(readall=lambda: self.blobs[name][1].encode())


def test_each_batch_is_fetched_once_and_again_only_when_it_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_EMBED_CACHE_DB", str(tmp_path / "_embed_cache.db"))
    monkeypatch.setattr(cache_blob, "_MERGED", {})
    cc = _CC({"_cache_rows/embed/a.jsonl": ("e1", '{"key": "a"}\n'), "_cache_rows/embed/b.jsonl": ("e2", '{"key": "b"}\n')})
    assert sum(len(r) for r in cache_blob._iter_batches(cc, "embed")) == 2
    assert sum(len(r) for r in cache_blob._iter_batches(cc, "embed")) == 0
    assert len(cc.downloads) == 2
    cc.blobs["_cache_rows/embed/b.jsonl"] = ("e3", '{"key": "c"}\n')
    assert sum(len(r) for r in cache_blob._iter_batches(cc, "embed")) == 1
    # A new process with the same local cache remembers; a wiped cache forgets.
    monkeypatch.setattr(cache_blob, "_MERGED", {})
    assert sum(len(r) for r in cache_blob._iter_batches(cc, "embed")) == 0
    monkeypatch.setattr(cache_blob, "_MERGED", {})
    (tmp_path / "_cache_rows_merged.json").unlink()
    assert sum(len(r) for r in cache_blob._iter_batches(cc, "embed")) == 2


def test_the_embedding_merge_runs_beside_the_compile(monkeypatch):
    import threading

    started, release = threading.Event(), threading.Event()

    def _slow():
        started.set()
        release.wait(5)
        return 0

    monkeypatch.setattr(cache_blob, "sync_embed_into_cache", _slow)
    monkeypatch.setattr(cache_blob, "_EMBED_SYNC", {"thread": None})
    assert cache_blob.sync_embed_into_cache_background() is True      # returns at once
    assert started.wait(2)
    assert cache_blob.sync_embed_into_cache_background() is False     # one at a time
    release.set()
    cache_blob._EMBED_SYNC["thread"].join(2)
    assert cache_blob.sync_embed_into_cache_background() is True
    cache_blob._EMBED_SYNC["thread"].join(2)


def test_an_ocr_lookup_waits_for_a_running_merge_and_never_otherwise(monkeypatch):
    import threading, time

    release = threading.Event()
    monkeypatch.setattr(cache_blob, "sync_ocr_into_cache", lambda: release.wait(5))
    monkeypatch.setattr(cache_blob, "_OCR_SYNC", {"thread": None})
    assert cache_blob.wait_for_ocr_merge(0.01) is True            # nothing running: no wait
    assert cache_blob.sync_ocr_into_cache_background() is True
    assert cache_blob.wait_for_ocr_merge(0.05) is False           # a lookup now waits
    release.set()
    assert cache_blob.wait_for_ocr_merge(2) is True
