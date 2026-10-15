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
