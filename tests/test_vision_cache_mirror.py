"""Vision replies survive a scale-to-zero: stored locally, mirrored, merged back."""
import json

from app.core import cache_blob, llm_client


class _Blob:
    def __init__(self, data):
        self._d = data

    def readall(self):
        return self._d


class _CC:
    def __init__(self):
        self.blobs = {}

    def upload_blob(self, name, data, overwrite=True):
        self.blobs[name] = data

    def list_blobs(self, name_starts_with=""):
        class B:
            def __init__(s, n):
                s.name, s.etag = n, n
        return [B(n) for n in self.blobs if n.startswith(name_starts_with)]

    def download_blob(self, name):
        return _Blob(self.blobs[name])


def test_vision_reply_round_trips_through_the_mirror(tmp_path, monkeypatch):
    cc = _CC()
    monkeypatch.setattr(cache_blob, "_container_client", lambda: cc)
    monkeypatch.setattr(cache_blob, "_seen_path", lambda: str(tmp_path / "seen.json"))
    monkeypatch.setattr(cache_blob, "_MERGED", {})
    monkeypatch.setenv("SOWSMITH_VISION_CACHE_DB", str(tmp_path / "a.db"))
    monkeypatch.setattr(llm_client, "_VCACHE_CONN", None)
    llm_client.VCACHE_NEW.clear()
    key = llm_client._vision_cache_key("m", "describe", "image/png", "AAAA")
    llm_client._vision_cache_put(key, "m", "a ceiling-mounted camera")
    assert cache_blob.mirror_vision()
    assert not llm_client.VCACHE_NEW
    # A fresh container: empty cache, then the merge brings the reply back.
    monkeypatch.setenv("SOWSMITH_VISION_CACHE_DB", str(tmp_path / "b.db"))
    monkeypatch.setattr(llm_client, "_VCACHE_CONN", None)
    assert llm_client._vision_cache_get(key) is None
    assert cache_blob.sync_vision_into_cache() == 1
    assert llm_client._vision_cache_get(key) == "a ceiling-mounted camera"
