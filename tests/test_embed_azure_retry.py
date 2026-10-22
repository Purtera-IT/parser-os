import io
import json
import urllib.error

import app.core.embedding_retrieval as E


def test_a_transient_failure_is_retried(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_EMBED_BASE", "https://example.invalid/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_EMBED_KEY", "k")
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("_ssl.c:999: The handshake operation timed out")
        return Resp(json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    assert E._embed_azure(["a line"]) == [[1.0, 0.0]]
    assert calls["n"] == 2


def test_a_bad_request_is_not_retried(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_EMBED_BASE", "https://example.invalid/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_EMBED_KEY", "k")
    calls = {"n": 0}

    def fake(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    try:
        E._embed_azure(["a line"])
    except urllib.error.HTTPError:
        pass
    assert calls["n"] == 1
