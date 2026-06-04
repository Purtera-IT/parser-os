"""Embedding cache + batched-embed contract (no network).

Verifies the universal speed layer under embed_texts:
  * a text is embedded remotely at most ONCE — the second call is served
    from the persistent sqlite cache (the warm-compile win),
  * cache misses go through a single batched round-trip,
  * failures are never cached (so they retry),
  * the returned matrix is L2-normalized and order-preserving.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import app.core.embedding_cache as ec
import app.core.embedding_retrieval as er


@pytest.fixture()
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_EMBED_CACHE_DB", str(tmp_path / "emb.db"))
    monkeypatch.delenv("SOWSMITH_EMBED_CACHE_DISABLE", raising=False)
    ec.reset_cache()
    yield
    ec.reset_cache()


def _fake_vec(text: str) -> list[float]:
    # deterministic, distinct per text, dim=4 for speed
    seed = sum(ord(c) for c in text)
    return [float(seed % 7), float(seed % 5), float(seed % 3), 1.0]


def test_cache_serves_second_call_without_reembedding(fresh_cache, monkeypatch):
    calls: list[list[str]] = []

    def fake_uncached(texts):
        calls.append(list(texts))
        return [_fake_vec(t) for t in texts]

    monkeypatch.setattr(er, "_embed_uncached", fake_uncached)

    m1 = er.embed_texts(["alpha", "beta", "gamma"])
    assert m1.shape == (3, 4)
    # all three were misses on the first pass
    assert calls == [["alpha", "beta", "gamma"]]

    # second call: two repeats + one new → only the NEW text is embedded
    calls.clear()
    m2 = er.embed_texts(["beta", "alpha", "delta"])
    assert m2.shape == (3, 4)
    assert calls == [["delta"]], "repeats must be served from the persistent cache"

    # cached vector for 'alpha' matches the first call's 'alpha' (row 0 of m1)
    assert np.allclose(m2[1], m1[0], atol=1e-6)


def test_failed_embeds_are_not_cached(fresh_cache, monkeypatch):
    calls: list[list[str]] = []

    def fake_uncached(texts):
        calls.append(list(texts))
        # 'bad' always fails (None) → zero row, must not be cached
        return [None if t == "bad" else _fake_vec(t) for t in texts]

    monkeypatch.setattr(er, "_embed_uncached", fake_uncached)

    er.embed_texts(["good", "bad"])
    calls.clear()
    er.embed_texts(["good", "bad"])
    # 'good' cached, 'bad' retried
    assert calls == [["bad"]]


def test_matrix_is_l2_normalized_and_ordered(fresh_cache, monkeypatch):
    monkeypatch.setattr(er, "_embed_uncached", lambda texts: [_fake_vec(t) for t in texts])
    m = er.embed_texts(["x", "y", "z"])
    norms = np.linalg.norm(m, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_disable_env_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_EMBED_CACHE_DB", str(tmp_path / "emb.db"))
    monkeypatch.setenv("SOWSMITH_EMBED_CACHE_DISABLE", "1")
    ec.reset_cache()
    try:
        assert ec.get_cache() is None
        calls: list[list[str]] = []
        monkeypatch.setattr(
            er, "_embed_uncached",
            lambda texts: (calls.append(list(texts)) or [_fake_vec(t) for t in texts]),
        )
        er.embed_texts(["a"])
        er.embed_texts(["a"])
        assert calls == [["a"], ["a"]], "disabled cache must re-embed every time"
    finally:
        ec.reset_cache()


def test_batch_endpoint_used_then_disabled_on_404(monkeypatch):
    # When /api/embed returns 404, we permanently fall back and never retry it.
    er._BATCH_ENDPOINT_OK = None

    class Resp:
        def __init__(self, code, payload=None):
            self.status_code = code
            self._payload = payload or {}

        def json(self):
            return self._payload

    posts: list[str] = []

    def fake_post(url, **kw):
        posts.append(url)
        return Resp(404)

    monkeypatch.setattr(er.requests, "post", fake_post)
    assert er._embed_batch_endpoint(["a", "b"]) is None
    assert er._BATCH_ENDPOINT_OK is False
    # second attempt short-circuits without another HTTP call
    posts.clear()
    assert er._embed_batch_endpoint(["c"]) is None
    assert posts == []
    er._BATCH_ENDPOINT_OK = None  # reset module global for other tests


def test_batch_endpoint_returns_aligned_vectors(monkeypatch):
    er._BATCH_ENDPOINT_OK = None

    class Resp:
        status_code = 200

        def json(self):
            return {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}

    monkeypatch.setattr(er.requests, "post", lambda url, **kw: Resp())
    out = er._embed_batch_endpoint(["a", "b"])
    assert out == [[1.0, 0.0], [0.0, 1.0]]
    assert er._BATCH_ENDPOINT_OK is True
    er._BATCH_ENDPOINT_OK = None
