# -*- coding: utf-8 -*-
"""A cached vector must decide the rule, whatever the endpoint is doing.

``SemanticRule.fires`` used to ask ``embedding_endpoint_reachable()`` BEFORE
it embedded anything, so a dead endpoint sent every rule down the regex
fallback -- including for lines whose vectors were already on disk. The
decision moved with the NETWORK rather than with the document.

Measured on live 01491cca with a warm cache, same files, same process:

    embedder up                          980 atoms  cf953ef36db5
    embedder down, cache consulted       980 atoms  cf953ef36db5
    embedder down, cache NOT consulted  1010 atoms  4ba644b0e597

Thirty atoms of a deal moving because a host was unreachable, and every one of
them carrying a different ``label_key``.

The cache is content-addressed by ``model || sha256(text)``, so a cached
vector IS the vector the endpoint would return. Reading it first cannot change
an answer; it can only stop the network from changing one.
"""
from __future__ import annotations

import numpy as np
import pytest

import app.core.embedding_retrieval as embedding_retrieval
from app.core.semantic_rules import SemanticRule


@pytest.fixture()
def rule(monkeypatch) -> SemanticRule:
    r = SemanticRule(
        name="test_rule",
        positives=["Unit Price", "Extended Price"],
        negatives=["Site", "Quantity"],
        threshold=0.5,
        lexical_fallback=lambda t: t.strip().lower() == "lexical only",
    )
    # Fixed prototypes: two positives at +x, two negatives at +y.
    pos = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    neg = np.array([[0.0, 1.0], [0.0, 1.0]], dtype="float32")
    monkeypatch.setattr(SemanticRule, "_protos", lambda self: (pos, neg))
    return r


def _unreachable(monkeypatch) -> None:
    monkeypatch.setattr(SemanticRule, "_reachable", lambda self: False)


def test_a_cached_vector_decides_even_when_the_endpoint_is_down(rule, monkeypatch) -> None:
    _unreachable(monkeypatch)
    monkeypatch.setattr(embedding_retrieval, "cached_embedding",
                        lambda text: [1.0, 0.0])

    def explode(*a, **kw):  # pragma: no cover - must never be reached
        raise AssertionError("went to the network despite a cache hit")

    monkeypatch.setattr(embedding_retrieval, "embed_texts", explode)

    # Nearest prototype is a positive, well clear of the threshold.
    assert rule.fires("Extended Price") is True


def test_a_cached_vector_that_matches_a_negative_does_not_fire(rule, monkeypatch) -> None:
    _unreachable(monkeypatch)
    monkeypatch.setattr(embedding_retrieval, "cached_embedding",
                        lambda text: [0.0, 1.0])
    assert rule.fires("Quantity") is False


def test_the_lexical_fallback_still_guards_a_cache_MISS(rule, monkeypatch) -> None:
    """The reachability gate is still what it was for: not blocking 180s on a
    wedged host for a line nobody has embedded yet."""
    _unreachable(monkeypatch)
    monkeypatch.setattr(embedding_retrieval, "cached_embedding", lambda text: None)

    def explode(*a, **kw):  # pragma: no cover
        raise AssertionError("embedded an uncached line against a dead endpoint")

    monkeypatch.setattr(embedding_retrieval, "embed_texts", explode)

    assert rule.fires("lexical only") is True     # the fallback fired
    assert rule.fires("something else") is False  # and it decided, not the net


def test_the_kill_switch_still_wins(rule, monkeypatch) -> None:
    """SOWSMITH_SEMANTIC_RULES=0 must stay an unconditional lexical path."""
    monkeypatch.setenv("SOWSMITH_SEMANTIC_RULES", "0")

    def explode(text):  # pragma: no cover
        raise AssertionError("consulted the cache despite the kill switch")

    monkeypatch.setattr(embedding_retrieval, "cached_embedding", explode)
    assert rule.fires("lexical only") is True
    assert rule.fires("Extended Price") is False


def test_cached_embedding_never_raises_and_never_blocks(monkeypatch) -> None:
    """It answers "not cached", never an exception, whatever the cache does."""
    class Boom:
        def get_many(self, model, texts):
            raise RuntimeError("sqlite is unhappy")

    monkeypatch.setattr("app.core.embedding_cache.get_cache", lambda: Boom())
    assert embedding_retrieval.cached_embedding("anything") is None
    assert embedding_retrieval.cached_embedding("") is None
