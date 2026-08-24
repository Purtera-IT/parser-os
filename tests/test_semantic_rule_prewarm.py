"""``semantic_rules.prewarm`` collapses N embedding round trips into one.

Each ``SemanticRule.fires`` call embeds a single text, i.e. one HTTP round trip
per candidate line. One RFP PDF asks about ~900 distinct lines, which against a
remote embedder (~2.5 s/call) is ~38 minutes of serial network wait for a parse
that should take seconds — the reason a full-pack compile looked like a hang.
``prewarm`` hands the whole candidate set to ``embed_texts``, which batches its
cache misses into one request, so every later ``fires()`` is a cache hit.

It is a pure cache-filling optimisation: it must never change a decision and
must never raise, whatever the embedder is doing.
"""
from __future__ import annotations

import app.core.semantic_rules as sr


def test_prewarm_sends_every_distinct_candidate_in_one_call(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(sr.SemanticRule, "_disabled", staticmethod(lambda: False))
    import app.core.embedding_retrieval as er
    monkeypatch.setattr(er, "embedding_endpoint_reachable", lambda: True)
    monkeypatch.setattr(er, "embed_texts", lambda texts: calls.append(list(texts)))

    n = sr.prewarm(["Scope of Work", "  Scope of Work  ", "Payment Terms", ""])

    assert len(calls) == 1, "one round trip, not one per line"
    assert calls[0] == ["Scope of Work", "Payment Terms"]
    assert n == 2


def test_prewarm_skips_long_prose(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(sr.SemanticRule, "_disabled", staticmethod(lambda: False))
    import app.core.embedding_retrieval as er
    monkeypatch.setattr(er, "embedding_endpoint_reachable", lambda: True)
    monkeypatch.setattr(er, "embed_texts", lambda texts: calls.append(list(texts)))

    sr.prewarm(["Scope of Work", "x" * (sr._PREWARM_MAX_CHARS + 1)])

    assert calls == [["Scope of Work"]]


def test_prewarm_is_a_noop_when_rules_are_disabled(monkeypatch):
    monkeypatch.setenv("SOWSMITH_SEMANTIC_RULES", "0")
    import app.core.embedding_retrieval as er
    monkeypatch.setattr(er, "embed_texts", lambda texts: (_ for _ in ()).throw(
        AssertionError("must not embed when rules are disabled")))
    assert sr.prewarm(["Scope of Work"]) == 0


def test_prewarm_is_a_noop_when_the_embedder_is_unreachable(monkeypatch):
    monkeypatch.setattr(sr.SemanticRule, "_disabled", staticmethod(lambda: False))
    import app.core.embedding_retrieval as er
    monkeypatch.setattr(er, "embedding_endpoint_reachable", lambda: False)
    monkeypatch.setattr(er, "embed_texts", lambda texts: (_ for _ in ()).throw(
        AssertionError("must not embed when the endpoint is down")))
    assert sr.prewarm(["Scope of Work"]) == 0


def test_prewarm_swallows_embedder_failures(monkeypatch):
    monkeypatch.setattr(sr.SemanticRule, "_disabled", staticmethod(lambda: False))
    import app.core.embedding_retrieval as er
    monkeypatch.setattr(er, "embedding_endpoint_reachable", lambda: True)
    monkeypatch.setattr(er, "embed_texts", lambda texts: (_ for _ in ()).throw(
        RuntimeError("embedder blew up")))
    assert sr.prewarm(["Scope of Work"]) == 0  # a warm-up must never fail a parse


def test_prewarm_caps_the_batch(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(sr.SemanticRule, "_disabled", staticmethod(lambda: False))
    monkeypatch.setattr(sr, "_PREWARM_MAX_TEXTS", 5)
    import app.core.embedding_retrieval as er
    monkeypatch.setattr(er, "embedding_endpoint_reachable", lambda: True)
    monkeypatch.setattr(er, "embed_texts", lambda texts: calls.append(list(texts)))

    assert sr.prewarm([f"line {i}" for i in range(50)]) == 5
    assert len(calls[0]) == 5
