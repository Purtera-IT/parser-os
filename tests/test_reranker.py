"""Cross-encoder reranker: retrieve-then-rerank stage-2 precision/recall lift.

The feedback store's bi-encoder (cosine / max-sim) is stage-1 recall — it embeds
query and exemplar independently, so it can mis-order two look-alike forms and
can let a paraphrased ghost slip just under the cosine threshold. A cross-encoder
scores the ``(query, exemplar)`` PAIR jointly, so retrieve-then-rerank can both
(a) re-order the bi-encoder's shortlist and (b) RESCUE a ghost the bi-encoder
under-scored — the ceiling-lift — while its own threshold lets it VETO a
bi-encoder fire it disagrees with.

These tests are hermetic: a content-addressed embedder places each text at a
chosen cosine to the query, and a deterministic ``rerank_fn`` is injected into
the store (no model download, no server, no network). The cross-encoder is
proven to (1) flip the winner, (2) rescue a sub-threshold ghost, (3) veto a
bi-encoder fire, (4) fail open to the bi-encoder when unreachable, and (5) be a
strict no-op when the flag/fn is off.
"""

from __future__ import annotations

import numpy as np
import pytest

import app.core.reranker as reranker
from app.core.decide import DecisionScope
from app.core.feedback_store import Correction, FeedbackStore

_REL = "rr_role"


def _unit(cos: float) -> np.ndarray:
    """2-D unit vector whose cosine to the query basis [1,0] is exactly ``cos``."""
    return np.array([cos, float(np.sqrt(max(0.0, 1.0 - cos * cos)))], dtype=np.float32)


def _make_embed(table: dict[str, np.ndarray], dim: int = 2):
    """Map known texts to chosen vectors; everything else is orthogonal to the
    query (cosine 0), so only the texts we place can match."""
    def emb(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            v = table.get(t.strip().lower())
            out[i] = v if v is not None else np.array([0.0, 1.0], dtype=np.float32)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n > 1e-9, n, 1.0)
    return emb


def _store(table, rerank_fn=None) -> FeedbackStore:
    s = FeedbackStore(
        ":memory:",
        embed_fn=_make_embed(table),
        reachable_fn=lambda: True,
        rerank_fn=rerank_fn,
    )
    # Isolate the bi-encoder / rerank path: the neural head (>=2 verdicts) is a
    # separate earlier tier we are not exercising here.
    s._enable_head = False
    return s


def _resolve(store: FeedbackStore, text: str, candidates: list[str]):
    return store.resolve(
        relation=_REL, text=text, candidates=candidates,
        context="", scope=DecisionScope(), instruction="", relations=None,
    )


# ── reranker module ────────────────────────────────────────────────────


def test_set_reranker_override_roundtrips():
    reranker.set_reranker(lambda q, docs: [0.7] * len(docs))
    try:
        assert reranker.rerank("x", ["a", "b", "c"]) == [0.7, 0.7, 0.7]
        assert reranker.rerank("x", []) == []
    finally:
        reranker.set_reranker(None)


def test_override_length_mismatch_is_none():
    reranker.set_reranker(lambda q, docs: [0.9])  # wrong length
    try:
        assert reranker.rerank("x", ["a", "b"]) is None
    finally:
        reranker.set_reranker(None)


def test_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("SOWSMITH_NEURAL_RERANK", raising=False)
    reranker.set_reranker(None)
    assert reranker.rerank("x", ["a"]) is None


def test_normalize_squashes_logits_passes_probs():
    # already-in-[0,1] → passthrough; out-of-range logits → sigmoid-squashed.
    assert reranker._normalize([0.1, 0.9]) == [0.1, 0.9]
    out = reranker._normalize([-4.0, 4.0])
    assert all(0.0 <= s <= 1.0 for s in out) and out[1] > 0.5 > out[0]


# ── store wiring ────────────────────────────────────────────────────────


def test_rerank_flips_the_winner():
    """Bi-encoder ranks the WRONG correction first (higher cosine); the
    cross-encoder re-orders the shortlist and the RIGHT verdict wins."""
    table = {
        "q": _unit(1.0),
        "exa": _unit(0.90),   # bi-encoder favors A
        "exb": _unit(0.85),   # B is the correct one
    }
    corrs = (
        Correction(id="A", relation=_REL, verdict="wrong", created_by="pm",
                   exemplars=["exa"]),
        Correction(id="B", relation=_REL, verdict="right", created_by="pm",
                   exemplars=["exb"]),
    )
    cands = ["wrong", "right"]

    # Bi-encoder only: higher cosine (A) wins → wrong verdict.
    s0 = _store(table)
    for c in corrs:
        s0.add(c)
    d0 = _resolve(s0, "q", cands)
    assert d0 is not None and d0.verdict == "wrong"

    # Rerank scores B above A → right verdict, with cross-encoder provenance.
    rr = lambda q, docs: [0.95 if d == "exb" else 0.10 for d in docs]
    s1 = _store(table, rerank_fn=rr)
    for c in corrs:
        s1.add(c)
    d1 = _resolve(s1, "q", cands)
    assert d1 is not None and d1.verdict == "right"
    assert "reranked" in (d1.rationale or "") and "cross-encoder" in (d1.rationale or "")
    assert d1.confidence == pytest.approx(0.95)


def test_rerank_rescues_sub_threshold_ghost():
    """Bi-encoder scores the ghost below 0.82 (abstain → keep); the
    cross-encoder rescues it (recall lift) and the drop fires."""
    table = {"q": _unit(1.0), "g": _unit(0.78)}  # 0.78 < 0.82 default threshold
    corr = Correction(id="G", relation=_REL, verdict="drop", created_by="pm",
                      exemplars=["g"])

    s0 = _store(table)
    s0.add(corr)
    assert _resolve(s0, "q", ["drop"]) is None  # bi-encoder abstains

    rr = lambda q, docs: [0.93 for _ in docs]
    s1 = _store(table, rerank_fn=rr)
    s1.add(corr)
    d = _resolve(s1, "q", ["drop"])
    assert d is not None and d.verdict == "drop"
    assert d.confidence == pytest.approx(0.93)


def test_rerank_vetoes_biencoder_fire():
    """Bi-encoder would fire (cosine 0.95 ≥ 0.82) but the cross-encoder scores
    below its threshold → veto → abstain (precision guard)."""
    table = {"q": _unit(1.0), "g": _unit(0.95)}
    corr = Correction(id="G", relation=_REL, verdict="drop", created_by="pm",
                      exemplars=["g"])

    s0 = _store(table)
    s0.add(corr)
    assert _resolve(s0, "q", ["drop"]).verdict == "drop"  # bi-encoder fires

    rr = lambda q, docs: [0.20 for _ in docs]  # below default 0.5
    s1 = _store(table, rerank_fn=rr)
    s1.add(corr)
    assert _resolve(s1, "q", ["drop"]) is None  # vetoed


def test_rerank_unreachable_falls_open_to_biencoder():
    """``rerank_fn`` returning None == reranker unreachable → fail open to the
    bi-encoder path (never lose a confident bi-encoder hit)."""
    table = {"q": _unit(1.0), "g": _unit(0.95)}
    corr = Correction(id="G", relation=_REL, verdict="drop", created_by="pm",
                      exemplars=["g"])
    s = _store(table, rerank_fn=lambda q, docs: None)
    s.add(corr)
    d = _resolve(s, "q", ["drop"])
    assert d is not None and d.verdict == "drop"
    assert "matched correction" in (d.rationale or "")  # bi-encoder rationale


def test_flag_off_is_pure_biencoder():
    """No rerank_fn and flag unset → byte-identical bi-encoder behavior."""
    table = {"q": _unit(1.0), "g": _unit(0.95)}
    s = _store(table)
    assert s._rerank_active() is False
    s.add(Correction(id="G", relation=_REL, verdict="drop", created_by="pm",
                     exemplars=["g"]))
    d = _resolve(s, "q", ["drop"])
    assert d is not None and d.verdict == "drop"
