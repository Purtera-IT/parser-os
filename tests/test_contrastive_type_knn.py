"""Runtime contrastive kNN: math + guess-free guarantees, no GPU/network.

Injected synthetic embed_fn so the store geometry is fully controlled. 4-D so an
OOD query can point into a dimension the store doesn't occupy (with L2-normalized
vectors, OOD shows up as low cosine to the nearest neighbor, not low magnitude).
"""
import numpy as np

from app.core.contrastive_type_knn import ContrastiveTypeKNN

# store occupies dims 0-2; dim 3 is the "unseen" direction for OOD tests.
_DIRS = {"keep": [1, 0, 0, 0], "site": [0, 1, 0, 0], "work": [0, 0, 1, 0]}


def _toy_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(4, dtype=np.float32)
        for kw, d in _DIRS.items():
            if kw in t.lower():
                v = v + np.array(d, dtype=np.float32)
        if not v.any():
            v = np.array([0.3, 0.3, 0.3, 0.0], dtype=np.float32)
        out.append(v)
    return np.array(out, dtype=np.float32)


def _const(vec):
    return lambda texts: np.array([vec] * len(texts), dtype=np.float32)


def _store(k=5):
    texts = (["keep clause"] * 8) + (["site address"] * 8) + (["work task"] * 8)
    labels = (["_keep"] * 8) + (["SITE"] * 8) + (["WORK"] * 8)
    emb = _toy_embed(texts)
    return ContrastiveTypeKNN(
        emb=emb / np.linalg.norm(emb, axis=1, keepdims=True),
        y=np.array(labels), text=np.array(texts),
        k=k, sim_floor=0.55, tau=0.30, mode="unified", embed_fn=_toy_embed,
    )


def test_confident_keep_classifies():
    res = _store().classify("keep this boilerplate")
    assert res is not None and res[0] == "_keep" and res[1] >= 0.30


def test_confident_facet_classifies():
    res = _store().classify("site address line")
    assert res is not None and res[0] == "SITE"


def test_ood_abstains():
    """Query points into the unseen dim 3 -> cosine 0 to every store point ->
    top-1 below sim_floor -> abstain (caller falls back to LLM)."""
    ck = _store()
    ck.embed_fn = _const([0.0, 0.0, 0.0, 1.0])
    assert ck.classify("anything") is None


def test_low_margin_abstains():
    """Equidistant between SITE and WORK with k spanning both full clusters ->
    vote margin ~0 -> abstain even though in-distribution (high top-1)."""
    ck = _store(k=16)
    ck.embed_fn = _const([0.0, 1.0, 1.0, 0.0])
    assert ck.classify("ambiguous") is None


def test_instant_learning_append_lifts_from_abstain():
    """A novel shape abstains (OOD); after teaching 3 exemplars there it
    classifies — the correction lands on the NEXT atom, no retrain."""
    novel = _const([0.0, 0.0, 0.0, 1.0])
    ck = _store()
    ck.embed_fn = novel
    assert ck.classify("novel shape") is None          # OOD -> abstain
    for _ in range(3):
        ck.append("novel marker", "SITE", persist=False)
    after = ck.classify("novel shape")
    assert after is not None and after[0] == "SITE"
    assert len(ck.y) == 27


def test_abstain_when_no_embedder():
    ck = _store()
    ck.embed_fn = None
    assert ck.classify("anything") is None
    assert ck.classify_batch(["a", "b"]) == [None, None]
