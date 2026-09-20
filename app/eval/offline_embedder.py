"""A deterministic, offline embedder for gates and fixtures.

Hashed bag of lower-cased tokens, L2-normalised. It has no notion of meaning:
two sentences are close exactly when they share words. That makes it the right
instrument for the transfer gate -- it cannot hide a wording key behind
semantic similarity, so "matches wording" and "matches work" separate cleanly.

It is NOT a stand-in for the production embedder when measuring real transfer;
a real-pair re-measure must use the pipeline's embedder.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

DIM = 1024
_TOKEN_RE = re.compile(r"[a-z0-9_:.\-]+")


def _bucket(token: str) -> int:
    return int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % DIM


def hash_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in _TOKEN_RE.findall(str(t or "").lower()):
            out[i, _bucket(tok)] += 1.0
        n = float(np.linalg.norm(out[i]))
        if n > 0:
            out[i] /= n
    return out


def offline_store(db_path: str = ":memory:"):
    """A FeedbackStore wired to the offline embedder, always reachable, no
    neural head, no reranker -- nearest-exemplar resolution only."""
    from app.core.feedback_store import FeedbackStore

    store = FeedbackStore(db_path, embed_fn=hash_embed, reachable_fn=lambda: True)
    store._enable_head = False
    store._enable_rerank = False
    return store


__all__ = ["DIM", "hash_embed", "offline_store"]
