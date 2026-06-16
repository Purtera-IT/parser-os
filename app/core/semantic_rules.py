"""Semantic rules — fire a fuzzy *linguistic* judgment by embedding similarity
instead of a keyword regex, so it generalizes to phrasings nobody wrote a keyword
for ("the vendor's responsibilities encompass:" fires the same as "...the
following services.").

A rule is a small set of POSITIVE prototype phrases (things that SHOULD fire) and
NEGATIVE ones (look similar but should NOT). At call time we embed the candidate
and fire iff its nearest prototype is a positive whose cosine clears ``threshold``.

Design principles:
  * STRUCTURE stays structural. This is only for linguistic judgments (is-this-a
    -lead-in / exclusion / boilerplate / section-type). Don't use it for things a
    flag already answers (hidden column, numPr list item, sheet role by shape).
  * SAFE OFFLINE. The qwen3 embedder lives on a box that sleeps/relays. If it is
    unreachable we fall back to the rule's ``lexical_fallback`` (the old regex),
    so a parse NEVER breaks or silently changes behaviour when embeddings are down.
  * SELF-HEALING. ``positives``/``negatives`` are just example lists — a PM/intern
    correction becomes a new example (no new regex), and the rule's behaviour shifts.
  * CHEAP. Prototypes embed once (process-cached); candidates hit the existing
    per-text embedding cache, and callers only ask about structurally-gated
    candidates, so the round-trips are bounded.
"""
from __future__ import annotations

import os
from typing import Callable, Sequence

_PROTO_CACHE: dict[str, object] = {}  # rule-name -> (pos_matrix, neg_matrix)


def _np():
    import numpy as np  # local import keeps parser import light
    return np


class SemanticRule:
    def __init__(
        self,
        name: str,
        positives: Sequence[str],
        negatives: Sequence[str] = (),
        threshold: float = 0.62,
        lexical_fallback: Callable[[str], bool] | None = None,
    ) -> None:
        self.name = name
        self.positives = list(positives)
        self.negatives = list(negatives)
        self.threshold = threshold
        self.lexical_fallback = lexical_fallback

    # -- env switches -----------------------------------------------------
    @staticmethod
    def _disabled() -> bool:
        # global kill-switch: force the lexical fallback everywhere (CI / offline
        # determinism / debugging a regression to the embedder).
        return os.environ.get("SOWSMITH_SEMANTIC_RULES", "1") == "0"

    def _reachable(self) -> bool:
        try:
            from app.core.embedding_retrieval import embedding_endpoint_reachable
            return bool(embedding_endpoint_reachable())
        except Exception:
            return False

    # -- prototype embedding (cached) -------------------------------------
    def _protos(self):
        cached = _PROTO_CACHE.get(self.name)
        if cached is not None:
            return cached
        from app.core.embedding_retrieval import embed_texts
        np = _np()
        texts = self.positives + self.negatives
        vecs = np.array(embed_texts(texts), dtype="float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        pos = vecs[: len(self.positives)]
        neg = vecs[len(self.positives) :]
        _PROTO_CACHE[self.name] = (pos, neg)
        return pos, neg

    # -- the decision -----------------------------------------------------
    def fires(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        # offline / disabled -> deterministic lexical fallback (never break a parse)
        if self._disabled() or not self._reachable():
            return bool(self.lexical_fallback(text)) if self.lexical_fallback else False
        try:
            from app.core.embedding_retrieval import embed_texts
            np = _np()
            pos, neg = self._protos()
            q = np.array(embed_texts([text])[0], dtype="float32")
            q /= np.linalg.norm(q) + 1e-9
            best_pos = float((pos @ q).max())
            best_neg = float((neg @ q).max()) if len(neg) else -1.0
            # fire iff the nearest prototype is a POSITIVE and it clears the floor
            return best_pos >= self.threshold and best_pos > best_neg
        except Exception:
            return bool(self.lexical_fallback(text)) if self.lexical_fallback else False

    def score(self, text: str) -> tuple[float, float]:
        """(nearest-positive cosine, nearest-negative cosine) — for calibration."""
        from app.core.embedding_retrieval import embed_texts
        np = _np()
        pos, neg = self._protos()
        q = np.array(embed_texts([text])[0], dtype="float32")
        q /= np.linalg.norm(q) + 1e-9
        bp = float((pos @ q).max())
        bn = float((neg @ q).max()) if len(neg) else -1.0
        return bp, bn
