"""Cross-encoder reranker — the precision/recall ceiling-lift over the bi-encoder.

The feedback store retrieves candidate corrections with a **bi-encoder**
(qwen3-embedding:8b) via cosine / max-sim. A bi-encoder embeds the query and
each exemplar *independently*, so it cannot model token-level interaction
between them: two short surface forms that look similar in isolation can be
conflated, and a paraphrased ghost can land just under the cosine threshold and
escape. A **cross-encoder** scores the ``(query, exemplar)`` PAIR jointly in one
forward pass — the model's attention sees both texts at once — which is strictly
more discriminating for the short, easily-confused entity tokens this store
deals in (site slugs, vendor names, role phrases).

Retrieve-then-rerank: the bi-encoder casts a wide net (cheap, high recall), the
cross-encoder re-scores the top-k (expensive, high precision). This is the
standard two-stage IR pattern and it is **one-sided safe** here — the
cross-encoder only confirms or vetoes what the bi-encoder surfaced, and carries
its own threshold, so it never invents a match from nothing.

Ollama has no rerank endpoint, so the cross-encoder is served *separately*. Two
universal backends, selected by ``SOWSMITH_RERANK_BACKEND``:

  * ``"st"`` (default): a local ``sentence_transformers.CrossEncoder``. No extra
    infra — runs on CPU, model pulled once from HF and cached on disk. Right for
    dev / a single box / CI.
  * ``"http"``: POST to a dedicated rerank server — HF TEI, infinity, or
    ``llama.cpp --reranking`` — at ``SOWSMITH_RERANK_URL``. Right for prod / GPU
    serving / horizontal scale. Response shapes from all three are handled.

Everything is behind ``SOWSMITH_NEURAL_RERANK`` (off by default), so the
pipeline is byte-identical until a deploy opts in. Scores are squashed to
``[0,1]`` (sigmoid of the raw relevance logit when a server returns logits) so a
single ``SOWSMITH_RERANK_THRESHOLD`` applies across every backend and model.

This module is **universal**: no deal-specific names, no regex, no keyword lists
— it scores arbitrary ``(query, document)`` text pairs with a learned model.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_DEFAULT_HTTP_TIMEOUT = 30

# Optional override hook. Tests install a deterministic scorer here; in prod the
# backend is chosen by env and this stays None.
_OVERRIDE: Optional[Callable[[str, list[str]], Optional[list[float]]]] = None

# Lazily-loaded sentence-transformers CrossEncoder singleton (model load is
# expensive — do it once per process).
_ST_MODEL = None
_ST_FAILED = False


# ────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────


def set_reranker(fn: Optional[Callable[[str, list[str]], Optional[list[float]]]]) -> None:
    """Install (or clear, with ``None``) a reranker function. Used by tests to
    inject a deterministic scorer without standing up a model or a server."""
    global _OVERRIDE
    _OVERRIDE = fn


def enabled() -> bool:
    return os.getenv("SOWSMITH_NEURAL_RERANK", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def backend() -> str:
    return os.getenv("SOWSMITH_RERANK_BACKEND", "st").strip().lower()


def threshold() -> float:
    try:
        return float(os.getenv("SOWSMITH_RERANK_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


def top_k() -> int:
    try:
        return max(1, int(os.getenv("SOWSMITH_RERANK_TOPK", "20")))
    except ValueError:
        return 20


def available() -> bool:
    """True iff a rerank call can actually run right now (override installed, or
    the configured backend is reachable). Lets the caller skip the wide-net
    retrieval entirely when reranking can't happen."""
    if _OVERRIDE is not None:
        return True
    if not enabled():
        return False
    b = backend()
    if b == "st":
        return _load_st() is not None
    if b == "http":
        return bool(os.getenv("SOWSMITH_RERANK_URL"))
    return False


# ────────────────────────────────────────────────────────────────────
# SCORE NORMALIZATION
# ────────────────────────────────────────────────────────────────────


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize(scores: list[float]) -> list[float]:
    """Map raw scores to [0,1]. Servers that already apply sigmoid return values
    in [0,1] (pass through); models that return raw logits get squashed."""
    if not scores:
        return scores
    if all(0.0 <= s <= 1.0 for s in scores):
        return [float(s) for s in scores]
    return [_sigmoid(float(s)) for s in scores]


# ────────────────────────────────────────────────────────────────────
# BACKENDS
# ────────────────────────────────────────────────────────────────────


def _load_st():
    global _ST_MODEL, _ST_FAILED
    if _ST_MODEL is not None:
        return _ST_MODEL
    if _ST_FAILED:
        return None
    try:
        from sentence_transformers import CrossEncoder
        model_name = os.getenv("SOWSMITH_RERANK_MODEL", _DEFAULT_MODEL)
        _ST_MODEL = CrossEncoder(model_name)
        logger.info("reranker: loaded CrossEncoder %s", model_name)
        return _ST_MODEL
    except Exception as e:  # offline / model missing / no sentence-transformers
        logger.warning("reranker: st backend unavailable: %s", e)
        _ST_FAILED = True
        return None


def _rerank_st(query: str, documents: list[str]) -> Optional[list[float]]:
    model = _load_st()
    if model is None:
        return None
    try:
        scores = model.predict([(query, d) for d in documents])
        return _normalize([float(s) for s in scores])
    except Exception as e:
        logger.warning("reranker: st predict failed: %s", e)
        return None


def _rerank_http(query: str, documents: list[str]) -> Optional[list[float]]:
    url = os.getenv("SOWSMITH_RERANK_URL")
    if not url:
        return None
    import requests
    timeout = int(os.getenv("SOWSMITH_RERANK_TIMEOUT", str(_DEFAULT_HTTP_TIMEOUT)))
    base = url.rstrip("/")
    endpoint = base if base.endswith("rerank") else base + "/rerank"
    try:
        r = requests.post(
            endpoint,
            json={"query": query, "texts": documents, "documents": documents},
            timeout=timeout,
        )
        if r.status_code != 200:
            logger.warning("reranker: http %s -> %s", endpoint, r.status_code)
            return None
        data = r.json()
        return _parse_http(data, len(documents))
    except Exception as e:
        logger.warning("reranker: http call failed: %s", e)
        return None


def _parse_http(data, n: int) -> Optional[list[float]]:
    """Map the response of a rerank server back to input order. Handles:
      * TEI:       ``[{"index": i, "score": s}, ...]``
      * infinity:  ``{"results": [{"index": i, "relevance_score": s}, ...]}``
      * llama.cpp: ``{"results": [{"index": i, "relevance_score": s}, ...]}``
    """
    rows = data.get("results") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None
    out: list[Optional[float]] = [None] * n
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        score = row.get("score", row.get("relevance_score"))
        if isinstance(idx, int) and 0 <= idx < n and score is not None:
            out[idx] = float(score)
    if any(v is None for v in out):
        return None
    return _normalize([float(v) for v in out])  # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────────────────────────


def rerank(query: str, documents: list[str]) -> Optional[list[float]]:
    """Relevance of each document to ``query``, in ``[0,1]``, aligned to input
    order. Returns ``None`` when reranking is disabled or the backend is
    unreachable — the caller then falls back to the bi-encoder score
    (fail-open; never a guess). Returns ``[]`` for an empty document list."""
    if not documents:
        return []
    if _OVERRIDE is not None:
        try:
            out = _OVERRIDE(query, documents)
        except Exception:
            return None
        if out is None:
            return None
        try:
            scored = [float(x) for x in out]
        except (TypeError, ValueError):
            return None
        if len(scored) != len(documents):
            return None
        return scored
    if not enabled():
        return None
    b = backend()
    if b == "st":
        return _rerank_st(query, documents)
    if b == "http":
        return _rerank_http(query, documents)
    return None


__all__ = [
    "rerank",
    "set_reranker",
    "available",
    "enabled",
    "backend",
    "threshold",
    "top_k",
]
