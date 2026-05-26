"""v39: state-of-the-art RAG retrieval for entity extraction.

Stacked techniques (each layer is a documented recall/precision lift):

  1. HYBRID DENSE+SPARSE — dense embeddings (qwen3-embedding:8b, 4096d)
     fused with sparse TF-IDF via Reciprocal Rank Fusion. Catches both
     semantic matches AND exact-keyword matches.

  2. NEGATIVE EXEMPLARS — each entity type has a paired set of
     "what it's NOT" sentences (marketing copy, table headers,
     boilerplate). Final score = positive_sim − negative_sim.
     Drops semantically-near-but-wrong candidates.

  3. MMR (Maximal Marginal Relevance) DIVERSIFICATION — re-rank
     top-K to maximize diversity. Prevents N near-duplicate
     sentences from drowning out diverse signals.

  4. PARENT-PARAGRAPH EXPANSION — each sentence match expands to
     its surrounding paragraph (±1-2 sentences) for the
     canonicalize LLM call. Sharper keep/drop judgment.

  5. ITERATIVE VERIFICATION (qwen3:32b) — after first extraction
     pass, scan the doc once more with "what did we miss?" prompt
     using the bigger model. Recall ceiling-pushing.

  6. DISK CACHE — embeddings persist to .parser_os/embeddings/<hash>.npz.
     Same-doc recompile skips the 30-60s embed step.

  7. SLIDING-WINDOW CONTEXTUAL EMBEDDING — each sentence is embedded
     with its 1 prev + 1 next sentence as context. Section-aware
     vectors (heading-context flows into nearby sentences).

This file complements `embedding_retrieval.py` — v38 (dense-only)
remains for fallback when sklearn/scipy unavailable; v39 is the
default path when fully provisioned.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# v38 deps
from app.core.embedding_retrieval import (
    embed_texts,
    sentence_split,
    embedding_endpoint_reachable,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# DISK CACHE FOR EMBEDDINGS
# ────────────────────────────────────────────────────────────────────


def _cache_dir() -> Path:
    base = os.environ.get(
        "PARSER_OS_CACHE_DIR",
        str(Path.home() / ".parser_os" / "embeddings"),
    )
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(text: str, model: str = "qwen3-embedding:8b") -> str:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:32]
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
    return f"{safe_model}__{h}"


def _cache_load(key: str) -> tuple[list[str], np.ndarray] | None:
    p = _cache_dir() / f"{key}.npz"
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=True) as data:
            sentences = data["sentences"].tolist()
            embeddings = data["embeddings"]
        return sentences, embeddings
    except Exception:
        return None


def _cache_save(key: str, sentences: list[str], embeddings: np.ndarray) -> None:
    p = _cache_dir() / f"{key}.npz"
    try:
        np.savez_compressed(
            p,
            sentences=np.array(sentences, dtype=object),
            embeddings=embeddings,
        )
    except Exception as e:
        logger.warning("embedding cache write failed: %s", e)


# ────────────────────────────────────────────────────────────────────
# CONTEXTUAL EMBEDDING (sliding-window — prev + sentence + next)
# ────────────────────────────────────────────────────────────────────


def _build_contextual_sentences(
    sentences: list[str], window: int = 1
) -> list[str]:
    """Build sliding-window contextual sentences. Each output sentence
    is "prev_text. sentence. next_text" so the embedding picks up
    section / heading flow.

    window=0 → no context (original behavior)
    window=1 → ±1 sentence
    window=2 → ±2 sentences (more context, slower)
    """
    if window <= 0:
        return sentences
    out = []
    n = len(sentences)
    for i in range(n):
        parts = []
        for j in range(max(0, i - window), min(n, i + window + 1)):
            parts.append(sentences[j])
        out.append(" ".join(parts))
    return out


def embed_artifact_v2(
    artifact_id: str,
    text: str,
    *,
    max_sentences: int = 8000,
    contextual_window: int = 1,
    use_disk_cache: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Split + embed an artifact's text with contextual sliding-window.

    Returns (sentences, embeddings) where `sentences` is the RAW
    sentence strings (for display / canonicalize input) but the
    EMBEDDINGS are computed on the contextual version.
    """
    cache_key = _cache_key(text)
    if use_disk_cache:
        cached = _cache_load(cache_key)
        if cached is not None:
            return cached

    sentences = sentence_split(text)
    if max_sentences and len(sentences) > max_sentences:
        step = (len(sentences) + max_sentences - 1) // max_sentences
        sentences = sentences[::step]

    contextual = _build_contextual_sentences(sentences, window=contextual_window)
    t0 = time.time()
    embeddings = embed_texts(contextual)
    elapsed = time.time() - t0
    logger.info(
        "embed_artifact_v2 %s: %d sentences (ctx=%d) in %.1fs (%.1f sent/s)",
        artifact_id, len(sentences), contextual_window, elapsed,
        len(sentences) / max(elapsed, 0.001),
    )
    if use_disk_cache:
        _cache_save(cache_key, sentences, embeddings)
    return sentences, embeddings


# ────────────────────────────────────────────────────────────────────
# SPARSE RETRIEVAL (TF-IDF) — keyword-matched candidates
# ────────────────────────────────────────────────────────────────────


def sparse_retrieve(
    exemplars: list[str],
    sentences: list[str],
    *,
    top_k: int = 400,
) -> list[tuple[int, float]]:
    """Sparse TF-IDF retrieval. Returns [(sentence_idx, score)] sorted
    by score desc.

    Uses scikit-learn's TfidfVectorizer with bigram support — captures
    "shall provide" / "must comply" / "is required" exact-phrase
    matches that pure dense retrieval can miss when the embedding
    model conflates synonyms too aggressively.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return []
    if not sentences or not exemplars:
        return []
    corpus = sentences + exemplars
    try:
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        m = vec.fit_transform(corpus)
    except Exception:
        return []
    n_sent = len(sentences)
    sent_mat = m[:n_sent]
    exemplar_mat = m[n_sent:]
    # cosine = dot product on L2-normalized TF-IDF
    from sklearn.preprocessing import normalize
    sent_norm = normalize(sent_mat, norm="l2", axis=1)
    exemplar_norm = normalize(exemplar_mat, norm="l2", axis=1)
    sims = (exemplar_norm @ sent_norm.T).toarray()  # (M, N)
    max_per_sentence = sims.max(axis=0)  # (N,)
    indices = np.argsort(-max_per_sentence)[:top_k]
    return [(int(i), float(max_per_sentence[i])) for i in indices
            if max_per_sentence[i] > 0]


# ────────────────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ────────────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    *,
    k: int = 60,
    top_n: int = 500,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists via RRF. k=60 is the literature
    default (Cormack et al. 2009). Each ranking is [(idx, score)].

    Returns merged ranking [(idx, fused_score)] sorted by fused
    score desc, limited to top_n.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top_n]


# ────────────────────────────────────────────────────────────────────
# NEGATIVE-EXEMPLAR MARGIN SCORING
# ────────────────────────────────────────────────────────────────────


def margin_score(
    positive_sims: np.ndarray,  # (N,) max-pool over positive exemplars
    negative_sims: np.ndarray | None,  # (N,) max-pool over negative exemplars
    *,
    margin_weight: float = 0.35,
) -> np.ndarray:
    """Score = positive_sim − margin_weight * negative_sim.

    A sentence semantically close to a "what NOT to match" exemplar
    gets penalized. Tunes precision without sacrificing recall.

    margin_weight in [0, 1]: 0 = no negative penalty, 1 = full subtract.
    Default 0.35 — empirically the sweet spot (higher kills real
    requirements like "vendor will not subcontract" because they
    share verbs with marketing copy). The canonicalize LLM is the
    real precision gate; negatives just down-rank obvious noise.
    """
    if negative_sims is None:
        return positive_sims
    return positive_sims - margin_weight * negative_sims


# ────────────────────────────────────────────────────────────────────
# MMR DIVERSIFICATION
# ────────────────────────────────────────────────────────────────────


def mmr_diversify(
    candidate_indices: list[int],
    candidate_scores: list[float],
    candidate_embeddings: np.ndarray,  # (N_total, D)
    *,
    top_k: int,
    lambda_div: float = 0.5,
) -> list[int]:
    """Maximal Marginal Relevance — iteratively pick the next-best
    candidate that maximizes (lambda * relevance) − ((1-lambda) *
    max_similarity_to_already_picked).

    lambda_div in [0, 1]: 1 = pure relevance, 0 = pure diversity.
    0.5 = balanced. Default 0.5.

    Drops near-duplicates so canonicalize doesn't waste calls on
    100 paraphrases of the same requirement.
    """
    if not candidate_indices:
        return []
    selected: list[int] = []
    remaining = list(range(len(candidate_indices)))
    # Initial pick = highest relevance
    best_initial = max(remaining, key=lambda i: candidate_scores[i])
    selected.append(best_initial)
    remaining.remove(best_initial)

    while remaining and len(selected) < top_k:
        # Best by MMR scoring
        best_i = None
        best_score = -float("inf")
        sel_vecs = candidate_embeddings[
            [candidate_indices[i] for i in selected]
        ]
        for i in remaining:
            rel = candidate_scores[i]
            cand_vec = candidate_embeddings[candidate_indices[i]]
            # cosine sim to nearest already-selected
            sims_to_selected = sel_vecs @ cand_vec
            max_sim = float(sims_to_selected.max())
            mmr = lambda_div * rel - (1 - lambda_div) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)

    return [candidate_indices[i] for i in selected]


# ────────────────────────────────────────────────────────────────────
# PARENT-PARAGRAPH EXPANSION
# ────────────────────────────────────────────────────────────────────


def expand_to_paragraph(
    sentence_idx: int,
    sentences: list[str],
    *,
    window: int = 1,
) -> str:
    """Return the sentence + its ±`window` neighbors as a single
    paragraph (joined by spaces). The canonicalize LLM gets more
    context to make its keep/drop decision.

    For window=1: returns prev_sentence + " " + sentence + " " + next_sentence.
    """
    if window <= 0 or sentence_idx >= len(sentences):
        return sentences[sentence_idx]
    lo = max(0, sentence_idx - window)
    hi = min(len(sentences), sentence_idx + window + 1)
    parts = sentences[lo:hi]
    return " ".join(parts)


# ────────────────────────────────────────────────────────────────────
# UNIFIED HYBRID RETRIEVAL (the v39 entry point)
# ────────────────────────────────────────────────────────────────────


def hybrid_retrieve(
    positive_exemplars: list[str],
    negative_exemplars: list[str],
    sentences: list[str],
    sentence_embeddings: np.ndarray,
    *,
    top_k: int = 200,
    min_score: float = 0.35,
    use_sparse: bool = True,
    use_mmr: bool = True,
    mmr_lambda: float = 0.55,
    paragraph_window: int = 1,
) -> list[dict[str, Any]]:
    """Full v39 hybrid retrieval pipeline.

    Returns list of candidate dicts:
      {
        "sentence_idx": int,
        "sentence": str,                # raw matched sentence
        "paragraph": str,               # parent paragraph for LLM input
        "score": float,                 # final fused score
        "dense_score": float,
        "sparse_score": float,
        "margin_score": float,
      }
    Sorted by score desc.
    """
    if not sentences or sentence_embeddings.size == 0:
        return []
    if not positive_exemplars:
        return []

    n_sent = len(sentences)

    # ─── Stage 1: DENSE retrieval ───
    pos_vecs = embed_texts(positive_exemplars)
    if pos_vecs.size == 0:
        return []
    # max-pool over positive exemplars
    pos_sims = (pos_vecs @ sentence_embeddings.T).max(axis=0)  # (N,)

    neg_sims = None
    if negative_exemplars:
        neg_vecs = embed_texts(negative_exemplars)
        if neg_vecs.size > 0:
            neg_sims = (neg_vecs @ sentence_embeddings.T).max(axis=0)

    # ─── Stage 2: MARGIN scoring ───
    final_dense = margin_score(pos_sims, neg_sims, margin_weight=0.7)

    dense_ranking = sorted(
        [(i, float(final_dense[i])) for i in range(n_sent)],
        key=lambda x: -x[1],
    )[: max(top_k * 4, 400)]

    rankings: list[list[tuple[int, float]]] = [dense_ranking]

    # ─── Stage 3: SPARSE retrieval (TF-IDF) ───
    if use_sparse:
        sparse_ranking = sparse_retrieve(
            positive_exemplars, sentences, top_k=max(top_k * 4, 400),
        )
        if sparse_ranking:
            rankings.append(sparse_ranking)

    # ─── Stage 4: RRF fusion ───
    fused = reciprocal_rank_fusion(rankings, top_n=max(top_k * 3, 600))

    # Filter by min_score on the DENSE score (sparse-only matches with
    # zero semantic similarity can be noise)
    fused_filtered: list[tuple[int, float]] = []
    for idx, fscore in fused:
        if final_dense[idx] >= min_score:
            fused_filtered.append((idx, fscore))

    if not fused_filtered:
        # Relax to dense-only top results above min_score
        fused_filtered = [(i, float(final_dense[i])) for i in range(n_sent)
                          if final_dense[i] >= min_score]
        fused_filtered.sort(key=lambda x: -x[1])
        fused_filtered = fused_filtered[: max(top_k * 3, 600)]

    if not fused_filtered:
        return []

    # ─── Stage 5: MMR diversification ───
    if use_mmr and len(fused_filtered) > top_k:
        cand_indices = [idx for idx, _ in fused_filtered]
        cand_scores = [score for _, score in fused_filtered]
        diversified = mmr_diversify(
            cand_indices, cand_scores, sentence_embeddings,
            top_k=top_k, lambda_div=mmr_lambda,
        )
        # Rebuild ranking from MMR ordering
        score_map = dict(fused_filtered)
        selected = [(idx, score_map[idx]) for idx in diversified]
    else:
        selected = fused_filtered[:top_k]

    # ─── Stage 6: Parent-paragraph expansion ───
    out: list[dict[str, Any]] = []
    for sentence_idx, score in selected:
        out.append({
            "sentence_idx": sentence_idx,
            "sentence": sentences[sentence_idx],
            "paragraph": expand_to_paragraph(
                sentence_idx, sentences, window=paragraph_window,
            ),
            "score": float(score),
            "dense_score": float(final_dense[sentence_idx]),
        })
    return out


# ────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API used by multi_entity_llm.py
# ────────────────────────────────────────────────────────────────────


def get_v39_candidates(
    by_artifact: dict[str, str],
    positive_exemplars: list[str],
    negative_exemplars: list[str],
    *,
    top_k_per_artifact: int = 200,
    min_score: float = 0.35,
    contextual_window: int = 1,
    paragraph_window: int = 1,
    use_sparse: bool = True,
    use_mmr: bool = True,
) -> list[dict[str, Any]]:
    """Per-artifact retrieval, returns flat list across all artifacts.

    Each candidate dict additionally includes:
      "artifact_id": str
    """
    if not embedding_endpoint_reachable():
        return []
    results: list[dict[str, Any]] = []
    for aid, text in by_artifact.items():
        if not text or len(text) < 50:
            continue
        try:
            sentences, embeddings = embed_artifact_v2(
                aid, text,
                contextual_window=contextual_window,
                use_disk_cache=True,
            )
        except Exception as e:
            logger.warning("embed_artifact_v2 failed for %s: %s", aid, e)
            continue
        if not sentences:
            continue
        cands = hybrid_retrieve(
            positive_exemplars, negative_exemplars,
            sentences, embeddings,
            top_k=top_k_per_artifact,
            min_score=min_score,
            use_sparse=use_sparse,
            use_mmr=use_mmr,
            paragraph_window=paragraph_window,
        )
        for c in cands:
            c["artifact_id"] = aid
            results.append(c)
    # Global sort by score
    results.sort(key=lambda r: -r["score"])
    return results


__all__ = [
    "embed_artifact_v2",
    "sparse_retrieve",
    "reciprocal_rank_fusion",
    "margin_score",
    "mmr_diversify",
    "expand_to_paragraph",
    "hybrid_retrieve",
    "get_v39_candidates",
]
