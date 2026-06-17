"""Embedding-driven candidate retrieval for entity extraction.

Architecture (v38):
  1. Split each artifact into sentences (custom splitter, no nltk dep).
  2. Embed every sentence via ollama /api/embeddings (qwen3-embedding:8b
     returns 4096-dim vectors).
  3. For each entity type, similarity-search a small set of curated
     "exemplar" sentences against the doc's sentence embeddings.
  4. Top-K candidates feed a single-sentence LLM canonicalize step
     (in multi_entity_llm.py) that decides keep/drop + canonical form.

Why this beats chunked extraction:
  - No chunk dropout (sentence is the atomic unit, no boundary loss).
  - No LLM self-limiting (each canonicalize call processes ONE
    candidate, so the model never "feels done" early).
  - Universal across entity types (same retrieval primitive, just
    different exemplar set per type).
  - Pure embedding-based retrieval: NO regex pattern matching.

Performance:
  - First embed of a 200-page doc: ~30-60s on qwen3-embedding:8b
    (batched 32 at a time).
  - Cached per-artifact via session cache (re-compile of same doc
    skips embedding step entirely).
  - Retrieval (cosine top-K): pure numpy, sub-second for 10K
    sentences × 15 exemplars.

API used by multi_entity_llm.py:
  - get_candidates_for_entity_type(by_artifact, entity_type, top_k)
    → list of {sentence, score, artifact_id, locator} candidates
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# OLLAMA EMBEDDING CONFIG
# ────────────────────────────────────────────────────────────────────

_DEFAULT_HOST = "http://100.114.102.122:11434"
_DEFAULT_MODEL = "qwen3-embedding:8b"
_DEFAULT_DIM = 4096  # qwen3-embedding:8b native dim; only used when no embed succeeds
_DEFAULT_TIMEOUT = 180  # was 60 — give a cold model load (after an idle unload /
                        # VRAM eviction) room to finish instead of timing out into
                        # a zero vector. keep_alive then holds it warm, so only the
                        # FIRST call after an unload pays this; the rest are fast.
_BATCH_SIZE = 8  # parallel HTTP calls; ollama queues internally
# Keep the model resident on the box between calls. ollama unloads a model after
# ~5 min idle by default; the next call then pays a multi-second reload that can
# blow the request timeout and come back as a ZERO vector — the "embedder
# flicker". Sending keep_alive on every request resets that timer, so the model
# stays warm as long as we're parsing. Override with OLLAMA_KEEP_ALIVE ("-1" =
# never unload). Stack this with `caffeinate` + OLLAMA_KEEP_ALIVE=-1 on the box.
_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

# ────────────────────────────────────────────────────────────────────
# SESSION CACHE (per-artifact embeddings, by text hash)
# ────────────────────────────────────────────────────────────────────

_EMBEDDING_CACHE: dict[str, tuple[list[str], np.ndarray]] = {}
_CACHE_MAX = 32  # LRU-evict on overflow


def _artifact_key(artifact_id: str, text: str) -> str:
    """Stable key from artifact_id + first 4KB of text hash."""
    h = hashlib.sha256(text[:4096].encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{artifact_id}::{h}"


# ────────────────────────────────────────────────────────────────────
# SENTENCE SPLITTER
# ────────────────────────────────────────────────────────────────────

# Common abbreviations that look like sentence-enders but aren't.
_ABBREV = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "inc", "llc", "ltd", "corp", "co",
    "etc", "e.g", "i.e", "vs", "vol",
    "no", "fig", "ref", "sec", "ch",
    "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "u.s.a", "p.o", "a.m", "p.m",
})


def sentence_split(text: str, max_sentence_chars: int = 500) -> list[str]:
    """Split text into sentences using punctuation + capitalization
    heuristics. No NLP library dependency.

    Strategy:
      1. Split paragraphs on \\n\\n (hard boundary).
      2. Within each paragraph: find candidate split points (.!?
         followed by whitespace + capital letter).
      3. Reject splits after known abbreviations.
      4. Collapse multi-space, strip, drop empty.
      5. Cap each sentence at max_sentence_chars (oversplit if longer).

    For lists/bullets: each line becomes its own "sentence" (common
    in RFPs where requirements are bullet-pointed).
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace within lines, preserve hard line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Treat bullet lines (lines starting with -, *, •, ▪, ■, or numbered)
    # as forced sentence boundaries
    bullet_re = re.compile(
        r"^\s*(?:[-*•▪■●○]|\d{1,3}\.|\d{1,3}\))\s+",
        flags=re.MULTILINE,
    )
    # Insert paragraph break before each bullet to force split
    text = bullet_re.sub(lambda m: "\n\n" + m.group(0), text)

    paragraphs = re.split(r"\n{2,}", text)
    sentences: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Single-line paragraph — treat as one or many sentences
        # Replace mid-line newlines with spaces (PDF flow)
        para_flat = re.sub(r"\n+", " ", para)
        para_flat = re.sub(r"\s+", " ", para_flat).strip()
        if not para_flat:
            continue

        # Find split candidates: . ! ? followed by space + capital
        # (ignore abbreviations)
        split_points: list[int] = []
        i = 0
        n = len(para_flat)
        while i < n:
            ch = para_flat[i]
            if ch in ".!?":
                # Look back to check abbreviation
                back_start = max(0, i - 10)
                back = para_flat[back_start:i].lower()
                last_word_m = re.search(r"\b([a-z]+)$", back)
                if last_word_m and last_word_m.group(1) in _ABBREV:
                    i += 1
                    continue
                # Look forward: space + capital or end-of-text
                j = i + 1
                if j >= n:
                    split_points.append(j)
                    break
                if para_flat[j] in " \t":
                    # Skip whitespace
                    k = j
                    while k < n and para_flat[k] in " \t":
                        k += 1
                    if k < n and (para_flat[k].isupper() or para_flat[k] in "“\""):
                        split_points.append(j)
            i += 1

        # Cut paragraph at split points
        prev = 0
        for sp in split_points:
            piece = para_flat[prev:sp].strip()
            if piece:
                sentences.append(piece)
            prev = sp
        tail = para_flat[prev:].strip()
        if tail:
            sentences.append(tail)

    # Post-process: cap length, drop noise
    out: list[str] = []
    for s in sentences:
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        # Drop super-short (likely headings / table cell fragments)
        if len(s) < 10:
            continue
        # Drop super-long (split aggressively)
        if len(s) > max_sentence_chars:
            # Hard cut at max_sentence_chars on word boundary
            while len(s) > max_sentence_chars:
                cut = s.rfind(" ", 0, max_sentence_chars)
                if cut < max_sentence_chars // 2:
                    cut = max_sentence_chars
                out.append(s[:cut].strip())
                s = s[cut:].strip()
            if s:
                out.append(s)
        else:
            out.append(s)
    return out


# ────────────────────────────────────────────────────────────────────
# OLLAMA EMBEDDING CLIENT
# ────────────────────────────────────────────────────────────────────


def _embed_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", _DEFAULT_MODEL)


def _embed_one(text: str) -> list[float] | None:
    """POST to /api/embeddings. Returns 4096-dim vector or None on failure."""
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")
    model = _embed_model()
    timeout = int(os.environ.get("SOWSMITH_EMBED_TIMEOUT", str(_DEFAULT_TIMEOUT)))
    try:
        r = requests.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": text, "keep_alive": _KEEP_ALIVE},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        emb = data.get("embedding")
        if isinstance(emb, list) and emb:
            return emb
    except Exception:
        return None
    return None


# Newer Ollama exposes /api/embed which accepts {"input": [..]} and returns
# ALL embeddings in ONE round-trip. That collapses N serial HTTP calls into 1,
# which is the dominant cold-compile cost over a high-latency (Tailscale) link.
# Older servers 404 here → we fall back to the per-text parallel path.
_BATCH_ENDPOINT_OK: bool | None = None  # None=untried, True=use it, False=404'd


def _embed_batch_endpoint(texts: list[str]) -> list[list[float] | None] | None:
    """Try the single-round-trip /api/embed endpoint. Returns per-text vectors
    (None entry on individual failure) or None if the endpoint is unavailable
    (so the caller falls back to the legacy per-text path)."""
    global _BATCH_ENDPOINT_OK
    if _BATCH_ENDPOINT_OK is False:
        return None
    if os.environ.get("SOWSMITH_EMBED_NO_BATCH"):
        _BATCH_ENDPOINT_OK = False
        return None
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")
    model = _embed_model()
    timeout = int(os.environ.get("SOWSMITH_EMBED_TIMEOUT", str(_DEFAULT_TIMEOUT)))
    try:
        r = requests.post(
            f"{host}/api/embed",
            json={"model": model, "input": texts, "keep_alive": _KEEP_ALIVE},
            timeout=timeout,
        )
        if r.status_code == 404:
            _BATCH_ENDPOINT_OK = False
            return None
        if r.status_code != 200:
            return None  # transient; don't disable the endpoint permanently
        data = r.json()
        embs = data.get("embeddings")
        if isinstance(embs, list) and len(embs) == len(texts):
            _BATCH_ENDPOINT_OK = True
            return [e if (isinstance(e, list) and e) else None for e in embs]
    except Exception:
        return None
    return None


def _embed_uncached(texts: list[str]) -> list[list[float] | None]:
    """Embed texts with no caching. Prefers the one-round-trip batch endpoint;
    falls back to the legacy ThreadPoolExecutor per-text path."""
    batched = _embed_batch_endpoint(texts)
    if batched is not None:
        return batched
    parallel = int(os.environ.get("SOWSMITH_EMBED_PARALLEL", str(_BATCH_SIZE)))
    out: list[list[float] | None] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_embed_one, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                out[i] = fut.result()
            except Exception:
                out[i] = None
    return out


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns (N, D) normalized numpy matrix.

    Two universal speed layers wrap the remote call so EVERY embedding
    consumer (entity enrichment, typed classifier, feedback store, neural
    head, GNN) benefits at once:

      1. Persistent content-addressed cache (sqlite, keyed by model +
         sha256(text)) — a re-compile of the same/similar corpus never
         re-embeds. See embedding_cache.py.
      2. Single-round-trip batch endpoint (/api/embed) for cache MISSES —
         one HTTP call for the whole miss set instead of N serial calls.

    Failed embeds → zero vector at that row (caller filters; never cached).
    """
    if not texts:
        return np.zeros((0, _DEFAULT_DIM), dtype=np.float32)

    model = _embed_model()
    out: list[list[float] | None] = [None] * len(texts)

    # 1) cache lookup -------------------------------------------------------
    from app.core.embedding_cache import get_cache
    cache = get_cache()
    miss_idx: list[int] = []
    if cache is not None:
        hits = cache.get_many(model, texts)
        for i, vec in enumerate(hits):
            if vec is not None:
                out[i] = vec
            else:
                miss_idx.append(i)
    else:
        miss_idx = list(range(len(texts)))

    # 2) embed only the misses (batched round-trip), then persist ----------
    if miss_idx:
        miss_texts = [texts[i] for i in miss_idx]
        fetched = _embed_uncached(miss_texts)
        to_store: list[tuple[str, list[float]]] = []
        for pos, emb in zip(miss_idx, fetched):
            out[pos] = emb
            if cache is not None and emb:
                to_store.append((texts[pos], emb))
        if cache is not None and to_store:
            cache.put_many(model, to_store)

    # 3) assemble + L2-normalize (re-normalizing cached unit vectors is a
    #    no-op; failed rows stay zero and auto-drop in cosine top-K). Dim is
    #    derived from the first success so swapping the embed model can't
    #    silently zero every row on a length mismatch.
    dim = next((len(e) for e in out if e), _DEFAULT_DIM)
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for i, emb in enumerate(out):
        if emb and len(emb) == dim:
            mat[i] = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 1e-9, norms, 1.0)
    mat = mat / norms
    return mat


def embed_artifact(
    artifact_id: str,
    text: str,
    *,
    max_sentences: int = 8000,
) -> tuple[list[str], np.ndarray]:
    """Split + embed an artifact's text. Cached by artifact_id + content hash.

    Returns (sentences, embeddings) where embeddings is (N, D) normalized.
    """
    key = _artifact_key(artifact_id, text)
    if key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[key]
    sentences = sentence_split(text)
    if max_sentences and len(sentences) > max_sentences:
        # Sample every Nth sentence to stay under cap on huge docs
        step = (len(sentences) + max_sentences - 1) // max_sentences
        sentences = sentences[::step]
    t0 = time.time()
    embeddings = embed_texts(sentences)
    elapsed = time.time() - t0
    logger.info(
        "embedded artifact %s: %d sentences in %.1fs (%.1f sent/s)",
        artifact_id, len(sentences), elapsed,
        len(sentences) / max(elapsed, 0.001),
    )
    # LRU evict if cache full
    while len(_EMBEDDING_CACHE) >= _CACHE_MAX:
        oldest = next(iter(_EMBEDDING_CACHE))
        del _EMBEDDING_CACHE[oldest]
    _EMBEDDING_CACHE[key] = (sentences, embeddings)
    return sentences, embeddings


# ────────────────────────────────────────────────────────────────────
# COSINE SIMILARITY RETRIEVAL
# ────────────────────────────────────────────────────────────────────


def retrieve_candidates(
    exemplars: list[str],
    sentences: list[str],
    sentence_embeddings: np.ndarray,
    *,
    top_k: int = 200,
    min_score: float = 0.45,
) -> list[tuple[str, float]]:
    """For a set of exemplar query texts, return top-K sentences most
    similar to ANY exemplar. Threshold by min_score (cosine sim).

    Returns list of (sentence, score) sorted by score desc.
    """
    if not exemplars or not sentences or sentence_embeddings.size == 0:
        return []
    # Embed exemplars (small set, single batch)
    exemplar_vecs = embed_texts(exemplars)
    if exemplar_vecs.size == 0:
        return []
    # Cosine similarity = dot product of L2-normalized vectors
    # exemplar_vecs: (M, D), sentence_embeddings: (N, D)
    # sims: (M, N)
    sims = exemplar_vecs @ sentence_embeddings.T
    # Max-pool over exemplars: each sentence's score = best match
    # to ANY exemplar (high recall: "matches at least one example")
    max_per_sentence = sims.max(axis=0)  # (N,)
    # Threshold + sort
    indices = np.argsort(-max_per_sentence)  # desc
    out: list[tuple[str, float]] = []
    for idx in indices:
        score = float(max_per_sentence[idx])
        if score < min_score:
            break
        out.append((sentences[idx], score))
        if len(out) >= top_k:
            break
    return out


# ────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API (called from multi_entity_llm.py)
# ────────────────────────────────────────────────────────────────────


def get_candidates_for_entity_type(
    by_artifact: dict[str, str],
    exemplars: list[str],
    *,
    top_k_per_artifact: int = 200,
    min_score: float = 0.45,
) -> list[dict[str, Any]]:
    """For each artifact, retrieve candidate sentences matching the
    given exemplar set. Returns flat list of:

      {"sentence": str, "score": float, "artifact_id": str}

    Sorted by score desc within and across artifacts.
    """
    results: list[dict[str, Any]] = []
    for artifact_id, text in by_artifact.items():
        if not text or len(text) < 50:
            continue
        try:
            sentences, embeddings = embed_artifact(artifact_id, text)
        except Exception as e:
            logger.warning("embed_artifact failed for %s: %s", artifact_id, e)
            continue
        if not sentences:
            continue
        candidates = retrieve_candidates(
            exemplars, sentences, embeddings,
            top_k=top_k_per_artifact, min_score=min_score,
        )
        for sentence, score in candidates:
            results.append({
                "sentence": sentence,
                "score": score,
                "artifact_id": artifact_id,
            })
    # Sort globally by score
    results.sort(key=lambda r: -r["score"])
    return results


def embedding_endpoint_reachable() -> bool:
    """Quick health check — used to fall back to chunked extraction
    when Griffin's Mac is unreachable or the embed model isn't loaded."""
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = [m.get("name", "") for m in r.json().get("models", [])]
        model = os.environ.get("OLLAMA_EMBED_MODEL", _DEFAULT_MODEL)
        return any(model in m for m in models)
    except Exception:
        return False


__all__ = [
    "sentence_split",
    "embed_texts",
    "embed_artifact",
    "retrieve_candidates",
    "get_candidates_for_entity_type",
    "embedding_endpoint_reachable",
]
