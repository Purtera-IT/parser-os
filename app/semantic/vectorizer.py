from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.normalizers import normalize_text
from app.core.schemas import EvidenceAtom
from app.domain.schemas import DomainPack


SENTENCE_TRANSFORMER_FEATURE_FLAG = "PURTERA_ENABLE_SENTENCE_TRANSFORMER"
SENTENCE_TRANSFORMER_MODEL = "PURTERA_SENTENCE_TRANSFORMER_MODEL"


def sentence_transformer_enabled() -> bool:
    value = os.getenv(SENTENCE_TRANSFORMER_FEATURE_FLAG, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def atom_representation(atom: EvidenceAtom, domain_pack: DomainPack | None = None) -> str:
    values = [
        normalize_text(atom.normalized_text),
        normalize_text(atom.raw_text),
        atom.atom_type.value,
        " ".join(sorted(atom.entity_keys)),
    ]
    if isinstance(atom.value, dict):
        for key in sorted(atom.value):
            values.append(f"{key}:{normalize_text(str(atom.value[key]))}")
    if domain_pack is not None:
        for entity_key in atom.entity_keys:
            prefix, _, canonical = entity_key.partition(":")
            if prefix == "device" and canonical in domain_pack.device_aliases:
                values.append(" ".join(normalize_text(alias) for alias in domain_pack.device_aliases[canonical]))
    return " ".join(part for part in values if part).strip()


def tfidf_char_ngram_similarity(texts: Iterable[str]) -> np.ndarray:
    rows = list(texts)
    if not rows:
        return np.zeros((0, 0), dtype=float)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    matrix = vectorizer.fit_transform(rows)
    similarities = cosine_similarity(matrix)
    return np.asarray(similarities, dtype=float)


def sentence_transformer_similarity(texts: Iterable[str]) -> np.ndarray:
    rows = list(texts)
    if not rows:
        return np.zeros((0, 0), dtype=float)
    from sentence_transformers import SentenceTransformer  # pragma: no cover - optional dependency

    model_name = os.getenv(SENTENCE_TRANSFORMER_MODEL, "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(rows, normalize_embeddings=True)
    similarities = cosine_similarity(embeddings)
    return np.asarray(similarities, dtype=float)


def best_effort_similarity(texts: Iterable[str]) -> tuple[np.ndarray, str]:
    rows = list(texts)
    if not rows:
        return np.zeros((0, 0), dtype=float), "tfidf_char_ngram"
    if sentence_transformer_enabled():
        try:
            return sentence_transformer_similarity(rows), "sentence_transformer"
        except Exception:
            pass
    return tfidf_char_ngram_similarity(rows), "tfidf_char_ngram"
