"""A heading that lists the customer's places is not a customer-duties section.

``docx_subsection_blocks_lift`` blocks a vendor preamble from lifting onto the
bullets under an exclusion or customer-responsibility sub-heading. Its
"Customer ..." positives pulled on the word "Customer" alone, so a location
heading sat right at the threshold. Location negatives keep the nearest
prototype on the allow side. A toy bag-of-words embedder stands in for the
real model so the geometry is deterministic offline.
"""
from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

import app.core.embedding_retrieval as embedding_retrieval
import app.core.semantic_rules as semantic_rules
from app.core.semantic_rules import SemanticRule
from app.parsers.docx_parser import DocxParser


def _toy_embed(texts):
    """Bag of words, with "customer" weighted the way the real model leans on
    it: the shared word alone is enough to pull two headings together."""
    out = []
    for t in texts:
        v = np.zeros(4096, dtype="float32")
        for w in re.split(r"[^a-z0-9]+", t.lower()):
            if w:
                v[zlib.crc32(w.encode()) % 4096] += 2.0 if w == "customer" else 1.0
        out.append(v.tolist())
    return out


@pytest.fixture()
def toy_embedder(monkeypatch):
    monkeypatch.delenv("SOWSMITH_SEMANTIC_RULES", raising=False)
    monkeypatch.setattr(semantic_rules, "_PROTO_CACHE", {})
    monkeypatch.setattr(DocxParser, "_SUBSECTION_BLOCK_RULE", None)
    monkeypatch.setattr(SemanticRule, "_reachable", lambda self: True)
    monkeypatch.setattr(embedding_retrieval, "cached_embedding", lambda text: None)
    monkeypatch.setattr(embedding_retrieval, "embed_texts", _toy_embed)
    yield
    DocxParser._SUBSECTION_BLOCK_RULE = None


def test_location_headings_do_not_block_the_lift(toy_embedder) -> None:
    assert DocxParser._subsection_blocks_lift("Customer Locations") is False
    assert DocxParser._subsection_blocks_lift("Customer-Designated Locations") is False
    assert DocxParser._subsection_blocks_lift("Customer Sites") is False


def test_contradiction_headings_still_block_the_lift(toy_embedder) -> None:
    assert DocxParser._subsection_blocks_lift("Customer Responsibilities") is True
    assert DocxParser._subsection_blocks_lift("Out of Scope") is True
    assert DocxParser._subsection_blocks_lift("Exclusions") is True
