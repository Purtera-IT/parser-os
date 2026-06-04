"""Semantic site fusion: a site CODE and its FRIENDLY NAME for the same physical
place merge via the decide() chokepoint, taught as text into the feedback store.

The deterministic alias passes in entity_resolution only merge site keys whose
slugs are identical (modulo a numeric suffix). They cannot see that
``site:atl_air_03`` and ``site:atlanta_air_office`` are the same location —
different slugs, different tokens. This is the real-world dupe that inflated a
live deal to 9 "sites" when ~4 exist.

The fix routes the merge judgment through decide() -> STORE (kNN) -> LLM ->
UNDECIDED. A PM teaches the merge ONCE as a text correction; the store fires on
the same (and, live, semantically-near) pair. Off by default and a no-op with
no store wired, so the default pipeline is byte-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

import app.core.embedding_retrieval as embedding_retrieval
import app.core.semantic_role as semantic_role
from app.core.decide import DecisionScope, set_store
from app.core.entity_resolution import semantic_site_fusion_groups
from app.core.feedback_store import Correction, FeedbackStore

_D = 64


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic content-addressed embedder: identical text -> identical
    vector (cosine 1.0 on an exact taught exemplar), different text -> a
    near-orthogonal vector (cosine well below the 0.82 store threshold)."""
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t.lower().strip()))
        out[i, h % _D] = 1.0
        out[i, (h // _D) % _D] += 0.5
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Force the LLM tier to abstain so ONLY a store hit can cause a merge —
    the test asserts the store path, not the model."""
    monkeypatch.setattr(
        semantic_role, "classify_role",
        lambda *a, **k: (None, 0.0),
    )
    # Keep the seam's batched cache-warming hermetic (no network embed call).
    monkeypatch.setattr(
        embedding_retrieval, "embed_texts",
        lambda texts, *a, **k: np.zeros((len(texts), _D), dtype=np.float32),
    )
    yield
    set_store(None)


# The keys the live deal produced: the air site appears as a code AND a friendly
# name; the HQ site is genuinely distinct.
_AIR_CODE = "site:atl_air_03"
_AIR_NAME = "site:atlanta_air_office"
_HQ = "site:atl_hq_01"
# Pair text the fusion builds (keys sorted, slug "_"->" "): "atl_air_03" sorts
# before "atlanta_air_office" because '_' < 'a'.
_TAUGHT_PAIR = "atl air 03 || atlanta air office"


def _store_with(*corrections: Correction) -> FeedbackStore:
    s = FeedbackStore(":memory:", embed_fn=_fake_embed, reachable_fn=lambda: True)
    for c in corrections:
        s.add(c)
    return s


def test_flag_off_is_noop(monkeypatch):
    """Default (flag unset): no fusion attempted, even with a wired store."""
    monkeypatch.delenv("SOWSMITH_NEURAL_SITE_FUSION", raising=False)
    set_store(_store_with(Correction(
        id="c1", relation="same_physical_site", verdict="same_site",
        exemplars=[_TAUGHT_PAIR], created_by="human",
    )))
    assert semantic_site_fusion_groups({_AIR_CODE, _AIR_NAME, _HQ}) == []


def test_no_store_is_noop(monkeypatch):
    """Flag on but no store wired: decide() falls through to the (abstaining)
    LLM, so nothing merges — byte-identical to the deterministic pipeline."""
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_FUSION", "1")
    set_store(None)
    assert semantic_site_fusion_groups({_AIR_CODE, _AIR_NAME, _HQ}) == []


def test_taught_correction_merges_code_and_friendly(monkeypatch):
    """A single PM-taught text correction merges the code and its friendly name;
    the genuinely-distinct HQ site is left alone."""
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_FUSION", "1")
    set_store(_store_with(Correction(
        id="c_air", relation="same_physical_site", verdict="same_site",
        exemplars=[_TAUGHT_PAIR], created_by="human",
    )))

    groups = semantic_site_fusion_groups({_AIR_CODE, _AIR_NAME, _HQ})

    # Exactly one merged group, containing the code + friendly name.
    assert len(groups) == 1
    merged = groups[0]
    assert merged == {_AIR_CODE, _AIR_NAME}
    # The distinct HQ site never joined a group.
    assert _HQ not in merged


def test_correction_for_other_relation_does_not_fire(monkeypatch):
    """A correction grounded on a DIFFERENT relation must not merge sites —
    corrections are relation-scoped, so an address-role rule can't leak here."""
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_FUSION", "1")
    set_store(_store_with(Correction(
        id="c_wrong", relation="physical_site", verdict="same_site",
        exemplars=[_TAUGHT_PAIR], created_by="human",
    )))
    assert semantic_site_fusion_groups({_AIR_CODE, _AIR_NAME, _HQ}) == []


def test_single_key_is_noop(monkeypatch):
    """Fewer than two site keys: nothing to compare."""
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_FUSION", "1")
    set_store(_store_with(Correction(
        id="c_air", relation="same_physical_site", verdict="same_site",
        exemplars=[_TAUGHT_PAIR], created_by="human",
    )))
    assert semantic_site_fusion_groups({_AIR_CODE}) == []
