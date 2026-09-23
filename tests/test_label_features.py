"""A categorical fact must reach the head as a number, not as prose.

The measurement that forced this (bge-small, on the strings the ingest really
produces): the two `blocked_on` answers -- a reseller writing to us versus us
writing to a reseller, same 104 characters -- sit at cosine 0.9967. Worse, a
DIFFERENT sentence with the SAME answer sits at 0.9138. So on the raw text the
wrong pair is closer than the right pair, and a head over it is worse than
useless. Every text shape was tried; the best reached 0.9925.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.learning.label_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_WEIGHT,
    augment,
    features_for,
)

THEIRS = {"company": "cdw.com", "role": "reseller", "side": "theirs"}
OURS = {"company": "purtera-it.com", "role": "internal", "side": "ours"}


def lab(by, to, **kw):
    return {"said_by": by, "said_to": [to] if to else [], **kw}


def test_the_direction_is_a_dimension_of_its_own():
    a = features_for(lab(THEIRS, OURS))
    b = features_for(lab(OURS, THEIRS))
    assert a != b
    i = FEATURE_NAMES.index("from_side:theirs")
    assert a[i] == 1.0 and b[i] == 0.0
    # and both cross sides, which is its own dimension so a projection does
    # not have to compose two one-hots to see it
    j = FEATURE_NAMES.index("crosses_sides")
    assert a[j] == b[j] == 1.0


def test_an_internal_note_is_marked_and_does_not_cross():
    f = features_for(lab(OURS, OURS, internal_only=True))
    assert f[FEATURE_NAMES.index("internal_only")] == 1.0
    assert f[FEATURE_NAMES.index("crosses_sides")] == 0.0


def test_an_unknown_party_is_a_class_not_a_hole():
    f = features_for({})
    assert f[FEATURE_NAMES.index("from_side:unknown")] == 1.0
    assert f[FEATURE_NAMES.index("to_side:unknown")] == 1.0
    assert f[FEATURE_NAMES.index("from_role:unknown")] == 1.0
    assert sum(f) == 3.0


def test_the_same_sentence_from_opposite_sides_comes_apart():
    # The whole point. One embedding, two feature blocks.
    rng = np.random.default_rng(0)
    e = rng.normal(size=(1, 384)).astype(np.float32)
    E = np.vstack([e, e])
    F = np.array([features_for(lab(THEIRS, OURS)), features_for(lab(OURS, THEIRS))], dtype=np.float32)
    Z = augment(E, F)
    cos = float(Z[0] @ Z[1])
    assert cos < 0.80, f"identical text, opposite sides still at {cos:.4f}"


def test_the_words_still_decide_between_different_sentences():
    # A weight that only separates sides has replaced a text head with a party
    # head. Same block, different text must stay apart.
    rng = np.random.default_rng(1)
    E = rng.normal(size=(2, 384)).astype(np.float32)
    f = features_for(lab(THEIRS, OURS))
    Z = augment(E, np.array([f, f], dtype=np.float32))
    assert float(Z[0] @ Z[1]) < 0.75


def test_the_result_is_a_unit_vector_the_head_can_use():
    rng = np.random.default_rng(2)
    E = rng.normal(size=(5, 384)).astype(np.float32)
    F = np.array([features_for(lab(THEIRS, OURS))] * 5, dtype=np.float32)
    Z = augment(E, F)
    assert Z.shape == (5, 384 + FEATURE_DIM)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)
    assert 0.0 < FEATURE_WEIGHT < 1.0


def test_a_block_of_the_wrong_shape_is_refused_loudly():
    # Silently zero-filling would put the row at a point in space that means
    # "nobody spoke", which is a different claim.
    with pytest.raises(ValueError):
        augment(np.zeros((2, 384), dtype=np.float32), np.zeros((2, FEATURE_DIM - 1), dtype=np.float32))
    with pytest.raises(ValueError):
        augment(np.zeros((2, 384), dtype=np.float32), np.zeros((3, FEATURE_DIM), dtype=np.float32))


def test_the_order_is_the_contract():
    # A head fitted on this block reads the same dimensions at query time.
    # Appending is safe; reordering silently changes what every stored head
    # believes it is looking at.
    assert FEATURE_NAMES[0] == "from_side:ours"
    assert FEATURE_NAMES[-1] == "has_below"
    assert len(set(FEATURE_NAMES)) == FEATURE_DIM


def test_a_head_abstains_rather_than_answer_from_the_wrong_space():
    # A head fitted on embedding+features and one fitted on the embedding
    # alone live in different spaces. A query from the wrong one is not a
    # worse answer, it is a different question -- so it routes to the LLM.
    from app.core.neural_head import NeuralHead

    rng = np.random.default_rng(7)
    E = rng.normal(size=(12, 64)).astype(np.float32)
    f_a, f_b = features_for(lab(THEIRS, OURS)), features_for(lab(OURS, THEIRS))
    F = np.array([f_a if i % 2 else f_b for i in range(12)], dtype=np.float32)
    X = augment(E, F)
    y = ["us" if i % 2 else "partner" for i in range(12)]
    head = NeuralHead(min_per_class=3).fit(X, y)
    assert head.input_dim == X.shape[1]

    wide = head.classify(X[0], ["us", "partner"])
    bare = head.classify(E[0] / np.linalg.norm(E[0]), ["us", "partner"])
    assert bare.verdict is None and bare.route_llm is True
    assert wide.verdict in {"us", "partner"} or wide.route_llm

    # and it survives a round trip, including through a head saved before
    # input_dim existed (absent -> 0 -> do not check)
    back = NeuralHead.from_state(head.to_state())
    assert back.input_dim == head.input_dim
    old = head.to_state()
    old.pop("input_dim")
    assert NeuralHead.from_state(old).input_dim == 0
