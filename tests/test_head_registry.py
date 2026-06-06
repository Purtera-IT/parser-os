"""Head registry + NeuralHead persistence — the durable memory of the
self-improving loop (#72).

These tests prove, with synthetic embeddings (no network):

  * a fitted :class:`NeuralHead` survives a save → load round-trip byte-for-byte
    in behavior (same verdict, same calibrated probability, same projection);
  * :meth:`NeuralHead.fit` honors ``sample_weight`` — a heavily-weighted
    subcluster pulls its class prototype toward itself (so one PM gold row is
    not drowned by abundant LLM silver);
  * :class:`HeadRegistry` register → promote → rollback behaves as a versioned
    champion store: the champion is selectable, the previous champion is kept,
    and rollback restores it; status is reflected in per-version metadata.
"""

from __future__ import annotations

import numpy as np

from app.core.neural_head import NeuralHead
from app.learning.head_registry import HeadRegistry, get_head_registry, set_head_registry


def _separable(n_per=8, D=24, seed=0):
    """Two cleanly-separated classes (axis 0 vs axis 1) + deterministic noise."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, axis in (("type_a", 0), ("type_b", 1)):
        for _ in range(n_per):
            v = rng.standard_normal(D).astype(np.float32) * 0.03
            v[axis] += 2.0
            X.append(v)
            y.append(label)
    X = np.asarray(X, dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X, y


# ── NeuralHead persistence ──────────────────────────────────────────────────
def test_head_save_load_roundtrip_preserves_behavior(tmp_path):
    X, y = _separable(seed=1)
    head = NeuralHead(min_per_class=3).fit(X, y)
    assert head.trained  # enough signal → projection engaged

    p = str(tmp_path / "head.npz")
    head.save(p)
    loaded = NeuralHead.load(p)

    # Structural identity.
    assert loaded.classes_ == head.classes_
    assert loaded.trained == head.trained
    assert np.allclose(loaded._W, head._W)
    assert np.allclose(loaded._protos, head._protos)
    assert loaded._temp == head._temp
    assert loaded._radius == head._radius

    # Behavioral identity on a fresh query squarely in type_a.
    am = X[[i for i, t in enumerate(y) if t == "type_a"]].mean(axis=0)
    q = (am / np.linalg.norm(am)).astype(np.float32)
    d0 = head.classify(q, ["type_a", "type_b"])
    d1 = loaded.classify(q, ["type_a", "type_b"])
    assert d0.verdict == d1.verdict == "type_a"
    assert abs(d0.confidence - d1.confidence) < 1e-6
    assert d0.route_llm == d1.route_llm


def test_cold_start_head_roundtrips_as_identity(tmp_path):
    # Single class → identity projection (W is None). Round-trip must keep W None.
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 16)).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    head = NeuralHead(min_per_class=3).fit(X, ["type_a"] * 4)
    assert head.trained is False
    assert head._W is None

    p = str(tmp_path / "cold.npz")
    head.save(p)
    loaded = NeuralHead.load(p)
    assert loaded._W is None
    assert loaded.trained is False
    assert loaded.classes_ == ["type_a"]


def test_sample_weight_pulls_prototype_toward_weighted_subcluster():
    # Force identity projection (min_per_class huge) so prototypes live in raw
    # space and the weighting effect is isolated from metric learning.
    D = 16
    u1 = np.zeros(D, dtype=np.float32); u1[2] = 1.0
    u2 = np.zeros(D, dtype=np.float32); u2[3] = 1.0
    # class "a": two subclusters (u1, u2); class "b": elsewhere (so >=2 classes).
    Xa = np.vstack([u1] * 4 + [u2] * 4)
    ya = ["a"] * 8
    ub = np.zeros(D, dtype=np.float32); ub[5] = 1.0
    Xb = np.vstack([ub] * 8)
    yb = ["b"] * 8
    X = np.vstack([Xa, Xb]).astype(np.float32)
    y = ya + yb

    uniform = NeuralHead(min_per_class=1000).fit(X, y)
    assert uniform.trained is False  # identity confirmed

    # Weight the u1 subcluster 10x; everything else 1x.
    w = np.ones(len(y), dtype=np.float32)
    w[0:4] = 10.0
    weighted = NeuralHead(min_per_class=1000).fit(X, y, sample_weight=w)
    assert weighted.trained is False

    ai = uniform.classes_.index("a")
    proto_u = uniform._protos[ai]
    proto_w = weighted._protos[ai]
    # The weighted prototype must sit closer to u1 than the uniform one.
    assert float(proto_w @ u1) > float(proto_u @ u1) + 1e-3


# ── HeadRegistry register / promote / rollback ───────────────────────────────
def test_register_does_not_promote(tmp_path):
    reg = HeadRegistry(str(tmp_path / "reg"))
    X, y = _separable(seed=3)
    head = NeuralHead(min_per_class=3).fit(X, y)
    meta = reg.register("atom_type", head, accuracy=0.99, coverage=0.8, ready=True)
    assert meta.status == "candidate"
    assert reg.champion_version("atom_type") is None  # registering != serving
    assert meta.version in [m.version for m in reg.history("atom_type")]


def test_promote_then_rollback_restores_previous(tmp_path):
    reg = HeadRegistry(str(tmp_path / "reg"))
    X, y = _separable(seed=4)
    h1 = NeuralHead(min_per_class=3).fit(X, y)
    h2 = NeuralHead(min_per_class=3).fit(X, y)
    m1 = reg.register("atom_type", h1, accuracy=0.95, coverage=0.7, ready=True)
    m2 = reg.register("atom_type", h2, accuracy=0.97, coverage=0.8, ready=True)

    reg.promote("atom_type", m1.version)
    assert reg.champion_version("atom_type") == m1.version
    assert reg.meta("atom_type", m1.version).status == "champion"

    reg.promote("atom_type", m2.version)
    assert reg.champion_version("atom_type") == m2.version
    assert reg.meta("atom_type", m2.version).status == "champion"
    assert reg.meta("atom_type", m1.version).status == "retired"

    restored = reg.rollback("atom_type")
    assert restored == m1.version
    assert reg.champion_version("atom_type") == m1.version
    assert reg.meta("atom_type", m1.version).status == "champion"
    assert reg.meta("atom_type", m2.version).status == "retired"


def test_rollback_with_no_history_returns_none(tmp_path):
    reg = HeadRegistry(str(tmp_path / "reg"))
    assert reg.rollback("atom_type") is None


def test_champion_loads_a_working_head(tmp_path):
    reg = HeadRegistry(str(tmp_path / "reg"))
    X, y = _separable(seed=5)
    head = NeuralHead(min_per_class=3).fit(X, y)
    meta = reg.register("atom_type", head, accuracy=0.99, coverage=0.8, ready=True)
    reg.promote("atom_type", meta.version)

    got = reg.champion("atom_type")
    assert got is not None
    champ_head, champ_meta = got
    assert champ_meta.version == meta.version
    am = X[[i for i, t in enumerate(y) if t == "type_a"]].mean(axis=0)
    q = (am / np.linalg.norm(am)).astype(np.float32)
    assert champ_head.classify(q, ["type_a", "type_b"]).verdict == "type_a"


def test_promote_unknown_version_raises(tmp_path):
    reg = HeadRegistry(str(tmp_path / "reg"))
    import pytest
    with pytest.raises(ValueError):
        reg.promote("atom_type", "nope")


def test_unset_history_and_summary_are_empty(tmp_path):
    reg = HeadRegistry(str(tmp_path / "reg"))
    assert reg.relations() == []
    assert reg.history("atom_type") == []
    assert reg.summary() == {}


def test_env_accessor_default_off(monkeypatch):
    monkeypatch.delenv("SOWSMITH_HEAD_REGISTRY_DIR", raising=False)
    set_head_registry(None)
    assert get_head_registry() is None
    set_head_registry(None)  # leave global clean for other tests
