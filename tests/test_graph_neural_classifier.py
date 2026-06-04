"""Neural graph edge classifier — offline, deterministic.

Covers the contracts that keep it safe to wire into graph_builder:
  * cold start (no labels) → abstains, so the caller keeps the deterministic
    edge (never a silent gain/loss),
  * entity-key adjacency + diffusion are well-formed,
  * given separable geometry it LEARNS the relation (supports / contradicts /
    no_edge) and decides without fallback,
  * it abstains (route_fallback) on out-of-distribution pairs.
"""
from __future__ import annotations

import numpy as np

from app.core.graph_neural_classifier import (
    NO_EDGE,
    GraphNeuralClassifier,
    build_adjacency,
    diffuse,
    pair_feature,
)


class _Atom:
    def __init__(self, entity_keys):
        self.entity_keys = entity_keys


def test_adjacency_links_only_shared_keys():
    atoms = [
        _Atom(["device:ap", "site:a"]),
        _Atom(["device:ap"]),          # shares device:ap with atom 0
        _Atom(["site:z"]),             # shares nothing
    ]
    A = build_adjacency(atoms)
    assert A.shape == (3, 3)
    # 0<->1 linked, 2 isolated
    assert A[0, 1] > 0 and A[1, 0] > 0
    assert A[0, 2] == 0 and A[2, 0] == 0
    # isolated node keeps full self-mass; every row sums to ~1 (stochastic)
    assert np.isclose(A[2, 2], 1.0)
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-5)


def test_diffuse_is_noop_without_edges_and_normalized():
    X = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    atoms = [_Atom([]), _Atom([])]          # no shared keys → identity adjacency
    A = build_adjacency(atoms)
    H = diffuse(X, A, hops=2)
    # rows L2-normalized; directions preserved (no neighbor to mix with)
    assert np.allclose(np.linalg.norm(H, axis=1), 1.0, atol=1e-5)
    assert np.allclose(H[0], [1.0, 0.0], atol=1e-5)
    assert np.allclose(H[1], [0.0, 1.0], atol=1e-5)


def test_cold_start_abstains():
    gnc = GraphNeuralClassifier(hops=0)
    X = _l2(np.random.default_rng(0).standard_normal((6, 8)))
    atoms = [_Atom([]) for _ in range(6)]
    gnc.fit_graph(X, atoms, labeled_pairs=[])   # no labels
    pred = gnc.predict(0, 1, [NO_EDGE, "supports", "contradicts"])
    assert pred.route_fallback is True
    assert pred.edge_type is None
    assert pred.trained is False


def _l2(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(n > 1e-9, n, 1.0)).astype(np.float32)


def _make_pool(rng, d=12, per=14):
    """Three node clusters whose pairwise geometry encodes the relation:
    P (+axis0), M (-axis0), O (+axis1)."""
    base_p = np.zeros(d); base_p[0] = 1.0
    base_m = np.zeros(d); base_m[0] = -1.0
    base_o = np.zeros(d); base_o[1] = 1.0
    P = _l2(base_p + 0.05 * rng.standard_normal((per, d)))
    M = _l2(base_m + 0.05 * rng.standard_normal((per, d)))
    O = _l2(base_o + 0.05 * rng.standard_normal((per, d)))
    X = np.vstack([P, M, O]).astype(np.float32)
    idx = {"P": list(range(0, per)),
           "M": list(range(per, 2 * per)),
           "O": list(range(2 * per, 3 * per))}
    return X, idx


def test_learns_separable_relation():
    rng = np.random.default_rng(7)
    X, idx = _make_pool(rng)
    atoms = [_Atom([]) for _ in range(X.shape[0])]

    pairs: list[tuple[int, int, str]] = []
    # supports: both endpoints from P (aligned, dot~+1)
    for a, b in zip(idx["P"][:10], idx["P"][1:11]):
        pairs.append((a, b, "supports"))
    # contradicts: P vs M (opposed, dot~-1)
    for a, b in zip(idx["P"][:10], idx["M"][:10]):
        pairs.append((a, b, "contradicts"))
    # no_edge: P vs O (orthogonal, dot~0)
    for a, b in zip(idx["P"][:10], idx["O"][:10]):
        pairs.append((a, b, NO_EDGE))

    gnc = GraphNeuralClassifier(hops=0)
    gnc.fit_graph(X, atoms, pairs)
    assert gnc.trained is True

    cands = ["supports", "contradicts", NO_EDGE]
    # held-out pairs of each geometry (different cluster members)
    s = gnc.predict(idx["P"][11], idx["P"][12], cands)
    c = gnc.predict(idx["P"][11], idx["M"][12], cands)
    n = gnc.predict(idx["P"][11], idx["O"][12], cands)

    assert s.edge_type == "supports" and s.route_fallback is False and s.is_edge
    assert c.edge_type == "contradicts" and c.route_fallback is False and c.is_edge
    assert n.edge_type == NO_EDGE and n.is_edge is False


def test_predict_out_of_range_indices_abstain():
    rng = np.random.default_rng(1)
    X, idx = _make_pool(rng)
    atoms = [_Atom([]) for _ in range(X.shape[0])]
    gnc = GraphNeuralClassifier(hops=0)
    gnc.fit_graph(X, atoms, [(0, 1, "supports")])
    pred = gnc.predict(0, 9999, ["supports", NO_EDGE])
    assert pred.route_fallback is True


# ── neural_edge_gate ─────────────────────────────────────────────────
from app.core.graph_neural_classifier import neural_edge_gate  # noqa: E402


class _GAtom:
    def __init__(self, id, vec, keys):
        self.id = id
        self.normalized_text = id
        self.raw_text = id
        self._vec = vec
        self.entity_keys = keys


class _Edge:
    def __init__(self, a, b, edge_type, family):
        self.from_atom_id = a
        self.to_atom_id = b
        self.edge_type = edge_type
        self.metadata = {"edge_family": family}


def test_edge_gate_drops_spurious_ambiguous_keeps_high_precision():
    rng = np.random.default_rng(3)
    d = 12
    # Build atoms: a contradiction cluster (P vs M) sharing a part key, plus an
    # orthogonal atom O that the deterministic builder spuriously semantic-links.
    p = np.zeros(d); p[0] = 1.0
    m = np.zeros(d); m[0] = -1.0
    o = np.zeros(d); o[1] = 1.0

    atoms = []
    vecs = {}
    for k in range(8):  # P-cluster, share part:x
        a = _GAtom(f"p{k}", _l2((p + 0.04 * rng.standard_normal(d)).reshape(1, -1))[0], ["part:x"])
        atoms.append(a); vecs[a.id] = a._vec
    for k in range(8):  # M-cluster, share part:x (contradiction targets)
        a = _GAtom(f"m{k}", _l2((m + 0.04 * rng.standard_normal(d)).reshape(1, -1))[0], ["part:x"])
        atoms.append(a); vecs[a.id] = a._vec
    for k in range(4):  # O atoms, orthogonal, no shared key
        a = _GAtom(f"o{k}", _l2((o + 0.04 * rng.standard_normal(d)).reshape(1, -1))[0], [f"site:{k}"])
        atoms.append(a); vecs[a.id] = a._vec

    embed_fn = lambda texts: np.vstack([vecs[t] for t in texts]).astype(np.float32)

    edges = []
    # high-precision contradictions P<->M (must be preserved)
    for k in range(6):
        edges.append(_Edge(f"p{k}", f"m{k}", "contradicts", "part_number_quantity_conflict"))
    # spurious ambiguous semantic_link between a P atom and an orthogonal O atom
    spurious = _Edge("p7", "o0", "supports", "semantic_link")
    edges.append(spurious)

    kept, dropped = neural_edge_gate(
        atoms, edges,
        embed_fn=embed_fn,
        high_precision_families={"part_number_quantity_conflict"},
        ambiguous_families={"semantic_link"},
        drop_confidence=0.6,
    )
    # all 6 high-precision edges survive
    hp = [e for e in kept if e.metadata["edge_family"] == "part_number_quantity_conflict"]
    assert len(hp) == 6
    # the orthogonal semantic_link was dropped
    assert dropped >= 1
    assert spurious not in kept


def test_edge_gate_noop_when_undertrained():
    atoms = [_GAtom(f"a{i}", _l2(np.random.default_rng(i).standard_normal((1, 8)))[0], []) for i in range(4)]
    vecs = {a.id: a._vec for a in atoms}
    edges = [_Edge("a0", "a1", "supports", "semantic_link")]
    kept, dropped = neural_edge_gate(
        atoms, edges,
        embed_fn=lambda texts: np.vstack([vecs[t] for t in texts]),
        high_precision_families={"value_support"},
        ambiguous_families={"semantic_link"},
    )
    assert kept == edges and dropped == 0
