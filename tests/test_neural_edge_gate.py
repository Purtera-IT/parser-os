"""Tests for neural_edge_gate cold-start fast path."""

from __future__ import annotations

from app.core.graph_neural_classifier import gate_can_run, neural_edge_gate


class _Atom:
    def __init__(self, aid: str, text: str = "x", keys=None):
        self.id = aid
        self.normalized_text = text
        self.raw_text = text
        self.entity_keys = keys or []


class _Edge:
    def __init__(self, from_id: str, to_id: str, family: str, edge_type: str = "supports"):
        self.from_atom_id = from_id
        self.to_atom_id = to_id
        self.metadata = {"edge_family": family}
        self.edge_type = type("ET", (), {"value": edge_type})()


def test_gate_skips_embed_when_no_ambiguous_edges() -> None:
    atoms = [_Atom("a1", keys=["site:one"]), _Atom("a2"), _Atom("a3")]
    edges = [_Edge("a1", "a2", "value_support")]
    called = []

    def embed_fn(texts):
        called.append(len(texts))
        raise AssertionError("should not embed")

    kept, dropped = neural_edge_gate(
        atoms,
        edges,
        embed_fn=embed_fn,
        high_precision_families={"value_support"},
        ambiguous_families={"semantic_link"},
    )
    assert kept == edges
    assert dropped == 0
    assert called == []


def test_gate_skips_embed_when_head_stays_cold() -> None:
    atoms = [_Atom(f"a{i}", keys=[f"device:d{i}"]) for i in range(8)]
    edges = [
        _Edge("a0", "a1", "value_support"),
        _Edge("a0", "a2", "semantic_link"),
    ]
    called = []

    def embed_fn(texts):
        called.append(len(texts))
        raise AssertionError("should not embed")

    assert gate_can_run(
        atoms,
        edges,
        high_precision_families={"value_support"},
        ambiguous_families={"semantic_link"},
    ) is False

    kept, dropped = neural_edge_gate(
        atoms,
        edges,
        embed_fn=embed_fn,
        high_precision_families={"value_support"},
        ambiguous_families={"semantic_link"},
    )
    assert kept == edges
    assert dropped == 0
    assert called == []
