"""Phase-2 reconciliation: a verdict layer whose every update has a receipt.

The COPPER shape is the founding fixture: a customer email (rank 90), a
vendor quote (rank 65) and a machine-extracted roster row (rank 40) disagree
about a quantity. The resolution must name the winner, the losers, their
ranks, and the edges that bound them -- and must REFUSE when the top of the
lattice ties, because picking between two rank-90 emails is a PM's judgment.
"""

from __future__ import annotations

from app.core.reconcile import (
    TIE_MARGIN,
    authority_potential,
    build_reconciliation,
    reconcile,
)
from app.core.schemas import AuthorityClass, EdgeType


class _Atom:
    def __init__(self, id: str, text: str, authority: AuthorityClass,
                 quantity=None) -> None:
        self.id = id
        self.raw_text = text
        self.authority_class = authority
        self.value = {"quantity": quantity} if quantity is not None else {}
        self.entity_keys = ["material:cat6_drop", "site:atl_01"]


class _Edge:
    def __init__(self, id: str, a: str, b: str,
                 edge_type: EdgeType = EdgeType.contradicts,
                 metadata=None) -> None:
        self.id = id
        self.from_atom_id = a
        self.to_atom_id = b
        self.edge_type = edge_type
        self.metadata = metadata or {"comparable_scope": "cat6 drops @ ATL-01"}


def _copper_fixture():
    atoms = [
        _Atom("atm_email", "Please plan for 56 drops at ATL-01.",
              AuthorityClass.customer_current_authored, quantity=56),
        _Atom("atm_quote", "Cat6 drop, qty 40, ATL-01.",
              AuthorityClass.vendor_quote, quantity=40),
        _Atom("atm_roster", "ATL-01 | drops | 48",
              AuthorityClass.machine_extractor, quantity=48),
    ]
    edges = [
        _Edge("edg_1", "atm_email", "atm_quote"),
        _Edge("edg_2", "atm_quote", "atm_roster"),
    ]
    return atoms, edges


def test_the_copper_shape_resolves_with_a_full_receipt() -> None:
    """'Quantity became 56 because the rank-90 email beat the rank-65 quote'
    is a sentence the structure must literally be able to emit."""
    atoms, edges = _copper_fixture()
    updates = reconcile(atoms, edges)
    assert len(updates) == 1
    update = updates[0]
    assert update.resolved
    assert update.winner.atom_id == "atm_email"
    assert update.winner.value == "56"
    assert update.winner.rank == 90
    assert {c.atom_id for c in update.superseded} == {"atm_quote", "atm_roster"}
    assert sorted(update.edge_ids) == ["edg_1", "edg_2"]
    explanation = update.explain()
    assert "56" in explanation and "customer_current_authored" in explanation
    assert "40" in explanation, "the superseded claim must appear in the receipt"


def test_a_top_tier_tie_refuses_and_surfaces_both() -> None:
    """Two rank-90 customer emails disagreeing is not the lattice's call."""
    atoms = [
        _Atom("atm_email_a", "Plan for 56 drops.",
              AuthorityClass.customer_current_authored, quantity=56),
        _Atom("atm_email_b", "Plan for 64 drops.",
              AuthorityClass.customer_current_authored, quantity=64),
    ]
    edges = [_Edge("edg_1", "atm_email_a", "atm_email_b")]
    updates = reconcile(atoms, edges)
    assert len(updates) == 1
    update = updates[0]
    assert not update.resolved
    assert update.winner is None
    assert len(update.superseded) == 2, "BOTH receipts surface on a refusal"
    assert "tie" in update.reason


def test_the_tie_margin_matches_the_lattice_s_narrowest_gap() -> None:
    """pm_confirmed (95) vs customer_authored (90) is the closest real pair;
    a margin wider than 5 would erase a distinction the lattice makes, and a
    narrower one would resolve same-tier noise."""
    assert TIE_MARGIN == 5
    atoms = [
        _Atom("a", "PM confirmed 56.", AuthorityClass.pm_confirmed, quantity=56),
        _Atom("b", "Customer wrote 64.", AuthorityClass.customer_current_authored,
              quantity=64),
    ]
    # 95 - 90 < margin is FALSE (5 is not < 5): pm_confirmed governs.
    index = authority_potential([
        __import__("app.core.reconcile", fromlist=["Claim"]).Claim(
            atom_id=a.id, rank=95 if a.id == "a" else 90,
            value=str(a.value.get("quantity")), text=a.raw_text,
            authority=str(a.authority_class),
        )
        for a in atoms
    ])
    assert index == 0, "a real lattice gap must resolve, not tie"


def test_reconciliation_never_mutates_an_atom() -> None:
    """A verdict layer over the evidence, not a rewrite of it -- receipts
    must keep replaying byte-exact afterwards."""
    atoms, edges = _copper_fixture()
    before = [(a.id, a.raw_text, a.value.copy()) for a in atoms]
    reconcile(atoms, edges)
    after = [(a.id, a.raw_text, a.value.copy()) for a in atoms]
    assert before == after


def test_determinism_and_stable_ordering() -> None:
    atoms, edges = _copper_fixture()
    first = [u.as_dict() for u in reconcile(atoms, edges)]
    second = [u.as_dict() for u in reconcile(atoms, edges)]
    assert first == second


def test_non_contradiction_edges_are_ignored() -> None:
    atoms, edges = _copper_fixture()
    edges.append(_Edge("edg_s", "atm_email", "atm_roster",
                       edge_type=EdgeType.supports))
    updates = reconcile(atoms, edges)
    assert len(updates) == 1
    assert "edg_s" not in updates[0].edge_ids


def test_the_envelope_surface_counts_and_carries_the_precision_context() -> None:
    """A conflict here is a lead, not a verdict: the proposing rule measured
    41% precise on 447 labelled pairs, and the surface says so rather than
    letting the list read as ground truth."""
    atoms, edges = _copper_fixture()
    surface = build_reconciliation(atoms, edges)
    assert surface["counts"] == {"conflict_sets": 1, "resolved": 1, "open": 0}
    assert surface["edge_rule_precision"]["contradicts"] == 0.41
    assert surface["resolved"][0]["winner"]["atom_id"] == "atm_email"


def test_clusters_split_by_scope_key() -> None:
    """Two conflicts about different materials are two conflict sets, not one
    blended verdict."""
    atoms = [
        _Atom("a1", "56 drops.", AuthorityClass.customer_current_authored, 56),
        _Atom("a2", "40 drops.", AuthorityClass.vendor_quote, 40),
        _Atom("b1", "12 cameras.", AuthorityClass.customer_current_authored, 12),
        _Atom("b2", "9 cameras.", AuthorityClass.vendor_quote, 9),
    ]
    edges = [
        _Edge("e1", "a1", "a2", metadata={"comparable_scope": "drops"}),
        _Edge("e2", "b1", "b2", metadata={"comparable_scope": "cameras"}),
    ]
    updates = reconcile(atoms, edges)
    assert len(updates) == 2
    assert {u.scope_key for u in updates} == {"drops", "cameras"}
    assert all(u.resolved for u in updates)
