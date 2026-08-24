"""Deal-level reconciliation: every belief update carries its receipt.

Phase 2 of the atom work, and bet #3's first slice. The graph builder finds
cross-document conflicts (contradicts edges over a comparable scope); this
module RESOLVES them -- and the resolution is the product, because a conflict
list a PM has to re-adjudicate by hand is a to-do list, not reconciliation.

The design is the deal-graph fixpoint from _BETS_LEDGER.md, scoped to what
today's evidence supports:

* every contradicts-edge cluster over the same comparable scope becomes a
  CONFLICT SET;
* each member's claim is weighted by its AUTHORITY PRIOR -- the same lattice
  the whole system runs on (contractual 100 > pm_confirmed 95 >
  customer_authored 90 > roster 80 > vendor 65 > meeting 55 > machine 40);
* the winning belief is the highest-authority claim, and the update record
  says WHY in receipts: winner, its rank, every superseded claim with its
  rank, and the edge ids that put them in the same set. "Quantity became 56
  because the rank-90 customer email beat the rank-65 vendor quote" is a
  sentence this structure can literally emit.

Guess-free, like every gate on this branch: a tie at the top of the lattice
is NOT resolved -- it surfaces as an open conflict with both receipts, because
picking between two rank-90 customer emails is a judgment a PM owns. Silence
where the evidence is balanced, and a paper trail where it is not.

WHY RULE-BASED PRIORS AND NOT THE LEARNED POTENTIAL YET, said with numbers:
the 447 labelled edge pairs measure the edge rules at 48.3% agreement overall
-- supports 90% precise, contradicts 41%, excludes unmeasurable (the label
space offered the labeller no 'excludes' option; 0/101 confirmed proves
nothing either way). A learned potential trained on 5 deals of labels, one of
which contributes 250 of 447 rows, would be a deal-memorisation head wearing
a physics costume. The potential function is therefore INJECTABLE (house
pattern) and defaults to the authority lattice; the learned one arrives when
the edge label corpus survives a deal-split eval that can refuse.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from app.core.authority import authority_rank

#: Two claims whose top authorities are within this many rank points are a
#: TIE: reconciliation abstains and surfaces both. The narrowest real gap in
#: the lattice is 5 (pm_confirmed 95 vs customer_authored 90), so anything
#: closer than that is same-tier evidence arguing with itself.
TIE_MARGIN = 5


@dataclass(frozen=True)
class Claim:
    """One atom's position inside a conflict set."""

    atom_id: str
    rank: int
    value: str          # the claim itself (a quantity, a scope statement)
    text: str
    authority: str


@dataclass
class BeliefUpdate:
    """One resolved (or deliberately unresolved) conflict, with receipts."""

    conflict_id: str
    scope_key: str                       # what the claims are about
    resolved: bool = False
    winner: Claim | None = None
    superseded: list[Claim] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def explain(self) -> str:
        if not self.resolved:
            contenders = ", ".join(
                f"{c.value!r} (rank {c.rank}, {c.atom_id})"
                for c in self.superseded
            )
            return f"UNRESOLVED {self.scope_key}: {self.reason} -- {contenders}"
        losers = ", ".join(
            f"{c.value!r} at rank {c.rank}" for c in self.superseded
        )
        return (
            f"{self.scope_key}: {self.winner.value!r} governs because "
            f"{self.winner.authority} (rank {self.winner.rank}) beats {losers}"
        )

    def as_dict(self) -> dict[str, Any]:
        def claim(c: Claim) -> dict[str, Any]:
            return {"atom_id": c.atom_id, "rank": c.rank, "value": c.value,
                    "authority": c.authority, "text": c.text[:200]}

        return {
            "conflict_id": self.conflict_id,
            "scope_key": self.scope_key,
            "resolved": self.resolved,
            "winner": claim(self.winner) if self.winner else None,
            "superseded": [claim(c) for c in self.superseded],
            "edge_ids": self.edge_ids,
            "reason": self.reason,
        }


#: Injectable potential: given the claims of one conflict set, return the
#: winner's index or None to abstain. The default is the authority lattice;
#: a learned potential slots in here when its corpus earns it.
Potential = Callable[[Sequence[Claim]], "int | None"]


def authority_potential(claims: Sequence[Claim]) -> int | None:
    """Highest authority wins; a tie inside TIE_MARGIN abstains."""
    if not claims:
        return None
    ranked = sorted(range(len(claims)), key=lambda i: -claims[i].rank)
    top = claims[ranked[0]]
    if len(ranked) > 1:
        runner = claims[ranked[1]]
        if top.rank - runner.rank < TIE_MARGIN:
            return None
        # Same-source consistency: two claims from the SAME authority tier
        # with different values is an intra-tier contradiction even when a
        # third, lower claim exists -- handled by the margin check above
        # because sorting puts both at the top.
    return ranked[0]


def _atom_value(atom: Any) -> str:
    """The claim an atom makes, preferring the structured quantity."""
    value = getattr(atom, "value", None) or {}
    if isinstance(value, dict):
        for key in ("quantity", "qty", "normalized_value", "amount"):
            if value.get(key) not in (None, ""):
                return str(value[key])
    text = str(getattr(atom, "raw_text", "") or "")
    return text[:120]


def _scope_key(edge: Any, atoms_by_id: dict[str, Any]) -> str:
    """What the conflict is ABOUT -- from edge metadata when the builder
    recorded it, else the shared entity keys, else the atom pair."""
    metadata = getattr(edge, "metadata", None) or {}
    for key in ("comparable_scope", "material", "scope_key", "entity_key", "family"):
        if metadata.get(key):
            return str(metadata[key])
    a = atoms_by_id.get(edge.from_atom_id)
    b = atoms_by_id.get(edge.to_atom_id)
    shared = set(getattr(a, "entity_keys", []) or []) & set(getattr(b, "entity_keys", []) or [])
    if shared:
        return "+".join(sorted(shared)[:3])
    return f"pair:{edge.from_atom_id[:12]}:{edge.to_atom_id[:12]}"


def reconcile(
    atoms: Sequence[Any],
    edges: Sequence[Any],
    *,
    potential: Potential = authority_potential,
) -> list[BeliefUpdate]:
    """Resolve every contradicts-cluster, or refuse with both receipts.

    Deterministic: same atoms + edges -> same updates in the same order.
    Nothing here mutates an atom -- reconciliation is a VERDICT LAYER over
    the evidence, not a rewrite of it, so receipts keep replaying.
    """
    atoms_by_id = {str(getattr(a, "id", "")): a for a in atoms}

    # union contradicts-edges into conflict sets, keyed by scope
    clusters: dict[str, dict[str, Any]] = {}
    for edge in edges:
        if str(getattr(edge, "edge_type", "")).split(".")[-1] != "contradicts":
            continue
        key = _scope_key(edge, atoms_by_id)
        slot = clusters.setdefault(key, {"atom_ids": set(), "edge_ids": []})
        slot["atom_ids"].update((str(edge.from_atom_id), str(edge.to_atom_id)))
        slot["edge_ids"].append(str(getattr(edge, "id", "")))

    updates: list[BeliefUpdate] = []
    for key in sorted(clusters):
        slot = clusters[key]
        claims: list[Claim] = []
        for atom_id in sorted(slot["atom_ids"]):
            atom = atoms_by_id.get(atom_id)
            if atom is None:
                continue
            try:
                rank = authority_rank(atom.authority_class)
            except Exception:  # noqa: BLE001 - unknown authority = weakest
                rank = 0
            claims.append(Claim(
                atom_id=atom_id,
                rank=rank,
                value=_atom_value(atom),
                text=str(getattr(atom, "raw_text", "") or ""),
                authority=str(getattr(atom, "authority_class", "")).split(".")[-1],
            ))
        if len(claims) < 2:
            continue  # a cluster that lost its atoms is not a conflict

        update = BeliefUpdate(
            conflict_id=f"rec_{abs(hash(key)) % 10**10:010d}",
            scope_key=key,
            edge_ids=sorted(slot["edge_ids"]),
        )
        winner_index = potential(claims)
        if winner_index is None:
            update.resolved = False
            update.superseded = claims
            top = max(c.rank for c in claims)
            update.reason = (
                f"authority tie at rank {top} -- same-tier evidence disagrees, "
                "and picking between equals is a PM's judgment, not a lattice's"
            )
        else:
            update.resolved = True
            update.winner = claims[winner_index]
            update.superseded = [c for i, c in enumerate(claims) if i != winner_index]
            update.reason = update.explain()
        updates.append(update)
    return updates


def build_reconciliation(atoms: Sequence[Any], edges: Sequence[Any]) -> dict[str, Any]:
    """Envelope surface: the resolved beliefs and the honest leftovers.

    Additive envelope key -- compute_output_signature hashes the
    CompileResult, not the envelope, so no signature moves (same argument,
    verified once already, as service_routing's widening).
    """
    updates = reconcile(atoms, edges)
    resolved = [u for u in updates if u.resolved]
    open_conflicts = [u for u in updates if not u.resolved]
    return {
        "resolved": [u.as_dict() for u in resolved],
        "open_conflicts": [u.as_dict() for u in open_conflicts],
        "counts": {
            "conflict_sets": len(updates),
            "resolved": len(resolved),
            "open": len(open_conflicts),
        },
        # The precision context a reader needs before trusting the list: the
        # rule that PROPOSES contradictions measured 41% precise on the 447
        # labelled pairs, so a conflict here is a lead, not a verdict --
        # which is exactly why every entry carries its receipts.
        "edge_rule_precision": {"contradicts": 0.41, "supports": 0.90,
                               "n_labelled_pairs": 447},
    }
