"""Long-deal windows that keep every evidence edge.

Dense attention is refused above the atom budget. This planner never drops a
saved edge and never truncates text. An edge that does not fit in one window
becomes an explicit two-line bridge, in time order. A judgment is attached
only to a window that already contains every subject line, or to its own
subject bundle. A deal-level judgment on an over-budget deal stays uncovered
rather than being taught from a fragment.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .data import Batch, DealExample, _Lines, featurize, judgment_subject
from .schema import DEAL, Schema


@dataclass
class LongDealPlan:
    windows: list[Batch]
    bridges: list[Batch]
    covered_edges: int
    uncovered_judgments: int
    complete: bool
    notes: list[str] = field(default_factory=list)


def _bundle(deal: DealExample, atoms: list, edges: list, judgments: list, schema: Schema,
            max_atoms: int) -> Batch:
    part = replace(deal, atoms=atoms, edges=edges, judgments=judgments,
                   source_supervision={**deal.source_supervision, "judgments": judgments,
                                       "deal_answers": []})
    return featurize(part, schema, max_atoms=max_atoms)


def plan_long_deal(deal: DealExample, schema: Schema, *, max_atoms: int = 4096,
                   overlap: int = 32) -> LongDealPlan:
    if max_atoms < 2:
        raise ValueError("Atom budget must leave room for an evidence pair")
    if overlap < 0 or overlap >= max_atoms:
        raise ValueError("Window overlap must be smaller than the atom budget")
    atoms = list(deal.atoms)
    n = len(atoms)
    if n <= max_atoms:
        return LongDealPlan([featurize(deal, schema, max_atoms=max_atoms)], [],
                            len(deal.edges), 0, True)
    step = max(1, max_atoms - overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + max_atoms)
        spans.append((start, end))
        if end == n:
            break
        start += step
    lines = _Lines(atoms)
    owned: list[list[dict]] = [[] for _ in spans]
    loose: list[tuple[tuple[int, ...], dict]] = []
    uncovered = 0
    for judgment in deal.judgments:
        if not isinstance(judgment, dict):
            continue
        head = str(judgment.get("head") or "")
        opp = schema.by_key().get(f"jdg:{head}")
        size = opp.size if opp is not None else "line"
        if size == DEAL:
            uncovered += 1
            continue
        idx = judgment_subject(judgment, size, lines)
        if not idx:
            uncovered += 1
            continue
        owners = [w for w, (begin, end) in enumerate(spans) if all(begin <= i < end for i in idx)]
        if owners:
            owned[owners[0]].append(judgment)
        elif len(idx) <= max_atoms:
            loose.append((idx, judgment))
        else:
            uncovered += 1
    covered: set[tuple[str, str, str]] = set()
    windows: list[Batch] = []
    for (begin, end), judgments in zip(spans, owned):
        subset = atoms[begin:end]
        keys = {a.key for a in subset}
        inside = [edge for edge in deal.edges if edge[0] in keys and edge[1] in keys]
        covered.update(inside)
        windows.append(_bundle(deal, subset, inside, judgments, schema, max_atoms))
    bridges: list[Batch] = []
    by_key = {a.key: a for a in atoms}
    for edge in deal.edges:
        if edge in covered or edge[0] not in by_key or edge[1] not in by_key:
            continue
        pair = sorted((by_key[edge[0]], by_key[edge[1]]), key=lambda a: (a.entered_at, a.order))
        bridges.append(_bundle(deal, pair, [edge], [], schema, max_atoms))
        covered.add(edge)
    for idx, judgment in loose:
        chosen = [atoms[i] for i in idx]
        keys = {a.key for a in chosen}
        inside = [edge for edge in deal.edges if edge[0] in keys and edge[1] in keys]
        bridges.append(_bundle(deal, chosen, inside, [judgment], schema, max_atoms))
    missing = [edge for edge in deal.edges if edge not in covered]
    if missing:
        raise RuntimeError(f"{len(missing)} evidence edges were not placed")
    notes = ["deal-level judgments are not taught from a truncated window"] if uncovered else []
    return LongDealPlan(windows, bridges, len(covered), uncovered, uncovered == 0, notes)
