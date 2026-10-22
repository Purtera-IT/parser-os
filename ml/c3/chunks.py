"""Split a long deal without dropping an evidence edge.

Dense attention stays inside a chunk. An edge whose ends land in different
chunks is returned beside those chunks. It is not truncated and not retargeted
onto a different line.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import DealExample


@dataclass(frozen=True)
class DealChunk:
    deal: DealExample
    cross_edges: tuple[tuple[str, str, str], ...]
    uncovered_judgments: int = 0


def split_deal(deal: DealExample, max_atoms: int, schema=None) -> list[DealChunk]:
    """Pack atoms in deal order. Every original edge is intra-chunk or cross."""
    if max_atoms < 1:
        raise ValueError("Chunk size must be at least one atom")
    atoms = list(deal.atoms)
    chunks: list[list] = []
    for atom in atoms:
        if not chunks or len(chunks[-1]) >= max_atoms:
            chunks.append([])
        chunks[-1].append(atom)
    owners = {atom.key: n for n, group in enumerate(chunks) for atom in group}
    intra: list[list[tuple[str, str, str]]] = [[] for _ in chunks]
    cross: list[list[tuple[str, str, str]]] = [[] for _ in chunks]
    seen: set[tuple[str, str, str]] = set()
    for edge in deal.edges:
        src, dst, _rel = edge
        if edge in seen:
            continue
        seen.add(edge)
        if src not in owners or dst not in owners:
            raise ValueError("Edge endpoint is not an atom in this deal")
        if owners[src] == owners[dst]:
            intra[owners[src]].append(edge)
        else:
            cross[owners[src]].append(edge)
    if len(seen) != len(set(deal.edges)):
        raise ValueError("Deal edges were not all preserved")
    placed = _judgments_in_chunks(deal, chunks, schema)
    out = []
    for group, inside, outside, judgments in zip(chunks, intra, cross, placed):
        out.append(DealChunk(
            deal=DealExample(
                deal_id=deal.deal_id, atoms=list(group), company=deal.company,
                company_policy=deal.company_policy, edges=inside,
                outcome=dict(deal.outcome), judgments=judgments,
                edge_notes={k: v for k, v in deal.edge_notes.items() if k in inside},
                source_supervision={}),
            cross_edges=tuple(outside),
            uncovered_judgments=0))
    saved = [j for j in deal.judgments if isinstance(j, dict)]
    attached = [j for chunk in out for j in chunk.deal.judgments]
    missing = len(saved) - len(attached)
    if out and missing:
        first = out[0]
        out[0] = DealChunk(deal=first.deal, cross_edges=first.cross_edges,
                           uncovered_judgments=missing)
    preserved = [edge for chunk in out for edge in (*chunk.deal.edges, *chunk.cross_edges)]
    if sorted(preserved) != sorted(set(deal.edges)):
        raise ValueError("Chunking dropped or duplicated an evidence edge")
    return out


def _judgments_in_chunks(deal: DealExample, chunks: list[list], schema) -> list[list[dict]]:
    """Keep a judgment only on a chunk that already holds every subject line.

    Deal-level judgments and subjects that span chunks stay uncovered. A
    fragment is not labeled as the whole deal.
    """
    owned: list[list[dict]] = [[] for _ in chunks]
    if schema is None:
        return owned
    from .data import _Lines, judgment_subject
    from .schema import DEAL

    lines = _Lines(deal.atoms)
    keys = [{atom.key for atom in group} for group in chunks]
    for judgment in deal.judgments:
        if not isinstance(judgment, dict):
            continue
        head = str(judgment.get("head") or "")
        opp = schema.by_key().get(f"jdg:{head}")
        size = opp.size if opp is not None else "line"
        if size == DEAL:
            continue
        idx = judgment_subject(judgment, size, lines)
        if not idx:
            continue
        subject = {deal.atoms[i].key for i in idx}
        for n, group_keys in enumerate(keys):
            if subject <= group_keys:
                owned[n].append(judgment)
                break
    return owned
