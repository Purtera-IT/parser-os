"""Teacher-free terms that make a link's explanation shape the relation (run E).

A saved link carries the labeler's note on why the two lines connect
(data.py keeps it on the edge; featurize puts its universal part, verdict
words masked, in ``Batch.edge_notes``). Until these terms, only the teacher
read it: no student loss did, so the explanations never trained. Two parts of
``c3_loss``, named as LossWeights fields (weight 0 turns each off), in the
spirit of why_carry.py's ``why_contrast`` / ``why_alone``:

``link_why_contrast``  the edge's pair representation, as the relation head
                       scores it (``s_i * t_j`` folded by the edge's relation:
                       its sum is the pair score), toward the encoded note of
                       that edge, against the part's other *distinct* notes
                       (identical notes are one column, never a negative). The
                       notes of near-miss edges (an end is a near-miss partner
                       of this edge's end) and of other-relation edges on the
                       same lines count ``link_why_hard_negative`` times. Both
                       directions. ``governs`` is scored by containment boxes,
                       not pairs, so its edges are no rows here; their notes
                       are still columns.
``link_why_alone``     the relation head must name the relation from the note
                       alone: the note's embedding stands in for both ends of
                       the pair and every relation (``governs`` too) scores it
                       through its own folds; cross-entropy on the saved
                       relation.

The notes are encoded once per step (one ``aux_text`` call, shared by both
terms) and projected by ``judgment_note`` into the head width, as the judgment
rationale does. The encoder side is detached (no WHY gradient into the line
encoder through the target); the projection and the relation head learn. No
extra forward pass: the terms read ``out.rel_x`` from the step's own pass.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from .data import Batch
from .folds import FoldStack
from .model import C3Model, C3Output
from .schema import RELATION


def _norm(t: str) -> str:
    return " ".join(str(t).split())


def link_note_rows(model: C3Model, batch: Batch) -> list[tuple[int, int, str, str]]:
    """The edges whose note can train: a relation the schema has and the
    deal view does not mask (head_scale ``rel:<name>`` 0)."""
    known = {o.field for o in model.schema.select(kind=RELATION)}
    return [(i, j, rel, _norm(t)) for i, j, rel, t in batch.edge_notes
            if rel in known and _norm(t) and batch.head_scale.get(f"rel:{rel}", 1.0)
            and i not in batch.link_ignored.get(rel, ())]


def note_columns(rows: list[tuple[int, int, str, str]]) -> tuple[list[str], list[int]]:
    """Distinct note texts and each edge's column: edges that share a note
    share a column, so a copy of an edge's own note is never its negative."""
    texts: list[str] = []
    seen: dict[str, int] = {}
    col = []
    for _, _, _, t in rows:
        if t not in seen:
            seen[t] = len(texts)
            texts.append(t)
        col.append(seen[t])
    return texts, col


def note_embeddings(model: C3Model, texts: list[str]) -> torch.Tensor:
    """[U, d_head]: the encoder read once, detached; the projection learns."""
    return model.judgment_note(model.aux_text(texts).detach())


def _relations(model: C3Model) -> tuple[list[str], list[str]]:
    """(pair relations in the head's order, every relation in schema order)."""
    every = [o.field for o in model.schema.select(kind=RELATION)]
    return [r for r in every if r != "governs"], every


def _near(batch: Batch) -> set[tuple[int, int]]:
    pairs = {(i, j) for i, j, _ in batch.near_misses}
    pairs |= {tuple(p) for p in batch.edges.get("near_miss", [])}
    return pairs | {(j, i) for i, j in pairs}


def hard_columns(rows: list[tuple[int, int, str, str]], col: list[int], batch: Batch
                 ) -> list[set[int]]:
    """Per edge, the columns that are hard negatives: the notes of near-miss
    edges and of other-relation edges on the same lines (never its own)."""
    near = _near(batch)
    out: list[set[int]] = []
    for m, (i, j, rel, _) in enumerate(rows):
        hard = set()
        for n, (a, b, rel2, _) in enumerate(rows):
            if col[n] == col[m]:
                continue
            if rel2 != rel and {i, j} & {a, b}:
                hard.add(col[n])
            elif {(i, a), (j, b), (i, b), (j, a)} & near:
                hard.add(col[n])
        out.append(hard)
    return out


def link_why_contrast_loss(model: C3Model, out: C3Output, batch: Batch, *,
                           e: torch.Tensor | None = None,
                           rows: list[tuple[int, int, str, str]] | None = None,
                           desc: dict[str, Any] | None = None,
                           tau: float = 0.1, hard: float = 4.0) -> torch.Tensor:
    """Supervised InfoNCE of edge pair representations against distinct notes."""
    zero = out.r.new_zeros(())
    if out.rel_x is None:
        return zero
    rows = link_note_rows(model, batch) if rows is None else rows
    texts, col = note_columns(rows)
    if len(texts) < 2:
        return zero
    pair_rels, _ = _relations(model)
    at = {r: k for k, r in enumerate(pair_rels)}
    keep = [m for m, (i, j, rel, _) in enumerate(rows)
            if rel in at and (s := out.relations.get(rel)) is not None and torch.isfinite(s[i, j])]
    if not keep:
        return zero
    if e is None:
        e = note_embeddings(model, texts)
    e = F.normalize(e.float(), dim=-1)
    desc = desc if desc is not None else model.describe()
    rd = torch.stack([desc[f"rel:{r}"][0] for r in pair_rels])
    head = model.relation_head
    ps, pd = head.src.params(rd), head.dst.params(rd)
    k = torch.tensor([at[rows[m][2]] for m in keep], device=out.rel_x.device)
    src = torch.tensor([rows[m][0] for m in keep], device=out.rel_x.device)
    dst = torch.tensor([rows[m][1] for m in keep], device=out.rel_x.device)
    pick = lambda p: {name: v[k] for name, v in p.items()}  # noqa: E731
    s = FoldStack.apply(head.a(out.rel_x[src]), pick(ps))
    t = FoldStack.apply(head.b(out.rel_x[dst]), pick(pd))
    pair = (s * t).float()                                              # [E, d_head]: sums to the score
    sim = F.normalize(pair, dim=-1) @ e.T / tau                         # [E, U]
    bias = torch.zeros_like(sim)
    lh = math.log(max(hard, 1e-6))
    hcols = hard_columns(rows, col, batch)
    for r_, m in enumerate(keep):
        for c in hcols[m]:
            bias[r_, c] = lh
    target = torch.tensor([col[m] for m in keep], device=sim.device)
    edge_to_note = F.cross_entropy(sim + bias, target)
    # Each note toward the edges that hold it; only notes some kept edge holds.
    member = F.one_hot(target, len(texts)).T.bool()                     # [U, E]
    held = member.any(-1)
    st = (sim + bias).T[held]
    note_to_edge = -(torch.logsumexp(st.masked_fill(~member[held], float("-inf")), -1)
                     - torch.logsumexp(st, -1))
    return 0.5 * (edge_to_note + note_to_edge.mean())


def link_why_alone_logits(model: C3Model, e: torch.Tensor,
                          desc: dict[str, Any] | None = None) -> torch.Tensor:
    """[U, R] the relation head's score of each note embedding as both ends
    of a pair, under every relation's folds (schema order, governs too)."""
    _, every = _relations(model)
    desc = desc if desc is not None else model.describe()
    rd = torch.stack([desc[f"rel:{r}"][0] for r in every])
    head = model.relation_head
    ps, pd = head.src.params(rd), head.dst.params(rd)
    a, b = head.a(e), head.b(e)
    cols = []
    for k in range(len(every)):
        s = FoldStack.apply(a, ps, k)
        t = FoldStack.apply(b, pd, k)
        cols.append((s * t).sum(-1) / s.shape[-1] ** 0.5)
    return torch.stack(cols, -1)


def link_why_alone_loss(model: C3Model, out: C3Output, batch: Batch, *,
                        e: torch.Tensor | None = None,
                        rows: list[tuple[int, int, str, str]] | None = None,
                        desc: dict[str, Any] | None = None) -> torch.Tensor:
    """The note alone must name its edge's relation, through the relation head."""
    zero = out.r.new_zeros(())
    rows = link_note_rows(model, batch) if rows is None else rows
    if not rows:
        return zero
    texts, col = note_columns(rows)
    if e is None:
        e = note_embeddings(model, texts)
    _, every = _relations(model)
    at = {r: k for k, r in enumerate(every)}
    logits = link_why_alone_logits(model, e, desc)[torch.tensor(col, device=e.device)].float()
    y = torch.tensor([at[r[2]] for r in rows], device=logits.device)
    return F.cross_entropy(logits, y)


def link_why_losses(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any] | None,
                    w) -> dict[str, torch.Tensor]:
    """The parts with a nonzero weight, named as LossWeights fields. The notes
    are encoded once and shared; ``desc`` is the step's encoded descriptions,
    reused (encoding them again would put another graph on the step)."""
    on_c, on_a = getattr(w, "link_why_contrast", 0.0), getattr(w, "link_why_alone", 0.0)
    if not (on_c or on_a):
        return {}
    rows = link_note_rows(model, batch)
    parts: dict[str, torch.Tensor] = {}
    zero = out.r.new_zeros(())
    if not rows:
        if on_c:
            parts["link_why_contrast"] = zero
        if on_a:
            parts["link_why_alone"] = zero
        return parts
    texts, _ = note_columns(rows)
    e = note_embeddings(model, texts)
    if on_c:
        parts["link_why_contrast"] = link_why_contrast_loss(
            model, out, batch, e=e, rows=rows, desc=desc, hard=float(getattr(w, "link_why_hard_negative", 4.0)))
    if on_a:
        parts["link_why_alone"] = link_why_alone_loss(model, out, batch, e=e, rows=rows, desc=desc)
    return parts


__all__ = ["hard_columns", "link_note_rows", "link_why_alone_logits", "link_why_alone_loss",
           "link_why_contrast_loss", "link_why_losses", "note_columns", "note_embeddings"]
