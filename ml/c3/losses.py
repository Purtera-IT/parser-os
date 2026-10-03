"""Every loss C3 trains with, one function each, combined in ``c3_loss``.

Nothing here runs a training loop: the training thread owns the schedule
(the stages in base-architecture-v3.md section 3). Each term is a named
entry in the returned dict, so an ablation is "set its weight to 0".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .data import IGNORE, Batch
from .model import CLAIM_SLOTS, C3Model, C3Output, log_gauss_mixture
from .schema import NUMBER


@dataclass
class LossWeights:
    heads: float = 1.0
    numbers: float = 0.2
    relations: float = 0.5
    governs: float = 0.5
    why_align: float = 0.5        # r_i toward the encoded WHY (InfoNCE)
    why_sufficiency: float = 0.5  # the WHY alone must carry the line to its answer
    residual: float = 0.01        # keep decisions going through r_i
    hindsight: float = 0.5        # Hindsight-JEPA
    variance: float = 0.1         # VICReg variance term on q, against collapse
    absence: float = 0.2
    adversary: float = 0.1
    policy_l1: float = 0.01
    conduct: float = 1.0
    space: dict[str, float] = field(default_factory=dict)  # optional per-space scale


def head_losses(model: C3Model, out: C3Output, batch: Batch, w: LossWeights,
                layer: str) -> tuple[torch.Tensor, torch.Tensor]:
    dev = out.r.device
    ce, num, terms = out.r.new_zeros(()), out.r.new_zeros(()), 0
    for opp in model.schema.select(layer=layer):
        if opp.key not in out.logits:
            continue
        y = torch.tensor(batch.targets[opp.key], device=dev)
        if (y != IGNORE).any():
            ce = ce + w.space.get(opp.space, 1.0) * F.cross_entropy(
                out.logits[opp.key], y, ignore_index=IGNORE)
            terms += 1
        if opp.kind == NUMBER and opp.key in out.numbers:
            vals = batch.numbers_target.get(opp.key, [])
            mask = torch.tensor([v is not None for v in vals], device=dev)
            if mask.any():
                t = torch.tensor([v or 0.0 for v in vals], device=dev)
                num = num + F.smooth_l1_loss(out.numbers[opp.key][mask], torch.log1p(t[mask]))
    return ce / max(terms, 1), num


def relation_loss(out: C3Output, batch: Batch) -> torch.Tensor:
    """For each source with a gold edge: softmax over every allowed target."""
    total, terms = out.r.new_zeros(()), 0
    for rel, pairs in batch.edges.items():
        s = out.relations.get(rel)
        if s is None:
            continue
        for src, dst in pairs:
            row = s[src]
            if torch.isfinite(row[dst]):
                total = total - row.log_softmax(-1)[dst]
                terms += 1
    return total / max(terms, 1)


def governs_loss(out: C3Output, batch: Batch) -> torch.Tensor:
    pairs = batch.edges.get("governs", [])
    if not pairs or out.governs is None:
        return out.r.new_zeros(())
    n = out.governs.shape[0]
    pos = torch.zeros(n, n, dtype=torch.bool, device=out.r.device)
    for p, c in pairs:
        pos[p, c] = True
    parents = pos.any(1)
    lab = torch.tensor(batch.labeled, device=out.r.device)
    # Negatives: other labeled lines under a parent that has gold children.
    neg = parents.view(-1, 1) & lab.view(1, -1) & ~pos & ~torch.eye(n, dtype=torch.bool,
                                                                   device=out.r.device)
    logp = out.governs.clamp(max=-1e-6)
    l_pos = -logp[pos].mean()
    l_neg = -torch.log1p(-logp[neg].exp() + 1e-6).mean() if neg.any() else 0.0
    return l_pos + l_neg


def why_losses(model: C3Model, out: C3Output, batch: Batch, desc, w: LossWeights,
               tau: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """The two ways the WHY shapes the space.

    align:       r_i is pulled toward its own WHY and away from the other
                 lines' WHYs (InfoNCE), verdict words masked.
    sufficiency: the encoded WHY replaces r_i, and the universal heads must
                 still reach the gold answer through the same folds. The WHY
                 becomes a displacement that carries a line to its answer;
                 r_i must learn to predict it for lines nobody explained.
    """
    idx = [i for i, t in enumerate(batch.why) if t]
    if len(idx) < 2:
        z = out.r.new_zeros(())
        return z, z
    why = model.why_proj(model.text([batch.why[i] for i in idx]))
    r = F.normalize(out.r[idx], dim=-1)
    sim = r @ F.normalize(why, dim=-1).T / tau
    target = torch.arange(len(idx), device=sim.device)
    align = 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target))

    stand_in = out.r.clone()
    stand_in[idx] = why + model.res_up(out.residual[idx])
    sub = C3Output(h=out.h, z_c=out.z_c, q_mu=out.q_mu, q_logvar=out.q_logvar,
                   q_logit=out.q_logit, r=stand_in, residual=out.residual)
    model.universal_heads(sub, stand_in, out.q_mean, desc)
    keep = torch.zeros(len(batch), dtype=torch.bool, device=sim.device)
    keep[idx] = True
    suff, terms = out.r.new_zeros(()), 0
    for key, lg in sub.logits.items():
        y = torch.tensor(batch.targets[key], device=sim.device).masked_fill(~keep, IGNORE)
        if (y != IGNORE).any():
            suff = suff + F.cross_entropy(lg, y, ignore_index=IGNORE)
            terms += 1
    return align, suff / max(terms, 1)


def hindsight_loss(model: C3Model, out: C3Output, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Hindsight-JEPA: q_i (past only) must put mass on the latent the EMA
    teacher computes for line i from the whole finished deal."""
    target = model.hindsight_targets(batch.inputs())
    nll = -log_gauss_mixture(target, out.q_mu, out.q_logvar, out.q_logit).mean() / target.shape[-1]
    q = out.q_mean
    var = F.relu(1.0 - torch.sqrt(q.var(0) + 1e-4)).mean() if q.shape[0] > 1 else q.new_zeros(())
    return nll, var


def absence_targets(batch: Batch) -> torch.Tensor:
    """Free labels from the timeline: slot s is empty at line i (no earlier or
    current line fills it) and some later line fills it -> 1; empty now and
    never filled -> 0; already filled -> ignored (-1)."""
    n = len(batch)
    keys = {s: [f"read:{r}" for r in reads] for s, reads in CLAIM_SLOTS.items()}
    filled = torch.zeros(n, len(keys), dtype=torch.bool)
    for j, (slot, ks) in enumerate(keys.items()):
        for k in ks:
            col = batch.targets.get(k)
            if col is None:
                continue
            filled[:, j] |= torch.tensor([c not in (IGNORE, 0) for c in col])
    seen = filled.cumsum(0) > 0
    later = filled.flip(0).cumsum(0).flip(0) > 0
    later_strict = torch.cat([later[1:], torch.zeros_like(later[:1])])
    t = later_strict.float()
    t[seen] = -1.0
    return t


def absence_loss(out: C3Output, batch: Batch) -> torch.Tensor:
    t = absence_targets(batch).to(out.r.device)
    m = t >= 0
    if not m.any():
        return out.r.new_zeros(())
    return F.binary_cross_entropy_with_logits(out.absence[m], t[m])


def c3_loss(model: C3Model, batch: Batch, w: LossWeights | None = None,
            adv_lambda: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    w = w or LossWeights()
    desc = model.describe()
    out = model(batch.inputs(), company=batch.company, desc=desc, adv_lambda=adv_lambda)
    parts: dict[str, torch.Tensor] = {}
    parts["heads"], parts["numbers"] = head_losses(model, out, batch, w, "universal")
    parts["conduct"], _ = head_losses(model, out, batch, w, "company")
    parts["relations"] = relation_loss(out, batch)
    parts["governs"] = governs_loss(out, batch)
    parts["why_align"], parts["why_sufficiency"] = why_losses(model, out, batch, desc, w)
    parts["residual"] = out.residual.pow(2).mean()
    parts["hindsight"], parts["variance"] = hindsight_loss(model, out, batch)
    parts["absence"] = absence_loss(out, batch)
    if len(model.companies) > 1 and batch.company in model.companies:
        y = torch.full((len(batch),), model.companies.index(batch.company), device=out.r.device)
        parts["adversary"] = F.cross_entropy(out.company_logits, y)
    parts["policy_l1"] = out.alpha.abs().sum() if out.alpha is not None else out.r.new_zeros(())
    total = sum(getattr(w, k) * v for k, v in parts.items())
    return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in parts.items()}
