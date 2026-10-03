"""Every loss C3 trains with, one function each, combined in ``c3_loss``.

Nothing here runs a training loop: the training thread owns the schedule
(the stages in base-architecture-v3.md section 3). Each term is a named
entry in the returned dict, so an ablation is "set its weight to 0".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .data import IGNORE, Batch, FlipTarget
from .explain import ExplanationBank
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
    explanation_only: float = 0.5  # the explanations' votes alone must reach the answer
    changes: float = 0.3           # what each line changes (hours, crew, sites, price...)
    claims: float = 0.3            # a WHY's compiled claim matches its line's changes
    rule_links: float = 0.5        # follows_rule / exception_to supervise a rule's region
    why_echo: float = 0.2          # r_to_text(r_i) lands on the encoded WHY (ask.py)
    clause_use: float = 0.1        # removing any non-statement clause must change the output
    pointers: float = 0.3          # hint_refs: the lines a decision came from
    entities: float = 0.2          # entity_keys: lines naming the same entity pull together
    judgments: float = 1.0         # judgment tabs about two lines, a group or the deal
    negatives: float = 0.3         # answers known wrong (the parser's type a person overruled)
    near_misses: float = 0.5       # lines that read alike but answer differently stay apart
    near_miss_margin: float = 1.0
    space: dict[str, float] = field(default_factory=dict)  # optional per-space scale


def row_weights(batch: Batch, device) -> torch.Tensor:
    """Per-line weight from weight_tier (load_bearing 3, ordinary 1, slight 0.3)."""
    w = batch.weights if len(batch.weights) == len(batch) else [1.0] * len(batch)
    return torch.tensor(w, device=device, dtype=torch.float32)


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged with row weights over the lines with a label."""
    m = y != IGNORE
    ce = F.cross_entropy(logits, y.clamp(min=0), reduction="none")
    return (ce * rw * m).sum() / (rw * m).sum().clamp(min=1e-6)


def pointer_loss(out: C3Output, batch: Batch) -> torch.Tensor:
    """hint_refs name the lines a decision came from: the pointer head puts
    its mass on the earlier ones (the heads see only the past)."""
    if out.pointers is None or not batch.hint_lines:
        return out.r.new_zeros(())
    total, terms = out.r.new_zeros(()), 0
    for i, js in enumerate(batch.hint_lines):
        js = [j for j in js if j < i]
        if js:
            total = total - torch.logsumexp(out.pointers[i, js], 0)
            terms += 1
    return total / max(terms, 1)


def entity_loss(out: C3Output, batch: Batch, tau: float = 0.1) -> torch.Tensor:
    """entity_keys: lines naming the same site, person or device pull
    together in r, against every other line of the deal (supervised
    contrastive)."""
    ents = [set(e) for e in (batch.entities or [])]
    n = out.r.shape[0]
    if len(ents) != n or sum(bool(e) for e in ents) < 2:
        return out.r.new_zeros(())
    pos = torch.tensor([[a != b and bool(ents[a] & ents[b]) for b in range(n)] for a in range(n)],
                       device=out.r.device)
    has = pos.any(1)
    if not has.any():
        return out.r.new_zeros(())
    z = F.normalize(out.r, dim=-1)
    sim = (z @ z.T / tau).masked_fill(torch.eye(n, dtype=torch.bool, device=z.device), -1e4)
    logp = sim.log_softmax(-1)
    return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


def head_losses(model: C3Model, out: C3Output, batch: Batch, w: LossWeights,
                layer: str) -> tuple[torch.Tensor, torch.Tensor]:
    dev = out.r.device
    ce, num, terms = out.r.new_zeros(()), out.r.new_zeros(()), 0
    rw = row_weights(batch, dev)
    for opp in model.schema.select(layer=layer):
        if opp.key not in out.logits:
            continue
        y = torch.tensor(batch.targets[opp.key], device=dev)
        if (y != IGNORE).any():
            ce = ce + w.space.get(opp.space, 1.0) * weighted_ce(out.logits[opp.key], y, rw)
            terms += 1
        if opp.kind == NUMBER and opp.key in out.numbers:
            vals = batch.numbers_target.get(opp.key, [])
            mask = torch.tensor([v is not None for v in vals], device=dev)
            if mask.any():
                t = torch.tensor([v or 0.0 for v in vals], device=dev)
                num = num + F.smooth_l1_loss(out.numbers[opp.key][mask], torch.log1p(t[mask]))
    return ce / max(terms, 1), num


def judgment_loss(model: C3Model, out: C3Output, batch: Batch,
                  desc: dict | None = None) -> torch.Tensor:
    """Cross-entropy on each judgment-tab verdict about two lines, a group of
    lines or the whole deal, one batch per question. Company questions train
    only when the company layer ran."""
    by: dict[str, list] = {}
    for j in batch.judged:
        by.setdefault(j.key, []).append(j)
    opps = model.schema.by_key()
    total, terms = out.r.new_zeros(()), 0
    for key, js in by.items():
        o = opps.get(key)
        if o is None or (not o.universal and out.conduct_x is None):
            continue
        logits = model.judge(out, key, [j.lines for j in js], desc)
        y = torch.tensor([j.answer for j in js], device=logits.device)
        total = total + F.cross_entropy(logits, y)
        terms += 1
    return total / max(terms, 1)


def negative_loss(out: C3Output, batch: Batch) -> torch.Tensor:
    """Push probability off answers a person ruled out: -log(1 - p(wrong))."""
    total, terms = out.r.new_zeros(()), 0
    for key, rows in batch.negatives.items():
        lg = out.logits.get(key)
        if lg is None:
            continue
        p = lg.softmax(-1)
        for i, bad in enumerate(rows):
            if bad:
                total = total - torch.log1p(-p[i, bad].sum().clamp(max=1 - 1e-6))
                terms += 1
    return total / max(terms, 1)


def moved_heads(model: C3Model, out: C3Output, rows: list[int], move: torch.Tensor,
                desc) -> C3Output:
    """The universal heads on lines ``rows`` with r moved by ``move``: what the
    heads would say if each line were a little different."""
    r = out.r[rows] + move
    sub = C3Output(h=out.h[rows], z_c=out.z_c[rows], q_mu=out.q_mu[rows],
                   q_logvar=out.q_logvar[rows], q_logit=out.q_logit[rows], r=r,
                   residual=out.residual[rows])
    model.universal_heads(sub, r, out.q_mean[rows], desc)
    return sub


def near_miss_loss(out: C3Output, batch: Batch, margin: float = 1.0) -> torch.Tensor:
    """Lines that read almost alike but were labeled differently: each must
    prefer its own answer over its twin's by ``margin`` (in logits), so the
    heads learn the small difference instead of averaging the two."""
    total, terms = out.r.new_zeros(()), 0
    for i, j, key in batch.near_misses:
        lg = out.logits.get(key)
        if lg is None:
            continue
        yi, yj = batch.targets[key][i], batch.targets[key][j]
        total = total + F.relu(margin - (lg[i, yi] - lg[i, yj])) + F.relu(margin - (lg[j, yj] - lg[j, yi]))
        terms += 2
    return total / max(terms, 1)


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


def explanation_only_loss(out: C3Output, batch: Batch) -> torch.Tensor:
    """CE of the explanation votes by themselves. Trains the reader to apply
    another line's reasoning (its own WHY is masked) to this line."""
    total, terms = out.r.new_zeros(()), 0
    for key, votes in out.explained.items():
        y = torch.tensor(batch.targets[key], device=votes.device)
        if (y != IGNORE).any():
            total = total + F.cross_entropy(votes, y, ignore_index=IGNORE)
            terms += 1
    return total / max(terms, 1)


def why_echo_loss(model: C3Model, out: C3Output, batch: Batch) -> torch.Tensor:
    """The model's voiced reason, r_to_text(r_i), should land on the line's
    encoded WHY (cosine). It is what ask.py compiles as a pseudo-explanation."""
    idx = [i for i, t in enumerate(batch.why) if t]
    if not idx:
        return out.r.new_zeros(())
    target = model.text([batch.why[i] for i in idx]).detach()
    return (1 - F.cosine_similarity(model.r_to_text(out.r[idx]), target, -1)).mean()


def clause_use_loss(model: C3Model, out: C3Output, batch: Batch, bank: ExplanationBank,
                    desc, margin: float = 0.02) -> torch.Tensor:
    """Nothing in a paragraph may be dead weight. Pick one condition,
    exception, cause, consequence, evidence or quantity clause of one
    explanation at random, delete it, and require the answers to move by at
    least ``margin`` (mean KL over the questions the explanations touched).
    Costs one extra forward pass."""
    import dataclasses
    import random

    from .clauses import STATEMENT, split_clauses

    cands = []
    for j, e in enumerate(bank.items):
        cs = split_clauses(e.text)
        if len(cs) > 1:
            cands += [(j, ci, cs) for ci, c in enumerate(cs) if c.role != STATEMENT]
    keys = [k for k in out.gates if k in out.logits]
    if not cands or not keys:
        return out.r.new_zeros(())
    j, ci, cs = random.choice(cands)
    shorter = " ".join(c.text for i, c in enumerate(cs) if i != ci)
    items = list(bank.items)
    items[j] = dataclasses.replace(items[j], text=shorter)
    out2 = model(batch.inputs(), company=batch.company, desc=desc, bank=ExplanationBank(items))
    kl = torch.stack([F.kl_div(out2.logits[k].log_softmax(-1), out.logits[k].log_softmax(-1),
                               log_target=True, reduction="batchmean") for k in keys]).mean()
    return F.relu(margin - kl)


def c3_loss(model: C3Model, batch: Batch, w: LossWeights | None = None,
            adv_lambda: float = 1.0, bank: ExplanationBank | None = None,
            rules: ExplanationBank | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """``bank``: explanations for the v5 reader to read. Default: none, unless
    ``rules`` are given or ``C3Config.legacy_reasons`` is set; then this
    deal's WHYs and company lines (each line masked from its own), plus
    ``rules``. In real training
    the bank should hold other deals' explanations too, which is what the
    model will have at inference."""
    w = w or LossWeights()
    desc = model.describe()
    # v6: reading explanations is the brain's job (brain.py, brain_loss).
    # The v5 bank (reader votes, compiled operators, rule cards, clause
    # slots) runs only when asked for, as a baseline to beat.
    if bank is None and (rules is not None or model.cfg.legacy_reasons):
        bank = ExplanationBank.from_batch(batch, model.schema)
    if rules is not None:
        bank = bank.extend(rules.items)
    out = model(batch.inputs(), company=batch.company, desc=desc, adv_lambda=adv_lambda,
                bank=bank)
    parts: dict[str, torch.Tensor] = {}
    parts["heads"], parts["numbers"] = head_losses(model, out, batch, w, "universal")
    parts["conduct"], _ = head_losses(model, out, batch, w, "company")
    parts["relations"] = relation_loss(out, batch)
    parts["governs"] = governs_loss(out, batch)
    parts["why_align"], parts["why_sufficiency"] = why_losses(model, out, batch, desc, w)
    parts["residual"] = out.residual.pow(2).mean()
    parts["hindsight"], parts["variance"] = hindsight_loss(model, out, batch)
    parts["absence"] = absence_loss(out, batch)
    parts["explanation_only"] = explanation_only_loss(out, batch)
    from .consequence import changes_loss, claims_loss, rule_link_loss  # noqa: PLC0415 (cycle)

    parts["changes"] = changes_loss(out, batch)
    parts["claims"] = claims_loss(out, batch, bank)
    parts["rule_links"] = rule_link_loss(out, batch, bank)
    parts["why_echo"] = why_echo_loss(model, out, batch)
    parts["pointers"] = pointer_loss(out, batch)
    parts["entities"] = entity_loss(out, batch)
    parts["judgments"] = judgment_loss(model, out, batch, desc)
    parts["negatives"] = negative_loss(out, batch)
    parts["near_misses"] = near_miss_loss(out, batch, w.near_miss_margin)
    if w.clause_use and bank is not None:
        parts["clause_use"] = clause_use_loss(model, out, batch, bank, desc)
    if len(model.companies) > 1 and batch.company in model.companies:
        y = torch.full((len(batch),), model.companies.index(batch.company), device=out.r.device)
        parts["adversary"] = F.cross_entropy(out.company_logits, y)
    parts["policy_l1"] = out.alpha.abs().sum() if out.alpha is not None else out.r.new_zeros(())
    total = sum(getattr(w, k) * v for k, v in parts.items())
    return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in parts.items()}


# ---------------------------------------------------------------- v6: the brain
@dataclass
class BrainWeights:
    direct: float = 1.0        # the label, answered from the bare page
    reasoned: float = 1.0      # the label, answered with the human's WHY on the page
    rationale: float = 1.0     # write the human's WHY, token by token
    distill: float = 0.5       # the direct answer moves toward what the reasoning concludes
    consolidate: float = 0.5   # notes written into weights must act as notes on the page
    flip: float = 0.5          # with a look-alike line supposed, answer what that line was labeled
    numbers: float = 0.2


def _answer_ce(out, batch: Batch, keys) -> torch.Tensor:
    total, terms = None, 0
    for k in keys:
        if k not in out.logits:
            continue
        y = torch.tensor(batch.targets[k], device=out.logits[k].device)
        if (y != IGNORE).any():
            ce = F.cross_entropy(out.logits[k], y, ignore_index=IGNORE)
            total = ce if total is None else total + ce
            terms += 1
    if total is None:
        return next(iter(out.logits.values())).new_zeros(())
    return total / terms


def supposed_twins(batch: Batch) -> tuple[list[FlipTarget], list[tuple[str, int]]]:
    """Practice for the supposition pass, from labels alone: line i's page
    with "Suppose instead: <line j's text>", for look-alike lines i and j,
    must answer what j was labeled (on the head where they differ for a near
    miss, the type for twins). The teacher learns to read a supposed case
    and apply the page's rule to it, so its reading of a WHY's sentences
    (supercharge.teach_flip) comes from comprehension, not from words."""
    sup: list[FlipTarget] = []
    gold: list[tuple[str, int]] = []
    pairs = [(i, j, k) for i, j, k in batch.near_misses] + \
            [(i, j, "col:label_type") for i, j in batch.twins]
    for i, j, k in pairs:
        for a, b in ((i, j), (j, i)):
            sup.append(FlipTarget(a, batch.texts[b]))
            gold.append((k, batch.targets[k][b]))
    return sup, gold


def brain_loss(brain, batch: Batch, w: BrainWeights | None = None,
               graph: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Every v6 term for one deal (brain.py has the table). Notes come from
    the deal's other lines; the reasoned pass and the rationale use each
    line's own WHY, which the direct pass never sees."""
    from .brain import company_field_notes, notes_from_deal, told_why  # noqa: PLC0415
    from .consolidate import context_distillation_loss  # noqa: PLC0415

    w = w or BrainWeights()
    desc = brain.describe()
    keys = [k for k in batch.targets]
    notes, cnotes = notes_from_deal(batch)
    company = batch.company or None
    parts: dict[str, torch.Tensor] = {}

    direct = brain(batch, company=company, notes=notes, company_notes=cnotes, graph=graph, desc=desc)
    why = told_why(batch, brain.schema)
    cfield = company_field_notes(batch, brain.schema)
    reasoned = brain(batch, company=company, why=why, notes=notes,
                     company_notes=[c + f for c, f in zip(cnotes, cfield)], graph=graph, desc=desc)
    parts["direct"] = _answer_ce(direct, batch, keys)
    parts["reasoned"] = _answer_ce(reasoned, batch, keys)
    parts["rationale"] = brain.rationale_loss(batch, notes, graph)

    has_why = torch.tensor([bool(t) for t in why], device=parts["direct"].device)
    if has_why.any():
        parts["distill"] = torch.stack([
            F.kl_div(direct.logits[k][has_why].log_softmax(-1),
                     reasoned.logits[k][has_why].detach().log_softmax(-1),
                     log_target=True, reduction="batchmean") for k in direct.logits]).mean()
    else:
        parts["distill"] = parts["direct"].new_zeros(())

    # Context distillation: the deal's notes, once on the page (teacher) and
    # once written into weights with the page bare (student).
    uni = sorted({t for ns in notes for t in ns})
    com = sorted({t for ns in cnotes for t in ns})
    if uni or com:
        with torch.no_grad():
            teacher = brain(batch, company=company, notes=[uni] * len(batch),
                            company_notes=[com] * len(batch), graph=graph, desc=desc)
        student = brain(batch, company=company, graph=graph, desc=desc,
                        deltas=brain.memory_universal(brain.lm, uni),
                        company_deltas=brain.memory_company(brain.lm, com))
        parts["consolidate"] = context_distillation_loss(teacher.logits, student.logits)
    else:
        parts["consolidate"] = parts["direct"].new_zeros(())

    parts["flip"] = parts["direct"].new_zeros(())
    sup, gold = supposed_twins(batch)
    if sup:
        flipped = brain.flip_read(batch, why, sup, notes, desc, graph)
        terms = [F.cross_entropy(flipped.logits[k][n].unsqueeze(0),
                                 torch.tensor([y], device=flipped.logits[k].device))
                 for n, (k, y) in enumerate(gold) if k in flipped.logits]
        if terms:
            parts["flip"] = torch.stack(terms).mean()

    num = parts["direct"].new_zeros(())
    for k, pred in direct.numbers.items():
        vals = batch.numbers_target.get(k, [])
        m = torch.tensor([v is not None for v in vals], device=pred.device, dtype=torch.bool)
        if m.any():
            t = torch.tensor([v or 0.0 for v in vals], device=pred.device)
            num = num + F.smooth_l1_loss(pred[m], torch.log1p(t[m]))
    parts["numbers"] = num
    total = sum(getattr(w, k) * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
