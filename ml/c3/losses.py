"""Every loss C3 trains with, one function each, combined in ``c3_loss``.

Nothing here runs a training loop: the training thread owns the schedule
(the stages in base-architecture-v3.md section 3). Each term is a named
entry in the returned dict, so an ablation is "set its weight to 0".
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

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
    clause_use: float = 0.0        # legacy clause/compiler ablation; opt in explicitly
    pointers: float = 0.3          # hint_refs: the lines a decision came from
    entities: float = 0.2          # entity_keys: lines naming the same entity pull together
    judgments: float = 1.0         # judgment tabs about two lines, a group or the deal
    judgment_rationale: float = 0.2 # privileged card paragraphs align card representations
    negatives: float = 0.3         # answers known wrong (the parser's type a person overruled)
    near_misses: float = 0.5       # lines that read alike but answer differently stay apart
    near_miss_margin: float = 1.0
    literals: float = 0.4          # unique in-line spans of open value strings
    deal_numbers: float = 0.2      # deal-answer site count
    space: dict[str, float] = field(default_factory=dict)  # optional per-space scale


def row_weights(batch: Batch, device) -> torch.Tensor:
    """Per-line weight from weight_tier (load_bearing 3, ordinary 1, slight 0.3)."""
    w = batch.weights if len(batch.weights) == len(batch) else [1.0] * len(batch)
    return torch.tensor(w, device=device, dtype=torch.float32)


def why_row_weights(batch: Batch, device) -> torch.Tensor:
    """Per-line weight of the WHY: 1.0, or DRAFT_WHY_WEIGHT for a model's
    draft nobody saved (data.why_weight)."""
    w = batch.why_weights if len(batch.why_weights) == len(batch) else [1.0] * len(batch)
    return torch.tensor(w, device=device, dtype=torch.float32)


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged with row weights over the lines with a label."""
    m = y != IGNORE
    ce = F.cross_entropy(logits, y.clamp(min=0), reduction="none")
    return (ce * rw * m).sum() / (rw * m).sum().clamp(min=1e-6)


def answer_loss(logits: torch.Tensor, batch: Batch, key: str, rw: torch.Tensor,
                keep: torch.Tensor | None = None) -> torch.Tensor:
    """Saved sets use independent sigmoid answers; unknown rows teach nothing."""
    if key not in batch.multi_targets:
        y = torch.tensor(batch.targets[key], device=logits.device)
        if keep is not None:
            y = y.masked_fill(~keep, IGNORE)
        return weighted_ce(logits, y, rw)
    values = batch.multi_targets[key]
    mask = torch.tensor([v is not None for v in values], device=logits.device)
    if keep is not None:
        mask = mask & keep
    if not mask.any():
        return logits.sum() * 0
    target = torch.zeros_like(logits)
    for i, ids in enumerate(values):
        if ids:
            target[i, list(ids)] = 1
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(-1)
    return (per * rw * mask).sum() / (rw * mask).sum().clamp(min=1e-6)


def has_answers(batch: Batch, key: str) -> bool:
    return (any(v is not None for v in batch.multi_targets[key]) if key in batch.multi_targets
            else any(v != IGNORE for v in batch.targets[key]))


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
    # Line x entity incidence: two lines share an entity when their rows meet.
    # A Python N x N loop is minutes per step on an 8,192-line window.
    vocab = {e: k for k, e in enumerate(sorted({e for s in ents for e in s}))}
    rows = [i for i, s in enumerate(ents) for _ in s]
    cols = [vocab[e] for s in ents for e in s]
    inc = torch.zeros(n, len(vocab), device=out.r.device)
    inc[rows, cols] = 1.0
    pos = (inc @ inc.T) > 0
    pos.fill_diagonal_(False)
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
        if has_answers(batch, opp.key):
            ce = ce + w.space.get(opp.space, 1.0) * answer_loss(out.logits[opp.key], batch, opp.key, rw)
            terms += 1
        if opp.kind == NUMBER and opp.key in out.numbers:
            vals = batch.numbers_target.get(opp.key, [])
            mask = torch.tensor([v is not None for v in vals], device=dev)
            if mask.any():
                t = torch.tensor([v or 0.0 for v in vals], device=dev)
                num = num + F.smooth_l1_loss(out.numbers[opp.key][mask], torch.log1p(t[mask]))
    return ce / max(terms, 1), num


def judgment_loss(model: C3Model, out: C3Output, batch: Batch,
                  desc: dict | None = None, *,
                  sizes: tuple[str, ...] | None = None) -> torch.Tensor:
    """Cross-entropy on each judgment-tab verdict about two lines, a group of
    lines or the whole deal, one batch per question. Company questions train
    only when the company layer ran. ``sizes`` limits the pass to those card
    sizes; None keeps every card."""
    by: dict[str, list] = {}
    for j in batch.judged:
        by.setdefault(j.key, []).append(j)
    opps = model.schema.by_key()
    total, terms = out.r.new_zeros(()), 0
    for key, js in by.items():
        o = opps.get(key)
        if o is None or (not o.universal and out.conduct_x is None):
            continue
        if sizes is not None and o.size not in sizes:
            continue
        logits = model.judge(out, key, [j.lines for j in js], desc, queries=[j.query for j in js])
        y = torch.tensor([j.answer for j in js], device=logits.device)
        if o.multiple:
            truth = torch.zeros_like(logits)
            for i, j in enumerate(js):
                if j.answers is None:
                    raise ValueError("Set-valued judgment needs its complete answer set")
                truth[i, list(j.answers)] = 1
            total = total + F.binary_cross_entropy_with_logits(logits, truth)
        else:
            total = total + F.cross_entropy(logits, y)
        terms += 1
    return total / max(terms, 1)


def judgment_rationale_loss(model: C3Model, out: C3Output, batch: Batch, *,
                            sizes: tuple[str, ...] | None = None) -> torch.Tensor:
    """Privileged card WHY alignment. Notes never enter task inference inputs.

    This is a paragraph embedding loss, not a substitute for the pretrained
    teacher's reasoning/grounding objectives or free-text output consumers.
    """
    by: dict[str, list] = {}
    for j in batch.judged:
        if j.note and j.why_weight > 0:
            by.setdefault(j.key, []).append(j)
    total, terms = out.r.new_zeros(()), 0
    for key, js in by.items():
        o = model.schema.by_key()[key]
        if sizes is not None and o.size not in sizes:
            continue
        if not o.universal and out.conduct_x is None:
            continue
        features = model.judgment_features(out, key, [j.lines for j in js], [j.query for j in js])
        # No WHY gradients flow into the source encoder via the target branch.
        notes = model.judgment_note(model.aux_text([j.note for j in js]).detach())
        loss = 1 - F.cosine_similarity(features, notes, dim=-1)
        weights = loss.new_tensor([j.why_weight for j in js])
        total = total + (loss * weights).sum() / weights.sum().clamp(min=1e-6)
        terms += 1
    return total / max(terms, 1)


def literal_span_loss(model: C3Model, out: C3Output, batch: Batch) -> torch.Tensor:
    """Put pointer mass on the unique word span of a saved open string.

    A value that is not a unique span of its line is not assigned a guessed
    location; the teacher token loss still sees the exact string.
    """
    grouped: dict[int, set[int]] = {}
    for item in batch.literals:
        if item.words:
            grouped.setdefault(item.line, set()).update(item.words)
    if not grouped:
        return out.r.new_zeros(())
    weights = model.word_weights(out, batch.texts, rows=set(grouped))
    total, terms = out.r.new_zeros(()), 0
    for i, gold in grouped.items():
        logp = weights[i]
        if logp is None:
            continue
        idx = [w for w in gold if w < logp.shape[0]]
        if not idx:
            continue
        total = total - torch.logsumexp(logp[idx], 0)
        terms += 1
    return total / max(terms, 1)


def deal_number_loss(model: C3Model, out: C3Output, batch: Batch) -> torch.Tensor:
    count = batch.deal_numbers.get("declared_site_count")
    if count is None:
        return out.r.new_zeros(())
    pred = model.deal_number(out.r.mean(0)).squeeze(-1)
    return F.smooth_l1_loss(pred, torch.log1p(pred.new_tensor(float(count))))


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
                   residual=out.residual[rows],
                   slots=out.slots[rows] if out.slots is not None else None)
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
    """Gold edges pick their target. A labeled line with no such edge picks no link.

    Other lines in the row are the "not this one" choices. Abstain is the last
    class when the model has a no-link score. Unlabeled lines are left alone:
    a missing card is not a lesson that nothing connects.
    """
    total, terms = out.r.new_zeros(()), 0
    labeled = {i for i, on in enumerate(batch.labeled) if on}
    gold: dict[str, dict[int, list[int]]] = {}
    for rel, pairs in batch.edges.items():
        for src, dst in pairs:
            gold.setdefault(rel, {}).setdefault(src, []).append(dst)
    rels = set(out.relations) | set(gold)
    for rel in rels:
        s = out.relations.get(rel)
        if s is None:
            continue
        none = out.link_none.get(rel)
        by_src = gold.get(rel, {})
        away = set(batch.link_elsewhere.get(rel, ()))
        for src, dsts in by_src.items():
            row = s[src]
            if none is not None:
                row = torch.cat([row, none[src].view(1)])
            logp = row.log_softmax(-1)
            for dst in dsts:
                if torch.isfinite(s[src, dst]):
                    total = total - logp[dst]
                    terms += 1
        if none is None:
            continue
        for src in labeled:
            if src in by_src or src in away:
                continue
            row = torch.cat([s[src], none[src].view(1)])
            total = total - row.log_softmax(-1)[-1]
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
    why = model.why_proj(model.aux_text([batch.why[i] for i in idx]))
    r = F.normalize(out.r[idx], dim=-1)
    sim = r @ F.normalize(why, dim=-1).T / tau
    target = torch.arange(len(idx), device=sim.device)
    ww = why_row_weights(batch, sim.device)[idx]
    align = 0.5 * ((F.cross_entropy(sim, target, reduction="none") * ww).sum()
                   + (F.cross_entropy(sim.T, target, reduction="none") * ww).sum()) / ww.sum()

    stand_in = out.r.clone()
    stand_in[idx] = why + model.res_up(out.residual[idx])
    sub = C3Output(h=out.h, z_c=out.z_c, q_mu=out.q_mu, q_logvar=out.q_logvar,
                   q_logit=out.q_logit, r=stand_in, residual=out.residual)
    model.universal_heads(sub, stand_in, out.q_mean, desc)
    keep = torch.zeros(len(batch), dtype=torch.bool, device=sim.device)
    keep[idx] = True
    rw = why_row_weights(batch, sim.device)
    suff, terms = out.r.new_zeros(()), 0
    for key, lg in sub.logits.items():
        if has_answers(batch, key):
            suff = suff + answer_loss(lg, batch, key, rw, keep)
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
        if has_answers(batch, key):
            total = total + answer_loss(votes, batch, key, votes.new_ones(len(batch)))
            terms += 1
    return total / max(terms, 1)


def why_echo_loss(model: C3Model, out: C3Output, batch: Batch) -> torch.Tensor:
    """The model's voiced reason, r_to_text(r_i), should land on the line's
    encoded WHY (cosine). It is what ask.py compiles as a pseudo-explanation."""
    idx = [i for i, t in enumerate(batch.why) if t]
    if not idx:
        return out.r.new_zeros(())
    target = model.aux_text([batch.why[i] for i in idx]).detach()
    ww = why_row_weights(batch, out.r.device)[idx]
    return ((1 - F.cosine_similarity(model.r_to_text(out.r[idx]), target, -1)) * ww).sum() / ww.sum()


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
            rules: ExplanationBank | None = None,
            judgment_sizes: tuple[str, ...] | None = None,
            keep: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """``keep``, when given, receives the forward pass (``out``) and the
    encoded descriptions (``desc``), so a lesson on the same step reuses them
    instead of running the trunk and the encoder again.

    ``bank``: explanations for the v5 reader to read. Default: none, unless
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
    if keep is not None:
        keep["out"], keep["desc"] = out, desc
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
    parts["judgments"] = judgment_loss(model, out, batch, desc, sizes=judgment_sizes)
    parts["judgment_rationale"] = judgment_rationale_loss(model, out, batch, sizes=judgment_sizes)
    parts["literals"] = literal_span_loss(model, out, batch)
    parts["deal_numbers"] = deal_number_loss(model, out, batch)
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
    card_direct: float = 1.0   # separate card subjects/evidence
    card_reasoned: float = 1.0 # full card WHY used by training-only teacher
    rationale: float = 1.0     # write the human's WHY, token by token
    values: float = 1.0        # exact saved value strings, token by token
    distill: float = 0.5       # the direct answer moves toward what the reasoning concludes
    consolidate: float = 0.5   # notes written into weights must act as notes on the page
    flip: float = 0.5          # with a look-alike line supposed, answer what that line was labeled
    numbers: float = 0.2


def _answer_ce(out, batch: Batch, keys) -> torch.Tensor:
    total, terms = None, 0
    for k in keys:
        if k not in out.logits:
            continue
        if has_answers(batch, k):
            ce = answer_loss(out.logits[k], batch, k, out.logits[k].new_ones(len(batch)))
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


def teacher_rows(batch: Batch, why: list[str | None]) -> tuple[list[int] | None, Batch]:
    """The lines the teacher's per-line terms can learn from: a saved answer,
    a number or a WHY. Context-only lines (a window's overlap, an edge pack,
    the digest) and unlabeled lines carry none, and reading them was most of
    the teacher's GPU time. Returns ``(rows, targets of those rows)``;
    ``rows`` is None when every line counts."""
    n = len(batch)
    keep = [bool(why[i]) for i in range(n)]
    for vals in batch.targets.values():
        for i, v in enumerate(vals):
            keep[i] = keep[i] or v != IGNORE
    for table in (batch.multi_targets, batch.numbers_target):
        for vals in table.values():
            for i, v in enumerate(vals):
                keep[i] = keep[i] or v is not None
    rows = [i for i in range(n) if keep[i]] or [0]
    if len(rows) == n:
        return None, batch
    return rows, rows_slice(batch, rows)


def rows_slice(batch: Batch, rows: list[int]) -> Batch:
    """The targets of ``rows`` only, for scoring a reading of those lines."""
    pick = lambda vals: [vals[i] for i in rows]  # noqa: E731
    return replace(batch, texts=pick(batch.texts),
                   targets={k: pick(v) for k, v in batch.targets.items()},
                   multi_targets={k: pick(v) for k, v in batch.multi_targets.items()},
                   numbers_target={k: pick(v) for k, v in batch.numbers_target.items()})


def teacher_context(brain, batch: Batch) -> dict:
    """What every chunk of a part's teacher loss shares: notes from the
    deal's other lines and each line's WHY (both read the whole part)."""
    from .brain import company_field_notes, notes_from_deal, told_why  # noqa: PLC0415

    notes, cnotes = notes_from_deal(batch)
    return {"notes": notes, "cnotes": cnotes, "why": told_why(batch, brain.schema),
            "cfield": company_field_notes(batch, brain.schema)}


# Grad passes a line costs in one piece: direct and reasoned, each on the
# universal and the company page, plus its WHY and its value. A card costs
# its direct and its reasoned page. A flip costs one page.
ROW_PASSES, CARD_PASSES, FLIP_PASSES = 6, 2, 1


@dataclass
class Piece:
    """One backward's worth of a part's teacher loss. A piece holds one kind
    of term (lines, cards or flips); ``scale`` is 1 / pieces of its kind, so
    every term weighs what it would on the whole part."""
    rows: list[int]
    cards: list
    flips: list[int]
    extras: bool
    scale: float
    tokens: int


def _pack(items: list, cost: list[int], size: int, budget: int) -> list[list]:
    """Greedy, cheapest first: at most ``size`` items and ``budget`` cost per
    group; an item over the budget on its own goes alone."""
    order = sorted(range(len(items)), key=lambda k: cost[k])
    groups, cur, used = [], [], 0
    for k in order:
        if cur and ((size and len(cur) >= size) or (budget and used + cost[k] > budget)):
            groups.append(cur)
            cur, used = [], 0
        cur.append(k)
        used += cost[k]
    if cur:
        groups.append(cur)
    return groups


def teacher_chunks(brain, batch: Batch, size: int, budget: int = 0, *,
                   caps: tuple[int, int, int] = (0, 0, 0), rng=None) -> list[Piece]:
    """A part's teacher loss in pieces of at most ``size`` lines, cards or
    flips and about ``budget`` grad tokens each. Cards the teacher cannot read
    whole are left out (``brain.teacher_cards``). ``size`` and ``budget`` 0 is
    one piece holding everything. Deal-level values ride the first piece.

    ``caps`` (lines, cards, flips; 0 is all) with ``rng`` teach a fresh random
    sample of the part each step, so a step costs the same on any part and
    the steps together cover it."""
    from .brain import teacher_cards, told_why  # noqa: PLC0415

    why = told_why(batch, brain.schema)
    sup, _ = supposed_twins(batch)
    rows, _ = teacher_rows(batch, why)
    rows = list(range(len(batch))) if rows is None else rows
    judged = teacher_cards(brain, batch)
    flip_ids = list(range(len(sup)))
    if rng is not None:
        def take(items, cap):
            return sorted(rng.sample(items, cap)) if cap and len(items) > cap else items
        rows = take(rows, caps[0])
        picked = take(list(range(len(judged))), caps[1])
        judged = [judged[k] for k in picked]
        flip_ids = take(flip_ids, caps[2])
    if size <= 0 and budget <= 0:
        return [Piece(None if rng is None else rows, judged, flip_ids, True, 1.0, 0)]
    count = brain.lm.token_count
    row_cost = [ROW_PASSES * count(brain.page(batch, i) + _why_tail(why, i)) for i in rows]
    card_cost = [CARD_PASSES * count(brain.card_page(batch, j, with_why=True)) for j in judged]
    flip_cost = [FLIP_PASSES * count(brain.page(batch, sup[k].line) + _why_tail(why, sup[k].line)
                                     + sup[k].condition) for k in flip_ids]
    pieces: list[Piece] = []
    for kind, items, cost in (("rows", rows, row_cost), ("cards", judged, card_cost),
                              ("flips", flip_ids, flip_cost)):
        groups = _pack(items, cost, size, budget)
        for g in groups:
            picked = [items[k] for k in g]
            pieces.append(Piece(
                rows=sorted(picked) if kind == "rows" else [],
                cards=picked if kind == "cards" else [],
                flips=sorted(picked) if kind == "flips" else [],
                extras=False, scale=1.0 / len(groups), tokens=sum(cost[k] for k in g)))
    if not pieces:
        pieces.append(Piece([], [], [], True, 1.0, 0))
    pieces[0].extras = True
    return pieces


def _why_tail(why, i: int) -> str:
    return f"Why: {why[i]}\n" if why[i] else ""


def brain_loss(brain, batch: Batch, w: BrainWeights | None = None,
               graph: torch.Tensor | None = None, *, rows: list[int] | None = None,
               judged: list | None = None, extras: bool = True, desc=None,
               context: dict | None = None,
               flips: list[int] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Every v6 term for one deal (brain.py has the table). Notes come from
    the deal's other lines; the reasoned pass and the rationale use each
    line's own WHY, which the direct pass never sees.

    ``rows`` / ``judged`` / ``flips`` / ``extras`` score one piece of the
    part (``teacher_chunks``); ``context`` lets the pieces share the notes.
    A term weighted 0 is not computed."""
    from .consolidate import context_distillation_loss  # noqa: PLC0415
    from .supercharge import _kl  # noqa: PLC0415

    w = w or BrainWeights()
    desc = desc if desc is not None else brain.describe()
    keys = [k for k in batch.targets]
    context = context or teacher_context(brain, batch)
    notes, cnotes, why, cfield = context["notes"], context["cnotes"], context["why"], context["cfield"]
    company = batch.company or None
    parts: dict[str, torch.Tensor] = {}
    zero = next(brain.parameters()).new_zeros(())

    # Only lines with something to teach get a page; the rest of the part is
    # still on those pages as earlier lines.
    if rows is None:
        rows, taught = teacher_rows(batch, why)
    else:
        taught = rows_slice(batch, rows)
    if rows == []:
        direct = reasoned = None
        parts["direct"] = parts["reasoned"] = zero
    else:
        direct = brain(batch, company=company, notes=notes, company_notes=cnotes, graph=graph,
                       desc=desc, rows=rows)
        reasoned = brain(batch, company=company, why=why, notes=notes,
                         company_notes=[c + f for c, f in zip(cnotes, cfield)], graph=graph, desc=desc,
                         rows=rows)
        parts["direct"] = _answer_ce(direct, taught, keys)
        parts["reasoned"] = _answer_ce(reasoned, taught, keys)
    parts["rationale"] = brain.rationale_loss(batch, notes, graph, rows=rows)
    parts["values"] = brain.value_loss(batch, rows=rows, deal_level=extras)
    cards = batch if judged is None else replace(batch, judged=list(judged))
    for name, with_why in (("card_direct", False), ("card_reasoned", True)):
        predictions = brain.card_logits(cards, with_why=with_why, desc=desc)
        grouped = {}
        for j in cards.judged:
            grouped.setdefault(j.key, []).append(j)
        terms = []
        for key, logits in predictions.items():
            js = grouped[key]
            if brain.schema.by_key()[key].multiple:
                truth = torch.zeros_like(logits)
                for i, j in enumerate(js):
                    truth[i, list(j.answers)] = 1
                per_row = F.binary_cross_entropy_with_logits(logits, truth, reduction="none").mean(-1)
            else:
                truth = torch.tensor([j.answer for j in js], device=logits.device)
                per_row = F.cross_entropy(logits, truth, reduction="none")
            weights = per_row.new_tensor([j.why_weight if with_why else 1.0 for j in js])
            terms.append((per_row * weights).sum() / weights.sum().clamp(min=1e-6))
        parts[name] = torch.stack(terms).mean() if terms else zero

    has_why = torch.tensor([bool(why[i]) for i in (rows if rows is not None else range(len(batch)))],
                           device=zero.device, dtype=torch.bool)
    if direct is not None and has_why.any():
        parts["distill"] = torch.stack([
            _kl(direct.logits[k][has_why], reasoned.logits[k][has_why].detach(), 1.0,
                brain.schema.by_key()[k].multiple) for k in direct.logits]).mean()
    else:
        parts["distill"] = parts["direct"].new_zeros(())

    # Context distillation: the deal's notes, once on the page (teacher) and
    # once written into weights with the page bare (student).
    uni = sorted({t for ns in notes for t in ns})
    com = sorted({t for ns in cnotes for t in ns})
    if (uni or com) and direct is not None and w.consolidate:
        with torch.no_grad():
            teacher = brain(batch, company=company, notes=[uni] * len(batch),
                            company_notes=[com] * len(batch), graph=graph, desc=desc, rows=rows)
        student = brain(batch, company=company, graph=graph, desc=desc, rows=rows,
                        deltas=brain.memory_universal(brain.lm, uni),
                        company_deltas=brain.memory_company(brain.lm, com))
        parts["consolidate"] = context_distillation_loss(teacher.logits, student.logits,
            multiple={o.key for o in brain.schema.opportunities if o.multiple})
    else:
        parts["consolidate"] = parts["direct"].new_zeros(())

    parts["flip"] = zero
    sup, gold = supposed_twins(batch) if (w.flip and (extras or flips)) else ([], [])
    if flips is not None and sup:
        sup, gold = [sup[k] for k in flips], [gold[k] for k in flips]
    if sup:
        flipped = brain.flip_read(batch, why, sup, notes, desc, graph)
        terms = [F.cross_entropy(flipped.logits[k][n].unsqueeze(0),
                                 torch.tensor([y], device=flipped.logits[k].device))
                 for n, (k, y) in enumerate(gold) if k in flipped.logits]
        if terms:
            parts["flip"] = torch.stack(terms).mean()

    num = zero
    for k, pred in (direct.numbers.items() if direct is not None else ()):
        vals = taught.numbers_target.get(k, [])
        m = torch.tensor([v is not None for v in vals], device=pred.device, dtype=torch.bool)
        if m.any():
            t = torch.tensor([v or 0.0 for v in vals], device=pred.device)
            num = num + F.smooth_l1_loss(pred[m], torch.log1p(t[m]))
    parts["numbers"] = num
    total = sum(getattr(w, k) * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
