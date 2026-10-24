"""Run E why2: the written reasoning that no loss read, and the text path on its own.

arch_review.md (§5 ranks 1-4, §6, §7, §8) found that about half of the
reasoning words reached no teacher-free loss, every company decision trained
from labels alone, and the described (text) path was a residual under free
rows that read the line's raw state ``h``. The parts here, each a named part
of ``c3_loss`` (LossWeights field; 0 turns it off), close those gaps without a
teacher. None changes the read path: serve reads as before.

``described_alone``  head CE on the described logits alone (``out.logits -
                     out.free_logits``) for the free-row heads that matter
                     (free_rows.R_ONLY_KEYS), on ``described_alone_p`` of the
                     steps: the text path must stand on its own, and the free
                     row becomes a correction, not the classifier.
``card_alone``       each line card's own note (a line-size judgment note, a
                     reason-only card's note in ``Batch.reason_notes``, a
                     ``<field>_note``, a ruled-out reading's reason: all of
                     ``Batch.field_notes`` but the link notes) stands in for r
                     (``why_proj``), with h and the asked slots blank, and must
                     reach *that head's* answer (class CE / set BCE / number
                     smooth L1). Up to ``card_cap`` notes a step.
``card_contrast``    per head, r_i toward its card note against that head's
                     other *distinct* notes (InfoNCE both ways): the card's
                     reason has a place in r.
``co_why_contrast``  the conduct layer's line (``out.conduct_x``) toward its
                     ``[company]`` line (``Batch.policy_note``: the label's
                     [purtera] line plus the [purtera] parts of its cards and
                     link notes, and the company heads' own card notes) through
                     ``company_why_proj``, against the other distinct lines.
``co_why_alone``     the projected company line stands in for the conduct
                     line and must reach the company answers through the
                     conduct head (and the company free rows): the policy line
                     on every company head the line answers; a company card
                     note on its own head. Only the conduct path reads either,
                     so company text never reaches the base.
``number_why_alone`` the label WHY alone (as why_alone) must reach the line's
                     number heads (smooth L1 on log1p, as ``numbers``).
``judgment_why_alone`` a pair / group / deal card's note alone (projected by
                     ``judgment_note``, as judgment_rationale does; a company
                     card's by ``company_why_proj``) stands in for the card's
                     pooled feature and must reach its verdict. Trains with the
                     cards (only when ``judgments`` is on, on the stage's sizes).

``before_forward`` sets the two knobs that are not parts: the r-only free
rows' share of h (``1 - free_h_out``, free_rows.py) and the company layer's
gradient scale into the trunk (``company_r_grad``).

Every encoded note goes through ``model.aux_text``: no gradient into a
released pretrained encoder from the target side; the projections learn.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from . import free_rows
from .data import IGNORE, Batch
from .losses import why_mean
from .model import C3Model, C3Output
from .schema import NUMBER, RELATION

#: The heads whose described logits must stand alone (with their free rows' share left out).
DESCRIBED_KEYS = free_rows.R_ONLY_KEYS


def _norm(t: Any) -> str:
    return " ".join(str(t or "").split())


def before_forward(model: C3Model, w) -> None:
    """The step's knobs: r-only rows' h share and the company gradient scale."""
    model.company_r_grad = float(getattr(w, "company_r_grad", 1.0))
    if free_rows.r_only_keys(model):
        free_rows.set_h_keep(model, 1.0 - float(getattr(w, "free_h_out", 1.0)))


# ---------------------------------------------------------------- targets

def _has_target(batch: Batch, key: str, i: int, kind: str) -> bool:
    if kind == NUMBER:
        vals = batch.numbers_target.get(key)
        return vals is not None and i < len(vals) and vals[i] is not None
    if key in batch.multi_targets:
        return batch.multi_targets[key][i] is not None
    col = batch.targets.get(key)
    return col is not None and col[i] != IGNORE


def _row_loss(model: C3Model, logits: dict[str, torch.Tensor], numbers: dict[str, torch.Tensor],
              batch: Batch, groups: dict[str, list[tuple[int, int]]]) -> torch.Tensor:
    """Mean over heads of each head's mean loss on its (row in logits, line) pairs."""
    by = model.schema.by_key()
    total, terms = None, 0
    for key, pairs in groups.items():
        o = by[key]
        m = [p for p, _ in pairs]
        lines = [i for _, i in pairs]
        if o.kind == NUMBER:
            pred = numbers.get(key)
            if pred is None:
                continue
            t = pred.new_tensor([float(batch.numbers_target[key][i]) for i in lines])
            loss = F.smooth_l1_loss(pred[m].float(), torch.log1p(t.float()))
        else:
            lg = logits.get(key)
            if lg is None:
                continue
            lg = lg[m].float()
            if key in batch.multi_targets:
                truth = torch.zeros_like(lg)
                for r, i in enumerate(lines):
                    ids = batch.multi_targets[key][i]
                    if ids:
                        truth[r, list(ids)] = 1
                loss = F.binary_cross_entropy_with_logits(lg, truth)
            else:
                y = torch.tensor([batch.targets[key][i] for i in lines], device=lg.device)
                loss = F.cross_entropy(lg, y)
        scale = batch.head_scale.get(key, 1.0)
        total = scale * loss if total is None else total + scale * loss
        terms += 1
    return total / terms if total is not None else None


def _nce(state: torch.Tensor, e: torch.Tensor, target: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """InfoNCE both ways: each row toward its column, each column toward the
    rows that hold it. Columns are distinct texts, so a copy is never a negative."""
    sim = F.normalize(state.float(), dim=-1) @ F.normalize(e.float(), dim=-1).T / tau
    row_to_col = F.cross_entropy(sim, target)
    member = F.one_hot(target, e.shape[0]).T.bool()
    held = member.any(-1)
    st = sim.T[held]
    col_to_row = -(torch.logsumexp(st.masked_fill(~member[held], float("-inf")), -1)
                   - torch.logsumexp(st, -1))
    return 0.5 * (row_to_col + col_to_row.mean())


def _columns(texts: list[str]) -> tuple[list[str], list[int]]:
    seen: dict[str, int] = {}
    out: list[str] = []
    col = []
    for t in texts:
        if t not in seen:
            seen[t] = len(out)
            out.append(t)
        col.append(seen[t])
    return out, col


def _sample(rows: list, cap: int) -> list:
    if cap and len(rows) > cap:
        keep = torch.randperm(len(rows))[:cap].sort().values.tolist()
        return [rows[k] for k in keep]
    return rows


# ---------------------------------------------------------------- the notes

def card_rows(model: C3Model, batch: Batch) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """(universal, company) card notes as (line, head, note), each on a head the
    line answers: ``field_notes`` (line cards, field notes, ruled-out readings;
    the link notes are link_why's) and the reason-only cards' notes."""
    by = model.schema.by_key()
    uni: list[tuple[int, str, str]] = []
    co: list[tuple[int, str, str]] = []
    seen = set()
    items = [(i, k, t) for i, notes in enumerate(batch.field_notes) for k, t in notes.items()]
    items += list(getattr(batch, "reason_notes", []) or [])
    for i, key, note in items:
        o = by.get(key)
        t = _norm(note)
        if o is None or o.kind == RELATION or not t or (i, key) in seen:
            continue
        if key.startswith("read:") and not _has_target(batch, key, i, o.kind):
            # A field note on a line whose field has no answer (a rate_note on a
            # rate card row that leaves the rate reading blank) still explains
            # the line: a WHY of its type.
            alt = by.get("col:label_type")
            if alt is not None and alt.layer == o.layer and (i, alt.key) not in seen \
                    and _has_target(batch, alt.key, i, alt.kind):
                key, o = alt.key, alt
        if not batch.head_scale.get(key, 1.0) or not _has_target(batch, key, i, o.kind):
            continue
        seen.add((i, key))
        (uni if o.universal else co).append((i, key, t))
    return uni, co


def company_rows(model: C3Model, batch: Batch, co_cards: list[tuple[int, str, str]]
                 ) -> list[tuple[int, str | None, str]]:
    """The company layer's lines: each line's [company] note (head None: every
    company head the line answers) and the company heads' own card notes."""
    rows: list[tuple[int, str | None, str]] = []
    for i, t in enumerate(batch.policy_note or []):
        t = _norm(t)
        if t:
            rows.append((i, None, t))
    rows += co_cards
    return rows


def _encode_why(model: C3Model, texts: list[str]) -> torch.Tensor:
    return model.why_proj(model.aux_text(texts))


def _encode_company(model: C3Model, texts: list[str]) -> torch.Tensor:
    return model.company_why_proj(model.aux_text(texts).detach())


# ---------------------------------------------------------------- the parts

def described_alone_loss(model: C3Model, out: C3Output, batch: Batch, w) -> torch.Tensor:
    """Head CE on ``logits - free_logits`` for the critical heads, on a share of steps."""
    from .losses import answer_loss, has_answers, row_weights  # noqa: PLC0415 (cycle)

    zero = out.r.new_zeros(())
    p = float(getattr(w, "described_alone_p", 0.5))
    if p <= 0 or float(torch.rand(())) >= p:
        return zero
    rw = row_weights(batch, out.r.device)
    total, terms = zero, 0.0
    for key in DESCRIBED_KEYS:
        free = out.free_logits.get(key)
        if free is None or key not in out.logits or not has_answers(batch, key):
            continue
        share = batch.head_scale.get(key, 1.0)
        if not share:
            continue
        total = total + share * answer_loss(out.logits[key] - free, batch, key, rw)
        terms += 1.0
    return total / max(terms, 1.0)


def _stand_in(model: C3Model, out: C3Output, lines: list[int], r: torch.Tensor,
              desc: dict[str, Any]) -> C3Output:
    """The universal heads on ``lines`` with ``r`` in place of r_i and the
    surface paths (h, the asked slots) blank."""
    ix = torch.tensor(lines, device=out.r.device, dtype=torch.long)
    h = out.h.new_zeros(len(lines), out.h.shape[-1])
    slots = out.slots.new_zeros((len(lines),) + tuple(out.slots.shape[1:])) if out.slots is not None else None
    r = r.to(out.r.dtype)
    sub = C3Output(h=h, z_c=out.z_c[ix], q_mu=out.q_mu[ix], q_logvar=out.q_logvar[ix],
                   q_logit=out.q_logit[ix], r=r, residual=out.residual[ix], slots=slots)
    model.universal_heads(sub, r, out.q_mean[ix], desc)
    return sub


def _bare_stand_in(model: C3Model, out: C3Output, r: torch.Tensor, desc: dict[str, Any]) -> C3Output:
    """The universal heads with ``r`` and nothing else: no line's h, q, z or
    asked slots (a note whose line is not in the deal's input)."""
    k = r.shape[0]
    h = out.h.new_zeros(k, out.h.shape[-1])
    slots = out.slots.new_zeros((k,) + tuple(out.slots.shape[1:])) if out.slots is not None else None
    r = r.to(out.r.dtype)
    sub = C3Output(h=h, z_c=out.z_c.new_zeros(k, *out.z_c.shape[1:]),
                   q_mu=out.q_mu.new_zeros(k, *out.q_mu.shape[1:]),
                   q_logvar=out.q_logvar.new_zeros(k, *out.q_logvar.shape[1:]),
                   q_logit=out.q_logit.new_zeros(k, *out.q_logit.shape[1:]),
                   r=r, residual=out.residual.new_zeros(k, *out.residual.shape[1:]), slots=slots)
    model.universal_heads(sub, r, out.q_mean.new_zeros(k, *out.q_mean.shape[1:]), desc)
    return sub


def unplaced_alone_loss(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any],
                        rows: list[tuple[str, int, str]]) -> tuple[torch.Tensor | None, int]:
    """(Sum over heads of each head's mean CE, heads): a card's note whose line
    is not a model input (``Batch.unplaced_notes``), alone in r's place with no
    line context, reaches the card's verdict."""
    by = model.schema.by_key()
    rows = [(k, a, t) for k, a, t in rows if k in by and by[k].universal and by[k].kind != NUMBER
            and batch.head_scale.get(k, 1.0) and _norm(t)]
    if not rows:
        return None, 0
    texts, col = _columns([_norm(t) for _, _, t in rows])
    e = _encode_why(model, texts)
    sub = _bare_stand_in(model, out, e[torch.tensor(col, device=e.device)], desc)
    groups: dict[str, list[tuple[int, int]]] = {}
    for m, (key, ans, _) in enumerate(rows):
        groups.setdefault(key, []).append((m, ans))
    total, terms = None, 0
    for key, pairs in groups.items():
        lg = sub.logits.get(key)
        if lg is None:
            continue
        lg = lg[[m for m, _ in pairs]].float()
        loss = F.cross_entropy(lg, torch.tensor([a for _, a in pairs], device=lg.device))
        loss = batch.head_scale.get(key, 1.0) * loss
        total = loss if total is None else total + loss
        terms += 1
    return total, terms


def card_alone_loss(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any],
                    rows: list[tuple[int, str, str]], e: torch.Tensor, col: list[int],
                    unplaced: list[tuple[str, int, str]] | None = None) -> torch.Tensor:
    """Each line card's note alone reaches its head's answer; with ``unplaced``,
    so does each card note whose line is not in the input (no line context).
    The mean over heads of both."""
    zero = out.r.new_zeros(())
    loss, heads = None, 0
    if rows:
        sub = _stand_in(model, out, [i for i, _, _ in rows], e[torch.tensor(col, device=e.device)], desc)
        groups: dict[str, list[tuple[int, int]]] = {}
        for m, (i, key, _) in enumerate(rows):
            groups.setdefault(key, []).append((m, i))
        placed = _row_loss(model, sub.logits, sub.numbers, batch, groups)
        if placed is not None:
            heads = len(groups)
            loss = placed * heads
    if unplaced:
        more, n = unplaced_alone_loss(model, out, batch, desc, unplaced)
        if more is not None:
            loss = more if loss is None else loss + more
            heads += n
    return zero if loss is None else loss / max(heads, 1)


def card_contrast_loss(out: C3Output, rows: list[tuple[int, str, str]], e: torch.Tensor,
                       col: list[int]) -> torch.Tensor:
    """Per head: r_i toward its note's column among that head's distinct notes."""
    zero = out.r.new_zeros(())
    by: dict[str, list[int]] = {}
    for m, (_, key, _) in enumerate(rows):
        by.setdefault(key, []).append(m)
    total, n = zero, 0
    for key, ms in by.items():
        cols = sorted({col[m] for m in ms})
        if len(cols) < 2:
            continue
        at = {c: k for k, c in enumerate(cols)}
        state = out.r[torch.tensor([rows[m][0] for m in ms], device=out.r.device)]
        target = torch.tensor([at[col[m]] for m in ms], device=out.r.device)
        total = total + len(ms) * _nce(state, e[torch.tensor(cols, device=e.device)], target)
        n += len(ms)
    return total / n if n else zero


def conduct_logits(model: C3Model, y: torch.Tensor, desc: dict[str, Any]
                   ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The conduct head (and the company free rows) on stand-in lines ``y``."""
    opps = [o for o in model.schema.select(layer="company") if o.kind != RELATION]
    logits, nums = model.conduct_head(y, torch.stack([desc[o.key][0] for o in opps]),
                                      [desc[o.key][1] for o in opps])
    lg: dict[str, torch.Tensor] = {}
    nm: dict[str, torch.Tensor] = {}
    for j, (o, x) in enumerate(zip(opps, logits)):
        lg[o.key] = x + model.free_heads[o.key](y).to(x.dtype) if o.key in model.free_heads else x
        if o.kind == NUMBER:
            nm[o.key] = nums[:, j]
    return lg, nm


def co_why_contrast_loss(out: C3Output, rows: list[tuple[int, str | None, str]], e: torch.Tensor,
                         col: list[int]) -> torch.Tensor:
    zero = out.r.new_zeros(())
    if out.conduct_x is None or len(set(col)) < 2:
        return zero
    state = out.conduct_x[torch.tensor([i for i, _, _ in rows], device=out.r.device)]
    return _nce(state, e, torch.tensor(col, device=e.device))


def co_why_alone_loss(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any],
                      rows: list[tuple[int, str | None, str]], e: torch.Tensor, col: list[int]) -> torch.Tensor:
    zero = out.r.new_zeros(())
    if out.conduct_x is None or not rows:
        return zero
    by = model.schema.by_key()
    company = [o for o in model.schema.select(layer="company") if o.kind != RELATION]
    groups: dict[str, list[tuple[int, int]]] = {}
    for m, (i, key, _) in enumerate(rows):
        keys = [key] if key else [o.key for o in company
                                  if batch.head_scale.get(o.key, 1.0) and _has_target(batch, o.key, i, o.kind)]
        for k in keys:
            if k in by:
                groups.setdefault(k, []).append((m, i))
    if not groups:
        return zero
    lg, nm = conduct_logits(model, e[torch.tensor(col, device=e.device)], desc)
    loss = _row_loss(model, lg, nm, batch, groups)
    return zero if loss is None else loss


def number_why_alone_loss(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any]) -> torch.Tensor:
    """The label WHY alone (r stand-in, surface blank) reaches the line's number heads."""
    zero = out.r.new_zeros(())
    ww = batch.why_weights if len(batch.why_weights) == len(batch) else [1.0] * len(batch)
    nums = [o for o in model.schema.select(layer="universal", kind=NUMBER) if batch.head_scale.get(o.key, 1.0)]
    groups: dict[str, list[int]] = {}
    for o in nums:
        for i, t in enumerate(batch.why):
            if t and ww[i] > 0 and _has_target(batch, o.key, i, NUMBER):
                groups.setdefault(o.key, []).append(i)
    lines = sorted({i for v in groups.values() for i in v})
    if not lines:
        return zero
    at = {i: m for m, i in enumerate(lines)}
    sub = _stand_in(model, out, lines, _encode_why(model, [_norm(batch.why[i]) for i in lines]), desc)
    loss = _row_loss(model, sub.logits, sub.numbers, batch,
                     {k: [(at[i], i) for i in v] for k, v in groups.items()})
    return zero if loss is None else loss


def judgment_why_alone_loss(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any],
                            sizes: tuple[str, ...] | None = None) -> torch.Tensor:
    """A card's note alone, in the card feature's place, reaches its verdict."""
    zero = out.r.new_zeros(())
    by = model.schema.by_key()
    groups: dict[str, list] = {}
    for j in batch.judged:
        o = by.get(j.key)
        if o is None or j.why_weight <= 0 or not batch.head_scale.get(j.key, 1.0):
            continue
        if sizes is not None and o.size not in sizes:
            continue
        text = _norm(j.note) if o.universal else _norm(" ".join(x for x in (j.note, j.policy) if x))
        if text:
            groups.setdefault(j.key, []).append((j, text))
    total, terms = zero, 0
    for key, items in groups.items():
        o = by[key]
        texts = [t for _, t in items]
        if o.universal:
            f, head = model.judgment_note(model.aux_text(texts).detach()), model.judgment_head
        else:
            f, head = _encode_company(model, texts), model.judgment_head_company
        logits = head(f, desc[key][0].unsqueeze(0), [desc[key][1]])[0][0].float()
        js = [j for j, _ in items]
        wts = logits.new_tensor([j.why_weight for j in js])
        if o.multiple:
            truth = torch.zeros_like(logits)
            for r, j in enumerate(js):
                truth[r, list(j.answers or (j.answer,))] = 1
            per = F.binary_cross_entropy_with_logits(logits, truth, reduction="none").mean(-1)
        else:
            per = F.cross_entropy(logits, torch.tensor([j.answer for j in js], device=logits.device),
                                  reduction="none")
        total = total + batch.head_scale.get(key, 1.0) * why_mean(per, wts)
        terms += 1
    return total / max(terms, 1)


def why2_losses(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any], w,
                judgment_sizes: tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:
    """The parts with a nonzero weight, named as LossWeights fields."""
    on = {k: float(getattr(w, k, 0.0)) for k in
          ("described_alone", "card_alone", "card_contrast", "co_why_contrast", "co_why_alone",
           "number_why_alone", "judgment_why_alone")}
    parts: dict[str, torch.Tensor] = {}
    zero = out.r.new_zeros(())
    if on["described_alone"]:
        parts["described_alone"] = described_alone_loss(model, out, batch, w)
    uni, co_cards = card_rows(model, batch) if (on["card_alone"] or on["card_contrast"]
                                                or on["co_why_contrast"] or on["co_why_alone"]) else ([], [])
    if on["card_alone"] or on["card_contrast"]:
        rows = _sample(uni, int(getattr(w, "card_cap", 256)))
        unplaced = _sample(list(getattr(batch, "unplaced_notes", []) or []), int(getattr(w, "card_cap", 256)))
        texts, col = _columns([t for _, _, t in rows]) if rows else ([], [])
        e = _encode_why(model, texts) if rows else None
        if on["card_alone"]:
            parts["card_alone"] = (card_alone_loss(model, out, batch, desc, rows, e, col, unplaced)
                                   if rows or unplaced else zero)
        if on["card_contrast"]:
            parts["card_contrast"] = card_contrast_loss(out, rows, e, col) if rows else zero
    if on["co_why_contrast"] or on["co_why_alone"]:
        rows = _sample(company_rows(model, batch, co_cards), int(getattr(w, "card_cap", 256)))
        if rows and out.conduct_x is not None:
            texts, col = _columns([t for _, _, t in rows])
            e = _encode_company(model, texts)
            if on["co_why_contrast"]:
                parts["co_why_contrast"] = co_why_contrast_loss(out, rows, e, col)
            if on["co_why_alone"]:
                parts["co_why_alone"] = co_why_alone_loss(model, out, batch, desc, rows, e, col)
        else:
            for k in ("co_why_contrast", "co_why_alone"):
                if on[k]:
                    parts[k] = zero
    if on["number_why_alone"]:
        parts["number_why_alone"] = number_why_alone_loss(model, out, batch, desc)
    # Card verdicts train from the cards stage on (gold_weights turns judgments on).
    if on["judgment_why_alone"] and float(getattr(w, "judgments", 0.0)):
        parts["judgment_why_alone"] = judgment_why_alone_loss(model, out, batch, desc, judgment_sizes)
    return parts


__all__ = ["DESCRIBED_KEYS", "before_forward", "card_alone_loss", "card_contrast_loss", "card_rows",
           "co_why_alone_loss", "co_why_contrast_loss", "company_rows", "conduct_logits",
           "described_alone_loss", "judgment_why_alone_loss", "number_why_alone_loss", "why2_losses"]
