"""Teacher-free terms that make the WHY carry the decision (run E).

Run C trained the WHY into the heads through ``why_align`` (InfoNCE of r_i
against the line's encoded WHY, weight 0.1), ``why_sufficiency`` and
``why_echo``. Three things kept that from deciding anything:

* the InfoNCE negatives were the other lines' WHYs *including identical
  ones*: 82 site rows sharing one WHY pushed each row away from 81 copies of
  its own reason, so the term mostly fought itself;
* nothing asked the heads to *follow* the WHY: a head could reach the gold
  answer through the paths that never see it, the free heads (``out.h``,
  the line's own text features) and the asked slots (lattice over ``h``);
* ``why_sufficiency`` swapped the WHY in for r but the free heads still read
  the real line's ``h``, so "the WHY alone" was never alone.

The terms here, each a named part of ``c3_loss`` (weight 0 turns it off):

``why_contrast``     r_i toward its own WHY, against the other *distinct*
                     WHYs of the part (identical WHYs are one column, never a
                     negative), with the near-miss partner's WHY as a hard
                     negative (``why_hard_negative`` times the weight). Both
                     directions: each WHY toward every line that holds it.
``why_swap``         counterfactual: line i's state with its WHY direction
                     replaced by line j's (r_i + |r_i| (e_j - e_i)) must give
                     j's answer when j answers differently (near misses
                     first, then other labeled pairs), and i's own answer when
                     j is a twin. r is detached and h and the asked slots
                     are blank: only the heads learn, and what they learn is
                     that the WHY direction outranks the surface paths.
``why_alone``        ``why_sufficiency`` with the surface paths blanked: the
                     encoded WHY stands in for r and the free heads read a
                     zero ``h`` on those lines, so the WHY really is alone.
``keyword_dropout``  one more forward with the surface tokens a decision
                     could lean on masked or paraphrased on labeled lines
                     (field labels like "City:", header words, "site", "CSV",
                     file extensions, a few random words); the label and the
                     WHY stay. The masked line must give its saved answers,
                     match its clean reading (KL, detached) and still find
                     its own WHY (``why_contrast`` on the masked state).

The swap gate (~/c3-local/runE/swap_gate.py) measures the same property on
held-out deals with perturbations these terms never train (entity swaps).
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import replace
from typing import Any

import torch
from torch.nn import functional as F

from .data import IGNORE, Batch, number_tokens
from .model import C3Model, C3Output

# ---------------------------------------------------------------- surface masks

#: A cell's field label at the start of the text or after a cell break.
_FIELD = re.compile(r"(^|\|\s*|\t)([A-Za-z][\w ./()#-]{0,40}?):(\s+)")
_EXT = re.compile(r"\.(csv|tsv|xlsx|xlsm|xls|docx|doc|pdf|pptx|ppt|txt|json|zip|msg|eml)\b", re.I)
#: Words that name a layout or a container, not what a line is for.
SURFACE_WORDS = {
    "site": ("location", "place", ""), "sites": ("locations", "places", ""),
    "location": ("site", "place", ""), "locations": ("sites", "places", ""),
    "csv": ("file", ""), "xlsx": ("file", ""), "spreadsheet": ("file", "document", ""),
    "workbook": ("file", "document", ""), "sheet": ("page", "tab", ""), "tab": ("page", ""),
    "list": ("set", ""), "table": ("set", "grid", ""), "row": ("entry", "line", ""),
    "rows": ("entries", "lines", ""), "column": ("field", ""), "columns": ("fields", ""),
    "header": ("heading", ""), "attached": ("included", "enclosed", ""),
    "attachment": ("file", "enclosure", ""), "address": ("addr", "street", ""),
}
#: Paraphrases of common field labels; anything else becomes a neutral label or none.
FIELD_SYNONYMS = {
    "city": "Town", "zip": "Postal code", "state": "Province", "address": "Street",
    "site": "Location", "site id": "Ref", "site_id": "Ref", "site no.": "No.", "site name": "Name",
    "location": "Place", "region": "Area", "phone": "Tel", "contact": "Person", "email": "Mail",
    "qty": "Count", "quantity": "Count", "description": "Details", "total hrs": "Sum hours",
    "hrs": "Hours", "hours": "Hrs", "floors": "Levels", "sq ft": "Area sqft", "bldg sqft": "Area sqft",
}
_WORD = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")


def _field(m: re.Match, rng: random.Random, k: list[int]) -> str:
    lead, name, space = m.group(1), m.group(2).strip(), m.group(3)
    k[0] += 1
    pick = rng.random()
    if pick < 0.4:
        return lead
    if pick < 0.7 and name.lower() in FIELD_SYNONYMS:
        return f"{lead}{FIELD_SYNONYMS[name.lower()]}:{space}"
    return f"{lead}f{k[0]}:{space}"


def mask_surface(text: str, rng: random.Random | None = None, word_drop: float = 0.1) -> str | None:
    """The line with its surface cues masked or paraphrased, or None when it
    has none to mask. Numbers, names and the rest of the wording stay, so the
    label and the WHY still hold. No word decides an answer here: the list
    names what the answer must *not* rest on."""
    rng = rng or random
    t = str(text or "")
    k = [0]
    out = _FIELD.sub(lambda m: _field(m, rng, k), t)
    out = _EXT.sub(lambda m: "" if rng.random() < 0.5 else ".file", out)

    def word(m: re.Match) -> str:
        w = m.group(0)
        alts = SURFACE_WORDS.get(w.lower())
        if alts is not None:
            alt = rng.choice(alts)
            return alt.upper() if w.isupper() and alt else (alt.capitalize() if w[:1].isupper() else alt)
        return "" if rng.random() < word_drop else w

    out = _WORD.sub(word, out)
    out = re.sub(r"[ ]{2,}", " ", out).strip()
    return out if out and out != t.strip() else None


# ---------------------------------------------------------------- the WHY columns

def _why_rows(batch: Batch) -> list[int]:
    ww = batch.why_weights if len(batch.why_weights) == len(batch) else [1.0] * len(batch)
    return [i for i, t in enumerate(batch.why) if t and ww[i] > 0]


def _columns(batch: Batch, idx: list[int]) -> tuple[list[str], list[int]]:
    """Distinct WHY texts and each row's column: lines that share a WHY share
    a column, so a copy of a line's own reason is never its negative."""
    texts: list[str] = []
    seen: dict[str, int] = {}
    col = []
    for i in idx:
        t = " ".join(str(batch.why[i]).split())
        if t not in seen:
            seen[t] = len(texts)
            texts.append(t)
        col.append(seen[t])
    return texts, col


def why_embeddings(model: C3Model, texts: list[str]) -> torch.Tensor:
    return model.why_proj(model.aux_text(texts))


def why_contrast_loss(model: C3Model, out: C3Output, batch: Batch, *, tau: float = 0.1,
                      hard: float = 4.0, r: torch.Tensor | None = None,
                      rows: list[int] | None = None) -> torch.Tensor:
    """Supervised InfoNCE of line state against distinct WHYs.

    ``r``/``rows``: score another pass's states (keyword dropout) on a subset
    of lines; default every line with a WHY in ``out``."""
    zero = out.r.new_zeros(())
    idx = _why_rows(batch)
    if rows is not None:
        keep = set(rows)
        idx = [i for i in idx if i in keep]
    if len(idx) < 2:
        return zero
    texts, col = _columns(batch, idx)
    if len(texts) < 2:
        return zero
    e = F.normalize(why_embeddings(model, texts).float(), dim=-1)
    state = (r if r is not None else out.r)[idx].float()
    sim = F.normalize(state, dim=-1) @ e.T / tau                       # [L, U]
    # Hard negatives: the WHY of a line that reads alike and answers differently.
    pos = {i: k for k, i in enumerate(idx)}
    bias = torch.zeros_like(sim)
    lh = math.log(max(hard, 1e-6))
    for i, j, _ in batch.near_misses:
        if i in pos and j in pos and col[pos[i]] != col[pos[j]]:
            bias[pos[i], col[pos[j]]] = lh
            bias[pos[j], col[pos[i]]] = lh
    target = torch.tensor(col, device=sim.device)
    from .losses import why_row_weights  # noqa: PLC0415 (cycle)

    ww = why_row_weights(batch, sim.device)[idx]
    line_to_why = F.cross_entropy(sim + bias, target, reduction="none")
    # Each WHY toward the lines that hold it (several positives).
    member = F.one_hot(target, len(texts)).T.bool()                    # [U, L]
    st = sim.T + bias.T
    why_to_line = -(torch.logsumexp(st.masked_fill(~member, float("-inf")), -1)
                    - torch.logsumexp(st, -1))
    cw = torch.zeros(len(texts), device=sim.device).index_add_(0, target, ww)
    return 0.5 * ((line_to_why * ww).sum() / ww.sum().clamp(min=1e-6)
                  + (why_to_line * cw).sum() / cw.sum().clamp(min=1e-6))


def swap_pairs(batch: Batch, cap: int = 64, rng: random.Random | None = None
               ) -> list[tuple[int, int, str, bool]]:
    """(i, j, key, flip): move line i's WHY direction to j's. ``flip`` pairs
    answer differently on ``key`` (near misses, then sampled labeled pairs
    with a different line type); the rest are twins and must hold."""
    rng = rng or random
    have = set(_why_rows(batch))
    distinct = {i: " ".join(str(batch.why[i]).split()) for i in have}
    out: list[tuple[int, int, str, bool]] = []
    seen = set()

    def add(i, j, key, flip):
        if i in have and j in have and distinct[i] != distinct[j] and (i, j, key) not in seen:
            y = batch.targets.get(key)
            if y is None or IGNORE in (y[i], y[j]) or (y[i] != y[j]) != flip:
                return
            seen.add((i, j, key))
            out.append((i, j, key, flip))

    for i, j, key in batch.near_misses:
        add(i, j, key, True)
        add(j, i, key, True)
    near = len(out)
    y = batch.targets.get("col:label_type")
    if y is not None:
        rows = [i for i in have if y[i] != IGNORE]
        by: dict[int, list[int]] = {}
        for i in rows:
            by.setdefault(y[i], []).append(i)
        if len(by) > 1:
            for _ in range(4 * cap):
                if len(out) - near >= cap // 2:
                    break
                i, j = rng.sample(rows, 2)
                if y[i] != y[j]:
                    add(i, j, "col:label_type", True)
    for i, j in batch.twins:
        add(i, j, "col:label_type", False)
        add(j, i, "col:label_type", False)
    flips = [p for p in out if p[3]][:cap]
    holds = [p for p in out if not p[3]][: max(8, cap // 4)]
    return flips + holds


def why_swap_loss(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any], *,
                  cap: int = 64, rng: random.Random | None = None) -> torch.Tensor:
    """Swapped WHYs must flip the decision (and twins' must hold it)."""
    from .losses import moved_heads  # noqa: PLC0415 (cycle)

    zero = out.r.new_zeros(())
    pairs = swap_pairs(batch, cap, rng)
    if not pairs:
        return zero
    lines = sorted({i for i, j, _, _ in pairs} | {j for _, j, _, _ in pairs})
    texts = [" ".join(str(batch.why[i]).split()) for i in lines]
    with torch.no_grad():
        e = F.normalize(why_embeddings(model, texts).float(), dim=-1)
    at = {i: k for k, i in enumerate(lines)}
    src = [i for i, _, _, _ in pairs]
    r = out.r.detach()
    move = (e[[at[j] for _, j, _, _ in pairs]] - e[[at[i] for i in src]]) * r[src].float().norm(dim=-1, keepdim=True)
    # h and the asked slots blank, as why_alone: a free row reading h could
    # otherwise hold the answer whatever the swapped WHY says (arch_review §3).
    held = C3Output(h=torch.zeros_like(out.h), z_c=out.z_c.detach(), q_mu=out.q_mu.detach(),
                    q_logvar=out.q_logvar.detach(), q_logit=out.q_logit.detach(), r=r,
                    residual=out.residual.detach(),
                    slots=torch.zeros_like(out.slots) if out.slots is not None else None)
    sub = moved_heads(model, held, src, move.to(r.dtype), desc)
    total, terms = zero, 0
    for key in {k for _, _, k, _ in pairs}:
        lg = sub.logits.get(key)
        if lg is None:
            continue
        sel = [n for n, p in enumerate(pairs) if p[2] == key]
        y = torch.tensor([batch.targets[key][pairs[n][1] if pairs[n][3] else pairs[n][0]] for n in sel],
                         device=lg.device)
        total = total + F.cross_entropy(lg[sel].float(), y, reduction="sum")
        terms += len(sel)
    return total / max(terms, 1)


def why_alone_loss(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any]) -> torch.Tensor:
    """The encoded WHY in r's place, with the free heads' ``h`` and the asked
    slots blank on those lines: the heads must reach the answer from the
    reason alone, not from the line's own text read around it."""
    from .losses import answer_loss, has_answers, why_row_weights  # noqa: PLC0415 (cycle)

    zero = out.r.new_zeros(())
    idx = _why_rows(batch)
    if not idx:
        return zero
    texts = [" ".join(str(batch.why[i]).split()) for i in idx]
    why = why_embeddings(model, texts)
    stand_in = out.r.clone()
    stand_in[idx] = why.to(stand_in.dtype)
    h = out.h.clone()
    h[idx] = 0
    slots = None
    if out.slots is not None:
        slots = out.slots.clone()
        slots[idx] = 0
    sub = C3Output(h=h, z_c=out.z_c, q_mu=out.q_mu, q_logvar=out.q_logvar, q_logit=out.q_logit,
                   r=stand_in, residual=out.residual, slots=slots)
    model.universal_heads(sub, stand_in, out.q_mean, desc)
    keep = torch.zeros(len(batch), dtype=torch.bool, device=out.r.device)
    keep[idx] = True
    rw = why_row_weights(batch, out.r.device)
    total, terms = zero, 0
    for key, lg in sub.logits.items():
        if has_answers(batch, key):
            total = total + answer_loss(lg, batch, key, rw, keep)
            terms += 1
    return total / max(terms, 1)


def keyword_dropout_loss(model: C3Model, out: C3Output, batch: Batch, w, *,
                         rng: random.Random | None = None,
                         desc: dict[str, Any] | None = None) -> torch.Tensor:
    """One more forward with surface cues masked on a sample of labeled lines."""
    from .losses import head_losses, row_weights  # noqa: PLC0415 (cycle)

    rng = rng or random
    zero = out.r.new_zeros(())
    p, cap = float(getattr(w, "keyword_dropout_p", 0.5)), int(getattr(w, "keyword_dropout_cap", 256))
    texts, numbers = list(batch.texts), list(batch.numbers)
    lines = []
    for i, on in enumerate(batch.labeled):
        if not on or rng.random() >= p:
            continue
        t = mask_surface(batch.texts[i], rng)
        if t is None:
            continue
        texts[i], numbers[i] = t, number_tokens(t)
        lines.append(i)
        if len(lines) >= cap:
            break
    if not lines:
        return zero
    keep = set(lines)
    base = batch.weights if len(batch.weights) == len(batch) else [1.0] * len(batch)
    b = replace(batch, texts=texts, numbers=numbers, layout_rewrites=[],
                weights=[x if i in keep else 0.0 for i, x in enumerate(base)])
    moved = model(b.inputs(), company=batch.company, desc=desc)
    ce, _ = head_losses(model, moved, b, w, "universal")
    co, _ = head_losses(model, moved, b, w, "company")
    rows = torch.tensor(lines, device=out.r.device)
    rw = row_weights(b, out.r.device)[rows]
    kl, n = zero, 0
    for key, logits in moved.logits.items():
        ref = out.logits.get(key)
        opp = model.schema.by_key().get(key)
        if ref is None or opp is None or opp.multiple or logits.shape != ref.shape:
            continue
        per = F.kl_div(F.log_softmax(logits[rows].float(), -1),
                       F.log_softmax(ref[rows].detach().float(), -1),
                       reduction="none", log_target=True).sum(-1)
        kl = kl + (per * rw).sum() / rw.sum().clamp(min=1e-6)
        n += 1
    tied = why_contrast_loss(model, moved, batch, hard=float(getattr(w, "why_hard_negative", 4.0)),
                             rows=lines)
    return ce + w.conduct * co + (kl / n if n else zero) + tied


def why_carry_losses(model: C3Model, out: C3Output, batch: Batch, desc: dict[str, Any], w,
                     rng: random.Random | None = None) -> dict[str, torch.Tensor]:
    """The parts with a nonzero weight, named as LossWeights fields. The
    sampling (masks, swap pairs) draws its seed from torch's generator, so a
    seeded run repeats exactly."""
    if rng is None:
        rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,))))
    parts: dict[str, torch.Tensor] = {}
    if getattr(w, "why_contrast", 0.0):
        parts["why_contrast"] = why_contrast_loss(model, out, batch,
                                                  hard=float(getattr(w, "why_hard_negative", 4.0)))
    if getattr(w, "why_swap", 0.0):
        parts["why_swap"] = why_swap_loss(model, out, batch, desc,
                                          cap=int(getattr(w, "why_swap_cap", 64)), rng=rng)
    if getattr(w, "why_alone", 0.0):
        parts["why_alone"] = why_alone_loss(model, out, batch, desc)
    if getattr(w, "keyword_dropout", 0.0):
        parts["keyword_dropout"] = keyword_dropout_loss(model, out, batch, w, rng=rng, desc=desc)
    return parts


__all__ = ["mask_surface", "swap_pairs", "why_alone_loss", "why_carry_losses", "why_contrast_loss",
           "why_swap_loss", "keyword_dropout_loss"]
