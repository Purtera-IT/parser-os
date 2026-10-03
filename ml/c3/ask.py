"""Ask for the reason that would reshape the model most (architecture v5).

v4's question engine asks the *customer* the question worth the most money.
This asks the *labeler* for the explanation worth the most to the model.

For every line, the model voices its own predicted reason: ``r_to_text(r_j)``
puts its rationale latent into the space explanations are encoded in
(trained by ``why_echo`` in losses.py to land on the line's real WHY). That
pseudo-explanation is compiled and applied like a rule card, and the score is
how much it would move the model's answers on *the other lines*:

    value(j) = Σ_{i ≠ j} Σ_o KL( p_o(i | bank + reason_j) || p_o(i | bank) )

A line whose reason would reshape many other lines is the one worth a careful
WHY or a rule card. A line whose reason changes nothing elsewhere can get a
short WHY. On the labeling page that is a "explain this one first" order
(proposed, v5 section 6).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .data import Batch
from .explain import Explanation, ExplanationBank, WHY_NOTE


@torch.no_grad()
def explanation_value(model, batch: Batch, bank: ExplanationBank | None = None,
                      layer: str = "universal", lines: list[int] | None = None) -> list[tuple[int, float]]:
    """[(line, value)] sorted high to low. One extra forward pass per line."""
    model.eval()
    bank = bank or ExplanationBank()
    inp = batch.inputs()
    base_emb = model.text([e.text for e in bank.items]) if len(bank) else None
    base = model(inp, company=batch.company, bank=bank, bank_emb=base_emb)
    keys = [k for k in base.logits if (model.schema.by_key()[k].layer == layer)]
    base_p = {k: base.logits[k].log_softmax(-1) for k in keys}
    voiced = model.r_to_text(base.r)                                    # [N, d_text]
    scores = []
    for j in (lines if lines is not None else range(len(batch))):
        pseudo = Explanation(f"<predicted reason for line {j}>", layer,
                             company=batch.company if layer == "company" else "",
                             source=WHY_NOTE, ref=f"ask#{j}", line=j)
        emb = voiced[j:j + 1] if base_emb is None else torch.cat([base_emb, voiced[j:j + 1]])
        out = model(inp, company=batch.company, bank=bank.extend([pseudo]), bank_emb=emb)
        others = torch.arange(len(batch)) != j
        v = 0.0
        for k in keys:
            new = out.logits[k].log_softmax(-1)
            v += float(F.kl_div(base_p[k][others], new[others], log_target=True,
                                reduction="batchmean"))
        scores.append((j, v))
    return sorted(scores, key=lambda s: -s[1])
