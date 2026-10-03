"""Read a paragraph once, keep it: explanations compiled into weights (v6).

A person told something once does not need it read aloud before every
decision afterwards. ``brain.Brain`` can read notes in context, but a page
only holds a few, and a company will write thousands. This module turns a
paragraph into a small change to the LM's weights, so it keeps acting after
it leaves the page.

``ParagraphMemory`` is a hypernetwork: the LM reads the paragraph (mean
hidden state), and the network writes a low-rank delta (A, B) into every
adapter slot of the LM (lm.LoRALinear). Several paragraphs stack along the
rank, so their effects add, and removing one removes exactly its part. With
no paragraphs, the LM is untouched, bit for bit.

It is trained by **context distillation**: the teacher is the brain with the
paragraphs on the page; the student is the same brain with the page bare and
the paragraphs written into weights. The student must give the teacher's
answers (KL), so the hypernetwork learns to write what the LM understood
from reading, not a keyword match.

One memory per layer: ``universal`` writes from WHYs, the company memory from
that company's lines. The universal pass never receives company deltas
(brain.py), so a company's paragraphs can never move a base answer.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .lm import LMBase


class ParagraphMemory(nn.Module):
    """Paragraph -> one (A, B) per adapter slot.

    The deltas are mixtures of a small learned basis per slot (``n_basis``),
    so the parameter count stays small even for a large LM.
    """

    def __init__(self, lm: LMBase, rank: int = 4, n_basis: int = 16, scale: float = 1.0):
        super().__init__()
        self.rank, self.scale = rank, scale
        slots = lm.adapters()
        self.coef = nn.Linear(lm.dim, len(slots) * n_basis * 2)
        self.n_basis = n_basis
        self.basis_a = nn.ParameterList(
            nn.Parameter(torch.randn(n_basis, rank, s.in_features) * s.in_features ** -0.5)
            for s in slots)
        self.basis_b = nn.ParameterList(
            nn.Parameter(torch.randn(n_basis, s.out_features, rank) * 0.02) for s in slots)

    def forward(self, lm: LMBase, paragraphs: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if not paragraphs:
            return None
        with lm.adapted(None):
            e = lm.embed(paragraphs)                                       # [P, dim]
        c = self.coef(e).view(len(paragraphs), len(self.basis_a), 2, self.n_basis)
        c = c.softmax(-1)
        out = []
        for s, (ba, bb) in enumerate(zip(self.basis_a, self.basis_b)):
            a = torch.einsum("pm,mri->pri", c[:, s, 0], ba)                # [P, r, in]
            b = torch.einsum("pm,mor->por", c[:, s, 1], bb)                # [P, out, r]
            a = a.reshape(-1, a.shape[-1])                                 # ranks stacked
            b = b.permute(1, 0, 2).reshape(b.shape[1], -1)
            out.append((a, self.scale * b))
        return out


def context_distillation_loss(teacher_logits: dict[str, torch.Tensor],
                              student_logits: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean KL(teacher || student) over the heads both produced. The teacher
    is detached: only the memory (and what reads it) learns from this."""
    keys = [k for k in teacher_logits if k in student_logits]
    if not keys:
        return next(iter(student_logits.values())).new_zeros(())
    return torch.stack([
        F.kl_div(student_logits[k].log_softmax(-1), teacher_logits[k].detach().log_softmax(-1),
                 log_target=True, reduction="batchmean") for k in keys]).mean()
