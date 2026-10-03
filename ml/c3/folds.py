"""Described heads: the labeling text builds the output layer and folds the space.

This is the part of C3 that is new. Every labeling opportunity comes with
text a person wrote for a person: the head's question, the reading's
description, each answer's description (schema.py). A conventional head
throws that text away and learns a fresh weight vector per class. Here the
text does three jobs:

1. **Answers are points, not weights.** The logit for answer *v* is the
   similarity between the line's (folded) latent and the encoded description
   of *v*. A new reading or a new answer gets a working head from its text
   alone, and two answers described alike start alike.

2. **The question folds the space.** From the opportunity's description a
   small hypernetwork generates a stack of folds. A fold is a reflection of
   the half-space on one side of a hyperplane onto the other:

       s = <x, n> - b          x  <-  x + 2·σ·relu(-s)·n

   With σ = 1 it maps every point to the side where s >= 0 and is an
   isometry on each side: it *identifies* two regions of the space, so the
   head is blind to which side a line was on. That is what a question does
   to a reader: "is it in the SOW?" does not care who said it; "who owns the
   task?" does. Each question gets the invariances its text implies, and
   questions whose text is alike get alike folds, so a rare head borrows
   the geometry of a common one through its words. σ in [0, 1] lets a fold
   be partial (a crease).

3. **The WHY is the path from the line to its answer.** A labeled line has
   a WHY that says what it is and what it changes. In training, the WHY's
   encoding is (a) the target the rationale latent ``r_i`` is pulled toward
   and (b) a substitute for ``r_i`` that must on its own carry the line to
   the right answer through the same folds (``why_sufficiency`` in
   losses.py). So the WHY is not a side label; it is a displacement in the
   same space the descriptions live in, and the model at inference must
   predict that displacement for lines nobody explained.

Prototypes are folded by the same folds as the line, so both live on the same
sheet of the folded space.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FoldStack(nn.Module):
    """Generate k folds and a FiLM from one description embedding, then apply them."""

    def __init__(self, d_desc: int, d: int, n_folds: int = 4):
        super().__init__()
        self.d, self.k = d, n_folds
        self.normals = nn.Linear(d_desc, n_folds * d)
        self.offsets = nn.Linear(d_desc, n_folds)
        self.strength = nn.Linear(d_desc, n_folds)
        self.film = nn.Linear(d_desc, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def params(self, e: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fold parameters for description embeddings ``e`` [O, d_desc]."""
        n = F.normalize(self.normals(e).view(-1, self.k, self.d), dim=-1)
        gamma, beta = self.film(e).chunk(2, dim=-1)
        return {"n": n, "b": self.offsets(e), "sigma": torch.sigmoid(self.strength(e)),
                "gamma": 1 + torch.tanh(gamma), "beta": beta}

    @staticmethod
    def apply(x: torch.Tensor, p: dict[str, torch.Tensor], o: int | None = None) -> torch.Tensor:
        """Fold ``x`` [..., d] with the folds of opportunity ``o`` (or all, broadcast).

        With ``o`` given, ``p`` holds every opportunity and ``x`` is [N, d].
        With ``o`` None, ``p`` holds one opportunity per leading row of ``x``.
        """
        sel = (lambda t: t[o]) if o is not None else (lambda t: t)
        n, b, sigma = sel(p["n"]), sel(p["b"]), sel(p["sigma"])
        x = x * sel(p["gamma"]) + sel(p["beta"])
        for j in range(n.shape[-2]):
            nj = n[..., j, :]
            s = (x * nj).sum(-1, keepdim=True) - b[..., j:j + 1]
            x = x + 2 * sigma[..., j:j + 1] * F.relu(-s) * nj
        return x


class DescribedHead(nn.Module):
    """Logits for every class-like opportunity, built from the schema's text.

    ``forward(x, desc, answers)``:
      x        [N, d]       the lines' latents for this head's space
      desc     [O, d_desc]  one embedding per opportunity description
      answers  list of [A_o, d_desc] answer-description embeddings
    returns a list of [N, A_o] logits, one per opportunity.
    """

    def __init__(self, d_desc: int, d: int, n_folds: int = 4, scale: float = 10.0):
        super().__init__()
        self.folds = FoldStack(d_desc, d, n_folds)
        self.proto = nn.Linear(d_desc, d)
        self.bias = nn.Linear(d_desc, 1)       # a prior per answer, also from its text
        self.number = nn.Linear(d_desc, d)     # a read-out direction for numeric answers
        self.log_scale = nn.Parameter(torch.tensor(float(scale)).log())

    def forward(self, x: torch.Tensor, desc: torch.Tensor,
                answers: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        p = self.folds.params(desc)
        logits = []
        numbers = []
        for o, a in enumerate(answers):
            xf = F.normalize(FoldStack.apply(x, p, o), dim=-1)
            pf = F.normalize(FoldStack.apply(self.proto(a), p, o), dim=-1)
            logits.append(self.log_scale.exp() * xf @ pf.T + self.bias(a).T)
            numbers.append(FoldStack.apply(x, p, o) @ F.normalize(self.number(desc[o]), dim=-1))
        return logits, torch.stack(numbers, -1) if numbers else x.new_zeros(x.shape[0], 0)

    def one(self, x: torch.Tensor, desc: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        """Logits [N, A] for a single opportunity (desc [d_desc], answers [A, d_desc]).
        Same computation as ``forward`` for that opportunity; used after a
        compiled reason has moved the lines (operators.py)."""
        logits, _ = self.forward(x, desc.unsqueeze(0), [answers])
        return logits[0]

    def prototypes(self, desc: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        """Answer points [A, d] in this head's (unfolded) space."""
        return self.proto(answers)


class DescribedRelation(nn.Module):
    """Pair scores for a relation, with both ends folded by the relation's text."""

    def __init__(self, d_desc: int, d: int, n_folds: int = 4):
        super().__init__()
        self.src = FoldStack(d_desc, d, n_folds)
        self.dst = FoldStack(d_desc, d, n_folds)
        self.a = nn.Linear(d, d, bias=False)
        self.b = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        """[R, N, N] scores: relation r from line i (source) to line j (target)."""
        ps, pd = self.src.params(desc), self.dst.params(desc)
        out = []
        for r in range(desc.shape[0]):
            s = FoldStack.apply(self.a(x), ps, r)
            t = FoldStack.apply(self.b(x), pd, r)
            out.append(s @ t.T / s.shape[-1] ** 0.5)
        return torch.stack(out)
