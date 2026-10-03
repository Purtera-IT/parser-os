"""Reasons compiled into operators on the latent space (architecture v5).

explain.py lets an explanation *vote*. This module lets it *act*: every
explanation (a WHY, a company line, a rule card, a question's guidance) is
compiled by one shared network into a small geometric program on a head's
space:

    REGION   where it applies:  μ_e(x) = exp( mean_j log σ((<x, n_ej> - b_ej) / τ) )
             a soft intersection of half-spaces (an "if" clause)
    TOPIC    which questions:   a(e, o) = σ(s · cos(T t_e, U e_o) + c) , or 1 if e names o
    ACTION   what it does there:
             fold    x ← x + σ_e · 2·relu(-<x, m_e>) · m_e
                     (makes the head blind to a distinction, inside the region)
             move    x ← x + δ_e + β · (p_{o,v} - mean_a p_{o,a})
                     (a displacement in head space, plus a pull toward the
                     answer v the explanation concludes, when it states one)

and applied, for line i and opportunity o, in precedence order:

    g_e = a(e, o) · μ_e(x) · reliability_e · Π_{f overrules e} (1 - g_f)
    x ← x + g_e · (fold_e(x) - x) + g_e · move_e,o

then the described head is evaluated on the moved x. With an empty bank
g = 0 and the head is exactly the base head.

What this buys over a vote:
* **Rules generalize by geometry.** The region is learned from the text, so a
  rule about "displays over 55 inches" covers every line in that region,
  including wordings nobody wrote down.
* **Rules compose.** Operators apply in sequence; an exception or an
  overruling switches the older operator off *only inside its own region*
  (case-law semantics), and the old one keeps working elsewhere.
* **Rules can be checked.** The region gives "does this rule apply to this
  line?" as a number. ``follows_rule`` / ``exception_to`` links (a proposed
  labeling field, v5 section 6) supervise it directly.
* **One compiler for all text.** The compiler is shared across WHYs, company
  lines and rule cards and is trained on all of them; a new rule card
  compiles with no new parameters.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .explain import Explanation
from .folds import DescribedHead


class ReasonCompiler(nn.Module):
    """Text embedding of an explanation -> region, topic, fold and move."""

    def __init__(self, d_text: int, d_head: int, k_region: int = 3, d_topic: int = 64,
                 n_change_slots: int = 6):
        super().__init__()
        self.k, self.d = k_region, d_head
        self.region_n = nn.Linear(d_text, k_region * d_head)
        self.region_b = nn.Linear(d_text, k_region)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.topic_e = nn.Linear(d_text, d_topic)
        self.topic_o = nn.Linear(d_text, d_topic)
        self.topic_s = nn.Parameter(torch.tensor(4.0))
        self.topic_c = nn.Parameter(torch.tensor(-1.0))
        self.fold_m = nn.Linear(d_text, d_head)
        self.fold_s = nn.Linear(d_text, 1)
        self.move = nn.Linear(d_text, d_head)
        self.beta = nn.Parameter(torch.tensor(0.5))
        # What the explanation claims will change (hours, crew, sites, price,
        # tasks, schedule) x (down, none, up): graded later by outcomes.
        self.claims = nn.Linear(d_text, n_change_slots * 3)
        self.n_slots = n_change_slots
        nn.init.zeros_(self.region_b.weight)
        nn.init.zeros_(self.region_b.bias)
        nn.init.normal_(self.move.weight, std=0.01)
        nn.init.zeros_(self.move.bias)

    def compile(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """t [M, d_text] -> operator parameters for M explanations."""
        m = t.shape[0]
        return {
            "n": F.normalize(self.region_n(t).view(m, self.k, self.d), dim=-1),
            "b": self.region_b(t),
            "m": F.normalize(self.fold_m(t), dim=-1),
            "sigma": torch.sigmoid(self.fold_s(t)).squeeze(-1),
            "move": self.move(t),
            "topic": self.topic_e(t),
            "claims": self.claims(t).view(m, self.n_slots, 3),
        }

    def region(self, x: torch.Tensor, ops: dict[str, torch.Tensor]) -> torch.Tensor:
        """μ [N, M]: how far each line sits inside each explanation's region."""
        s = torch.einsum("nd,mkd->nmk", x, ops["n"]) - ops["b"].unsqueeze(0)
        return F.logsigmoid(s / self.log_tau.exp()).mean(-1).exp()

    def topic(self, e_opp: torch.Tensor, ops: dict[str, torch.Tensor],
              exps: list[Explanation], opp_key: str) -> torch.Tensor:
        """a [M]: whether each explanation is about this question."""
        cos = F.cosine_similarity(ops["topic"], self.topic_o(e_opp).unsqueeze(0), -1)
        a = torch.sigmoid(self.topic_s * cos + self.topic_c)
        named = torch.tensor([any(c[0] == opp_key for c in e.conclusions) for e in exps],
                             dtype=torch.bool, device=a.device)
        return torch.where(named, torch.ones_like(a), a)

    def gates(self, x: torch.Tensor, ops: dict[str, torch.Tensor], exps: list[Explanation],
              e_opp: torch.Tensor, opp_key: str) -> torch.Tensor:
        """g [N, M]: how strongly each explanation acts on each line, for one question.

        Includes the leave-one-out mask (a line never applies its own WHY),
        reliability, and overruling (inside an overruler's region the
        overruled explanation is switched off)."""
        n = x.shape[0]
        g = self.region(x, ops) * self.topic(e_opp, ops, exps, opp_key).unsqueeze(0)
        rel = torch.tensor([e.reliability for e in exps], dtype=x.dtype, device=x.device)
        g = g * rel.unsqueeze(0)
        own = torch.tensor([e.line for e in exps], device=x.device)
        g = g.masked_fill(own.view(1, -1) == torch.arange(n, device=x.device).view(-1, 1), 0.0)
        refs = {e.ref: j for j, e in enumerate(exps) if e.ref}
        off = torch.ones_like(g)
        for j, e in enumerate(exps):
            for target in e.overrules:
                if target in refs:
                    off[:, refs[target]] = off[:, refs[target]] * (1 - g[:, j])
        return g * off

    def apply(self, x: torch.Tensor, ops: dict[str, torch.Tensor], g: torch.Tensor,
              exps: list[Explanation], opp_key: str, answers: list[str],
              protos: torch.Tensor) -> torch.Tensor:
        """Move the lines [N, d] by every operator in order; return the moved lines."""
        centre = protos.mean(0, keepdim=True)
        for j, e in enumerate(exps):
            gj = g[:, j:j + 1]
            if not bool((gj > 0).any()):
                continue
            m = ops["m"][j]
            s = (x * m).sum(-1, keepdim=True)
            folded = x + ops["sigma"][j] * 2 * F.relu(-s) * m
            move = ops["move"][j].unsqueeze(0)
            for key, value in e.conclusions:
                if key == opp_key and value in answers:
                    move = move + self.beta * (protos[answers.index(value)] - centre)
            x = x + gj * (folded - x) + gj * move
        return x


def apply_reasons(compiler: ReasonCompiler, head: DescribedHead, x: torch.Tensor,
                  e_opp: torch.Tensor, e_ans: torch.Tensor, opp_key: str, answers: list[str],
                  ops: dict[str, torch.Tensor], exps: list[Explanation]
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits [N, A] after the compiled reasons act, and the gates [N, M]."""
    g = compiler.gates(x, ops, exps, e_opp, opp_key)
    moved = compiler.apply(x, ops, g, exps, opp_key, answers, head.prototypes(e_opp, e_ans))
    return head.one(moved, e_opp, e_ans), g
