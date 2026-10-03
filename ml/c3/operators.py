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

from .clauses import (CAUSE, CONDITION, CONSEQUENCE, EVIDENCE, EXCEPTION, ROLES, STATEMENT,
                      split_clauses)
from .explain import Explanation
from .folds import DescribedHead
from .notes import extract_programs


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

        # Full paragraphs (v5.1): every clause gets its own slot by role, and
        # every word stays readable by the lines.
        self.role = nn.Embedding(len(ROLES), d_text)
        nn.init.normal_(self.role.weight, std=0.02)
        self.cond_n = nn.Linear(d_text, k_region * d_head)
        self.cond_b = nn.Linear(d_text, k_region)
        self.exc_n = nn.Linear(d_text, k_region * d_head)
        self.exc_b = nn.Linear(d_text, k_region)
        for lin in (self.cond_b, self.exc_b):
            nn.init.zeros_(lin.weight)
        nn.init.zeros_(self.cond_b.bias)
        nn.init.constant_(self.exc_b.bias, 0.0)
        self.clause_fold = nn.Linear(d_text, d_head)
        self.clause_move = nn.Linear(d_text, d_head)
        nn.init.normal_(self.clause_move.weight, std=0.01)
        self.clause_claims = nn.Linear(d_text, n_change_slots * 3)
        self.clause_topic = nn.Linear(d_text, d_topic)
        self.tok_q = nn.Linear(d_head, d_topic)
        self.tok_k = nn.Linear(d_text, d_topic)
        self.tok_v = nn.Linear(d_text, d_head)
        nn.init.normal_(self.tok_v.weight, std=0.01)
        self.detail_gain = nn.Parameter(torch.tensor(1.0))

    def compile(self, t: torch.Tensor, texts: list[str] | None = None,
                encoder: nn.Module | None = None) -> dict:
        """t [M, d_text] -> operator parameters for M explanations.

        With ``texts`` and ``encoder``, each explanation is also read clause by
        clause (clauses.py) and word by word, so a full paragraph is compiled
        whole: conditions AND together into the region, exceptions are cut out
        of it, every cause shapes the fold, every consequence adds to the move
        and the claims, arithmetic becomes programs, and the words stay
        available to each line (``detail``)."""
        ops = self._compile_whole(t)
        if texts is not None and encoder is not None:
            self._compile_clauses(ops, texts, encoder)
        return ops

    def _compile_clauses(self, ops: dict, texts: list[str], encoder: nn.Module) -> None:
        m = len(texts)
        dev = ops["move"].device
        clauses = [split_clauses(tx) for tx in texts]
        flat = [(j, c) for j, cs in enumerate(clauses) for c in cs]
        ops["clauses"] = clauses
        ops["programs"] = [extract_programs(tx) for tx in texts]
        if not flat:
            return
        owner = torch.tensor([j for j, _ in flat], device=dev)
        role = torch.tensor([ROLES.index(c.role) for _, c in flat], device=dev)
        e = encoder([c.text for _, c in flat]) + self.role(role)            # [C, d_text]

        def pick(r: str) -> torch.Tensor:
            return (role == ROLES.index(r))

        k, d = self.k, self.d
        ops["cond"] = (F.normalize(self.cond_n(e).view(-1, k, d), dim=-1), self.cond_b(e),
                       owner, pick(CONDITION))
        ops["exc"] = (F.normalize(self.exc_n(e).view(-1, k, d), dim=-1), self.exc_b(e),
                      owner, pick(EXCEPTION))
        ops["has_cond"] = torch.zeros(m, dtype=torch.bool, device=dev).index_fill(
            0, owner[pick(CONDITION)], True) if pick(CONDITION).any() else torch.zeros(
            m, dtype=torch.bool, device=dev)

        def add(base: torch.Tensor, part: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            w = mask.to(part.dtype).unsqueeze(-1)
            return base.index_add(0, owner, part * w.view(-1, *([1] * (part.dim() - 1))))

        fold = add(self.fold_m.weight.new_zeros(m, d), self.clause_fold(e), pick(CAUSE))
        ops["m"] = F.normalize(ops["m_raw"] + fold, dim=-1)
        conseq = pick(CONSEQUENCE)
        ops["move"] = add(ops["move"], self.clause_move(e), conseq)
        ops["claims"] = add(ops["claims"], self.clause_claims(e).view(-1, self.n_slots, 3), conseq)
        topical = pick(EVIDENCE) | pick(STATEMENT)
        ops["topic"] = add(ops["topic"], self.clause_topic(e), topical)

        tok, mask = encoder.tokens(texts)                                     # [M, T, d_text]
        ops["tok_k"], ops["tok_v"], ops["tok_mask"] = self.tok_k(tok), self.tok_v(tok), mask

    def _compile_whole(self, t: torch.Tensor) -> dict:
        m = t.shape[0]
        return {
            "m_raw": self.fold_m(t),
            "n": F.normalize(self.region_n(t).view(m, self.k, self.d), dim=-1),
            "b": self.region_b(t),
            "m": F.normalize(self.fold_m(t), dim=-1),
            "sigma": torch.sigmoid(self.fold_s(t)).squeeze(-1),
            "move": self.move(t),
            "topic": self.topic_e(t),
            "claims": self.claims(t).view(m, self.n_slots, 3),
        }

    def _half(self, x: torch.Tensor, n: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        s = torch.einsum("nd,mkd->nmk", x, n) - b.unsqueeze(0)
        return F.logsigmoid(s / self.log_tau.exp()).mean(-1)                 # log μ

    def region(self, x: torch.Tensor, ops: dict) -> torch.Tensor:
        """μ [N, M]: how far each line sits inside each explanation's region.

        Whole-text region by default. When the explanation has condition
        clauses, its region is their AND (product); every exception clause
        cuts its own region out (× (1 - μ_exception))."""
        log_mu = self._half(x, ops["n"], ops["b"])                           # [N, M]
        if "cond" in ops:
            n_c, b_c, owner, is_c = ops["cond"]
            if is_c.any():
                lc = self._half(x, n_c[is_c], b_c[is_c])                      # [N, C]
                summed = x.new_zeros(x.shape[0], log_mu.shape[1]).index_add(1, owner[is_c], lc)
                log_mu = torch.where(ops["has_cond"].unsqueeze(0), summed, log_mu)
            n_e, b_e, owner_e, is_e = ops["exc"]
            if is_e.any():
                le = self._half(x, n_e[is_e], b_e[is_e]).exp()
                keep = torch.log1p(-le.clamp(max=1 - 1e-6))
                log_mu = log_mu.index_add(1, owner_e[is_e], keep)
        return log_mu.exp()

    def detail(self, x: torch.Tensor, ops: dict) -> torch.Tensor | None:
        """[N, M, d]: what each line reads from each explanation's own words.
        A paragraph that names the site, the size and the step lets each line
        pick out the words that are about it, instead of one summary vector."""
        if "tok_k" not in ops:
            return None
        q = self.tok_q(x)                                                    # [N, d_t]
        att = torch.einsum("nd,mtd->nmt", q, ops["tok_k"]) / q.shape[-1] ** 0.5
        att = att.masked_fill(~ops["tok_mask"].unsqueeze(0), float("-inf")).softmax(-1)
        return self.detail_gain * torch.einsum("nmt,mtd->nmd", att, ops["tok_v"])

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
        detail = self.detail(x, ops)
        for j, e in enumerate(exps):
            gj = g[:, j:j + 1]
            if not bool((gj > 0).any()):
                continue
            m = ops["m"][j]
            s = (x * m).sum(-1, keepdim=True)
            folded = x + ops["sigma"][j] * 2 * F.relu(-s) * m
            move = ops["move"][j].unsqueeze(0)
            if detail is not None:
                move = move + detail[:, j]
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
