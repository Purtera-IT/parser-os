"""Explanations the model reads and applies: learn the connection from the text.

A person told "a display over 55 inches needs two technicians, because one
person cannot lift it safely" gets the next 65-inch line right without
seeing a hundred examples. This module gives the model the same route.

**The bank.** Every explanation is an ``Explanation``: its text, its layer
(universal or one company's), and optionally the opportunity and answer it
argues for. Three sources fill it:

* each labeled line's universal WHY, with the card it led to as its
  conclusions (``why_note``);
* each labeled line's company line, ``[purtera] reject: ...``, with that
  company's verdicts as conclusions (``policy_line``);
* **rule cards** a person writes once, in plain words, saying how a kind of
  line connects to an answer and why (``rule_card``; fixtures/rule_cards.json).
  A rule card needs no examples.

**The reader** (``ExplanationReader``). For every line, every opportunity and
every explanation in the same layer, it asks two things:

1. *Does this explanation apply to this line, for this question?* An
   attention score from the line's latent, the question's description and
   the explanation's text, with a "nothing applies" slot so it can abstain.
2. *If it applies, which answer does it argue for?* Read from the text
   itself (the explanation's encoding against each answer's description)
   plus the explanation's stated conclusion when it has one.

The votes are added to the head's logits. Because the reader is trained on
thousands of (line, someone else's explanation) pairs, it learns *how to
apply an explanation*. After that, a new rule card changes predictions the
moment it is added, with no retraining and no new parameters
(tests/test_c3_explain.py).

**What keeps it honest.**
* Universal heads read universal explanations only. Company heads read only
  the active company's. A Purtera rule card cannot move a universal output
  (tested bit-identical).
* A line never reads its own WHY (leave-one-out), so in training the reader
  must transfer another line's reasoning, which is the skill it needs at
  inference, when a new deal has no WHYs at all.
* ``explanation_only`` in losses.py trains the votes alone to reach the gold
  answer, so the reader cannot lean on the base head and ignore the text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .data import IGNORE, Batch
from .schema import Schema

WHY_NOTE, POLICY_LINE, RULE_CARD = "why_note", "policy_line", "rule_card"


@dataclass(frozen=True)
class Explanation:
    text: str
    layer: str                                  # "universal" | "company"
    company: str = ""                           # set for the company layer
    conclusions: tuple[tuple[str, str], ...] = ()  # (opportunity key, answer value)
    source: str = RULE_CARD
    ref: str = ""
    line: int = -1                              # line it came from in this deal, else -1


@dataclass
class ExplanationBank:
    items: list[Explanation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def extend(self, more: Iterable[Explanation]) -> "ExplanationBank":
        return ExplanationBank(self.items + list(more))

    def select(self, layer: str, company: str = "") -> list[int]:
        return [i for i, e in enumerate(self.items)
                if e.layer == layer and (layer == "universal" or e.company == company)]

    @staticmethod
    def from_batch(batch: Batch, schema: Schema, *, deal_lines: bool = True) -> "ExplanationBank":
        """The WHYs and company lines of a featurized deal, with the cards they led to.

        ``line`` is set so the reader can mask a line's own explanation.
        """
        opps = schema.by_key()
        out: list[Explanation] = []
        for i in range(len(batch)):
            uni, com = [], []
            for key, col in batch.targets.items():
                if col[i] == IGNORE:
                    continue
                o = opps[key]
                (uni if o.universal else com).append((key, o.answers[col[i]].value))
            ref = f"{batch.deal_id}#{i}"
            line = i if deal_lines else -1
            if batch.why[i]:
                out.append(Explanation(batch.why[i], "universal", conclusions=tuple(uni),
                                       source=WHY_NOTE, ref=ref, line=line))
            if batch.policy_note[i]:
                out.append(Explanation(batch.policy_note[i], "company", company=batch.company,
                                       conclusions=tuple(com), source=POLICY_LINE,
                                       ref=ref, line=line))
        return ExplanationBank(out)


def load_rule_cards(path: str | Path, schema: Schema | None = None) -> ExplanationBank:
    """Rule cards from JSON: a list of
    ``{"id", "layer", "company"?, "text", "opportunity"?, "answer"?}``.

    ``opportunity`` and ``answer`` are optional: a card that names neither is
    read against every question in its layer, and the reader decides where it
    applies. When given, they are checked against the schema.
    """
    cards = json.loads(Path(path).read_text(encoding="utf-8"))
    by_key = schema.by_key() if schema else {}
    out = []
    for c in cards:
        concl: tuple[tuple[str, str], ...] = ()
        if c.get("opportunity"):
            if by_key:
                o = by_key.get(c["opportunity"])
                if o is None:
                    raise ValueError(f"rule card {c.get('id')}: unknown opportunity {c['opportunity']}")
                if c.get("answer") and o.index(c["answer"]) is None:
                    raise ValueError(f"rule card {c.get('id')}: {c['answer']!r} is not an answer of {o.key}")
                if (o.layer == "universal") != (c["layer"] == "universal"):
                    raise ValueError(f"rule card {c.get('id')}: layer {c['layer']} but {o.key} is {o.layer}")
            if c.get("answer"):
                concl = ((c["opportunity"], c["answer"]),)
        out.append(Explanation(text=c["text"], layer=c["layer"], company=c.get("company", ""),
                               conclusions=concl, source=RULE_CARD, ref=c.get("id", "")))
    return ExplanationBank(out)


class ExplanationReader(nn.Module):
    """Relevance and votes from a bank of explanations, for one layer's heads."""

    def __init__(self, d_text: int, d_head: int, d_att: int = 64):
        super().__init__()
        self.q_line = nn.Linear(d_head, d_att)
        self.q_opp = nn.Linear(d_text, d_att)
        self.k_exp = nn.Linear(d_text, d_att)
        self.null = nn.Parameter(torch.zeros(1))          # "no explanation applies"
        self.topic_e = nn.Linear(d_text, d_att)            # is it about this question?
        self.topic_o = nn.Linear(d_text, d_att)
        self.argue_e = nn.Linear(d_text, d_att)            # which answer does its text argue for?
        self.argue_a = nn.Linear(d_text, d_att)
        self.kappa = nn.Parameter(torch.tensor(2.0))       # weight of a stated conclusion
        self.gain = nn.Linear(d_text, 1)                   # per-question trust in explanations
        nn.init.constant_(self.gain.bias, 1.0)

    def forward(self, x: torch.Tensor, e_opp: torch.Tensor, e_ans: torch.Tensor,
                opp_key: str, answers: list[str], exp_emb: torch.Tensor,
                exps: list[Explanation]) -> tuple[torch.Tensor, torch.Tensor]:
        """Votes [N, A] for one opportunity, and the attention [N, M+1] (last = null)."""
        n, m = x.shape[0], exp_emb.shape[0]
        d = self.q_line.out_features
        q = self.q_line(x) + self.q_opp(e_opp)                              # [N, d]
        k = self.k_exp(exp_emb)                                             # [M, d]
        topic = F.cosine_similarity(self.topic_e(exp_emb), self.topic_o(e_opp).unsqueeze(0), -1)
        named = torch.tensor([any(c[0] == opp_key for c in e.conclusions) for e in exps],
                             device=x.device, dtype=x.dtype)
        score = q @ k.T / d ** 0.5 + 2.0 * topic + 2.0 * named              # [N, M]
        own = torch.tensor([e.line for e in exps], device=x.device)
        mask = own.view(1, -1) == torch.arange(n, device=x.device).view(-1, 1)
        score = score.masked_fill(mask, float("-inf"))
        att = torch.cat([score, self.null.expand(n, 1)], -1).softmax(-1)    # [N, M+1]

        argue = F.normalize(self.argue_e(exp_emb), dim=-1) @ F.normalize(self.argue_a(e_ans), dim=-1).T
        stated = torch.zeros(m, len(answers), device=x.device, dtype=x.dtype)
        for j, e in enumerate(exps):
            for key, value in e.conclusions:
                if key == opp_key and value in answers:
                    stated[j, answers.index(value)] = 1.0
        vote = 4.0 * argue + self.kappa * stated                            # [M, A]
        return F.softplus(self.gain(e_opp)) * (att[:, :m] @ vote), att


def encode_bank(model, bank: ExplanationBank) -> torch.Tensor | None:
    """One embedding per explanation, through the model's own text encoder."""
    if not len(bank):
        return None
    return model.text([e.text for e in bank.items])
