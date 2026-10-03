"""Heads that read with a language model (architecture v6).

For every line, a causal LM (lm.py) reads a plain-text page: the earlier
lines of the deal (never later ones), the line itself, optionally notes from
other labeled lines, and in the reasoned pass the line's own WHY paragraph.
Every head then reads the LM's hidden states over that whole page, queried
by its own description, and scores each answer against the answer's
description, both embedded by the same LM. There is no keyword table, clause
splitter or if/and rule anywhere on this path: the paragraph goes in as
words, and what each word does is learned on top of what the LM already
knows about language.

Three passes over the same page teach it *why*, not just *what*:

=============  ==============================================  ============================
pass           the page ends with                               trained by
=============  ==============================================  ============================
direct         ``Answer:``                                      the label (CE), and toward
                                                                the reasoned pass (distill)
reasoned       ``Why: <the human's WHY paragraph> Answer:``     the label (CE)
rationale      the same page, scored token by token on the WHY  next-token loss: the model
                                                                learns to write the reason
=============  ==============================================  ============================

The reasoned pass makes the heads use the paragraph. The rationale loss makes
the LM able to produce the reasoning itself, so it learns the connection,
not the answer string. Distillation moves what the reasoning concludes into
the direct pass, so at inference the model answers fast and can still be
asked to write its WHY first (``explain``).

Notes from other lines are read in context (``notes``) or compiled into
weights by consolidate.py, so a paragraph read once keeps working after it
leaves the page.

**Firewall.** The universal heads read a page with universal notes and
universal adapters only. The company heads run a second pass whose page adds
the company's notes and whose weights add the company's adapters. Nothing
company-specific ever enters the universal pass, so universal outputs are
bit-identical under any company text (tests/test_c3_brain.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .data import IGNORE, Batch
from .lm import LMBase
from .schema import NUMBER, RELATION, Schema


@dataclass
class BrainOutput:
    logits: dict[str, torch.Tensor] = field(default_factory=dict)    # opportunity -> [N, A]
    numbers: dict[str, torch.Tensor] = field(default_factory=dict)   # opportunity -> [N]
    attention: dict[str, torch.Tensor] = field(default_factory=dict) # layer -> [N, O, T] over the page
    hidden: torch.Tensor | None = None                                # [N, T, dim] universal pass
    mask: torch.Tensor | None = None
    ids: torch.Tensor | None = None


class HeadReader(nn.Module):
    """Every head of one layer reads the page. Head i pools the hidden
    states with a query made from its own description; answers are scored
    against their descriptions. New heads and answers need no new weights."""

    def __init__(self, dim: int, d_att: int = 64):
        super().__init__()
        self.q = nn.Linear(dim, d_att)
        self.k = nn.Linear(dim, d_att)
        self.v = nn.Linear(dim, dim)
        self.ans = nn.Linear(dim, dim)
        self.num = nn.Linear(dim, 1)
        self.temp = nn.Parameter(torch.tensor(4.0))

    def forward(self, h: torch.Tensor, mask: torch.Tensor, e_opp: torch.Tensor,
                e_ans: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        q = self.q(e_opp)                                              # [O, a]
        k = self.k(h)                                                  # [N, T, a]
        s = torch.einsum("oa,nta->not", q, k) / q.shape[-1] ** 0.5
        s = s.masked_fill(~mask.unsqueeze(1), float("-inf"))
        att = s.softmax(-1)                                            # [N, O, T]
        pooled = torch.einsum("not,ntd->nod", att, self.v(h))          # [N, O, dim]
        p = F.normalize(pooled, dim=-1)
        logits = [self.temp * p[:, o] @ F.normalize(self.ans(a), dim=-1).T
                  for o, a in enumerate(e_ans)]
        return logits, self.num(pooled).squeeze(-1), att


class Brain(nn.Module):
    def __init__(self, schema: Schema, lm: LMBase, d_graph: int | None = None,
                 n_prefix: int = 2, context_lines: int = 6, d_att: int = 64):
        super().__init__()
        self.schema, self.lm = schema, lm
        self.context_lines = context_lines
        self.reader_universal = HeadReader(lm.dim, d_att)
        self.reader_company = HeadReader(lm.dim, d_att)
        self.n_prefix = n_prefix
        # The deal graph's rationale r_i (model.C3Model) as soft tokens at the
        # front of the page, so structure the text does not show (same
        # section, time order, hindsight-trained consequence) reaches the LM.
        self.graph = nn.Linear(d_graph, n_prefix * lm.dim) if d_graph else None
        self.opps = {
            layer: [o for o in schema.select(layer=layer) if o.kind != RELATION]
            for layer in ("universal", "company")
        }

    # ------------------------------------------------------------ text
    def describe(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Every head and answer description, embedded by the LM itself."""
        emb = self.lm.embed(self.schema.texts())
        out, i = {}, 0
        for o in self.schema.opportunities:
            out[o.key] = (emb[i], emb[i + 1:i + 1 + len(o.answers)])
            i += 1 + len(o.answers)
        return out

    def page(self, batch: Batch, i: int, notes: list[str] | None = None) -> str:
        """Line i as the LM reads it: earlier lines only (foresight), then the
        line, then any notes. Plain formatting, no rules."""
        lo = max(0, i - self.context_lines)
        def who(j: int) -> str:
            return ", ".join(x for x in (batch.role[j], batch.doc_kind[j]) if x) or "line"
        parts = []
        if lo < i:
            parts.append("Earlier in the deal:")
            parts += [f"- ({who(j)}) {batch.texts[j]}" for j in range(lo, i)]
        parts.append(f"Line ({who(i)}): {batch.texts[i]}")
        if notes:
            parts.append("Notes from labeled lines:")
            parts += [f"- {n}" for n in notes]
        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------ reading
    def read(self, pages: list[str], layer: str, desc, graph: torch.Tensor | None = None,
             deltas=None) -> tuple[BrainOutput, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, mask = self.lm.tokenize(pages)
        prefix = None
        if self.graph is not None and graph is not None:
            prefix = self.graph(graph).view(len(pages), self.n_prefix, self.lm.dim)
        with self.lm.adapted(deltas):
            h, lm_logits = self.lm(ids, mask, prefix=prefix)
        opps = self.opps[layer]
        reader = self.reader_universal if layer == "universal" else self.reader_company
        out = BrainOutput(hidden=h, mask=mask, ids=ids)
        if opps:
            logits, nums, att = reader(h, mask, torch.stack([desc[o.key][0] for o in opps]),
                                       [desc[o.key][1] for o in opps])
            out.attention[layer] = att
            for j, (o, lg) in enumerate(zip(opps, logits)):
                out.logits[o.key] = lg
                if o.kind == NUMBER:
                    out.numbers[o.key] = nums[:, j]
        return out, ids, mask, lm_logits

    def forward(self, batch: Batch, *, company: str | None = None,
                why: list[str | None] | None = None,
                notes: list[list[str]] | None = None,
                company_notes: list[list[str]] | None = None,
                deltas=None, company_deltas=None,
                graph: torch.Tensor | None = None, desc=None) -> BrainOutput:
        """Direct pass by default; pass ``why`` (one paragraph or None per
        line) for the reasoned pass. ``notes`` and ``deltas`` are universal;
        ``company_notes`` and ``company_deltas`` reach the company heads only."""
        desc = desc if desc is not None else self.describe()
        n = len(batch)
        notes = notes or [[] for _ in range(n)]
        pages = [self.page(batch, i, notes[i]) + _tail(why, i) for i in range(n)]
        out, _, _, _ = self.read(pages, "universal", desc, graph, deltas)
        if company is not None and self.opps["company"]:
            cn = company_notes or [[] for _ in range(n)]
            cpages = [self.page(batch, i, notes[i] + cn[i]) + _tail(why, i) for i in range(n)]
            both = _concat(deltas, company_deltas)
            cout, _, _, _ = self.read(cpages, "company", desc, graph, both)
            out.logits.update(cout.logits)
            out.numbers.update(cout.numbers)
            out.attention.update(cout.attention)
        return out

    # ------------------------------------------------------------ the WHY
    def rationale_loss(self, batch: Batch, notes: list[list[str]] | None = None,
                       graph: torch.Tensor | None = None) -> torch.Tensor:
        """Next-token loss on each human WHY, given the page: the LM learns to
        write the reasoning an expert wrote for this line."""
        idx = [i for i, w in enumerate(batch.why) if w]
        if not idx:
            return next(self.parameters()).new_zeros(())
        notes = notes or [[] for _ in range(len(batch))]
        heads = [self.page(batch, i, notes[i]) + "Why: " for i in idx]
        fulls = [h + batch.why[i] for h, i in zip(heads, idx)]
        _, hmask = self.lm.tokenize(heads)
        ids, mask = self.lm.tokenize(fulls)
        prefix = None
        if self.graph is not None and graph is not None:
            prefix = self.graph(graph[idx]).view(len(idx), self.n_prefix, self.lm.dim)
        _, logits = self.lm(ids, mask, prefix=prefix)
        start = hmask.sum(1)                                           # first WHY token
        pos = torch.arange(ids.shape[1], device=ids.device).view(1, -1)
        on_why = (pos >= start.view(-1, 1)) & mask & (pos > 0)
        tgt = ids.masked_fill(~on_why, IGNORE)[:, 1:]
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                               ignore_index=IGNORE)

    @torch.no_grad()
    def explain(self, batch: Batch, i: int, max_new: int = 120,
                notes: list[str] | None = None) -> str:
        """Greedy-write the model's own WHY for line i. Untrained: noise."""
        text = self.page(batch, i, notes) + "Why: "
        ids, mask = self.lm.tokenize([text])
        start = int(mask.sum())
        ids = ids[:, :start]
        for _ in range(max_new):
            _, logits = self.lm(ids, torch.ones_like(ids, dtype=torch.bool))
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if ids.shape[1] >= self.lm.max_len:
                break
        if hasattr(self.lm, "tokenizer"):
            return self.lm.tokenizer.decode(ids[0, start:].tolist(), skip_special_tokens=True)
        return bytes(t for t in ids[0, start:].tolist() if t < 256).decode("utf-8", "replace")


def notes_from_deal(batch: Batch, k: int = 3) -> tuple[list[list[str]], list[list[str]]]:
    """Universal and company notes per line from the deal's OTHER labeled
    lines (a line never reads its own WHY), in deal order. Retrieval by
    similarity is the training thread's choice; at inference the notes come
    from past deals."""
    uni, com = [], []
    for i in range(len(batch)):
        u = [batch.why[j] for j in range(len(batch)) if j != i and batch.why[j]]
        c = [batch.policy_note[j] for j in range(len(batch)) if j != i and batch.policy_note[j]]
        uni.append(u[:k])
        com.append(c[:k])
    return uni, com


def _tail(why: list[str | None] | None, i: int) -> str:
    if why is not None and why[i]:
        return f"Why: {why[i]}\nAnswer:"
    return "Answer:"


def _concat(a, b):
    """Two adapter sets as one: ranks stacked, so the deltas add."""
    if a is None:
        return b
    if b is None:
        return a
    return [(torch.cat([x[0], y[0]], 0), torch.cat([x[1], y[1]], 1)) for x, y in zip(a, b)]


def brain_targets(batch: Batch, key: str, device: Any) -> torch.Tensor:
    return torch.tensor(batch.targets[key], device=device)
