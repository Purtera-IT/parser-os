"""The training-only teacher that reads the long labels (architecture v6).

The product is the task heads (model.C3Model); they run alone, with no
language model. This module is the teacher they learn from in training
(supercharge.py), and is never loaded at run time. It can wrap a pretrained
LM or encoder so it already knows the language; the heads never depend on
it generating text.

For every line, a causal LM (lm.py) reads a plain-text page: the earlier
lines of the deal (never later ones), the line itself, optionally notes from
other labeled lines, and in the reasoned pass the line's own WHY paragraph.
The teacher's heads read the LM's hidden states over that whole page, queried
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

The reasoned pass makes the teacher use the paragraph. The rationale loss makes
the LM able to produce the reasoning itself, so it learns the connection,
not the answer string. Distillation moves what the reasoning concludes into
the direct pass, which is the read the teacher gives on lines nobody
labeled. ``explain`` lets a person see what WHY the teacher would write.

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
import torch
from torch import nn
from torch.nn import functional as F

from .consolidate import ParagraphMemory
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
        # Paragraphs compiled into weights (consolidate.py), one per layer.
        self.memory_universal = ParagraphMemory(lm)
        self.memory_company = ParagraphMemory(lm)
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

    # ------------------------------------------------------------ flips
    def flip_read(self, batch: Batch, why: list[str | None], flips: list,
                  notes: list[list[str]] | None = None, desc=None,
                  graph: torch.Tensor | None = None) -> BrainOutput:
        """The flip pass: for each near-miss contrast a labeler wrote ("if X,
        this would be Y"), the reasoned page with X assumed. Row n of every
        output is flip n. The page never states Y: the teacher has to apply
        the rule in the WHY to the changed case, which is what it then
        teaches the heads (supercharge.teach_flip)."""
        desc = desc if desc is not None else self.describe()
        notes = notes or [[] for _ in range(len(batch))]
        pages = [self.page(batch, f.line, notes[f.line])
                 + (f"Why: {why[f.line]}\n" if why[f.line] else "")
                 + f"Suppose instead: {f.condition}\nAnswer:" for f in flips]
        g = graph[[f.line for f in flips]] if graph is not None else None
        out, _, _, _ = self.read(pages, "universal", desc, g)
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

    # ------------------------------------------------------------ grounding
    def grounded_page(self, batch: Batch, i: int, window: int = 12
                      ) -> tuple[str, dict[int, tuple[int, int]], tuple[int, int]]:
        """The teacher's privileged page for line i: the deal's lines around
        it, before AND after (the teacher may see hindsight; the heads never
        do), each numbered, then the line's WHY. Returns the text, each
        line's character span, and the WHY's span."""
        lo, hi = max(0, i - window), min(len(batch), i + window + 1)
        text, spans = "Deal lines:\n", {}
        for j in range(lo, hi):
            role = ", ".join(x for x in (batch.role[j], batch.doc_kind[j]) if x) or "line"
            head = f"{'>>' if j == i else '-'} [{j}] ({role}) "
            spans[j] = (len(text) + len(head), len(text) + len(head) + len(batch.texts[j]))
            text += head + batch.texts[j] + "\n"
        text += f"Why line [{i}] is labeled as it is: "
        why = (len(text), len(text) + len(batch.why[i] or ""))
        return text + (batch.why[i] or ""), spans, why

    def grounding(self, batch: Batch, window: int = 12
                  ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """What each WHY rests on, read from the teacher itself, not parsed.

        For every line with a WHY, the teacher's log-likelihood of that WHY is
        traced back to the page by gradient x input: the weight on each
        token is how much the expert's reasoning, as the teacher reads it,
        depends on that token. Summed per line, it says which other lines
        and documents the reasoning points at (``pointers``: [N, N+1], last
        column = rests on the line itself). Summed per word of the line, it
        says which facts were cited and which were passed over, such as a
        price the WHY calls irrelevant (``words``: one distribution per line
        over text.word_spans). Rows without a WHY are zero / None.
        """
        from .text import word_spans  # noqa: PLC0415

        n = len(batch)
        dev = next(self.parameters()).device
        pointers = torch.zeros(n, n + 1, device=dev)
        words: list[torch.Tensor | None] = [None] * n
        was = self.training
        self.eval()
        for i in range(n):
            if not batch.why[i]:
                continue
            text, spans, (ws, we) = self.grounded_page(batch, i, window)
            ids, mask = self.lm.tokenize([text])
            offs = self.lm.offsets(text)[: ids.shape[1]]
            emb = self.lm.input_embeddings(ids).detach().requires_grad_(True)
            with torch.enable_grad():
                _, logits = self.lm(ids, mask, inputs_embeds=emb)
                start = next((t for t, (a, _) in enumerate(offs) if a >= ws and t > 0), len(offs))
                lp = logits[0, start - 1:-1].log_softmax(-1)
                ll = lp.gather(-1, ids[0, start:].unsqueeze(-1)).sum()
                (g,) = torch.autograd.grad(ll, emb)
            sal = (g * emb).sum(-1).abs()[0].detach()                     # [T]
            per_char = {}
            for t, (a, b) in enumerate(offs):
                if b > a and b <= ws:
                    for c in range(a, b):
                        per_char[c] = per_char.get(c, 0.0) + float(sal[t]) / (b - a)
            def mass(a: int, b: int) -> float:
                return sum(per_char.get(c, 0.0) for c in range(a, b))
            for j, (a, b) in spans.items():
                pointers[i, n if j == i else j] = mass(a, b)
            a0, _ = spans[i]
            w = [mass(a0 + a, a0 + b) for _, a, b in word_spans(batch.texts[i])]
            if w and sum(w) > 0:
                words[i] = torch.tensor(w, device=dev) / sum(w)
            tot = pointers[i].sum()
            if tot > 0:
                pointers[i] /= tot
        self.train(was)
        return pointers, words

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


def told_why(batch: Batch, schema: Schema) -> list[str | None]:
    """Each line's WHY plus its universal per-field notes ("sow_coverage: ..."),
    as the reasoned pass reads them. Company field notes go to the company
    pass only (``company_field_notes``)."""
    layer = {o.key: o.layer for o in schema.opportunities}
    out = []
    for i in range(len(batch)):
        parts = [batch.why[i]] if batch.why[i] else []
        notes = batch.field_notes[i] if i < len(batch.field_notes) else {}
        parts += [f"{k.split(':', 1)[1]}: {t}" for k, t in notes.items() if layer.get(k) == "universal"]
        out.append("\n".join(parts) or None)
    return out


def company_field_notes(batch: Batch, schema: Schema) -> list[list[str]]:
    layer = {o.key: o.layer for o in schema.opportunities}
    return [[f"{k.split(':', 1)[1]}: {t}" for k, t in (batch.field_notes[i].items()
             if i < len(batch.field_notes) else []) if layer.get(k) == "company"]
            for i in range(len(batch))]


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


