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
    #: Mean of each page's hidden states [N, dim]. Always set; ``hidden``,
    #: ``mask``, ``ids`` and ``attention`` are left empty when the pages were
    #: read in chunks, since padding differs per chunk.
    pooled: torch.Tensor | None = None
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
                 n_prefix: int = 2, context_lines: int = 6, d_att: int = 64,
                 page_batch: int = 0, read_chunk: int = 0):
        super().__init__()
        self.schema, self.lm = schema, lm
        self.context_lines = context_lines
        # 0 runs every line of a deal in one language-model pass. A positive
        # width splits that pass so a large teacher fits next to the student.
        self.page_batch = page_batch
        # Pages read (tokenized, through the LM and the readers) per chunk; 0
        # reads them all at once. Without it an 8,192-line part holds every
        # page's hidden states at once.
        self.read_chunk = read_chunk
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

    def _lm_rows(self, ids: torch.Tensor, mask: torch.Tensor,
                 prefix: torch.Tensor | None = None, *, logits: bool = True):
        """One language-model pass, or several row-slices of the same padded page.

        Slicing the batch does not change a line's tokens or its mask. The
        slices share one width, so the hidden states concatenate on the row.
        ``logits=False`` skips the vocabulary projection and returns
        ``(hidden, None)``.
        """
        def one(i, m, pre):
            return self.lm(i, m, prefix=pre) if logits else (self.lm.hidden(i, m, prefix=pre), None)

        width = self.page_batch
        if not width or ids.shape[0] <= width:
            return one(ids, mask, prefix)
        hidden, lgs = [], []
        for start in range(0, ids.shape[0], width):
            stop = start + width
            part = None if prefix is None else prefix[start:stop]
            h, lg = one(ids[start:stop], mask[start:stop], part)
            hidden.append(h)
            lgs.append(lg)
        return torch.cat(hidden, 0), (torch.cat(lgs, 0) if logits else None)

    # ------------------------------------------------------------ reading
    def read(self, pages: list[str], layer: str, desc, graph: torch.Tensor | None = None,
             deltas=None) -> tuple[BrainOutput, torch.Tensor | None, torch.Tensor | None, None]:
        """The heads' reading of each page. In chunks of ``read_chunk`` pages
        when set (longest pages together, so padding stays small); the rows
        come back in page order either way."""
        width = self.read_chunk
        if not width or len(pages) <= width:
            return self._read(pages, layer, desc, graph, deltas, keep=True)
        order = sorted(range(len(pages)), key=lambda i: len(pages[i]))
        parts = []
        for start in range(0, len(order), width):
            rows = order[start:start + width]
            g = graph[rows] if graph is not None else None
            parts.append((rows, self._read([pages[i] for i in rows], layer, desc, g, deltas, keep=False)[0]))
        back = torch.empty(len(pages), dtype=torch.long)
        back[[i for rows, _ in parts for i in rows]] = torch.arange(len(pages))
        back = back.to(parts[0][1].pooled.device)
        out = BrainOutput(pooled=torch.cat([o.pooled for _, o in parts])[back])
        for key in parts[0][1].logits:
            out.logits[key] = torch.cat([o.logits[key] for _, o in parts])[back]
        for key in parts[0][1].numbers:
            out.numbers[key] = torch.cat([o.numbers[key] for _, o in parts])[back]
        return out, None, None, None

    def _read(self, pages, layer, desc, graph, deltas, *, keep: bool):
        ids, mask = self.lm.tokenize(pages)
        prefix = None
        if self.graph is not None and graph is not None:
            prefix = self.graph(graph).view(len(pages), self.n_prefix, self.lm.dim)
        with self.lm.adapted(deltas):
            h, _ = self._lm_rows(ids, mask, prefix, logits=False)
        opps = self.opps[layer]
        reader = self.reader_universal if layer == "universal" else self.reader_company
        m = mask.unsqueeze(-1).to(h.dtype)
        out = BrainOutput(pooled=(h * m).sum(1) / m.sum(1).clamp(min=1))
        if keep:
            out.hidden, out.mask, out.ids = h, mask, ids
        if opps:
            logits, nums, att = reader(h, mask, torch.stack([desc[o.key][0] for o in opps]),
                                       [desc[o.key][1] for o in opps])
            if keep:
                out.attention[layer] = att
            for j, (o, lg) in enumerate(zip(opps, logits)):
                out.logits[o.key] = lg
                if o.kind == NUMBER:
                    out.numbers[o.key] = nums[:, j]
        return out, (ids if keep else None), (mask if keep else None), None

    def forward(self, batch: Batch, *, company: str | None = None,
                why: list[str | None] | None = None,
                notes: list[list[str]] | None = None,
                company_notes: list[list[str]] | None = None,
                deltas=None, company_deltas=None,
                graph: torch.Tensor | None = None, desc=None,
                rows: list[int] | None = None) -> BrainOutput:
        """Direct pass by default; pass ``why`` (one paragraph or None per
        line) for the reasoned pass. ``notes`` and ``deltas`` are universal;
        ``company_notes`` and ``company_deltas`` reach the company heads only.
        ``rows`` reads only those lines (each still with its earlier lines on
        the page); output row k is line ``rows[k]``."""
        desc = desc if desc is not None else self.describe()
        n = len(batch)
        g = graph if graph is None or rows is None else graph[list(rows)]
        rows = list(range(n)) if rows is None else list(rows)
        notes = notes or [[] for _ in range(n)]
        pages = [self.page(batch, i, notes[i]) + _tail(why, i) for i in rows]
        out, _, _, _ = self.read(pages, "universal", desc, g, deltas)
        if company is not None and self.opps["company"]:
            cn = company_notes or [[] for _ in range(n)]
            cpages = [self.page(batch, i, notes[i] + cn[i]) + _tail(why, i) for i in rows]
            both = _concat(deltas, company_deltas)
            cout, _, _, _ = self.read(cpages, "company", desc, g, both)
            out.logits.update(cout.logits)
            out.numbers.update(cout.numbers)
            out.attention.update(cout.attention)
        return out

    def card_page(self, batch: Batch, judgment, *, with_why: bool = False) -> str:
        """Source evidence and source question, isolated from saved target fields."""
        o = self.schema.by_key()[judgment.key]
        text = "Evidence:\n" + "\n".join(f"[{i}] {batch.texts[i]}" for i in judgment.lines)
        if judgment.query:
            text += "\nQuestion: " + judgment.query
        if not o.universal:
            text += "\nCompany policy: " + (batch.company_policy or "")
        if with_why:
            text += "\nWhy: " + judgment.note
            if not o.universal and judgment.policy:
                text += "\nCompany reasoning: " + judgment.policy
        return text + "\nAnswer:"

    def card_logits(self, batch: Batch, *, with_why: bool = False, desc=None,
                    graph: torch.Tensor | None = None):
        """Per-card teacher heads, preserving distinct questions sharing evidence."""
        desc = desc if desc is not None else self.describe()
        by = {}
        for j in batch.judged:
            by.setdefault(j.key, []).append(j)
        result = {}
        for key, js in by.items():
            pages = [self.card_page(batch, j, with_why=with_why) for j in js]
            if any(self.lm.token_count(t) > self.lm.max_len for t in pages):
                raise ValueError("Card source/WHY exceeds teacher context budget; do not truncate")
            ids, mask = self.lm.tokenize(pages)
            prefix = None
            if graph is not None and self.graph is not None:
                pooled = torch.stack([
                    graph[list(j.lines)].mean(0) if j.lines else graph.mean(0) for j in js])
                prefix = self.graph(pooled).view(len(js), self.n_prefix, self.lm.dim)
            h, _ = self._lm_rows(ids, mask, prefix, logits=False)
            reader = self.reader_universal if self.schema.by_key()[key].universal else self.reader_company
            logits, _, _ = reader(h, mask, desc[key][0].unsqueeze(0), [desc[key][1]])
            result[key] = logits[0]
        return result

    # ------------------------------------------------------------ flips
    def flip_read(self, batch: Batch, why: list[str | None], flips: list,
                  notes: list[list[str]] | None = None, desc=None,
                  graph: torch.Tensor | None = None) -> BrainOutput:
        """The supposition pass: for each (line, sentence), the reasoned page
        with the sentence assumed. Row n of every output is supposition n.
        Nothing tells the teacher whether the sentence changes the case: it
        reads it and applies the rule on the page, and that reading is what
        it teaches the heads (supercharge.teach_flip). It practices on
        look-alike lines, where the answer is known (losses.supposed_twins)."""
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
                       graph: torch.Tensor | None = None,
                       rows: list[int] | None = None) -> torch.Tensor:
        """Next-token loss on each human WHY, given the page: the LM learns to
        write the reasoning an expert wrote for this line. ``rows`` limits it
        to those lines."""
        idx = [i for i in (range(len(batch)) if rows is None else rows) if batch.why[i]]
        if not idx:
            return next(self.parameters()).new_zeros(())
        notes = notes or [[] for _ in range(len(batch))]
        heads = [self.page(batch, i, notes[i]) + "Why:" for i in idx]
        tails = [" " + batch.why[i] for i in idx]
        if any(self.lm.token_count(h + t) > self.lm.max_len for h, t in zip(heads, tails)):
            raise ValueError("Expert WHY exceeds teacher context budget; increase the budget "
                             "or prepare source context explicitly, never truncate the WHY")
        ids, mask, start = self.lm.tokenize_pairs(heads, tails)
        prefix = None
        if self.graph is not None and graph is not None:
            prefix = self.graph(graph[idx]).view(len(idx), self.n_prefix, self.lm.dim)
        _, logits = self._lm_rows(ids, mask, prefix)                  # start: first WHY token
        pos = torch.arange(ids.shape[1], device=ids.device).view(1, -1)
        on_why = (pos >= start.view(-1, 1)) & mask & (pos > 0)
        tgt = ids.masked_fill(~on_why, IGNORE)[:, 1:]
        token_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                     tgt.reshape(-1), ignore_index=IGNORE, reduction="none")
        token_loss = token_loss.view_as(tgt)
        valid = tgt != IGNORE
        if not valid.any(1).all():
            raise ValueError("Expert WHY has no supervised tokens")
        row_loss = token_loss.sum(1) / valid.sum(1).clamp(min=1)
        weights = row_loss.new_tensor([batch.why_weights[i] if batch.why_weights else 1.0 for i in idx])
        return (row_loss * weights).sum() / weights.sum().clamp(min=1e-6)

    def value_loss(self, batch: Batch, rows: list[int] | None = None,
                   deal_level: bool = True) -> torch.Tensor:
        """Next-token loss on each exact saved value string. Nothing is clipped.
        ``rows`` limits the line values to those lines; ``deal_level`` keeps
        the deal's own values (line -1)."""
        keep = None if rows is None else set(rows)
        rows = [(item.line, str(item.text).strip()) for item in batch.literals if str(item.text or "").strip()
                and ((item.line < 0 and deal_level) or (item.line >= 0 and (keep is None or item.line in keep)))]
        if not rows:
            return next(self.parameters()).new_zeros(())
        heads = ["Deal\nValue:" if i < 0 else batch.texts[i] + "\nValue:" for i, _ in rows]
        tails = [" " + text for _, text in rows]
        if any(self.lm.token_count(h + t) > self.lm.max_len for h, t in zip(heads, tails)):
            raise ValueError("Saved value exceeds teacher context budget; do not truncate it")
        ids, mask, start = self.lm.tokenize_pairs(heads, tails)
        _, logits = self._lm_rows(ids, mask)
        pos = torch.arange(ids.shape[1], device=ids.device).view(1, -1)
        on_value = (pos >= start.view(-1, 1)) & mask & (pos > 0)
        tgt = ids.masked_fill(~on_value, IGNORE)[:, 1:]
        token_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                     tgt.reshape(-1), ignore_index=IGNORE, reduction="none").view_as(tgt)
        valid = tgt != IGNORE
        lost = int((~valid.any(1)).sum())
        if lost:
            raise ValueError(f"{lost} of {len(rows)} saved values have no supervised tokens")
        return token_loss[valid].mean()

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
    def write(self, prompt: str, max_new: int = 120, stop: str | None = None) -> str:
        """Greedy-continue ``prompt``; cut at ``stop`` when given."""
        ids, mask = self.lm.tokenize([prompt])
        start = int(mask.sum())
        ids = ids[:, :start]
        for _ in range(max_new):
            _, logits = self.lm(ids, torch.ones_like(ids, dtype=torch.bool))
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if ids.shape[1] >= self.lm.max_len:
                break
        if hasattr(self.lm, "tokenizer"):
            text = self.lm.tokenizer.decode(ids[0, start:].tolist(), skip_special_tokens=True)
        else:
            text = bytes(t for t in ids[0, start:].tolist() if t < 256).decode("utf-8", "replace")
        return text.split(stop)[0] if stop else text

    def explain(self, batch: Batch, i: int, max_new: int = 120,
                notes: list[str] | None = None) -> str:
        """Greedy-write the model's own WHY for line i. Untrained: noise."""
        return self.write(self.page(batch, i, notes) + "Why: ", max_new)

    def rewrite(self, batch: Batch, i: int, why: str, change: str | None = None,
                max_new: int = 80) -> str:
        """Line i written again by the teacher, with its WHY on the page.
        Without ``change``: other words, the same reason (what the answer
        depends on stays). With ``change`` (a sentence of the WHY): the
        smallest edit that makes the sentence hold. The teacher's reading of
        the result, not this prompt, decides what the new line answers."""
        ask = (f"Rewrite the line so that this holds: {change}" if change else
               "Rewrite the line in other words, keeping everything the reason depends on.")
        return self.write(self.page(batch, i) + f"Why: {why}\n{ask}\nLine: ", max_new, "\n").strip()


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


