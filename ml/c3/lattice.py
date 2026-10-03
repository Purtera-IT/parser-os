"""Asked attention: every question puts its own query into the trunk.

In the plain trunk (model.DealEncoder) each line is read once, the same way
for every question, and the heads read the finished vector. Here the
questions take part in the reading. Each line carries one slot per question
space (content, consequence, sites, money...), seeded from the space's own
written description, and each slot looks back over the deal for what *its*
question needs::

  line i, after the deal graph:  h_i
        │
        ├─ slot s = seed(description of space s) + h_i        one per space
        │
        └─ asked layer × L
             1. slots of the line talk to each other          sites inform scope
                (one-way valve: base slots read base slots
                only; the company slot reads all of them)
             2. each slot attends back over the deal,         where it looks depends
                foresight-masked (j entered no later than i),  on what it is asked;
                with same-document / same-section biases      the attention is its
                                                              citation
             3. MLP
        │
        └─ slots[i, s]  → the heads of space s (with r_i)

Why it matters here:

* **Asked, not just read.** A sites question and a price question about the
  same line look at different earlier lines. A new head written as a
  description gets a slot seeded from that description: it asks its own
  question without new layers.
* **The firewall is the wiring.** Base slots are computed from base slots
  and the deal only; the company slot is computed after them and nothing
  reads it back. Base outputs are bit-identical whatever the company
  (tests/test_c3_lattice.py), by construction rather than by a penalty.
* **Citations come out of the reading.** Each slot's attention over earlier
  lines is a distribution over evidence (``citations``), the same shape the
  teacher's pointers teach.

Cost: per line, S slots attend over at most N earlier lines, so S·N² per
layer (11 spaces, 2,000 lines: about 44M scores per head and layer, done in
chunks of ``chunk`` lines). Nothing here runs a language model.
"""
from __future__ import annotations

import torch
from torch import nn


class _Attend(nn.Module):
    """Multi-head attention from queries [B, Q, d] to keys [B, K, d]."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, bias: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        b, nq, d = x.shape
        nk = mem.shape[1]
        q = self.q(x).view(b, nq, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(mem).view(b, nk, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        s = q @ k.transpose(-1, -2) / (d // self.h) ** 0.5
        if bias is not None:
            s = s + bias
        a = s.softmax(-1)
        y = (a @ v).transpose(1, 2).reshape(b, nq, d)
        return self.o(y), a


class AskedLayer(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.ln_s, self.ln_c, self.ln_m = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.slot_attn = _Attend(d, heads)
        self.deal_attn = _Attend(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.w_doc = nn.Parameter(torch.zeros(heads))
        self.w_sec = nn.Parameter(torch.zeros(heads))

    def talk(self, base: torch.Tensor, company: torch.Tensor | None
             ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Slots of one line talk. The valve: base slots attend only to base
        slots (computed alone, so the company slot cannot change a bit of
        them); the company slot attends to everything."""
        nb = self.ln_s(base)
        y, _ = self.slot_attn(nb, nb)
        base2 = base + self.drop(y)
        if company is None:
            return base2, None
        nc = self.ln_s(company)
        y, _ = self.slot_attn(nc, torch.cat([nb, nc], 1))
        return base2, company + self.drop(y)

    def look(self, slots: torch.Tensor, h: torch.Tensor, allowed: torch.Tensor,
             same_doc: torch.Tensor, same_sec: torch.Tensor, chunk: int
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each slot of line i attends over lines j with allowed[i, j]."""
        n, s, d = slots.shape
        mem = self.ln_m(h)
        outs, cites = [], []
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            q = self.ln_c(slots[a:b]).reshape(1, (b - a) * s, d)
            bias = (self.w_doc.view(-1, 1, 1) * same_doc[a:b] + self.w_sec.view(-1, 1, 1) * same_sec[a:b])
            bias = bias.masked_fill(~allowed[a:b], float("-inf"))           # [H, b-a, N]
            bias = bias.repeat_interleave(s, 1).unsqueeze(0)                 # [1, H, (b-a)*S, N]
            y, att = self.deal_attn(q, mem.unsqueeze(0), bias)
            outs.append(y.view(b - a, s, d))
            cites.append(att[0].mean(0).view(b - a, s, n))
        return slots + self.drop(torch.cat(outs)), torch.cat(cites)

    def forward(self, base, company, h, allowed, same_doc, same_sec, chunk):
        base, company = self.talk(base, company)
        k = base.shape[1]
        both = base if company is None else torch.cat([base, company], 1)
        both, cites = self.look(both, h, allowed, same_doc, same_sec, chunk)
        both = both + self.drop(self.mlp(self.ln_s(both)))
        return both[:, :k], (both[:, k:] if company is not None else None), cites[:, :k]


class AskedLattice(nn.Module):
    """The asked layers over the deal graph's output. ``spaces`` fixes the
    slot order; a space's seed is its written description, encoded by the
    same text encoder the heads use."""

    def __init__(self, d: int, d_text: int, spaces: list[str], n_layers: int, heads: int,
                 dropout: float, chunk: int = 256):
        super().__init__()
        self.spaces, self.chunk = list(spaces), chunk
        self.seed = nn.Linear(d_text, d)
        self.company_seed = nn.Linear(d_text, d)
        self.layers = nn.ModuleList(AskedLayer(d, heads, dropout) for _ in range(n_layers))

    def forward(self, h: torch.Tensor, space_desc: torch.Tensor, allowed: torch.Tensor,
                same_doc: torch.Tensor, same_sec: torch.Tensor,
                company_desc: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """h [N, d]; space_desc [S, d_text]; company_desc [d_text] or None.
        Returns base slots [N, S, d], the company slot [N, 1, d] or None, and
        the base slots' citations [N, S, N] from the last layer."""
        base = h.unsqueeze(1) + self.seed(space_desc).unsqueeze(0)
        company = (h.unsqueeze(1) + self.company_seed(company_desc).view(1, 1, -1)
                   if company_desc is not None else None)
        cites = h.new_zeros(h.shape[0], len(self.spaces), h.shape[0])
        for layer in self.layers:
            base, company, cites = layer(base, company, h, allowed, same_doc, same_sec, self.chunk)
        return base, company, cites


def space_descriptions(schema, desc: dict) -> dict[str, torch.Tensor]:
    """A space's description: the mean of its universal questions' encoded
    descriptions (a new question changes its space's seed, nothing else)."""
    out: dict[str, list[torch.Tensor]] = {}
    for o in schema.opportunities:
        if o.universal and o.key in desc:
            out.setdefault(o.space, []).append(desc[o.key][0])
    return {s: torch.stack(v).mean(0) for s, v in out.items()}


def company_description(schema, desc: dict) -> torch.Tensor | None:
    vs = [desc[o.key][0] for o in schema.select(layer="company") if o.key in desc]
    return torch.stack(vs).mean(0) if vs else None


__all__ = ["AskedLattice", "AskedLayer", "company_description", "space_descriptions"]
