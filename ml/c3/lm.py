"""The language model inside the training-only teacher (architecture v6).

A person who is told "the 65-inch panels are a two-person lift, so every
mount needs two techs and the hours double, unless the customer's staff do
the lifting" understands it because they already know the language: what a
lift is, what "unless" does, that doubling techs doubles labor. A pretrained
language model carries that knowledge, learned from trillions of words
before it sees a deal. v6 uses one only as the **teacher** (brain.py) that
reads our WHY paragraphs in training; the task heads learn from it
(supercharge.py) and run without it. Nothing on the teacher's path splits
sentences with keywords or applies if/and rules: the paragraph goes in as
words.

This file holds the two interchangeable LMs and the adapter slots that let a
paragraph be compiled into weights (consolidate.py):

* ``TinyCausalLM``: a small byte-level transformer in plain torch. It has no
  pretrained knowledge. It exists so the tests and shape checks run on CPU
  with nothing downloaded. Its outputs mean nothing until trained.
* ``HFCausalLM``: any pretrained causal LM from ``transformers`` (the v6 doc
  names candidates). This is the real teacher. Optional import, as with
  ``text.HFEncoder``.

Both expose the same interface: ``tokenize``, ``forward(ids, mask, prefix)
-> (hidden, logits)``, ``embed(texts)`` (mean-pooled hidden states, used for
the head and answer descriptions so they live in the LM's own space), and
``adapters()``, the ``LoRALinear`` layers a hypernetwork can write into.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A linear layer with a writable low-rank slot.

    ``base(x) + scale * x Aᵀ Bᵀ`` where (A, B) are set per call by
    ``LMBase.adapted`` (consolidate.py writes them from paragraphs). With no
    adapter set, the output is exactly ``base(x)``.
    """

    def __init__(self, base: nn.Linear, scale: float = 1.0):
        super().__init__()
        self.base = base
        self.scale = scale
        self.delta: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.delta is None:
            return y
        a, b = self.delta                        # A [r, in], B [out, r]
        return y + self.scale * (x @ a.T) @ b.T


class LMBase(nn.Module):
    dim: int
    max_len: int

    def adapters(self) -> list[LoRALinear]:
        return [m for m in self.modules() if isinstance(m, LoRALinear)]

    @contextmanager
    def adapted(self, deltas: list[tuple[torch.Tensor, torch.Tensor]] | None) -> Iterator[None]:
        """Run with one (A, B) per adapter slot written in; restored after."""
        slots = self.adapters()
        if deltas is not None and len(deltas) != len(slots):
            raise ValueError(f"{len(deltas)} deltas for {len(slots)} adapter slots")
        try:
            for s, d in zip(slots, deltas or [None] * len(slots)):
                s.delta = d
            yield
        finally:
            for s in slots:
                s.delta = None

    def tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, ids: torch.Tensor, mask: torch.Tensor,
                prefix: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> torch.Tensor:
        """Mean-pooled hidden states [B, dim]."""
        ids, mask = self.tokenize(texts)
        h, _ = self(ids, mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)


class _Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q = LoRALinear(nn.Linear(d, d))
        self.k = nn.Linear(d, d)
        self.v = LoRALinear(nn.Linear(d, d))
        self.o = nn.Linear(d, d)
        self.up = nn.Linear(d, 4 * d)
        self.down = LoRALinear(nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        y = self.ln1(x)
        q, k, v = (f(y).view(b, t, self.h, d // self.h).transpose(1, 2)
                   for f in (self.q, self.k, self.v))
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        allowed = causal.view(1, 1, t, t) & pad.view(b, 1, 1, t)
        allowed = allowed | torch.eye(t, dtype=torch.bool, device=x.device).view(1, 1, t, t)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class TinyCausalLM(LMBase):
    """Byte-level causal transformer, plain torch, untrained. A stand-in."""

    PAD, BOS = 256, 257

    def __init__(self, dim: int = 64, layers: int = 2, heads: int = 4, max_len: int = 512):
        super().__init__()
        self.dim, self.max_len = dim, max_len
        self.tok = nn.Embedding(258, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 258, bias=False)

    def tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [[self.BOS] + list(str(t).encode("utf-8"))[: self.max_len - 1] for t in texts]
        t = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), t), self.PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        dev = self.tok.weight.device
        return ids.to(dev), (ids != self.PAD).to(dev)

    def input_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self.tok(ids)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor,
                prefix: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tok(ids) if inputs_embeds is None else inputs_embeds
        if prefix is not None:                    # soft tokens from the deal graph
            x = torch.cat([prefix, x], 1)
            mask = torch.cat([torch.ones(prefix.shape[:2], dtype=torch.bool,
                                         device=mask.device), mask], 1)
        t = x.shape[1]
        x = x + self.pos(torch.arange(t, device=x.device).clamp(max=self.max_len - 1))
        for blk in self.blocks:
            x = blk(x, mask)
        h = self.ln(x)
        if prefix is not None:
            h = h[:, prefix.shape[1]:]
        return h, self.out(h)


class HFCausalLM(LMBase):
    """A pretrained causal LM. ``pip install transformers`` first.

    ``targets`` names the linear layers that get adapter slots (attention
    query/value projections by default, the usual LoRA choice).
    """

    def __init__(self, name: str, targets: tuple[str, ...] = ("q_proj", "v_proj"),
                 max_len: int = 1024):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer  # optional dependency

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.dim, self.max_len = self.model.config.hidden_size, max_len
        for parent in list(self.model.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name in targets and isinstance(child, nn.Linear):
                    setattr(parent, child_name, LoRALinear(child))

    def tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        dev = next(self.model.parameters()).device
        return enc["input_ids"].to(dev), enc["attention_mask"].bool().to(dev)

    def input_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor,
                prefix: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_embeddings(ids) if inputs_embeds is None else inputs_embeds
        if prefix is not None:
            x = torch.cat([prefix.to(x.dtype), x], 1)
            mask = torch.cat([torch.ones(prefix.shape[:2], dtype=torch.bool,
                                         device=mask.device), mask], 1)
        out = self.model(inputs_embeds=x, attention_mask=mask.long(), output_hidden_states=True)
        h, logits = out.hidden_states[-1], out.logits
        if prefix is not None:
            h, logits = h[:, prefix.shape[1]:], logits[:, prefix.shape[1]:]
        return h, logits
