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

import math
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

    def __init__(self, base: nn.Linear, scale: float = 1.0, rank: int = 0):
        super().__init__()
        self.base = base
        self.scale = scale
        self.delta: tuple[torch.Tensor, torch.Tensor] | None = None
        # Persistent low rank, zero at the start so the layer still matches
        # ``base`` until training moves it. Rank 0 leaves the layer unchanged.
        if rank > 0:
            self.A = nn.Parameter(torch.empty(rank, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.A is not None and self.B is not None:
            # The base may be bfloat16 on the H100 while these slots stay float,
            # so the addition does not crash on the dtype.
            a = self.A.to(dtype=x.dtype)
            b = self.B.to(dtype=x.dtype)
            y = y + self.scale * (x @ a.T) @ b.T
        if self.delta is None:
            return y
        a, b = self.delta                        # A [r, in], B [out, r]
        return y + self.scale * (x @ a.to(dtype=x.dtype).T) @ b.to(dtype=x.dtype).T


class LMBase(nn.Module):
    dim: int
    max_len: int
    #: Descriptions embedded per language-model pass.
    EMBED_CHUNK = 32
    EMBED_TOKENS = 32768      # padded tokens per embedding call

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

    def token_count(self, text: str) -> int:
        """Untruncated count needed to certify complete expert supervision."""
        raise NotImplementedError

    def encode(self, text: str, first: bool) -> list[int]:
        """Token ids of ``text``; ``first`` adds the start token, if any."""
        raise NotImplementedError

    @property
    def pad_id(self) -> int:
        raise NotImplementedError

    def tokenize_pairs(self, heads: list[str], tails: list[str]
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each head then its tail, encoded apart and joined, right-padded.

        Returns ids, mask and each row's first tail position. Encoding the
        tail on its own keeps every one of its tokens out of the head, even
        where a joint encoding would merge the two at the seam.
        """
        seqs, start = [], []
        for h, t in zip(heads, tails):
            a, b = self.encode(h, True), self.encode(t, False)
            if not b:
                raise ValueError("Supervised text encodes to no tokens")
            if len(a) + len(b) > self.max_len:
                raise ValueError("Teacher text exceeds context budget; explicit context preparation is required")
            seqs.append(a + b)
            start.append(len(a))
        width = max(len(q) for q in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.bool)
        for i, q in enumerate(seqs):
            ids[i, :len(q)] = torch.tensor(q)
            mask[i, :len(q)] = True
        dev = next(self.parameters()).device
        return ids.to(dev), mask.to(dev), torch.tensor(start, device=dev)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor,
                prefix: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def hidden(self, ids: torch.Tensor, mask: torch.Tensor,
               prefix: torch.Tensor | None = None,
               inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        """Last hidden states only. A causal LM's vocabulary logits are
        [pages, tokens, ~150k]; the heads never read them."""
        return self(ids, mask, prefix=prefix, inputs_embeds=inputs_embeds)[0]

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Vocabulary logits for the hidden states given, and only those."""
        raise NotImplementedError

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of each token of ``text`` as ``tokenize`` cuts it
        ((0, 0) for special tokens), so a token's weight can be traced back to
        the words and lines of the page."""
        raise NotImplementedError

    def embed(self, texts: list[str]) -> torch.Tensor:
        """Mean-pooled hidden states [B, dim], in length-sorted chunks so the
        schema's ~760 descriptions are never one padded batch."""
        size = [self.token_count(t) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: size[i])
        out: list[torch.Tensor | None] = [None] * len(texts)
        calls, cur = [], []
        for i in order:
            if cur and (len(cur) >= self.EMBED_CHUNK or (len(cur) + 1) * size[i] > self.EMBED_TOKENS):
                calls.append(cur)
                cur = []
            cur.append(i)
        if cur:
            calls.append(cur)
        for rows in calls:
            ids, mask = self.tokenize([texts[i] for i in rows])
            h = self.hidden(ids, mask)
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            for k, i in enumerate(rows):
                out[i] = pooled[k]
        return torch.stack(out)  # type: ignore[arg-type]


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

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(h)

    def token_count(self, text: str) -> int:
        return 1 + len(str(text).encode("utf-8"))

    def encode(self, text: str, first: bool) -> list[int]:
        return ([self.BOS] if first else []) + list(str(text).encode("utf-8"))

    @property
    def pad_id(self) -> int:
        return self.PAD

    def tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if any(self.token_count(t) > self.max_len for t in texts):
            raise ValueError("Teacher text exceeds context budget; explicit context preparation is required")
        seqs = [[self.BOS] + list(str(t).encode("utf-8")) for t in texts]
        t = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), t), self.PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        dev = self.tok.weight.device
        return ids.to(dev), (ids != self.PAD).to(dev)

    def input_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self.tok(ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        out = [(0, 0)]                                # BOS
        for ci, ch in enumerate(str(text)):
            out += [(ci, ci + 1)] * len(ch.encode("utf-8"))
        return out[: self.max_len]

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
                 max_len: int = 1024, trainable_rank: int = 0,
                 dtype: torch.dtype | None = None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer  # optional dependency

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Pages are split and trimmed by their real length; that needs the pad on the right.
        self.tokenizer.padding_side = "right"
        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.dim, self.max_len = self.model.config.hidden_size, max_len
        for parent in list(self.model.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name in targets and isinstance(child, nn.Linear):
                    setattr(parent, child_name, LoRALinear(child, rank=trainable_rank))

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(str(text), truncation=False)["input_ids"])

    def encode(self, text: str, first: bool) -> list[int]:
        return self.tokenizer(str(text), add_special_tokens=first, truncation=False)["input_ids"]

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if any(self.token_count(t) > self.max_len for t in texts):
            raise ValueError("Teacher text exceeds context budget; explicit context preparation is required")
        enc = self.tokenizer(list(texts), padding=True, truncation=False, return_tensors="pt")
        dev = next(self.model.parameters()).device
        return enc["input_ids"].to(dev), enc["attention_mask"].bool().to(dev)

    def input_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                             return_offsets_mapping=True)
        return [tuple(o) for o in enc["offset_mapping"]]

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

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.model.get_output_embeddings()(h)

    def hidden(self, ids: torch.Tensor, mask: torch.Tensor,
               prefix: torch.Tensor | None = None,
               inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        body = getattr(self.model, "model", None)
        if body is None:                          # not a *ForCausalLM wrapper
            return self(ids, mask, prefix=prefix, inputs_embeds=inputs_embeds)[0]
        x = self.input_embeddings(ids) if inputs_embeds is None else inputs_embeds
        if prefix is not None:
            x = torch.cat([prefix.to(x.dtype), x], 1)
            mask = torch.cat([torch.ones(prefix.shape[:2], dtype=torch.bool,
                                         device=mask.device), mask], 1)
        def read_body(embeddings, attention):
            return body(inputs_embeds=embeddings, attention_mask=attention).last_hidden_state

        # Optional bind-only recomputation: per-layer checkpoint inputs from
        # many pages otherwise coexist until the piece loss runs backward.
        # Preserve the complete page and graph-prefix derivative. Trainable
        # LM adapters retain their existing execution path.
        recompute = (getattr(self, "checkpoint_frozen_hidden", False)
                     and torch.is_grad_enabled() and x.requires_grad
                     and not any(p.requires_grad for p in self.parameters()))
        if recompute:
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(read_body, x, mask.long(), use_reentrant=False)
        else:
            h = read_body(x, mask.long())
        return h[:, prefix.shape[1]:] if prefix is not None else h
