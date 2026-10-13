"""Text encoders. One encoder reads the lines, the WHY notes and the schema's
descriptions, so all three land in the same space from the start.

``HashingEncoder`` needs no download and no tokenizer: hashed words and
character trigrams into a trainable embedding bag. It is what the tests and
the shape checks run on. ``HFEncoder`` wraps a pretrained encoder (the design
names ModernBERT-large) and is only imported when asked for.
"""
from __future__ import annotations

import re
import zlib

import torch
from torch import nn

_WORD = re.compile(r"[a-z0-9$\"]+|\[mask\]")


def word_spans(text: str) -> list[tuple[str, int, int]]:
    """The words of a line as every encoder here sees them, with character
    spans: what a per-word weight (supercharge.py) is defined over."""
    t = str(text or "")
    return [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(t.lower())]


def _hash(s: str, buckets: int) -> int:
    return zlib.crc32(s.encode("utf-8")) % buckets


class HashingEncoder(nn.Module):
    def __init__(self, dim: int = 256, buckets: int = 1 << 16):
        super().__init__()
        self.dim, self.buckets = dim, buckets
        self.bag = nn.EmbeddingBag(buckets, dim, mode="mean")
        nn.init.normal_(self.bag.weight, std=dim ** -0.5)

    def _ids(self, text: str) -> list[int]:
        words = _WORD.findall(str(text or "").lower())
        grams = [w[i:i + 3] for w in words for i in range(max(1, len(w) - 2))]
        toks = [f"w:{w}" for w in words] + [f"g:{g}" for g in grams]
        toks += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        return [_hash(t, self.buckets) for t in toks] or [_hash("<empty>", self.buckets)]

    def forward(self, texts: list[str]) -> torch.Tensor:
        ids, offsets = [], []
        for t in texts:
            offsets.append(len(ids))
            ids.extend(self._ids(t))
        dev = self.bag.weight.device
        return self.bag(torch.tensor(ids, device=dev), torch.tensor(offsets, device=dev))

    def tokens(self, texts: list[str], max_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-word vectors [B, T, d] and a mask [B, T]: every word kept, so a
        reader can attend to the exact words of a paragraph (operators.py)."""
        words = [_WORD.findall(str(t or "").lower()) or ["<empty>"] for t in texts]
        if any(len(w) > max_len for w in words):
            raise ValueError("Encoder paragraph exceeds token budget; do not truncate rich supervision")
        T = max(len(w) for w in words)
        ids, offsets = [], []
        for ws in words:
            for i in range(T):
                offsets.append(len(ids))
                if i < len(ws):
                    w = ws[i]
                    grams = [w[j:j + 3] for j in range(max(1, len(w) - 2))]
                    ids.extend(_hash(t, self.buckets) for t in [f"w:{w}"] + [f"g:{g}" for g in grams])
                else:
                    ids.append(0)
        dev = self.bag.weight.device
        vec = self.bag(torch.tensor(ids, device=dev), torch.tensor(offsets, device=dev))
        mask = torch.tensor([[i < len(ws) for i in range(T)] for ws in words], device=dev)
        return vec.view(len(texts), T, -1), mask


class HFEncoder(nn.Module):
    """A pretrained encoder, mean-pooled. ``pip install transformers`` first.

    Lines are encoded in length-sorted chunks of at most ``chunk_tokens``
    padded tokens, so a 4,096-line window never becomes one padded batch.
    While the body is frozen (no parameter requires grad) a line's vector
    never changes, so it is computed once and kept (``cache``); the deal
    graph and every head above it still train.
    """

    def __init__(self, name: str = "answerdotai/ModernBERT-large", max_length: int = 256,
                 dtype: torch.dtype | None = None, chunk_tokens: int = 32768):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # optional dependency

        self.tok = AutoTokenizer.from_pretrained(name)
        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self.model = AutoModel.from_pretrained(name, **kwargs)
        self.dim = self.model.config.hidden_size
        self.max_length = max_length
        self.chunk_tokens = chunk_tokens
        self.cache: dict[str, torch.Tensor] = {}

    def frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.parameters())

    def _ids(self, texts: list[str], budget: int) -> list[list[int]]:
        ids = self.tok([str(t) for t in texts], truncation=False)["input_ids"]
        if any(len(x) > budget for x in ids):
            raise ValueError("Encoder paragraph exceeds token budget; do not truncate rich supervision")
        return ids

    def _encode(self, ids: list[list[int]]) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out: list[torch.Tensor | None] = [None] * len(ids)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and len(ids[order[end]]) * (end + 1 - start) <= self.chunk_tokens:
                end += 1
            rows = order[start:end]
            enc = self.tok.pad({"input_ids": [ids[i] for i in rows]}, return_tensors="pt").to(dev)
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            for k, i in enumerate(rows):
                out[i] = pooled[k]
            start = end
        return torch.stack(out)  # type: ignore[arg-type]

    def forward(self, texts: list[str]) -> torch.Tensor:
        texts = [str(t) for t in texts]
        if not self.frozen():
            return self._encode(self._ids(texts, self.max_length))
        todo = sorted({t for t in texts if t not in self.cache})
        if todo:
            with torch.no_grad():
                vecs = self._encode(self._ids(todo, self.max_length))
            for t, v in zip(todo, vecs):
                self.cache[t] = v
        return torch.stack([self.cache[t] for t in texts])

    def fixed(self, texts: list[str]) -> torch.Tensor:
        """Encode texts that are not the deal's lines (WHYs, notes, queries,
        words, flip sentences) without a gradient into the encoder. Only the
        lines teach the released blocks; the rest would hold every one of
        those texts' activations for backward."""
        texts = [str(t) for t in texts]
        if self.frozen():
            return self(texts)
        with torch.no_grad():
            return self._encode(self._ids(texts, self.max_length))

    def _check_budget(self, texts: list[str], budget: int) -> None:
        self._ids(texts, budget)

    def tokens(self, texts: list[str], max_len: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        dev = next(self.model.parameters()).device
        self._check_budget(texts, max_len)
        enc = self.tok(list(texts), padding=True, truncation=False, return_tensors="pt").to(dev)
        return self.model(**enc).last_hidden_state, enc["attention_mask"].bool()


class TagEmbedding(nn.Module):
    """Small hashed vocabularies: doc kind, speaker role and side, number tokens."""

    def __init__(self, dim: int, buckets: int = 4096):
        super().__init__()
        self.buckets = buckets
        self.bag = nn.EmbeddingBag(buckets, dim, mode="sum")
        nn.init.normal_(self.bag.weight, std=0.02)

    def forward(self, groups: list[list[str]]) -> torch.Tensor:
        ids, offsets = [], []
        for g in groups:
            offsets.append(len(ids))
            ids.extend(_hash(t, self.buckets) for t in (g or ["<none>"]))
        dev = self.bag.weight.device
        return self.bag(torch.tensor(ids, device=dev), torch.tensor(offsets, device=dev))
