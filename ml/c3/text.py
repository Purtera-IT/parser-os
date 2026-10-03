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


class HFEncoder(nn.Module):
    """A pretrained encoder, mean-pooled. ``pip install transformers`` first."""

    def __init__(self, name: str = "answerdotai/ModernBERT-large", max_length: int = 256):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # optional dependency

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.dim = self.model.config.hidden_size
        self.max_length = max_length

    def forward(self, texts: list[str]) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        enc = self.tok(list(texts), padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt").to(dev)
        out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1)


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
