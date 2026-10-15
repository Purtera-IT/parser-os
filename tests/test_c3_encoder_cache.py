"""HFEncoder keeps line vectors for a read, so a repeated line or a deal
parsed again is encoded once, and drops them when training resumes."""
from __future__ import annotations

import torch
from torch import nn

from ml.c3.text import HFEncoder


def _encoder(trainable: bool):
    enc = HFEncoder.__new__(HFEncoder)
    nn.Module.__init__(enc)
    enc.model = nn.Linear(2, 2)
    for p in enc.model.parameters():
        p.requires_grad_(trainable)
    enc.cache = {}
    enc.calls = []

    def _encode(ids):
        enc.calls.append(len(ids))
        return torch.stack([torch.full((2,), float(len(x))) for x in ids])

    enc._encode = _encode
    enc._all_ids = lambda texts: [[0] * len(t) for t in texts]
    return enc


def test_a_read_encodes_each_distinct_line_once_across_calls():
    enc = _encoder(trainable=True).eval()
    with torch.no_grad():
        a = enc(["Best regards, the site team", "Mount four displays", "Best regards, the site team"])
        b = enc(["Mount four displays", "Best regards, the site team"])
    assert enc.calls == [2]
    assert torch.equal(a[0], a[2]) and torch.equal(b[0], a[1])


def test_training_never_reads_from_the_cache_and_clears_it():
    enc = _encoder(trainable=True).eval()
    with torch.no_grad():
        enc(["Mount four displays"])
    enc.train()
    assert enc.cache == {}
    enc(["Mount four displays", "Mount four displays"])
    assert enc.calls == [1, 2]


def test_a_frozen_body_keeps_its_cache_through_training():
    enc = _encoder(trainable=False)
    enc(["Mount four displays"])
    enc.train()
    enc(["Mount four displays"])
    assert enc.calls == [1]


def test_the_cache_starts_over_past_its_cap():
    enc = _encoder(trainable=True).eval()
    enc.CACHE_MAX = 2
    with torch.no_grad():
        enc(["a", "bb"])
        enc(["ccc"])
    assert enc.calls == [2, 1] and set(enc.cache) == {"ccc"}
