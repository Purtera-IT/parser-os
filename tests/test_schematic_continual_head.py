"""Continual symbol-type head: gets better/holds each run (eval-gated), retrains
on cached features (fast), caches re-ingests, persists champion. Synthetic
deal-distinct glyphs so leave-one-deal-out actually exercises."""
from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("PIL.Image")

from app.core.schematic_continual_head import ContinualSymbolHead


def _g(k, seed=0):
    from PIL import Image, ImageDraw

    rng = np.random.RandomState(seed)
    im = Image.new("RGB", (64, 64), "white")
    d = ImageDraw.Draw(im)
    o = int(rng.uniform(-5, 5)); r = int(22 * rng.uniform(0.8, 1.15)); w = int(rng.uniform(2, 4))
    if k == "circle":
        d.ellipse((32 - r + o, 32 - r + o, 32 + r + o, 32 + r + o), outline="black", width=w)
    elif k == "tri":
        d.polygon([(32 + o, 32 - r), (32 - r + o, 32 + r), (32 + r + o, 32 + r)], outline="black", width=w)
    elif k == "sq":
        d.rectangle((32 - r + o, 32 - r + o, 32 + r + o, 32 + r + o), outline="black", width=w)
    elif k == "dia":
        d.polygon([(32, 32 - r), (32 + r, 32), (32, 32 + r), (32 - r, 32)], outline="black", width=w)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


_SD = [1000]


def _crops(classes):
    out = []
    for k in classes:
        for _ in range(8):
            _SD[0] += 1
            out.append((_g(k, _SD[0]), k))
    return out


def test_continual_improves_then_holds_and_classifies():
    h = ContinualSymbolHead(":memory:")
    for d in ["D1", "D2", "D3"]:
        h.ingest(_crops(["circle", "tri", "sq", "dia"]), deal_id=d)
    r1 = h.retrain()
    assert r1.promoted and r1.new_f1 >= 0.7
    for d in ["D4", "D5", "D6"]:
        h.ingest(_crops(["circle", "tri", "sq", "dia"]), deal_id=d)
    r2 = h.retrain()
    # more data -> promoted (>=) or held; never a silent regression
    assert r2.promoted or r2.reason == "no_improvement"
    correct = sum(1 for k in ["circle", "tri", "sq", "dia"] if (p := h.classify(_g(k, 999))) and p[0] == k)
    assert correct >= 3


def test_reingest_is_cached_no_duplicate_rows():
    h = ContinualSymbolHead(":memory:")
    batch = _crops(["circle", "tri"])
    h.ingest(batch, deal_id="D1")
    n_after_first = h.n_rows()
    added = h.ingest(batch, deal_id="D1")  # identical crops again
    assert added == 0
    assert h.n_rows() == n_after_first


def test_untrained_head_abstains():
    h = ContinualSymbolHead(":memory:")
    assert h.classify(_g("circle", 1)) is None  # no champion yet


def test_champion_persists_to_disk(tmp_path):
    db = str(tmp_path / "cont.db")
    h = ContinualSymbolHead(db)
    for d in ["D1", "D2", "D3"]:
        h.ingest(_crops(["circle", "tri", "sq"]), deal_id=d)
    assert h.retrain().promoted
    # new instance on the same db must load the champion
    h2 = ContinualSymbolHead(db)
    assert h2.champion is not None
    assert h2.classify(_g("circle", 777)) is not None
