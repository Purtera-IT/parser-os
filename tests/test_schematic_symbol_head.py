"""Tests for the neural symbol-classifier head (distillation target for the
schematic symbol detector). Synthetic glyphs stand in for real legend symbols so
the test is deterministic and needs no PDF / network / VLM."""
from __future__ import annotations

import io

import pytest

from app.core.schematic_symbol_head import (
    SchematicSymbolStore,
    SymbolRow,
    TEACHER_VLM,
    crop_feature,
    feature_sha,
    train_symbol_head,
)

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")


def _glyph(kind: str, jitter: int = 0, scale: float = 1.0) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (100, 100), "white")
    d = ImageDraw.Draw(im)
    o = jitter
    if kind == "circle":
        d.ellipse((15 + o, 15 + o, 15 + o + int(50 * scale), 15 + o + int(50 * scale)), outline="black", width=3)
    elif kind == "triangle":
        d.polygon([(40 + o, 12 + o), (12 + o, 12 + o + int(56 * scale)), (12 + o + int(56 * scale), 12 + o + int(56 * scale))], outline="black", width=3)
    elif kind == "square":
        d.rectangle((16 + o, 16 + o, 16 + o + int(48 * scale), 16 + o + int(48 * scale)), outline="black", width=3)
    elif kind == "diamond":
        cx, cy, r = 45 + o, 45 + o, int(28 * scale)
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], outline="black", width=3)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def _seed_store() -> SchematicSymbolStore:
    st = SchematicSymbolStore(":memory:")
    rows = []
    for deal in ["MARRIOTT", "OPTBOT", "COPPER", "YONAH", "BANKS"]:
        for c in ["circle", "triangle", "square", "diamond"]:
            for j in range(6):
                f = crop_feature(_glyph(c, jitter=(j * 3 - 7), scale=0.8 + 0.08 * j))
                rows.append(SymbolRow(deal, "T1", feature_sha(f), c, TEACHER_VLM, 0.9, f))
    st.log(rows)
    return st


def test_feature_same_glyph_closer_than_cross_glyph():
    """Sparse line-art has low absolute pixel correlation, but the ink-bbox
    normalization must still make the SAME glyph (shifted/scaled) more similar
    than a DIFFERENT glyph — that relative ordering is what the head exploits."""
    import numpy as np

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

    d0 = crop_feature(_glyph("diamond", jitter=0, scale=1.0))
    d1 = crop_feature(_glyph("diamond", jitter=20, scale=1.5))
    sq = crop_feature(_glyph("square", jitter=0, scale=1.0))
    assert cos(d0, d1) > cos(d0, sq)


def test_head_trains_and_generalizes_across_deals():
    head = train_symbol_head(_seed_store())
    assert head is not None
    assert head.val_macro_f1 >= 0.75  # leave-one-deal-out
    assert set(head.classes) == {"circle", "triangle", "square", "diamond"}


def test_head_classifies_out_of_distribution_glyphs():
    head = train_symbol_head(_seed_store())
    correct = 0
    for c in ["circle", "triangle", "square", "diamond"]:
        r = head.classify(crop_feature(_glyph(c, jitter=15, scale=1.4)))
        if r and r[0] == c:
            correct += 1
    assert correct >= 3  # robust to shift+scale outside the training range


def test_head_returns_none_when_insufficient_data():
    st = SchematicSymbolStore(":memory:")
    f = crop_feature(_glyph("circle"))
    st.log([SymbolRow("D1", "T1", feature_sha(f), "circle", TEACHER_VLM, 0.9, f)])
    assert train_symbol_head(st) is None  # 1 class -> can't train
