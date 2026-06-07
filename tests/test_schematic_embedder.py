"""Tests for the contrastive ViT symbol embedder. Fast: builds the encoder and
checks the interface/shape/normalization + that it drops into LegendIndex. Does
NOT train (training is a slow offline job validated separately on the real
corpus, where it beats the deterministic feature on held-out-source retrieval)."""
from __future__ import annotations

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL.Image")

from app.core.schematic_embedder import EMB, SchematicEmbedder, ViTEncoder
from app.core.schematic_symbol_head import LegendIndex


def _glyph(kind: str) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (80, 80), "white")
    d = ImageDraw.Draw(im)
    if kind == "circle":
        d.ellipse((20, 20, 60, 60), outline="black", width=3)
    else:
        d.rectangle((20, 20, 60, 60), outline="black", width=3)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def test_encoder_outputs_unit_normalized_embedding():
    emb = SchematicEmbedder(ViTEncoder())
    v = emb.embed(_glyph("circle"))
    assert v.shape == (EMB,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3  # L2-normalized


def test_embedder_plugs_into_legend_index():
    emb = SchematicEmbedder(ViTEncoder())
    idx = LegendIndex(embed=emb.embed)
    idx.add_symbol("CAMERA", _glyph("circle"))
    idx.add_symbol("SPEAKER", _glyph("square"))
    assert len(idx) == 2
    ref, sim = idx.match(_glyph("circle"))
    assert ref is not None
    assert -1.0 <= sim <= 1.0


def test_save_load_roundtrip(tmp_path):
    emb = SchematicEmbedder(ViTEncoder())
    v1 = emb.embed(_glyph("circle"))
    p = tmp_path / "enc.pt"
    emb.save(str(p))
    emb2 = SchematicEmbedder.load(str(p))
    v2 = emb2.embed(_glyph("circle"))
    assert np.allclose(v1, v2, atol=1e-5)
