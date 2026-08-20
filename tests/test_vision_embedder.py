"""The glyph embedder's contract and its degradation path.

Accuracy is measured in scratch (crop_feature 62.5% -> two-view DINOv2 92.2%
on 8 symbols x 8 distortions); these pin the properties the compile path
depends on, which are cheaper to assert and the ones that break silently.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pytest

from app.core.vision_embedder import DinoV2Embedder, default_glyph_embedder


def _png(fill=255) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (64, 64), fill).save(buf, format="PNG")
    return buf.getvalue()


def test_disabled_by_env_yields_no_embedder():
    """An offline image must keep crop_feature rather than fail a compile."""
    os.environ["SOWSMITH_GLYPH_EMBED_DINOV2"] = "0"
    try:
        import app.core.vision_embedder as ve

        ve._TRIED = False
        ve._MODEL = ve._PROC = None
        ve._DEFAULT = None
        assert DinoV2Embedder().available is False
        assert default_glyph_embedder() is None
    finally:
        os.environ.pop("SOWSMITH_GLYPH_EMBED_DINOV2", None)
        import app.core.vision_embedder as ve

        ve._TRIED = False
        ve._MODEL = ve._PROC = None
        ve._DEFAULT = None


def test_an_unavailable_embedder_returns_a_zero_vector_not_an_exception():
    e = DinoV2Embedder.__new__(DinoV2Embedder)
    e._ok = False
    v = e.embed(_png())
    assert v.shape == (DinoV2Embedder.dim,)
    assert not v.any()


def test_undecodable_bytes_do_not_raise():
    e = DinoV2Embedder.__new__(DinoV2Embedder)
    e._ok = True
    assert e.embed(b"not a png").shape == (DinoV2Embedder.dim,)


def test_dim_is_two_views():
    """768, not 384: the raw and morphologically-closed views are concatenated
    so one vector still fixes dashed line style."""
    assert DinoV2Embedder.dim == 768


@pytest.mark.skipif(
    DinoV2Embedder().available is False,
    reason="dinov2 weights unavailable in this environment",
)
class TestWithWeights:
    def test_embedding_is_unit_norm_and_deterministic(self):
        e = DinoV2Embedder()
        a, b = e.embed(_png()), e.embed(_png())
        assert a.shape == (768,)
        assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-4
        # Deterministic: the flaky from-scratch ViT it replaces was not.
        assert float(np.dot(a, b)) > 0.9999

    def test_a_dashed_shape_matches_its_solid_self(self):
        """The case CLS alone got wrong 2 times in 8."""
        from PIL import Image, ImageDraw

        def draw(dashed: bool) -> bytes:
            img = Image.new("L", (64, 64), 255)
            d = ImageDraw.Draw(img)
            d.ellipse([10, 10, 54, 54], outline=0, width=2)
            if dashed:
                px = img.load()
                for y in range(64):
                    for x in range(64):
                        if px[x, y] < 128 and (x + y) % 6 < 3:
                            px[x, y] = 255
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        e = DinoV2Embedder()
        solid_circle = e.embed(draw(False))
        dashed_circle = e.embed(draw(True))

        square = Image.new("L", (64, 64), 255)
        ImageDraw.Draw(square).rectangle([10, 10, 54, 54], outline=0, width=2)
        buf = io.BytesIO()
        square.save(buf, format="PNG")
        solid_square = e.embed(buf.getvalue())

        assert float(np.dot(solid_circle, dashed_circle)) > float(
            np.dot(solid_square, dashed_circle)
        ), "a dashed circle must match a solid circle over a solid square"
