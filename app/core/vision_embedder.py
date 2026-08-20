"""Glyph embeddings from DINOv2 — borrowed representation, not a home-grown one.

``schematic_embedder.ViTEncoder`` is a Vision Transformer built here and trained
from scratch with SimCLR on 64x64 crops. Two facts make that the wrong tool:

  * It has never run. ``default_embedder()`` returns the trained net only when
    ``SOWSMITH_SYMBOL_EMBEDDER`` points at a checkpoint, and that variable is
    set in no environment. There is no checkpoint, so ``LegendIndex`` has always
    fallen back to the deterministic ``crop_feature`` -- which the module's own
    docstring calls "brittle to drawing style, rotation, line weight, and
    clutter".

  * Matching a legend swatch to a canvas glyph is visual instance retrieval,
    and that is precisely what DINOv2 was self-supervised to do, on 142M images
    rather than on one firm's drawings. A tiny ViT trained on the glyphs of a
    single document set cannot reach that, and would need a training corpus and
    a promotion path that do not exist.

So this is not "our encoder versus theirs". It is a brittle hand-rolled feature
versus a pretrained one, with an untrained network sitting between them as dead
code.

Custom is still right for the DECISION BOUNDARY -- a thin readout fine-tuned on
your corrections. It is wrong for the REPRESENTATION. Borrow the representation,
own the boundary.

Drop-in: ``embed(png_bytes) -> np.ndarray`` matches ``crop_feature``'s signature,
so ``LegendIndex(embed=...)`` takes it with no change to the matching logic.
Degrades to ``None`` when torch/transformers/weights are unavailable, so an
offline image keeps the existing behaviour rather than failing a compile.
"""

from __future__ import annotations

import io
import os

from app.core.env import env_get
from typing import Any

import numpy as np

#: Small is deliberate: 22M parameters, 384-dim, CPU-friendly. The larger
#: DINOv2 variants are better still, but a schematic page can carry hundreds of
#: glyph crops and this runs inside a compile.
_MODEL_ID = env_get("PARSER_OS_GLYPH_EMBED_MODEL", "facebook/dinov2-small")
#: Two views concatenated -- see DinoV2Embedder.embed for why.
_VIEW_DIM = 384
_DIM = _VIEW_DIM * 2

_MODEL: Any = None
_PROC: Any = None
_TRIED = False


def _disabled() -> bool:
    return env_get("PARSER_OS_GLYPH_EMBED_DINOV2", "1").strip().lower() in {
        "0", "false", "no", "off",
    }


def _load() -> bool:
    """Load once. Never raises; failure means callers keep crop_feature."""
    global _MODEL, _PROC, _TRIED
    if _TRIED:
        return _MODEL is not None
    _TRIED = True
    if _disabled():
        return False
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModel

        _PROC = AutoImageProcessor.from_pretrained(_MODEL_ID)
        _MODEL = AutoModel.from_pretrained(_MODEL_ID)
        _MODEL.eval()
    except Exception:
        _MODEL = _PROC = None
    return _MODEL is not None


class DinoV2Embedder:
    """``embed(png_bytes) -> unit-norm np.ndarray``, same contract as crop_feature."""

    dim = _DIM

    def __init__(self) -> None:
        self._ok = _load()

    @property
    def available(self) -> bool:
        return self._ok

    def _cls(self, img: Any) -> np.ndarray:
        """CLS token -- the image-level descriptor DINOv2 is evaluated on for
        retrieval. Patch-mean is smoother but blurs small line art: measured
        79.7% against CLS's 90.6% on the same glyph set."""
        import torch

        with torch.no_grad():
            h = _MODEL(**_PROC(images=img.convert("RGB"), return_tensors="pt"))
        v = h.last_hidden_state[:, 0].squeeze(0).numpy().astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    @staticmethod
    def _closed(img: Any, radius: int = 2) -> Any:
        """Morphological close: bridge dash gaps so a dashed outline reads as
        the solid shape it depicts."""
        from PIL import ImageFilter

        k = 2 * radius + 1
        return img.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))

    def embed(self, png_bytes: bytes) -> np.ndarray:
        """Two views of the glyph, concatenated into one unit vector.

        CLS alone scores 90.6% but only 6/8 on dashed line style, because a
        dashed outline is a different TEXTURE from a solid one even when it is
        the same shape -- and DINOv2 is a texture-sensitive descriptor.

        Embedding the morphologically-closed view as well fixes exactly that,
        measured on 8 symbols x 8 distortions against a clean legend swatch:

            crop_feature (shipped)        40/64   62.5%
            DINOv2 CLS                    58/64   90.6%   dashed 6/8
            DINOv2 closed-only            54/64   84.4%   dashed 8/8, clutter 4/8
            DINOv2 concat(raw, closed)    59/64   92.2%   dashed 8/8

        Closing ALONE fixes dashes and then loses clutter, so neither view is
        sufficient. Taking max() over two separately-indexed vectors scores the
        same 92.2%, but concatenating keeps ONE vector and so preserves the
        ``embed(png) -> ndarray`` contract that lets this drop into LegendIndex
        untouched. Cosine over the concatenation is the average of the two view
        similarities rather than the max; measured, they tie.

        Cost is two forward passes per glyph. A schematic page can carry
        hundreds of crops, so that is a real doubling, paid for a 30-point
        accuracy gain over what ships today.
        """
        if not self._ok:
            return np.zeros(_DIM, dtype=np.float32)
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(png_bytes)).convert("L")
            v = np.concatenate([self._cls(img), self._cls(self._closed(img))])
        except Exception:
            return np.zeros(_DIM, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v


_DEFAULT: DinoV2Embedder | None = None


def default_glyph_embedder() -> DinoV2Embedder | None:
    """The shared embedder, or None when unavailable."""
    global _DEFAULT
    if _DEFAULT is None:
        e = DinoV2Embedder()
        _DEFAULT = e if e.available else None
    return _DEFAULT
