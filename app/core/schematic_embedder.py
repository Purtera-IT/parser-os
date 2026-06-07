"""Self-supervised ViT symbol embedder — the neural upgrade behind LegendIndex.

Per-document grounding (``LegendIndex``) matches each canvas glyph to the nearest
legend swatch. The QUALITY of that match is only as good as the embedding. The
deterministic :func:`app.core.schematic_symbol_head.crop_feature` works zero-shot
but is brittle to drawing style, rotation, line weight, and clutter.

This module learns a better metric with a small **Vision Transformer** trained
**contrastively (SimCLR / NT-Xent)**: two random augmentations of the SAME glyph
crop must embed close; different crops must embed apart. It is **self-supervised**
— it needs only glyph crops, no labels — so it can train on ANY firm's drawings
(legend swatches + detected region crops) and generalize across vocabularies.

Drop-in: ``SchematicEmbedder.embed(png_bytes) -> np.ndarray`` matches
``crop_feature``'s signature, so ``LegendIndex(embed=emb.embed)`` swaps it in with
zero change to the matching logic. CPU-friendly (64x64, tiny ViT).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False

IMG = 64
PATCH = 8
EMB = 128


def _to_gray_tensor(png_bytes: bytes):
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(png_bytes)) as im:
        g = im.convert("L")
        inv = ImageOps.invert(g)
        bbox = inv.getbbox()
        if bbox:
            g = g.crop(bbox)
        g = g.resize((IMG, IMG))
    a = np.asarray(g, dtype=np.float32) / 255.0
    return torch.from_numpy(a)[None, None]  # (1,1,IMG,IMG)


def _augment(t):
    """Random style-preserving augmentation: rotate, scale, translate, noise,
    line-weight (erode/dilate via blur+threshold)."""
    import torchvision.transforms.functional as TF
    angle = float(np.random.uniform(-25, 25))
    scale = float(np.random.uniform(0.8, 1.25))
    tx = int(np.random.uniform(-5, 5))
    ty = int(np.random.uniform(-5, 5))
    img = TF.affine(t, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0], fill=1.0)
    if np.random.rand() < 0.5:
        img = img + torch.randn_like(img) * 0.05
    return img.clamp(0, 1)


class ViTEncoder(nn.Module):
    def __init__(self, img=IMG, patch=PATCH, emb=EMB, depth=3, heads=4):
        super().__init__()
        self.np_side = img // patch
        n_patches = self.np_side ** 2
        self.patch = nn.Conv2d(1, emb, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, n_patches, emb) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=emb * 2,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.tx = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb)
        self.proj = nn.Linear(emb, emb)

    def forward(self, x):
        x = self.patch(x)                       # (B,emb,h,w)
        x = x.flatten(2).transpose(1, 2)        # (B,patches,emb)
        x = x + self.pos
        x = self.tx(x)
        x = self.norm(x.mean(dim=1))            # mean-pool
        x = self.proj(x)
        return F.normalize(x, dim=-1)           # unit sphere


def _nt_xent(z1, z2, temp=0.2):
    """SimCLR loss: each (z1[i],z2[i]) is a positive pair; all others negatives."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], 0)                  # (2B,emb)
    sim = (z @ z.t()) / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], 0)
    return F.cross_entropy(sim, targets)


@dataclass
class SchematicEmbedder:
    model: "ViTEncoder"

    def embed(self, png_bytes: bytes) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            v = self.model(_to_gray_tensor(png_bytes))[0]
        return v.cpu().numpy().astype(np.float32)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> "SchematicEmbedder":
        m = ViTEncoder()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(m)


def train_embedder(glyph_pngs: list[bytes], *, steps: int = 400, batch: int = 16,
                   lr: float = 1e-3, seed: int = 0) -> SchematicEmbedder:
    """Self-supervised contrastive training on a pool of glyph crops (no labels).
    Returns a SchematicEmbedder. CPU-friendly for small pools."""
    if not _TORCH:
        raise RuntimeError("torch unavailable")
    torch.manual_seed(seed)
    np.random.seed(seed)
    tensors = [_to_gray_tensor(p) for p in glyph_pngs]
    model = ViTEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(tensors)
    for _ in range(steps):
        idx = np.random.randint(0, n, size=min(batch, n))
        v1 = torch.cat([_augment(tensors[i]) for i in idx], 0)
        v2 = torch.cat([_augment(tensors[i]) for i in idx], 0)
        z1, z2 = model(v1), model(v2)
        loss = _nt_xent(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()
    return SchematicEmbedder(model)
