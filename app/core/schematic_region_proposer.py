"""Neural region proposer — the learned replacement for the heuristic
``region_proposals.propose_regions``.

The heuristic clusters vector strokes with hardcoded point thresholds. Three
problems: (1) vector-ONLY, so it is blind on raster/scanned sheets; (2) fixed
size/distance bands miss off-scale or tightly-packed symbols; (3) no notion of
"is this actually a symbol" — it emits geometry, not confidence.

This module makes it neural, same MAX pattern as the rest:

* **Candidate generation** is modality-agnostic — a multi-scale sliding window
  over the rendered page (works on vector AND raster). Optionally seeded with the
  heuristic proposals so we never do worse than today.
* **Objectness head** (binary symbol/background) scores each candidate crop. It
  is a normal registered head in :mod:`app.core.schematic_heads`, so it is
  taught by the VLM teacher (silver) + PM corrections (gold), auto-selected,
  eval-gated, and abstains. Feature = the symbol embedder (or crop_feature).
* **NMS** collapses overlapping keeps into one box per physical symbol.

So region proposal becomes: dense candidates -> neural objectness filter -> NMS,
taught by the best vision model until it is our own. Until the head is trained it
returns [] (caller falls back to the heuristic / VLM), so wiring it in is safe.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

OBJECTNESS_HEAD = "symbol_objectness"
LABEL_SYMBOL = "symbol"
LABEL_BACKGROUND = "background"


@dataclass
class ScoredRegion:
    bbox_px: tuple[int, int, int, int]
    score: float


def sliding_windows(img, scales=(48, 80, 128), stride_frac: float = 0.5):
    """Yield (bbox_px, crop_png_bytes) over a multi-scale grid. Modality-agnostic
    — operates on the rendered pixels so it works on raster sheets too."""
    from PIL import Image

    W, H = img.size
    for s in scales:
        step = max(8, int(s * stride_frac))
        for y in range(0, max(1, H - s + 1), step):
            for x in range(0, max(1, W - s + 1), step):
                box = (x, y, min(W, x + s), min(H, y + s))
                buf = io.BytesIO()
                img.crop(box).save(buf, format="PNG")
                yield box, buf.getvalue()


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua else 0.0


def _nms(regions: list[ScoredRegion], iou_thresh: float = 0.3) -> list[ScoredRegion]:
    out: list[ScoredRegion] = []
    for r in sorted(regions, key=lambda r: r.score, reverse=True):
        if all(_iou(r.bbox_px, k.bbox_px) < iou_thresh for k in out):
            out.append(r)
    return out


def propose_regions_neural(registry, img, *, score_thresh: float = 0.6,
                           scales=(48, 80, 128), max_regions: int = 400,
                           seed_boxes: list[tuple] | None = None) -> list[ScoredRegion]:
    """Dense candidates -> objectness head -> NMS. Returns [] if the objectness
    head is not trained yet (caller falls back). ``registry`` is a
    :class:`app.core.schematic_heads.HeadRegistry` with OBJECTNESS_HEAD trained.
    ``img`` is a PIL grayscale/RGB page image."""
    head = registry._heads.get(OBJECTNESS_HEAD)
    if head is None or head.trained is None:
        return []
    cands = list(sliding_windows(img, scales=scales))
    if seed_boxes:
        from PIL import Image
        for box in seed_boxes:
            buf = io.BytesIO(); img.crop(box).save(buf, format="PNG")
            cands.append((tuple(box), buf.getvalue()))
    scored: list[ScoredRegion] = []
    for box, png in cands:
        pred = registry.predict(OBJECTNESS_HEAD, png)  # (label, prob) or None
        if pred and pred[0] == LABEL_SYMBOL and pred[1] >= score_thresh:
            scored.append(ScoredRegion(bbox_px=tuple(box), score=float(pred[1])))
    return _nms(scored)[:max_regions]


def register_objectness_head(registry) -> None:
    """Register the binary symbol/background objectness head on a registry, using
    the symbol embedder feature. Train it via registry.capture(...) +
    registry.train(OBJECTNESS_HEAD) on VLM-labeled candidate crops."""
    from app.core.schematic_heads import HeadSpec, page_feature
    registry.register(HeadSpec(OBJECTNESS_HEAD, page_feature,
                               db_env="SOWSMITH_OBJECTNESS_HEAD_DB"))
