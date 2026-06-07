"""End-to-end NEURAL symbol grounding — the replacement for the heuristic
``detect_symbols`` path that produced junk on real DDs ("device:typical",
"device:system" — common words matched as devices, 0 real recall, 2.5 hr).

Composes the neural stack built piece by piece, with NO text-tag word matching:

    page image
      -> neural region proposer (objectness head)  : WHERE are symbol-shaped things
      -> symbol embedder (contrastive ViT / crop_feature) : embed each region crop
      -> per-document LegendIndex                   : WHICH legend entry it matches
      -> abstain if similarity/objectness too low   : -> VLM teacher fallback

Every stage is the learned/per-document version, so:
  * it works on raster AND vector sheets (objectness runs on pixels),
  * it grounds glyphs to the *legend* (generalizes per drawing set),
  * it abstains instead of emitting garbage (precision over recall),
  * the slow VLM is consulted only on abstains, not 2,567 times.

Returns ``GroundedSymbol`` records with bbox + meaning + confidence so they
project onto the existing ``schematic_symbol_detection`` atoms (provenance
unchanged). Falls back cleanly when heads/legend aren't ready (returns []).
"""
from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass
class GroundedSymbol:
    bbox_px: tuple[int, int, int, int]
    meaning: str
    legend_entry_id: str | None
    confidence: float          # objectness * legend-match similarity
    source: str                # "neural" | "abstain"


def _img_from_page(page, dpi: int):
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def ground_page_image(img, *, registry, legend_index,
                      objectness_thresh: float = 0.6,
                      match_thresh: float = 0.45,
                      scales=(40, 64, 96, 128)) -> list[GroundedSymbol]:
    """Neural grounding on a single rendered page image. ``registry`` has the
    trained objectness head; ``legend_index`` is this document's LegendIndex
    (its embedder should match the one the regions are scored with)."""
    from app.core.schematic_region_proposer import propose_regions_neural

    if not legend_index or len(legend_index) == 0:
        return []
    regions = propose_regions_neural(
        registry, img, score_thresh=objectness_thresh, scales=scales,
    )
    out: list[GroundedSymbol] = []
    for r in regions:
        crop = img.crop(r.bbox_px)
        buf = io.BytesIO(); crop.save(buf, format="PNG")
        ref, sim = legend_index.match(buf.getvalue(), threshold=match_thresh)
        if ref is None:
            # objectness said "symbol" but no confident legend match -> abstain
            out.append(GroundedSymbol(r.bbox_px, "", None, float(r.score), "abstain"))
            continue
        out.append(GroundedSymbol(
            bbox_px=r.bbox_px, meaning=ref.meaning,
            legend_entry_id=ref.legend_entry_id,
            confidence=float(r.score) * float(sim), source="neural",
        ))
    # Cross-scale dedup: overlapping windows produce duplicate detections of one
    # physical symbol, which would inflate counts/takeoff. Keep the highest-
    # confidence detection per overlapping cluster (NMS over grounded boxes).
    return _dedup_grounded(out)


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _min_dim(b):
    return min(b[2] - b[0], b[3] - b[1])


def _dedup_grounded(symbols: list[GroundedSymbol], iou_thresh: float = 0.25) -> list[GroundedSymbol]:
    """NMS over grounded boxes. Cross-scale duplicates of one physical symbol
    have low IoU (small box inside big box) but nearly the same CENTER, so we
    suppress on center-proximity too — keeps counts/takeoff accurate."""
    kept: list[GroundedSymbol] = []
    for s in sorted(symbols, key=lambda g: g.confidence, reverse=True):
        dup = False
        scx, scy = _center(s.bbox_px)
        for k in kept:
            kcx, kcy = _center(k.bbox_px)
            dist = ((scx - kcx) ** 2 + (scy - kcy) ** 2) ** 0.5
            near = dist < 0.8 * max(_min_dim(s.bbox_px), _min_dim(k.bbox_px))
            if near or _iou(s.bbox_px, k.bbox_px) >= iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept


def ground_page(page, page_index: int, *, registry, legend_index,
                dpi: int = 200, **kw) -> list[GroundedSymbol]:
    """Render a fitz page and ground it. Returns [] if objectness head untrained
    or legend empty (caller falls back to heuristic/VLM)."""
    if not legend_index or len(legend_index) == 0:
        return []
    head = registry._heads.get("symbol_objectness")
    if head is None or head.trained is None:
        return []
    img = _img_from_page(page, dpi)
    return ground_page_image(img, registry=registry, legend_index=legend_index, **kw)


def abstained(symbols: list[GroundedSymbol]) -> list[GroundedSymbol]:
    """Regions that looked like symbols but didn't ground — the queue to send to
    the VLM teacher (and to capture as new training labels)."""
    return [s for s in symbols if s.source == "abstain"]
