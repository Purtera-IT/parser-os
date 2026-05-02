"""Regression and quality metrics for Parser OS overlays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


@dataclass(frozen=True)
class ImageDiff:
    changed_pixels: int
    total_pixels: int
    changed_ratio: float
    max_abs_channel_delta: int
    mean_abs_delta: float


@dataclass(frozen=True)
class BoxDiff:
    added: list[str]
    removed: list[str]
    changed: list[str]
    same: int


def load_detection_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def summarize_boxes(payload: dict[str, Any]) -> dict[str, Any]:
    boxes = payload.get("boxes", [])
    summary: dict[str, int] = {}
    for b in boxes:
        key = f"{b.get('color')}:{'synthetic' if b.get('synthetic') else 'real'}"
        summary[key] = summary.get(key, 0) + 1
    return {
        "box_count": len(boxes),
        "summary": summary,
        "image_width": payload.get("image_width") or payload.get("debug_stats", {}).get("W"),
        "image_height": payload.get("image_height") or payload.get("debug_stats", {}).get("H"),
    }


def compare_images(a_path: str | Path, b_path: str | Path) -> ImageDiff:
    if Image is None:
        raise RuntimeError("Pillow is required for image diff metrics")
    a = np.asarray(Image.open(a_path).convert("RGB"))
    b = np.asarray(Image.open(b_path).convert("RGB"))
    if a.shape != b.shape:
        # Pad by treating every pixel in the larger canvas as changed.
        total = max(a.shape[0] * a.shape[1], b.shape[0] * b.shape[1])
        return ImageDiff(total, total, 1.0, 255, 255.0)
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    changed = np.any(delta > 0, axis=2)
    changed_count = int(changed.sum())
    total = int(changed.size)
    return ImageDiff(
        changed_pixels=changed_count,
        total_pixels=total,
        changed_ratio=changed_count / float(max(1, total)),
        max_abs_channel_delta=int(delta.max()) if delta.size else 0,
        mean_abs_delta=float(delta.mean()) if delta.size else 0.0,
    )


def _by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(b.get("box_id")): b for b in payload.get("boxes", [])}


def compare_box_json(a_path: str | Path, b_path: str | Path) -> BoxDiff:
    a = _by_id(load_detection_json(a_path))
    b = _by_id(load_detection_json(b_path))
    a_ids = set(a)
    b_ids = set(b)
    added = sorted(b_ids - a_ids)
    removed = sorted(a_ids - b_ids)
    changed: list[str] = []
    for bid in sorted(a_ids & b_ids):
        av = a[bid]
        bv = b[bid]
        keys = ("color", "px_bbox", "parent_box_id", "children_count", "nested_depth", "synthetic")
        if any(av.get(k) != bv.get(k) for k in keys):
            changed.append(bid)
    return BoxDiff(added=added, removed=removed, changed=changed, same=len(a_ids & b_ids) - len(changed))


def quality_checks(payload: dict[str, Any]) -> dict[str, Any]:
    boxes = payload.get("boxes", [])
    W = int(payload.get("image_width") or payload.get("debug_stats", {}).get("W") or 0)
    H = int(payload.get("image_height") or payload.get("debug_stats", {}).get("H") or 0)
    zero_area: list[str] = []
    out_of_bounds: list[str] = []
    title_colhdr_conflicts: list[dict[str, Any]] = []
    title_bands = [b for b in boxes if str(b.get("box_id", "")).endswith(("_title", "_sublabel"))]
    colhdrs = [b for b in boxes if str(b.get("box_id", "")).startswith("colhdr_")]

    for b in boxes:
        x0, y0, x1, y1 = [int(v) for v in b.get("px_bbox", [0, 0, 0, 0])]
        if x1 <= x0 or y1 <= y0:
            zero_area.append(str(b.get("box_id")))
        if x0 < -2 or y0 < -2 or x1 > W + 2 or y1 > H + 2:
            out_of_bounds.append(str(b.get("box_id")))

    for t in title_bands:
        tx0, ty0, tx1, ty1 = [int(v) for v in t.get("px_bbox", [0, 0, 0, 0])]
        ta = max(1, (tx1 - tx0) * (ty1 - ty0))
        for h in colhdrs:
            hx0, hy0, hx1, hy1 = [int(v) for v in h.get("px_bbox", [0, 0, 0, 0])]
            ix0, iy0 = max(tx0, hx0), max(ty0, hy0)
            ix1, iy1 = min(tx1, hx1), min(ty1, hy1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            ha = max(1, (hx1 - hx0) * (hy1 - hy0))
            # This is a warning, not an automatic failure: some legitimate
            # header rings live inside a clipped title band. Large overlap flags
            # cases where the wash may obscure the header region.
            if ((ix1 - ix0) * (iy1 - iy0)) / float(ha) > 0.72:
                title_colhdr_conflicts.append({"title": t.get("box_id"), "colhdr": h.get("box_id")})

    return {
        "zero_area_count": len(zero_area),
        "zero_area_ids": zero_area[:50],
        "out_of_bounds_count": len(out_of_bounds),
        "out_of_bounds_ids": out_of_bounds[:50],
        "title_colhdr_overlap_warnings": len(title_colhdr_conflicts),
        "title_colhdr_overlap_examples": title_colhdr_conflicts[:25],
    }


def _pdf_pt_to_image_xy(x_pt: float, y_pt: float, page_w_pt: float, page_h_pt: float, scale: float, cw_quarter_turns: int) -> tuple[float, float]:
    """Map PyMuPDF top-left PDF points to rendered image pixels."""
    px = x_pt * scale
    py = y_pt * scale
    W_orig = page_w_pt * scale
    H_orig = page_h_pt * scale
    n = cw_quarter_turns % 4
    if n == 0:
        return px, py
    if n == 1:
        return H_orig - py, px
    if n == 2:
        return W_orig - px, H_orig - py
    return py, W_orig - px


def _pdf_bbox_to_image_bbox(bbox_pt: tuple[float, float, float, float], page_w_pt: float, page_h_pt: float, scale: float, cw_quarter_turns: int) -> tuple[int, int, int, int]:
    x0p, y0p, x1p, y1p = bbox_pt
    pts = [
        _pdf_pt_to_image_xy(x0p, y0p, page_w_pt, page_h_pt, scale, cw_quarter_turns),
        _pdf_pt_to_image_xy(x1p, y0p, page_w_pt, page_h_pt, scale, cw_quarter_turns),
        _pdf_pt_to_image_xy(x0p, y1p, page_w_pt, page_h_pt, scale, cw_quarter_turns),
        _pdf_pt_to_image_xy(x1p, y1p, page_w_pt, page_h_pt, scale, cw_quarter_turns),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))


def text_span_containment(pdf_path: str | Path, payload: dict[str, Any], page_index: int = 0) -> dict[str, Any]:
    """Count native PDF word centroids inside overlay classes.

    Uses PyMuPDF word boxes and the same top-left point -> rendered-pixel
    transform as the detector.  It avoids pypdfium/OpenCV so QA can run quickly
    even when only text containment is needed.
    """
    try:
        import fitz
    except Exception as e:  # pragma: no cover
        return {"available": False, "error": repr(e)}

    stats = payload.get("debug_stats", {})
    scale = float(stats.get("render_scale_used") or payload.get("scale") or 1.0)
    rotate_qt = int(stats.get("rotated_cw_quarter_turns") or 0)
    fdoc = fitz.open(str(pdf_path))
    try:
        page = fdoc[page_index]
        pw, ph = float(page.rect.width), float(page.rect.height)
        words = page.get_text("words")
    finally:
        fdoc.close()
    boxes = payload.get("boxes", [])

    buckets = {
        "BLUE": [], "ORANGE": [], "CYAN": [], "PURPLE": [], "GREEN": [], "UNCONTAINED": []
    }
    for w in words:
        if len(w) < 5 or not str(w[4]).strip():
            continue
        x0, y0, x1, y1 = _pdf_bbox_to_image_bbox(
            (float(w[0]), float(w[1]), float(w[2]), float(w[3])), pw, ph, scale, rotate_qt
        )
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        owners = []
        for b in boxes:
            bx0, by0, bx1, by1 = [int(v) for v in b.get("px_bbox", [0, 0, 0, 0])]
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                owners.append(str(b.get("color")))
        if not owners:
            buckets["UNCONTAINED"].append(str(w[4]))
        else:
            for color in ("PURPLE", "CYAN", "GREEN", "ORANGE", "BLUE"):
                if color in owners:
                    buckets[color].append(str(w[4]))
                    break
    total = sum(len(v) for v in buckets.values())
    return {
        "available": True,
        "total_words": total,
        "contained_words": total - len(buckets["UNCONTAINED"]),
        "uncontained_words": len(buckets["UNCONTAINED"]),
        "containment_ratio": (total - len(buckets["UNCONTAINED"])) / float(max(1, total)),
        "by_class": {k: len(v) for k, v in buckets.items()},
        "uncontained_examples": buckets["UNCONTAINED"][:30],
    }
