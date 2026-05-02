"""Semantic QA checks for overlay color placement.

These checks complement pixel diffs.  Pixel diffs tell us something moved;
semantic checks say whether known bad patterns remain: overbroad purple slabs,
body-sized blue title washes, cyan paragraph hulls, and mini-table bleed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SemanticIssue:
    kind: str
    box_id: str
    px_bbox: tuple[int, int, int, int]
    detail: str


def _bbox(b: dict[str, Any]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = b.get("px_bbox") or b.get("rect") or [0, 0, 0, 0]
    return int(x0), int(y0), int(x1), int(y1)


def _wh(b: dict[str, Any]) -> tuple[int, int]:
    x0, y0, x1, y1 = _bbox(b)
    return max(0, x1 - x0), max(0, y1 - y0)


def _semantic_issues(payload: dict[str, Any]) -> list[SemanticIssue]:
    W = int(payload.get("image_width") or payload.get("debug_stats", {}).get("W") or 0)
    H = int(payload.get("image_height") or payload.get("debug_stats", {}).get("H") or 0)
    by_id = {b.get("box_id"): b for b in payload.get("boxes", [])}
    issues: list[SemanticIssue] = []
    for b in payload.get("boxes", []):
        bid = str(b.get("box_id") or "")
        color = b.get("color")
        syn = bool(b.get("synthetic"))
        x0, y0, x1, y1 = _bbox(b)
        w, h = _wh(b)
        if color == "PURPLE" and syn and bid.startswith("titleblk"):
            if (h >= int(0.34 * H) and w <= int(0.18 * W)) or bid.startswith(("titleblkgrp_", "titleblkpanel_")):
                issues.append(SemanticIssue("false_purple_slab", bid, (x0, y0, x1, y1), "purple must be compact artwork, not text panel/sidebar"))
        if syn and bid.startswith("colhdr_"):
            if h > max(28, int(0.035 * H)) or (w * h) > int(0.006 * W * H):
                issues.append(SemanticIssue("false_colhdr_body_hull", bid, (x0, y0, x1, y1), "cyan header ring is too tall/large"))
        if syn and color == "BLUE" and bid.endswith("_title") and not bid.startswith("textsec_"):
            p = by_id.get(b.get("parent_box_id"))
            if p:
                _, ph = _wh(p)
                if ph > 0 and h >= max(34, int(0.18 * ph)):
                    issues.append(SemanticIssue("overbroad_blue_title_wash", bid, (x0, y0, x1, y1), "title wash would cover body content"))
        if "_ext" in bid and (bid.endswith("_mtrow") or "_mtcell" in bid):
            if w >= max(150, int(0.42 * W)):
                issues.append(SemanticIssue("mini_table_bleed", bid, (x0, y0, x1, y1), "mini-table extension spans main body"))
    return issues


def color_pixel_counts(overlay_png: str | Path) -> dict[str, int]:
    im = np.asarray(Image.open(overlay_png).convert("RGB"))
    # RGB approximations of BGR constants after cv2.imwrite.
    masks = {
        "orange": (im[:, :, 0] > 230) & (im[:, :, 1] > 95) & (im[:, :, 1] < 190) & (im[:, :, 2] < 80),
        "blue": (im[:, :, 2] > 170) & (im[:, :, 1] < 130) & (im[:, :, 0] < 80),
        "cyan": (im[:, :, 0] < 80) & (im[:, :, 1] > 180) & (im[:, :, 2] > 180),
        "purple": (im[:, :, 0] > 140) & (im[:, :, 2] > 140) & (im[:, :, 1] < 100),
        "green": (im[:, :, 1] > 130) & (im[:, :, 0] < 130) & (im[:, :, 2] < 130),
    }
    return {k: int(np.count_nonzero(v)) for k, v in masks.items()}


def orange_dark_alignment(overlay_png: str | Path, source_rgb_png: str | Path, radius_px: int = 2) -> dict[str, float | int]:
    overlay = np.asarray(Image.open(overlay_png).convert("RGB"))
    source = np.asarray(Image.open(source_rgb_png).convert("RGB"))
    if overlay.shape[:2] != source.shape[:2]:
        return {"orange_pixels": 0, "aligned_pixels": 0, "alignment_ratio": 0.0, "note": "shape_mismatch"}
    orange = (overlay[:, :, 0] > 230) & (overlay[:, :, 1] > 95) & (overlay[:, :, 1] < 190) & (overlay[:, :, 2] < 80)
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    dark = gray < 170
    if radius_px > 0:
        dark = cv2.dilate(dark.astype("uint8"), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius_px + 1, 2 * radius_px + 1))).astype(bool)
    orange_n = int(np.count_nonzero(orange))
    aligned = int(np.count_nonzero(orange & dark))
    ratio = aligned / orange_n if orange_n else 1.0
    return {"orange_pixels": orange_n, "aligned_pixels": aligned, "alignment_ratio": float(ratio)}


def semantic_report(payload: dict[str, Any], overlay_png: str | Path | None = None, source_rgb_png: str | Path | None = None) -> dict[str, Any]:
    issues = _semantic_issues(payload)
    report: dict[str, Any] = {
        "issue_count": len(issues),
        "issues": [asdict(i) for i in issues],
        "box_count": len(payload.get("boxes", [])),
        "cleanup": payload.get("debug_stats", {}).get("semantic_cleanup", {}),
        "raster_line_repairs": payload.get("debug_stats", {}).get("raster_line_repairs", 0),
    }
    if overlay_png is not None:
        report["color_pixels"] = color_pixel_counts(overlay_png)
    if overlay_png is not None and source_rgb_png is not None:
        report["orange_dark_alignment"] = orange_dark_alignment(overlay_png, source_rgb_png)
    return report
