"""Deterministic vector/text symbol detector (PR6 of the schematic upgrade).

Given a PyMuPDF page, a ``ResolvedLegend``, and a ``DetectionTargetSet``,
this module finds every instance of each target symbol on the drawing
body and returns a list of ``SymbolDetection`` records.

The detector composes two modalities:

1. **Text-tag matching.** For each target with a legend entry that has
   a short symbol token (``WN``, ``CR``, ``PTZ`` …), scan the page's
   text blocks for that token using word-boundary matching outside
   the legend / title-block regions. This is the primary modality
   for text-extractable PDFs and is sufficient for the Marriott-style
   floor plans.
2. **Glyph template matching.** For each target whose legend entry
   has a recorded ``symbol_bbox_pdf``, crop the legend swatch at the
   schematic replay DPI and run a deterministic OpenCV template
   match against the rendered page body. Used when the symbol is a
   pictogram rather than a text token.

Both modalities emit ``SymbolDetection`` records with bbox in PDF
points and a ``crop_sha256`` deterministically derived from the
200-DPI rendered crop. A unified deterministic NMS pass at the end
removes duplicate hits across modalities so a symbol matched both
ways yields one atom per physical instance.

No runtime LLM. No network calls. CPU-only. Re-running on the same
PDF produces byte-identical ``detection_id`` lists.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from app.parsers.schematic_models import (
    SCHEMATIC_REPLAY_DPI,
    DetectionTargetSet,
    Modality,
    ParsedLegend,
    SymbolDetection,
    crop_sha256_of_pixels,
)


_WS = re.compile(r"\s+")
_TEXT_TAG_PAD = 1.5  # PDF-point padding around a matched word


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Return ``[(token, start, end), ...]`` for word-boundary tokens.

    Word boundary = any non-alphanumeric character. We keep the
    start/end indices so we can slice the original block text and
    place a bbox via interpolation along the line.
    """
    out: list[tuple[str, int, int]] = []
    for m in re.finditer(r"[A-Z0-9][A-Z0-9\-/_.+]*", text):
        out.append((m.group(0), m.start(), m.end()))
    return out


def _interpolate_bbox(
    bbox: tuple[float, float, float, float],
    text_len: int,
    start: int,
    end: int,
) -> tuple[float, float, float, float]:
    """Estimate a sub-bbox for a substring inside a single-line block.

    Assumes monospaced character width along the line, which is
    plenty for tag-style symbols on schematics. Falls back to the
    full block bbox when the substring length is zero or out of
    range. Always returns a strictly positive bbox.
    """
    x0, y0, x1, y1 = bbox
    if text_len <= 0:
        return bbox
    width = max(1.0, x1 - x0)
    per_ch = width / text_len
    sub_x0 = x0 + per_ch * max(0, start) - _TEXT_TAG_PAD
    sub_x1 = x0 + per_ch * max(start + 1, end) + _TEXT_TAG_PAD
    sub_y0 = y0 - _TEXT_TAG_PAD
    sub_y1 = y1 + _TEXT_TAG_PAD
    if sub_x1 <= sub_x0:
        sub_x1 = sub_x0 + max(1.0, per_ch)
    if sub_y1 <= sub_y0:
        sub_y1 = sub_y0 + 1.0
    return (sub_x0, sub_y0, sub_x1, sub_y1)


def _crop_and_hash_page_region(
    page: Any,
    bbox: tuple[float, float, float, float],
) -> str:
    """Deterministic 200-DPI crop hash of a region of a PyMuPDF page."""

    import fitz  # type: ignore[import-not-found]

    zoom = SCHEMATIC_REPLAY_DPI / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        clip=fitz.Rect(*bbox),
        alpha=False,
        colorspace=fitz.csRGB,
    )
    return crop_sha256_of_pixels(pix.samples, pix.width, pix.height, pix.n)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    if area_a + area_b - inter <= 0:
        return 0.0
    return inter / (area_a + area_b - inter)


def _deterministic_nms(
    detections: Iterable[SymbolDetection],
    iou_threshold: float = 0.5,
) -> list[SymbolDetection]:
    """Per-target deterministic NMS.

    Two hits for the *same* target whose bboxes IoU > threshold are
    merged: highest confidence wins, then lowest detection_id.
    Detections for different targets do not suppress each other.
    """
    by_target: dict[str, list[SymbolDetection]] = {}
    for det in detections:
        by_target.setdefault(det.target_key, []).append(det)
    kept: list[SymbolDetection] = []
    for target_key, group in by_target.items():
        group.sort(key=lambda d: (-d.confidence, d.detection_id))
        local_kept: list[SymbolDetection] = []
        for det in group:
            if any(_bbox_iou(det.bbox_pdf, k.bbox_pdf) > iou_threshold for k in local_kept):
                continue
            local_kept.append(det)
        kept.extend(local_kept)
    kept.sort(key=lambda d: (d.page_index, d.target_key, d.bbox_pdf[1], d.bbox_pdf[0]))
    return kept


# ─────────────────────── text-tag matcher ──────────────────────────


def _text_tag_matches(
    *,
    page: Any,
    page_index: int,
    sheet_number: str | None,
    blocks: Sequence[Any],
    target_set: DetectionTargetSet,
    legend: ParsedLegend,
    excluded_bboxes: Sequence[tuple[float, float, float, float]],
) -> list[SymbolDetection]:
    """Find every legend-symbol token on the drawing body.

    A token is a candidate detection when:
      - the legend entry it came from is mapped to a target in
        ``target_set``,
      - the token's surrounding bbox does not intersect any
        ``excluded_bbox`` (legend region, title-block region), and
      - the token matches the legend's normalized symbol text with
        word-boundary matching.
    """
    out: list[SymbolDetection] = []
    entry_to_target = {
        t.legend_entry_id: t for t in target_set.targets if t.legend_entry_id
    }
    if not entry_to_target:
        return out
    symbol_to_entry: dict[str, Any] = {}
    for entry in legend.entries:
        if entry.entry_id not in entry_to_target:
            continue
        tok = (entry.normalized_symbol_text or "").upper()
        if tok and tok not in symbol_to_entry:
            symbol_to_entry[tok] = entry

    if not symbol_to_entry:
        return out

    for blk in blocks:
        if any(_bbox_iou(blk.bbox, ex) > 0.0 for ex in excluded_bboxes):
            continue
        tokens = _tokenize(blk.text)
        for token, start, end in tokens:
            tok_upper = token.upper()
            entry = symbol_to_entry.get(tok_upper)
            if entry is None:
                continue
            target = entry_to_target[entry.entry_id]
            if "text_tag" not in target.expected_modalities:
                continue
            sub_bbox = _interpolate_bbox(blk.bbox, len(blk.text), start, end)
            crop_hash = _crop_and_hash_page_region(page, sub_bbox)
            out.append(
                SymbolDetection.make(
                    page_index=page_index,
                    sheet_number=sheet_number,
                    target_key=target.target_key,
                    entity_key=target.entity_key,
                    legend_entry_id=entry.entry_id,
                    bbox_pdf=sub_bbox,
                    crop_sha256=crop_hash,
                    modality="text_tag",
                    confidence=0.9,
                    nearby_text=blk.text[: 80],
                )
            )
    return out


# ─────────────────────── glyph template matcher ────────────────────


def _glyph_template_matches(
    *,
    page: Any,
    page_index: int,
    sheet_number: str | None,
    target_set: DetectionTargetSet,
    legend: ParsedLegend,
    excluded_bboxes: Sequence[tuple[float, float, float, float]],
    legend_page: Any,
    threshold: float = 0.78,
) -> list[SymbolDetection]:
    """Match legend symbol crops against the drawing body with OpenCV.

    Skips silently when OpenCV / NumPy are unavailable. The match
    threshold is conservative; sub-threshold matches surface as
    nothing rather than as low-confidence detections, because the
    parser keeps the legend as authority and unknown candidates are
    expected to become ``unknown_symbol`` warnings from text-tag
    analysis instead.
    """
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        import fitz  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return []

    entry_to_target = {
        t.legend_entry_id: t for t in target_set.targets if t.legend_entry_id
    }
    if not entry_to_target:
        return []

    zoom = SCHEMATIC_REPLAY_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        page_pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
    except Exception:  # pragma: no cover
        return []
    page_arr = np.frombuffer(page_pix.samples, dtype=np.uint8).reshape(
        page_pix.height, page_pix.width, page_pix.n
    )
    page_gray = cv2.cvtColor(page_arr, cv2.COLOR_RGB2GRAY)

    out: list[SymbolDetection] = []

    for entry in legend.entries:
        if entry.entry_id not in entry_to_target:
            continue
        target = entry_to_target[entry.entry_id]
        if "glyph_template" not in target.expected_modalities:
            continue
        if entry.symbol_bbox_pdf is None:
            continue
        try:
            tmpl_pix = legend_page.get_pixmap(
                matrix=matrix,
                clip=fitz.Rect(*entry.symbol_bbox_pdf),
                alpha=False,
                colorspace=fitz.csRGB,
            )
        except Exception:
            continue
        tmpl = np.frombuffer(tmpl_pix.samples, dtype=np.uint8).reshape(
            tmpl_pix.height, tmpl_pix.width, tmpl_pix.n
        )
        tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_RGB2GRAY)
        if tmpl_gray.shape[0] < 4 or tmpl_gray.shape[1] < 4:
            continue
        if page_gray.shape[0] < tmpl_gray.shape[0] or page_gray.shape[1] < tmpl_gray.shape[1]:
            continue

        try:
            score_map = cv2.matchTemplate(page_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        except cv2.error:  # pragma: no cover
            continue
        ys, xs = np.where(score_map >= threshold)
        for y, x in sorted(zip(ys.tolist(), xs.tolist())):
            # Reject any match landing inside an excluded region.
            px_bbox_pt = (
                float(x) / zoom,
                float(y) / zoom,
                float(x + tmpl_gray.shape[1]) / zoom,
                float(y + tmpl_gray.shape[0]) / zoom,
            )
            if any(_bbox_iou(px_bbox_pt, ex) > 0.0 for ex in excluded_bboxes):
                continue
            crop_hash = _crop_and_hash_page_region(page, px_bbox_pt)
            score = float(score_map[y, x])
            out.append(
                SymbolDetection.make(
                    page_index=page_index,
                    sheet_number=sheet_number,
                    target_key=target.target_key,
                    entity_key=target.entity_key,
                    legend_entry_id=entry.entry_id,
                    bbox_pdf=px_bbox_pt,
                    crop_sha256=crop_hash,
                    modality="glyph_template",
                    confidence=min(0.99, score),
                )
            )
    return out


# ─────────────────────── public entry ──────────────────────────────


def detect_symbols(
    *,
    page: Any,
    page_index: int,
    sheet_number: str | None,
    blocks: Sequence[Any],
    target_set: DetectionTargetSet,
    legend: ParsedLegend,
    legend_page: Any | None = None,
    excluded_bboxes: Sequence[tuple[float, float, float, float]] = (),
    include_glyph: bool = True,
) -> list[SymbolDetection]:
    """Run all modalities and return a deterministic detection list.

    ``excluded_bboxes`` should include the legend's bbox on the
    legend page so the matcher does not "find" the legend rows
    themselves as detections on the drawing body.
    """
    text_hits = _text_tag_matches(
        page=page,
        page_index=page_index,
        sheet_number=sheet_number,
        blocks=blocks,
        target_set=target_set,
        legend=legend,
        excluded_bboxes=excluded_bboxes,
    )
    glyph_hits: list[SymbolDetection] = []
    if include_glyph and legend_page is not None:
        glyph_hits = _glyph_template_matches(
            page=page,
            page_index=page_index,
            sheet_number=sheet_number,
            target_set=target_set,
            legend=legend,
            excluded_bboxes=excluded_bboxes,
            legend_page=legend_page,
        )
    return _deterministic_nms([*text_hits, *glyph_hits])
