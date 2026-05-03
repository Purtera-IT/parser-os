"""Line-level aggregates from PyMuPDF ``get_text('dict')`` (size / bold / color hints)."""

from __future__ import annotations

import re
from typing import Any


def _rgb_from_int(c: int | None) -> list[int] | None:
    if c is None:
        return None
    try:
        ci = int(c)
    except (TypeError, ValueError):
        return None
    if ci < 0:
        return None
    r = (ci >> 16) & 255
    g = (ci >> 8) & 255
    b = ci & 255
    return [r, g, b]


def _span_bold(span: dict[str, Any]) -> bool:
    fn = str(span.get("font") or "").lower()
    if "bold" in fn or "heavy" in fn or "black" in fn:
        return True
    try:
        fl = int(span.get("flags") or 0)
    except (TypeError, ValueError):
        return False
    return bool(fl & 16)


def extract_pdf_span_line_signals(page: Any) -> list[dict[str, Any]]:
    """One record per text line in PDF reading order (span dict merged to a line).

    Coordinates are **PDF points** (not detector px). Consumers can align with
    overlay later using ``rotated_cw`` + scale transforms.
    """
    try:
        raw = page.get_text("dict")
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for block in raw.get("blocks") or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            if not spans:
                continue
            parts: list[str] = []
            sizes: list[float] = []
            bolds: list[bool] = []
            colors: list[int] = []
            for sp in spans:
                t = str(sp.get("text") or "")
                if not t:
                    continue
                parts.append(t)
                try:
                    sizes.append(float(sp.get("size") or 0.0))
                except (TypeError, ValueError):
                    sizes.append(0.0)
                bolds.append(_span_bold(sp))
                try:
                    colors.append(int(sp.get("color") or 0))
                except (TypeError, ValueError):
                    colors.append(0)
            text = "".join(parts).strip()
            text = re.sub(r"[\s\u00a0]+", " ", text)
            if not text:
                continue
            bbox = line.get("bbox") or [0, 0, 0, 0]
            max_sz = max(sizes) if sizes else 0.0
            bold_ratio = sum(1 for b in bolds if b) / max(len(bolds), 1)
            color0 = colors[0] if colors else None
            out.append(
                {
                    "text": text,
                    "bbox_pdf": [float(x) for x in bbox],
                    "font_size_pt_max": round(max_sz, 2),
                    "bold_span_ratio": round(bold_ratio, 3),
                    "color_rgb_sample": _rgb_from_int(color0),
                }
            )
    out.sort(key=lambda r: (r["bbox_pdf"][1], r["bbox_pdf"][0]))
    return out


def match_title_to_span_style(
    title: str, span_lines: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Best-effort: find a span line whose text contains the section ``title``."""
    if not title or not span_lines:
        return None
    tcf = title.casefold().strip()
    if len(tcf) < 3:
        return None
    best: dict[str, Any] | None = None
    best_len = 10**9
    for row in span_lines:
        tx = str(row.get("text") or "").strip()
        if not tx:
            continue
        tcf2 = tx.casefold()
        if tcf in tcf2 or tcf2 in tcf:
            ln = len(tx)
            if ln < best_len:
                best_len = ln
                best = {
                    "matched_span_text": tx[:240],
                    "font_size_pt_max": row.get("font_size_pt_max"),
                    "bold_span_ratio": row.get("bold_span_ratio"),
                    "color_rgb_sample": row.get("color_rgb_sample"),
                    "bbox_pdf": row.get("bbox_pdf"),
                }
    return best
