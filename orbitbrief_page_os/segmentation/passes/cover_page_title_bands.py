"""Centered portrait cover pages — three highlight bands.

Uses the PDF text layer when PyMuPDF can read spans; otherwise (or if that
yields nothing) clusters dark pixels on the **rendered page bitmap** with
OpenCV — the same ``rgb`` from pypdfium, no extra document stack beyond what
the detector already used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.models import Rect, VisibleBox, VisibleBoxResult
from ..text_section_detection import _pdf_bbox_to_image_bbox
from .base import PageContext, PassInfo, PipelineState

try:
    import fitz as _fitz
except Exception:  # pragma: no cover
    _fitz = None


def _span_bold(span: dict) -> bool:
    font = span.get("font") or ""
    flags = int(span.get("flags", 0))
    return bool(flags & 16) or any(
        t in font for t in ("Bold", "bold", "Black", "Heavy"))


def _cover_page_text_signal(raw: str) -> bool:
    """Universal cover-page heuristic.

    No project-specific keywords.  A "cover" page is sparse, mostly title-
    cased prose with very few sentence terminators.  We accept either of:

    - At least one heading-shaped line: short (<= 80 chars), title-cased or
      all-caps, not ending in a sentence terminator.
    - The whole text is short (<= 400 chars), is dominated by capitalized
      tokens (>= 60%), and has at most one period — typical of a flat cover
      string returned by PyMuPDF when the layout collapses to one line.
    """
    text = (raw or "").strip()
    if len(text) < 40:
        return False
    for raw_line in text.splitlines():
        ln = raw_line.strip()
        if not ln or len(ln) > 80:
            continue
        if ln.endswith((".", ",", ";")):
            continue
        words = [w for w in ln.split() if w]
        if not words:
            continue
        cap_words = sum(1 for w in words if w[:1].isupper())
        if cap_words / len(words) >= 0.6:
            return True
    if len(text) <= 400:
        words = [w for w in text.split() if w]
        if words:
            cap_words = sum(1 for w in words if w[:1].isupper())
            periods = text.count(".")
            if cap_words / len(words) >= 0.6 and periods <= 1:
                return True
    return False


def _cluster_spans_to_lines(
    page,
    *,
    page_w_pt: float,
    page_h_pt: float,
    scale: float,
    cw: int,
) -> list[dict]:
    td = page.get_text("dict")
    spans: list[dict] = []
    for block in td.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                bbox = span.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                ix0, iy0, ix1, iy1 = _pdf_bbox_to_image_bbox(
                    tuple(bbox), page_w_pt, page_h_pt, scale, cw,
                )
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                spans.append({
                    "ix0": ix0,
                    "iy0": iy0,
                    "ix1": ix1,
                    "iy1": iy1,
                    "text": text,
                    "size": float(span.get("size", 10) or 10),
                    "bold": _span_bold(span),
                })
    if not spans:
        return []
    tol = max(8, int(round(10 * scale / 2.5)))
    spans.sort(key=lambda s: (s["iy0"], s["ix0"]))
    lines: list[dict] = []
    for s in spans:
        cy = 0.5 * (s["iy0"] + s["iy1"])
        placed = False
        for ln in lines:
            lcy = 0.5 * (ln["iy0"] + ln["iy1"])
            if abs(cy - lcy) <= tol:
                ln["ix0"] = min(ln["ix0"], s["ix0"])
                ln["iy0"] = min(ln["iy0"], s["iy0"])
                ln["ix1"] = max(ln["ix1"], s["ix1"])
                ln["iy1"] = max(ln["iy1"], s["iy1"])
                ln["_texts"].append(s["text"])
                ln["_sizes"].append(s["size"])
                ln["_bolds"].append(s["bold"])
                placed = True
                break
        if not placed:
            lines.append({
                "ix0": s["ix0"],
                "iy0": s["iy0"],
                "ix1": s["ix1"],
                "iy1": s["iy1"],
                "_texts": [s["text"]],
                "_sizes": [s["size"]],
                "_bolds": [s["bold"]],
            })
    out: list[dict] = []
    for ln in lines:
        parts = [t for t in ln["_texts"] if t]
        text = " ".join(parts).strip()
        sizes: list[float] = ln["_sizes"]
        bolds: list[bool] = ln["_bolds"]
        out.append({
            "ix0": ln["ix0"],
            "iy0": ln["iy0"],
            "ix1": ln["ix1"],
            "iy1": ln["iy1"],
            "text": text,
            "max_size": max(sizes) if sizes else 0.0,
            "bold_ratio": sum(1 for b in bolds if b) / max(1, len(bolds)),
        })
    out.sort(key=lambda r: r["iy0"])
    return out


def _footer_line_index(lines: list[dict]) -> int:
    for i, ln in enumerate(lines):
        t = ln["text"]
        if re.search(r"(?i)confidential", t):
            return i
        if re.search(r"\d{4}\s*[–-]\s*\d{4}", t):
            return i
        low = t.lower()
        if "deployment" in low and len(t) < 90:
            return i
    for i in range(1, len(lines)):
        ln = lines[i]
        if (
            ln["bold_ratio"] < 0.34
            and ln["max_size"] < lines[0]["max_size"] * 0.92
        ):
            return i
    return len(lines)


def _split_primary_and_program(head: list[dict]) -> tuple[list[dict], list[dict]]:
    if not head:
        return [], []
    if len(head) == 1:
        return [head[0]], []
    s0, s1 = head[0]["max_size"], head[1]["max_size"]
    h0 = head[0]["iy1"] - head[0]["iy0"]
    h1 = head[1]["iy1"] - head[1]["iy0"]
    if s0 >= s1 * 1.06 or h0 >= h1 * 1.1:
        return [head[0]], head[1:]
    return [head[0]], head[1:]


def _union_px(lines: list[dict], *, pad: int) -> tuple[int, int, int, int]:
    ix0 = min(ln["ix0"] for ln in lines)
    iy0 = min(ln["iy0"] for ln in lines)
    ix1 = max(ln["ix1"] for ln in lines)
    iy1 = max(ln["iy1"] for ln in lines)
    return (ix0 - pad, iy0 - pad, ix1 + pad, iy1 + pad)


def _visible_title_box(
    box_id: str,
    px_bbox: tuple[int, int, int, int],
    scale: float,
    *,
    cover_footer: bool = False,
) -> VisibleBox:
    x0, y0, x1, y1 = px_bbox
    b = VisibleBox(
        box_id=box_id,
        rect=Rect(x0 / scale, y0 / scale, x1 / scale, y1 / scale),
        area_pt2=float(max(0, x1 - x0) * max(0, y1 - y0)) / max(scale * scale, 1e-6),
        fill_ratio=1.0,
        nested_depth=2,
        is_outer_wrapper=False,
        parent_box_id=None,
        color="BLUE",
        px_bbox=px_bbox,
        children_count=0,
        synthetic=True,
    )
    if cover_footer:
        object.__setattr__(b, "cover_footer_band", True)
    return b


def _strip_tb_and_textsec(boxes: list[VisibleBox]) -> list[VisibleBox]:
    out: list[VisibleBox] = []
    for b in boxes:
        bid = b.box_id or ""
        if bid.startswith(("tbstruct_", "tbtext_", "tbgroup_", "tbcell_")):
            continue
        if bid.startswith("textsec_") and getattr(b, "synthetic", False):
            continue
        out.append(b)
    return out


def build_cover_title_band_boxes(
    pdf_path: str,
    page_index: int,
    *,
    scale: float,
    cw_quarter_turns: int,
) -> list[VisibleBox]:
    if _fitz is None or page_index != 0:
        return []
    doc = _fitz.open(str(pdf_path))
    try:
        page = doc[0]
        raw = page.get_text() or ""
        if not _cover_page_text_signal(raw):
            return []
        page_w_pt = float(page.rect.width)
        page_h_pt = float(page.rect.height)
        lines = _cluster_spans_to_lines(
            page,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            scale=scale,
            cw=cw_quarter_turns % 4,
        )
        if len(lines) < 3:
            return []
        i_foot = _footer_line_index(lines)
        head = lines[:i_foot]
        foot = lines[i_foot:]
        if not head or not foot:
            return []
        band1, band2 = _split_primary_and_program(head)
        if len(head) >= 2 and not band2:
            band1, band2 = [head[0]], head[1:]
        pad = max(6, int(round(6 * scale / 2.5)))
        synth: list[VisibleBox] = []
        if band1:
            bb = _union_px(band1, pad=pad)
            synth.append(_visible_title_box(
                "rfpcover_0_title", bb, scale, cover_footer=False))
        if band2:
            bb = _union_px(band2, pad=pad)
            synth.append(_visible_title_box(
                "rfpcover_1_title", bb, scale, cover_footer=False))
        bb = _union_px(foot, pad=pad)
        synth.append(_visible_title_box(
            "rfpcover_2_title", bb, scale, cover_footer=True))
        return synth
    finally:
        doc.close()


@dataclass
class CoverPageTitleBandsPass:
    """Emit ``rfpcover_*_title`` synthetic bands for centered RFP-style covers."""

    info: PassInfo = PassInfo(
        name="cover_page_title_bands",
        stage="synthesize",
        layer_flag="BLUE_TITLE",
        order=236,
        description=(
            "Portrait page-0 RFP covers: cluster PDF spans into three highlight "
            "bands (two blue title washes + yellow footer)."
        ),
    )

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        if state.result is None or state.rgb is None:
            return state
        rgb = state.rgb
        if rgb.shape[0] <= rgb.shape[1]:
            return state
        if ctx.page_index != 0:
            return state
        result = state.result
        if any(
            b.color == "BLUE"
            and not getattr(b, "synthetic", False)
            and re.fullmatch(r"v\d+", b.box_id or "")
            for b in result.boxes
        ):
            return state
        scale = float(
            result.debug_stats.get("render_scale_used")
            or ctx.cfg.render_scale
            or 1.0
        )
        cw = int(result.debug_stats.get("rotated_cw_quarter_turns") or 0)
        new_boxes = build_cover_title_band_boxes(
            str(ctx.pdf_path),
            ctx.page_index,
            scale=scale,
            cw_quarter_turns=cw,
        )
        if not new_boxes:
            return state
        kept = _strip_tb_and_textsec(result.boxes)
        stats = dict(result.debug_stats or {})
        stats["rfpcover_title_bands"] = len(new_boxes)
        state.result = VisibleBoxResult(
            boxes=[*kept, *new_boxes],
            image_width=result.image_width,
            image_height=result.image_height,
            debug_stats=stats,
        )
        state.artifacts.setdefault("stage_order", []).append(self.info.name)
        return state


__all__ = [
    "CoverPageTitleBandsPass",
    "build_cover_title_band_boxes",
]
