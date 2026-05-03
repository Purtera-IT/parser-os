"""Table-of-contents page detector.

A TOC page has a recognizable shape: a heading like "Contents" / "Table of
Contents" near the top, followed by a stack of numbered entries each ending
with a page number (e.g. ``1 INTRODUCTION ............ 3`` and
``1.1 Purpose ............ 3``).

When a page matches this shape we want to ignore *every* generic detection —
the contour-based text-section bands, mini-table grids, prose layout, and the
right-margin title-block fallback all produce noise on a TOC.  Instead we
emit a clean, structural overlay:

- One BLUE ``toc_heading`` band around the "Contents" heading.
- One BLUE ``toc_outer`` wrapper around the full list of entries.
- One BLUE ``toc_top_*_title`` band per top-level entry (``N TITLE``).
- One RED  ``toc_sub_*_title`` band per sub-level entry (``N.M TITLE``).

Detection is universal: only the page text shape is used (numbered prefix +
page-number suffix), no project-specific keywords.
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


_HEADING_RE = re.compile(r"\b(table\s+of\s+contents|contents)\b", re.I)
# "1 TITLE ... 3"  or  "1.1 TITLE ... 3"
_TOC_ENTRY_RE = re.compile(
    r"""
    ^\s*
    (?P<num>\d+(?:\.\d+){0,3})    # 1   or 1.1   or 1.1.1
    [\.\)\s]+
    (?P<label>.+?)
    \s*\.{0,}\s*
    (?P<page>\d+)\s*$
    """,
    re.X,
)


def _line_summary(line: dict) -> tuple[str, float, tuple[float, float, float, float]]:
    parts: list[str] = []
    max_size = 0.0
    bbox = line.get("bbox") or (0, 0, 0, 0)
    for sp in line.get("spans", []):
        t = (sp.get("text") or "").strip()
        if t:
            parts.append(t)
        try:
            s = float(sp.get("size", 0) or 0)
        except Exception:
            s = 0.0
        if s > max_size:
            max_size = s
    return " ".join(parts).strip(), max_size, tuple(bbox)


def _collect_lines(page) -> list[dict]:
    """Return a flat list of ``{text, size, bbox}`` lines top-to-bottom.

    PyMuPDF often emits a single visual row of text as several ``Line``
    entries (a numeric prefix, the label, and a trailing page number can each
    live on their own ``Line`` with slightly different y-mid).  We cluster
    lines whose y-mids agree within ~4pt into one logical row, then sort
    contributors left-to-right and concatenate their text so the row reads as
    "1.1 PURPOSE 3" instead of three orphaned tokens.
    """
    raw: list[dict] = []
    td = page.get_text("dict")
    for block in td.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for ln in block.get("lines", []):
            text, size, bbox = _line_summary(ln)
            if not text:
                continue
            x0, y0, x1, y1 = bbox
            raw.append({
                "text": text,
                "size": float(size),
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
            })
    if not raw:
        return []
    raw.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    clusters: list[list[dict]] = []
    for ln in raw:
        cy = 0.5 * (ln["bbox"][1] + ln["bbox"][3])
        placed = False
        for cluster in clusters:
            cy_mid = 0.5 * (
                min(p["bbox"][1] for p in cluster)
                + max(p["bbox"][3] for p in cluster)
            )
            if abs(cy - cy_mid) <= 4.0:
                cluster.append(ln)
                placed = True
                break
        if not placed:
            clusters.append([ln])
    out: list[dict] = []
    for cluster in clusters:
        cluster.sort(key=lambda r: r["bbox"][0])
        text = " ".join(p["text"] for p in cluster).strip()
        size = max(p["size"] for p in cluster)
        ux0 = min(p["bbox"][0] for p in cluster)
        uy0 = min(p["bbox"][1] for p in cluster)
        ux1 = max(p["bbox"][2] for p in cluster)
        uy1 = max(p["bbox"][3] for p in cluster)
        out.append({
            "text": text,
            "size": size,
            "bbox": (ux0, uy0, ux1, uy1),
        })
    out.sort(key=lambda r: r["bbox"][1])
    return out


def _is_toc_page(lines: list[dict]) -> tuple[bool, int | None]:
    """Return (is_toc, heading_line_index_or_None)."""
    if not lines:
        return False, None
    heading_idx: int | None = None
    for i, ln in enumerate(lines[:8]):  # heading must be near the top
        if _HEADING_RE.fullmatch(ln["text"].strip()) or _HEADING_RE.search(
            ln["text"].strip()
        ) and len(ln["text"].strip()) <= 24:
            heading_idx = i
            break
    entry_count = sum(1 for ln in lines if _TOC_ENTRY_RE.match(ln["text"]))
    if entry_count >= 4 and heading_idx is not None:
        return True, heading_idx
    if entry_count >= 6:
        return True, heading_idx
    return False, None


def _classify_entry(text: str) -> tuple[str | None, str | None, str | None]:
    """Return (level, num, label) where level is "TOP" / "SUB" / None."""
    m = _TOC_ENTRY_RE.match(text)
    if not m:
        return None, None, None
    num = m.group("num")
    label = m.group("label").strip()
    if "." in num:
        return "SUB", num, label
    return "TOP", num, label


def _mk_box(
    *,
    box_id: str,
    px_bbox: tuple[int, int, int, int],
    scale: float,
    color: str = "BLUE",
    synthetic: bool = True,
) -> VisibleBox:
    x0, y0, x1, y1 = px_bbox
    return VisibleBox(
        box_id=box_id,
        rect=Rect(x0 / scale, y0 / scale, x1 / scale, y1 / scale),
        area_pt2=float(max(0, x1 - x0) * max(0, y1 - y0)) / max(scale * scale, 1e-6),
        fill_ratio=1.0,
        nested_depth=1,
        is_outer_wrapper=False,
        parent_box_id=None,
        color=color,
        px_bbox=px_bbox,
        children_count=0,
        synthetic=synthetic,
    )


def detect_toc_boxes(
    *,
    pdf_path: str,
    page_index: int,
    scale: float,
    cw_quarter_turns: int,
) -> list[VisibleBox]:
    if _fitz is None:
        return []
    doc = _fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        page_w_pt = float(page.rect.width)
        page_h_pt = float(page.rect.height)
        lines = _collect_lines(page)
        is_toc, heading_idx = _is_toc_page(lines)
        if not is_toc:
            return []

        cw = cw_quarter_turns % 4
        boxes: list[VisibleBox] = []

        if heading_idx is not None:
            ln = lines[heading_idx]
            x0, y0, x1, y1 = _pdf_bbox_to_image_bbox(
                ln["bbox"], page_w_pt, page_h_pt, scale, cw,
            )
            pad_y = 8
            pad_x = 8
            heading_box = _mk_box(
                box_id="toc_heading_title",
                px_bbox=(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y),
                scale=scale,
                color="BLUE",
            )
            boxes.append(heading_box)

        entry_indices: list[int] = []
        for i, ln in enumerate(lines):
            if heading_idx is not None and i == heading_idx:
                continue
            if _TOC_ENTRY_RE.match(ln["text"]):
                entry_indices.append(i)
        if not entry_indices:
            return boxes

        # Outer wrapper around the union of all entry rows.
        union_bbox_pt = (
            min(lines[i]["bbox"][0] for i in entry_indices),
            min(lines[i]["bbox"][1] for i in entry_indices),
            max(lines[i]["bbox"][2] for i in entry_indices),
            max(lines[i]["bbox"][3] for i in entry_indices),
        )
        ux0, uy0, ux1, uy1 = _pdf_bbox_to_image_bbox(
            union_bbox_pt, page_w_pt, page_h_pt, scale, cw,
        )
        outer_pad = 12
        # Stretch outer wrapper to a full-width-ish band: match the widest
        # entry but expand horizontally by a small pad so the TOC reads as a
        # single column unit.
        boxes.append(_mk_box(
            box_id="toc_outer",
            px_bbox=(ux0 - outer_pad, uy0 - outer_pad, ux1 + outer_pad, uy1 + outer_pad),
            scale=scale,
            color="BLUE",
            synthetic=False,
        ))

        # Per-entry rows.
        for entry_idx, line_idx in enumerate(entry_indices):
            ln = lines[line_idx]
            level, _num, _label = _classify_entry(ln["text"])
            if level is None:
                continue
            ex0, ey0, ex1, ey1 = _pdf_bbox_to_image_bbox(
                ln["bbox"], page_w_pt, page_h_pt, scale, cw,
            )
            pad_y = 4
            pad_x = 6
            row_bbox = (ex0 - pad_x, ey0 - pad_y, ex1 + pad_x, ey1 + pad_y)
            if level == "TOP":
                bx = _mk_box(
                    box_id=f"toc_top_{entry_idx}_title",
                    px_bbox=row_bbox,
                    scale=scale,
                    color="BLUE",
                )
                boxes.append(bx)
            else:  # SUB
                bx = _mk_box(
                    box_id=f"toc_sub_{entry_idx}_title",
                    px_bbox=row_bbox,
                    scale=scale,
                    color="BLUE",
                )
                # Re-use the existing red-band channel from the prose pass.
                object.__setattr__(bx, "subhdr_red_band", True)
                boxes.append(bx)

        return boxes
    finally:
        doc.close()


def _strip_for_toc(boxes: list[VisibleBox]) -> list[VisibleBox]:
    """Drop noisy boxes that obscure a TOC layout.

    Removes:
    - text-section blue outlines (``textsec_*``)
    - mini-table cells / rows / tables (``minitable_*``, ``mtcell*``, ``mtrow*``)
    - title-block right-margin noise (``tbstruct_*``, ``tbtext_*``, ``tbcell_*``,
      ``tbgroup_*``)
    - prose layout bands (``prose*``)
    - raw column-header rings (``colhdr_*``)
    """
    out: list[VisibleBox] = []
    for b in boxes:
        bid = b.box_id or ""
        if bid.startswith((
            "textsec_",
            "minitable_",
            "tbstruct_",
            "tbtext_",
            "tbcell_",
            "tbgroup_",
            "prose",
            "colhdr_",
        )):
            continue
        if "_mtcell" in bid or "_mtrow" in bid or "_mtable" in bid:
            continue
        out.append(b)
    return out


@dataclass
class TocLayoutBandsPass:
    """Table-of-contents specific overlay producer."""

    info: PassInfo = PassInfo(
        name="toc_layout_bands",
        stage="synthesize",
        layer_flag="BLUE_TITLE",
        order=237,
        description=(
            "Detect Table-of-Contents pages by heading + numbered/page-number "
            "entries and emit clean blue heading + per-entry blue (top-level) "
            "and red (sub-level) overlays, suppressing noisy generic passes."
        ),
    )

    def run(self, ctx: PageContext, state: PipelineState) -> PipelineState:
        if state.result is None or state.rgb is None:
            return state
        result = state.result
        scale = float(
            result.debug_stats.get("render_scale_used")
            or ctx.cfg.render_scale
            or 1.0
        )
        cw = int(result.debug_stats.get("rotated_cw_quarter_turns") or 0)
        toc_boxes = detect_toc_boxes(
            pdf_path=str(ctx.pdf_path),
            page_index=ctx.page_index,
            scale=scale,
            cw_quarter_turns=cw,
        )
        if not toc_boxes:
            return state
        kept = _strip_for_toc(result.boxes)
        new_result = VisibleBoxResult(
            boxes=[*kept, *toc_boxes],
            image_width=result.image_width,
            image_height=result.image_height,
            debug_stats={
                **dict(result.debug_stats or {}),
                "toc_layout_bands": len(toc_boxes),
            },
        )
        state.result = new_result
        state.artifacts.setdefault("stage_order", []).append(self.info.name)
        return state


__all__ = ["TocLayoutBandsPass", "detect_toc_boxes"]
