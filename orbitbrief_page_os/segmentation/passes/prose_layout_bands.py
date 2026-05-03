"""Fitz-driven prose layout overlays for RFP narrative pages.

When contour detection finds nothing and title-block raster fallback starts
boxing right-edge word snippets, use PDF text blocks directly:

- Big section titles (font size >= 13 + bold) -> BLUE title highlight
- Bold sub-headers (small bold lines)         -> RED title highlight
- Bullet-list runs (consecutive bullet blocks merged) -> single YELLOW
  highlight, optionally wrapped together with the preceding sub-header /
  colon intro by an outer BLUE wrapper.
- Plain prose paragraphs                      -> BLUE wrapper outline
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


def _is_v_wrapper(b: VisibleBox) -> bool:
    return (
        b.color == "BLUE"
        and not getattr(b, "synthetic", False)
        and re.fullmatch(r"v\d+", b.box_id or "")
    )


def _is_v_box(b: VisibleBox) -> bool:
    """Match ANY contour ``vN`` box regardless of color.

    The blue ``v0`` outer wrapper sometimes covers only part of the
    detected table (e.g. left column) while the orange ``v1..vN`` cell
    rectangles fill out the rest.  When suppressing prose detection
    over a table region, we want to honour ALL of them, not just the
    wrapper, so right-column cell text doesn't leak into a paragraph
    box that overruns the table.
    """
    return (
        not getattr(b, "synthetic", False)
        and bool(re.fullmatch(r"v\d+", b.box_id or ""))
    )


def _fit_outer_wrappers_to_children(result: VisibleBoxResult) -> VisibleBoxResult:
    """Expand BLUE outer wrappers to actually enclose all of their children.

    The contour pass marks some ``vN`` box as the table's outer wrapper but
    occasionally computes a bbox that is narrower than the union of its
    child cells.  Drawing that wrapper produces a blue outline whose right
    (or bottom) edge cuts through cell content.  This helper rebuilds any
    such wrapper with the union of its own bbox and every child box's
    bbox, so the wrapper visually contains them.

    Children are detected by ``parent_box_id`` matching the wrapper's
    ``box_id``.  Other fields (color, synthetic, depth) are preserved.
    """
    boxes = list(result.boxes or [])
    if not boxes:
        return result
    children_by_parent: dict[str, list[VisibleBox]] = {}
    for b in boxes:
        pid = b.parent_box_id or ""
        if pid:
            children_by_parent.setdefault(pid, []).append(b)
    changed = False
    for idx, b in enumerate(boxes):
        if not getattr(b, "is_outer_wrapper", False):
            continue
        children = children_by_parent.get(b.box_id or "", [])
        if not children:
            continue
        x0, y0, x1, y1 = b.px_bbox
        ux0, uy0, ux1, uy1 = x0, y0, x1, y1
        for c in children:
            cx0, cy0, cx1, cy1 = c.px_bbox
            ux0 = min(ux0, cx0)
            uy0 = min(uy0, cy0)
            ux1 = max(ux1, cx1)
            uy1 = max(uy1, cy1)
        if (ux0, uy0, ux1, uy1) == (x0, y0, x1, y1):
            continue
        # Re-create the wrapper with the union bbox.  ``Rect`` is in PDF
        # points (px / scale); recompute it from the new pixel bbox via
        # the existing rect's scaling factor to keep both views aligned.
        old_x0 = float(b.px_bbox[0])
        old_rect_x0 = float(b.rect.x0)
        scale = (old_x0 / old_rect_x0) if old_rect_x0 > 0 else 1.0
        if scale <= 0:
            scale = 1.0
        new_rect = Rect(ux0 / scale, uy0 / scale, ux1 / scale, uy1 / scale)
        boxes[idx] = VisibleBox(
            box_id=b.box_id,
            rect=new_rect,
            area_pt2=float(max(0, ux1 - ux0) * max(0, uy1 - uy0)) / max(scale * scale, 1e-6),
            fill_ratio=b.fill_ratio,
            nested_depth=b.nested_depth,
            is_outer_wrapper=b.is_outer_wrapper,
            parent_box_id=b.parent_box_id,
            color=b.color,
            px_bbox=(int(ux0), int(uy0), int(ux1), int(uy1)),
            children_count=b.children_count,
            synthetic=getattr(b, "synthetic", False),
        )
        changed = True
    if not changed:
        return result
    return VisibleBoxResult(
        boxes=boxes,
        image_width=result.image_width,
        image_height=result.image_height,
        debug_stats=result.debug_stats,
    )


def _is_tb_noise(b: VisibleBox) -> bool:
    bid = b.box_id or ""
    return bid.startswith(("tbstruct_", "tbtext_", "tbgroup_", "tbcell_"))


def _is_textsec(b: VisibleBox) -> bool:
    bid = b.box_id or ""
    return bid.startswith("textsec_")


def _bbox_inside(inner: tuple[int, int, int, int],
                 outer: tuple[int, int, int, int],
                 slack: int = 8) -> bool:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return (
        ix0 >= ox0 - slack
        and iy0 >= oy0 - slack
        and ix1 <= ox1 + slack
        and iy1 <= oy1 + slack
    )


def _bbox_overlaps_any(target: tuple[int, int, int, int],
                       regions: list[tuple[int, int, int, int]]) -> bool:
    tx0, ty0, tx1, ty1 = target
    for rx0, ry0, rx1, ry1 in regions:
        if tx1 <= rx0 or rx1 <= tx0:
            continue
        if ty1 <= ry0 or ry1 <= ty0:
            continue
        return True
    return False


_SUB_BULLET_GLYPHS = {"o", "\u25aa", "▪", "\u00ba"}
_SUB_SUB_BULLET_GLYPHS = {"\u25aa", "▪"}


def _bullet_rows_in_block(block: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Group block lines into bullet rows.

    PyMuPDF emits the bullet glyph on its own ``Line`` and the text on a
    sibling ``Line`` sharing roughly the same y range.  Returns a list of
    ``(glyph, union_bbox_pdf_pts)`` tuples in top-to-bottom order.
    """
    lines: list[tuple[float, float, float, float, str]] = []
    for ln in block.get("lines", []):
        bbox = ln.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        text_parts: list[str] = []
        for sp in ln.get("spans", []):
            t = (sp.get("text") or "").strip()
            if t:
                text_parts.append(t)
        text = " ".join(text_parts).strip()
        if not text:
            continue
        x0, y0, x1, y1 = bbox
        lines.append((float(x0), float(y0), float(x1), float(y1), text))
    if not lines:
        return []
    # Cluster lines into rows using y midpoints (~3pt slack).
    lines.sort(key=lambda r: (r[1], r[0]))
    rows: list[list[tuple[float, float, float, float, str]]] = []
    for ln in lines:
        cy = 0.5 * (ln[1] + ln[3])
        placed = False
        for row in rows:
            ry_mid = 0.5 * (
                min(p[1] for p in row) + max(p[3] for p in row)
            )
            if abs(cy - ry_mid) <= 3.0:
                row.append(ln)
                placed = True
                break
        if not placed:
            rows.append([ln])
    rich: list[dict] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda r: r[0])
        leftmost_text = row_sorted[0][4].strip()
        # A sub-bullet ``o`` / ``▪`` is emitted by PyMuPDF as a SOLO line
        # whose entire text is the glyph itself.  Without this guard, any
        # wrapped line whose first letter happens to be ``o`` (e.g. "other
        # aspect of the proposal …") would masquerade as a sub-bullet and
        # get a green box.
        if len(leftmost_text) <= 2:
            glyph_text = leftmost_text[:1]
        else:
            glyph_text = ""
        glyph_x0 = float(row_sorted[0][0])
        # text column x: the x of the next line on the same baseline (the
        # span sitting to the right of the glyph), else the glyph's own x.
        if len(row_sorted) >= 2:
            text_x0 = float(row_sorted[1][0])
        else:
            text_x0 = float(row_sorted[0][2]) + 4.0
        ux0 = min(p[0] for p in row)
        uy0 = min(p[1] for p in row)
        ux1 = max(p[2] for p in row)
        uy1 = max(p[3] for p in row)
        rich.append({
            "glyph": glyph_text,
            "glyph_x0": glyph_x0,
            "text_x0": text_x0,
            "bbox": (float(ux0), float(uy0), float(ux1), float(uy1)),
        })
    rich.sort(key=lambda r: r["bbox"][1])

    # Merge wrapped continuation rows into the previous bullet row: a row
    # whose leftmost x sits at (or to the right of) the previous row's
    # text column AND that does not begin with a bullet glyph belongs to
    # the same logical bullet line.
    merged: list[dict] = []
    for r in rich:
        if merged:
            prev = merged[-1]
            is_bullet_glyph = r["glyph"] in _SUB_BULLET_GLYPHS or r["glyph"] in ("•", "\u2022")
            indented_to_text_col = r["bbox"][0] >= prev["text_x0"] - 4.0
            if not is_bullet_glyph and indented_to_text_col:
                px0, py0, px1, py1 = prev["bbox"]
                rx0, ry0, rx1, ry1 = r["bbox"]
                prev["bbox"] = (
                    min(px0, rx0), min(py0, ry0),
                    max(px1, rx1), max(py1, ry1),
                )
                continue
        merged.append(r)

    return [(r["glyph"], r["bbox"]) for r in merged]


def _block_summary(block: dict) -> tuple[str, float, float]:
    """Return (text, max_font_size, bold_line_ratio) for a fitz block."""
    parts: list[str] = []
    bold_lines = 0
    total_lines = 0
    max_size = 0.0
    for ln in block.get("lines", []):
        line_text_parts: list[str] = []
        line_bold = False
        for sp in ln.get("spans", []):
            t = (sp.get("text") or "").strip()
            if not t:
                continue
            line_text_parts.append(t)
            try:
                size = float(sp.get("size", 0) or 0)
            except Exception:
                size = 0.0
            if size > max_size:
                max_size = size
            font = sp.get("font", "") or ""
            flags = int(sp.get("flags", 0))
            if (flags & 16) or any(
                tag in font for tag in ("Bold", "bold", "Black", "Heavy")
            ):
                line_bold = True
        if line_text_parts:
            parts.append(" ".join(line_text_parts))
            total_lines += 1
            if line_bold:
                bold_lines += 1
    text = " ".join(parts).strip()
    bold_ratio = (bold_lines / total_lines) if total_lines else 0.0
    return text, max_size, bold_ratio


_BULLET_GLYPHS = ("•", "\u2022", "▪", "\u25aa", "\ufffd")


_NUM_LIST_RE = re.compile(r"^\s*\d{1,3}\.\s+\S")
# Lettered list markers: ``A) Foo``, ``b) Bar``, ``A. Foo``.  Restrict to a
# single character followed by ``)`` or ``.`` so we don't catch words.
_ALPHA_LIST_RE = re.compile(r"^\s*[A-Za-z][\.\)]\s+\S")


def _is_bullet_text(text: str) -> bool:
    if not text:
        return False
    if any(g in text for g in _BULLET_GLYPHS):
        return True
    head = text.lstrip()[:2]
    if head in ("o ", "o\t"):
        return True
    # Treat numbered list items ("1. Foo", "2. Bar") as bullets so they
    # get the yellow highlight + intro group wrapper treatment.  This is
    # distinct from numbered SECTION headings ("1.1 Purpose"), which
    # ``_numbered_heading_depth`` matches via a stricter ``N(.N)+`` form.
    if _NUM_LIST_RE.match(text):
        return True
    # Lettered list items ("A) Foo", "B) Bar", "a. Baz") — same intent as
    # numbered list items, just an alphabetic counter.
    if _ALPHA_LIST_RE.match(text):
        return True
    return False


def _is_bullet_intro_text(text: str) -> bool:
    t = (text or "").strip()
    if not t.endswith(":"):
        return False
    return 12 <= len(t) <= 140


_NUM_PREFIX_RE = re.compile(r"^\s*(\d+)((?:\.\d+){0,3})\s+\S")


def _numbered_heading_depth(text: str) -> int | None:
    """Return depth of a leading section number ("1" → 0, "1.1" → 1, ...).

    Returns ``None`` when the line doesn't open with a section-number prefix.
    """
    m = _NUM_PREFIX_RE.match(text or "")
    if not m:
        return None
    return m.group(2).count(".")


def _classify(text: str, size: float, bold_ratio: float, body_size: float) -> str:
    """Return a row class: BIG_TITLE, SUB_HEADER, BULLET, INTRO, PARA, SKIP."""
    if not text:
        return "SKIP"
    if _is_bullet_text(text):
        return "BULLET"

    # Numbered section headings ("1 INTRODUCTION", "1.1 Purpose", etc.) have
    # the strongest signal — depth 0 (top-level) is BIG_TITLE, depth >= 1
    # (sub-section) is SUB_HEADER.  This dominates the size/bold check so a
    # 14pt "1.1 Purpose" doesn't get classified as a top-level title just
    # because it crosses the size threshold for the page.
    depth = _numbered_heading_depth(text)
    if depth is not None and bold_ratio >= 0.3 and len(text) <= 120:
        return "BIG_TITLE" if depth == 0 else "SUB_HEADER"

    big_title_size = max(13.0, body_size + 2.5)
    if size >= big_title_size and bold_ratio >= 0.5:
        return "BIG_TITLE"
    if (
        bold_ratio >= 0.5
        and len(text) <= 90
        and not text.endswith(":")
        and not _is_bullet_text(text)
        # Avoid mistaking inline "Label: value" content (contact info,
        # captions) for a section sub-header.  Short label-value lines
        # (≤30 chars, e.g. "Total Units : 323") are still treated as a
        # callout heading because they're typically standalone metadata
        # rows on cover sheets.
        and (": " not in text or len(text) <= 30)
    ):
        return "SUB_HEADER"
    if _is_bullet_intro_text(text):
        return "INTRO"
    if len(text) >= 30:
        return "PARA"
    return "SKIP"


def _mk_box(
    *,
    box_id: str,
    px_bbox: tuple[int, int, int, int],
    scale: float,
    synthetic: bool,
    color: str = "BLUE",
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


def _line_summary(line: dict) -> tuple[str, float, float, tuple[float, float, float, float]]:
    """Return (text, max_size, bold_ratio, pdf_bbox) for a single fitz line."""
    parts: list[str] = []
    bold_chars = 0
    total_chars = 0
    max_size = 0.0
    for sp in line.get("spans", []):
        t = sp.get("text") or ""
        stripped = t.strip()
        if not stripped:
            continue
        parts.append(stripped)
        try:
            size = float(sp.get("size", 0) or 0)
        except Exception:
            size = 0.0
        if size > max_size:
            max_size = size
        font = sp.get("font", "") or ""
        flags = int(sp.get("flags", 0))
        is_bold = (flags & 16) or any(
            tag in font for tag in ("Bold", "bold", "Black", "Heavy")
        )
        n = len(stripped)
        total_chars += n
        if is_bold:
            bold_chars += n
    text = " ".join(parts).strip()
    bold_ratio = (bold_chars / total_chars) if total_chars else 0.0
    bbox = tuple(line.get("bbox") or (0.0, 0.0, 0.0, 0.0))
    return text, max_size, bold_ratio, bbox


def _collect_prose_rows(
    page,
    *,
    page_w_pt: float,
    page_h_pt: float,
    scale: float,
    cw: int,
    excludes: list[tuple[int, int, int, int]],
) -> tuple[list[dict], dict[int, dict]]:
    """Return (rows, bullet_block_for_row).

    Each row is a single visual text line clustered with siblings on the
    same baseline (within ~4pt).  Working at line granularity (vs block
    granularity) is essential for mixed pages where fitz emits "1
    INTRODUCTION + 1.1 PURPOSE + body" as one giant block — line-level
    classification recovers per-line headings vs paragraphs.
    """
    raw: list[dict] = []
    line_to_block: list[dict] = []  # parallel to raw; block dict for each line
    td = page.get_text("dict")
    for block in td.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for ln in block.get("lines", []):
            text, max_size, bold_ratio, bbox_pt = _line_summary(ln)
            if not text:
                continue
            ix0, iy0, ix1, iy1 = _pdf_bbox_to_image_bbox(
                bbox_pt, page_w_pt, page_h_pt, scale, cw,
            )
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            if excludes and any(
                _bbox_inside((ix0, iy0, ix1, iy1), r) for r in excludes
            ):
                continue
            raw.append({
                "bbox": (ix0, iy0, ix1, iy1),
                "text": text,
                "size": max_size,
                "bold_ratio": bold_ratio,
                "x0_pt": float(bbox_pt[0]),
            })
            line_to_block.append(block)
    if not raw:
        return [], {}
    # Cluster lines into y-rows then merge text left-to-right.  ``slack``
    # is in IMAGE PIXELS (matching the bbox coordinate space) — we want
    # roughly 4 PDF points of tolerance regardless of render scale, so we
    # scale by ``scale``.  Without the scale factor a 2.5x render would
    # split a bullet glyph from its sibling text line by ~5–6 px.
    indices = list(range(len(raw)))
    indices.sort(key=lambda i: (raw[i]["bbox"][1], raw[i]["bbox"][0]))
    slack = max(4.0, scale * 4.0)
    clusters: list[list[int]] = []
    for i in indices:
        cy = 0.5 * (raw[i]["bbox"][1] + raw[i]["bbox"][3])
        placed = False
        for cluster in clusters:
            cy_mid = 0.5 * (
                min(raw[j]["bbox"][1] for j in cluster)
                + max(raw[j]["bbox"][3] for j in cluster)
            )
            if abs(cy - cy_mid) <= slack:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    rows: list[dict] = []
    for cluster in clusters:
        cluster.sort(key=lambda i: raw[i]["bbox"][0])
        text = " ".join(raw[i]["text"] for i in cluster).strip()
        if not text:
            continue
        max_size = max(raw[i]["size"] for i in cluster)
        bold_ratio = max(raw[i]["bold_ratio"] for i in cluster)
        ux0 = min(raw[i]["bbox"][0] for i in cluster)
        uy0 = min(raw[i]["bbox"][1] for i in cluster)
        ux1 = max(raw[i]["bbox"][2] for i in cluster)
        uy1 = max(raw[i]["bbox"][3] for i in cluster)
        # text_x0 = leftmost x of the second-or-later line in the cluster
        # (a bullet row has its glyph as the first line; the actual text
        # starts at the second).  For non-bullet rows this falls back to the
        # row's own leftmost x.
        xs_sorted = sorted({raw[i]["bbox"][0] for i in cluster})
        text_x0 = xs_sorted[1] if len(xs_sorted) >= 2 else xs_sorted[0]
        rows.append({
            "bbox": (ux0, uy0, ux1, uy1),
            "text": text,
            "size": max_size,
            "bold_ratio": bold_ratio,
            "x0_pt": min(raw[i]["x0_pt"] for i in cluster),
            "text_x0": text_x0,
            "line_h": uy1 - uy0,        # tracked for gap-threshold purposes
            "_first_block": line_to_block[cluster[0]],
        })
    rows.sort(key=lambda r: r["bbox"][1])

    # Continuation-line merge: a row that has no bullet glyph, sits at or
    # past the previous row's text column, has the same font size, and
    # follows it with only a tiny vertical gap (≈ one line height) is a
    # wrap continuation — merge it into the previous row.  This is what
    # makes "Retail locations." fold into the preceding paragraph and
    # "supporting all other …" fold into the last bullet line.
    #
    # The gap threshold is tied to the ORIGINAL single-line height of the
    # previous row (``line_h``), not to its merged bbox height — otherwise
    # each merge grows the prev box and the threshold balloons until
    # adjacent paragraphs and intro lines also get swallowed.
    merged: list[dict] = []
    for r in rows:
        if merged:
            prev = merged[-1]
            gap = r["bbox"][1] - prev["bbox"][3]
            same_size = abs(r["size"] - prev["size"]) <= 0.6
            indented_to_text_col = r["bbox"][0] >= prev["text_x0"] - 6
            no_numbered_prefix = _numbered_heading_depth(r["text"]) is None
            not_bullet = not _is_bullet_text(r["text"])
            # If the previous row is itself a numbered heading (e.g. "3.2.1
            # Sites in Scope"), do not let the body paragraph following it
            # collapse into the heading row — that turns the heading into a
            # PARA block and makes us lose the red sub-header highlight.
            prev_is_numbered_heading = (
                _numbered_heading_depth(prev["text"]) is not None
                and prev["bold_ratio"] >= 0.5
            )
            gap_limit = max(8, int(0.6 * prev["line_h"]))
            if (
                not_bullet
                and same_size
                and indented_to_text_col
                and no_numbered_prefix
                and not prev_is_numbered_heading
                and gap <= gap_limit
            ):
                px0, py0, px1, py1 = prev["bbox"]
                rx0, ry0, rx1, ry1 = r["bbox"]
                prev["bbox"] = (
                    min(px0, rx0), min(py0, ry0),
                    max(px1, rx1), max(py1, ry1),
                )
                prev["text"] = (prev["text"] + " " + r["text"]).strip()
                prev["bold_ratio"] = max(prev["bold_ratio"], r["bold_ratio"])
                continue
        merged.append(r)

    bullet_block_for_row: dict[int, dict] = {}
    final_rows: list[dict] = []
    for r in merged:
        idx = len(final_rows)
        if _is_bullet_text(r["text"]):
            bullet_block_for_row[idx] = r["_first_block"]
        # Drop the private ``_first_block`` field from the public row dict.
        final_rows.append({
            "bbox": r["bbox"],
            "text": r["text"],
            "size": r["size"],
            "bold_ratio": r["bold_ratio"],
            "x0_pt": r["x0_pt"],
            "text_x0": r["text_x0"],
        })
    return final_rows, bullet_block_for_row


def detect_prose_layout_boxes(
    *,
    pdf_path: str,
    page_index: int,
    scale: float,
    cw_quarter_turns: int,
    exclude_regions: list[tuple[int, int, int, int]] | None = None,
) -> tuple[list[VisibleBox], bool]:
    """Return (boxes, has_prose_signal).

    ``exclude_regions`` is a list of pixel bboxes (typically table wrappers)
    in which prose detection should be suppressed — any text line whose
    bbox falls inside one of those regions is treated as table content and
    skipped, so the contour-based table overlay remains the source of truth
    for that area.
    """
    if _fitz is None:
        return [], False
    excludes = list(exclude_regions or [])
    doc = _fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        page_w_pt = float(page.rect.width)
        page_h_pt = float(page.rect.height)

        rows, bullet_block_for_row = _collect_prose_rows(
            page,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            scale=scale,
            cw=cw_quarter_turns % 4,
            excludes=excludes,
        )

        if not rows:
            return [], False

        # Body font size = the most common non-bullet, non-bold size; used to
        # decide what counts as a "big" title vs a small bold sub-header.
        sizes = [r["size"] for r in rows if not _is_bullet_text(r["text"])]
        body_size = sorted(sizes)[len(sizes) // 2] if sizes else 10.0

        cls = [
            _classify(r["text"], r["size"], r["bold_ratio"], body_size)
            for r in rows
        ]

        # Group consecutive bullet rows into runs.
        bullet_runs: list[tuple[int, int, tuple[int, int, int, int]]] = []
        i = 0
        while i < len(rows):
            if cls[i] != "BULLET":
                i += 1
                continue
            start = i
            ux0, uy0, ux1, uy1 = rows[i]["bbox"]
            j = i + 1
            while j < len(rows) and cls[j] == "BULLET":
                bx0, by0, bx1, by1 = rows[j]["bbox"]
                # Only merge when vertically continuous (gap <= 22px).
                prev_y1 = rows[j - 1]["bbox"][3]
                if (by0 - prev_y1) > 22:
                    break
                ux0 = min(ux0, bx0)
                uy0 = min(uy0, by0)
                ux1 = max(ux1, bx1)
                uy1 = max(uy1, by1)
                j += 1
            bullet_runs.append((start, j - 1, (ux0, uy0, ux1, uy1)))
            i = j

        # Identify anchors (SUB_HEADER or INTRO immediately above a bullet run).
        anchored: dict[int, int] = {}  # run_idx -> anchor row index
        for r_idx, (s, _e, (_bx0, by0, _bx1, _by1)) in enumerate(bullet_runs):
            anchor = s - 1
            if anchor < 0:
                continue
            if cls[anchor] not in ("SUB_HEADER", "INTRO"):
                continue
            ay1 = rows[anchor]["bbox"][3]
            # Allow a small overlap (-15) since the bullet glyph row's bbox
            # often sneaks above the intro line's reported bottom by a pixel
            # or two when fitz reports tight ascent/descent boxes.
            if -15 <= (by0 - ay1) <= 50:
                anchored[r_idx] = anchor

        big_title_boxes: list[VisibleBox] = []
        sub_header_boxes: list[VisibleBox] = []
        para_boxes: list[VisibleBox] = []
        bullet_boxes: list[VisibleBox] = []
        subbullet_boxes: list[VisibleBox] = []
        prose_chars = 0
        consumed: set[int] = set()

        # Consumed indices that belong to a bullet run.
        for r_idx, (s, e, _bb) in enumerate(bullet_runs):
            for k in range(s, e + 1):
                consumed.add(k)
        for r_idx, anchor in anchored.items():
            consumed.add(anchor)

        # Emit one yellow region per bullet run + a blue group wrapper when
        # anchored.  Inside each bullet block, walk the line-rows and emit a
        # green sub-bullet box for every ``o`` / ``▪`` row.
        sub_idx = 0
        for r_idx, (s, e, bb) in enumerate(bullet_runs):
            x0, y0, x1, y1 = bb
            pad_y = 6
            pad_x = 8
            ybbox = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
            yb = _mk_box(
                box_id=f"prosebullet_{s}_title",
                px_bbox=ybbox,
                scale=scale,
                synthetic=True,
            )
            object.__setattr__(yb, "cover_footer_band", True)
            bullet_boxes.append(yb)

            seen_block_ids: set[int] = set()
            for k in range(s, e + 1):
                blk = bullet_block_for_row.get(k)
                if blk is None:
                    continue
                # Multiple line-rows can map back to the SAME pymupdf block
                # (e.g. several ``o`` sub-bullets inside one block).  Walking
                # ``_bullet_rows_in_block`` once per row would emit duplicate
                # green sub-bullet boxes — skip blocks we've already scanned
                # in this run.
                blk_key = id(blk)
                if blk_key in seen_block_ids:
                    continue
                seen_block_ids.add(blk_key)
                for glyph, rbb_pt in _bullet_rows_in_block(blk):
                    if glyph not in _SUB_BULLET_GLYPHS:
                        continue
                    gx0p, gy0p, gx1p, gy1p = _pdf_bbox_to_image_bbox(
                        rbb_pt, page_w_pt, page_h_pt,
                        scale, cw_quarter_turns % 4,
                    )
                    if gx1p <= gx0p or gy1p <= gy0p:
                        continue
                    spad_x = 6
                    spad_y = 2
                    is_sub_sub = glyph in _SUB_SUB_BULLET_GLYPHS
                    box_prefix = (
                        "prosesubsubbul" if is_sub_sub else "prosesubbul"
                    )
                    gbox = _mk_box(
                        box_id=f"{box_prefix}_{sub_idx}_title",
                        px_bbox=(
                            gx0p - spad_x,
                            gy0p - spad_y,
                            gx1p + spad_x,
                            gy1p + spad_y,
                        ),
                        scale=scale,
                        synthetic=True,
                    )
                    if is_sub_sub:
                        object.__setattr__(gbox, "subbullet_purple_band", True)
                    else:
                        object.__setattr__(gbox, "subbullet_green_band", True)
                    subbullet_boxes.append(gbox)
                    sub_idx += 1

            if r_idx in anchored:
                anchor = anchored[r_idx]
                ax0, ay0, ax1, ay1 = rows[anchor]["bbox"]
                gpad = 9
                gx0 = min(ax0, x0) - gpad
                gy0 = min(ay0, y0) - gpad
                gx1 = max(ax1, x1) + gpad
                gy1 = max(ay1, y1) + gpad
                para_boxes.append(_mk_box(
                    box_id=f"prosebulgrp_{anchor}",
                    px_bbox=(gx0, gy0, gx1, gy1),
                    scale=scale,
                    synthetic=False,
                ))

        # Big titles + sub-headers emit one box per row.
        for k, row in enumerate(rows):
            x0, y0, x1, y1 = row["bbox"]
            pad = 4
            bbp = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
            klass = cls[k]
            if klass == "BIG_TITLE":
                big_title_boxes.append(_mk_box(
                    box_id=f"prosehead_{k}_title",
                    px_bbox=bbp,
                    scale=scale,
                    synthetic=True,
                ))
                prose_chars += len(row["text"])
                continue
            if klass == "SUB_HEADER":
                sb = _mk_box(
                    box_id=f"prosesub_{k}_title",
                    px_bbox=bbp,
                    scale=scale,
                    synthetic=True,
                )
                object.__setattr__(sb, "subhdr_red_band", True)
                sub_header_boxes.append(sb)
                prose_chars += len(row["text"])
                continue
            if klass in ("INTRO", "BULLET"):
                prose_chars += len(row["text"])
                continue
            if k in consumed:
                prose_chars += len(row["text"])

        # Merge consecutive paragraph rows into single boxes.  Unanchored
        # ``INTRO`` rows (colon-ending lead-ins that did NOT find a bullet
        # block underneath) are folded in with adjacent paragraphs so their
        # text still ends up inside a wrapper.
        def _is_para_like(idx: int) -> bool:
            if idx in consumed:
                return False
            return cls[idx] == "PARA" or cls[idx] == "INTRO"

        i = 0
        while i < len(rows):
            if not _is_para_like(i):
                i += 1
                continue
            start = i
            ux0, uy0, ux1, uy1 = rows[i]["bbox"]
            line_h = max(1.0, uy1 - uy0)
            j = i + 1
            while j < len(rows) and _is_para_like(j):
                bx0, by0, bx1, by1 = rows[j]["bbox"]
                gap = by0 - rows[j - 1]["bbox"][3]
                if gap > 1.6 * line_h and gap > 14:
                    break
                ux0 = min(ux0, bx0)
                uy0 = min(uy0, by0)
                ux1 = max(ux1, bx1)
                uy1 = max(uy1, by1)
                line_h = max(line_h, by1 - by0)
                j += 1
            pad = 4
            para_boxes.append(_mk_box(
                box_id=f"prosepara_{start}",
                px_bbox=(ux0 - pad, uy0 - pad, ux1 + pad, uy1 + pad),
                scale=scale,
                synthetic=False,
            ))
            for k in range(start, j):
                prose_chars += len(rows[k]["text"])
            i = j

        all_boxes: list[VisibleBox] = [
            *big_title_boxes,
            *sub_header_boxes,
            *para_boxes,
            *bullet_boxes,
            *subbullet_boxes,
        ]
        has_signal = prose_chars >= 300 and bool(all_boxes)
        return all_boxes, bool(has_signal)
    finally:
        doc.close()


@dataclass
class ProseLayoutBandsPass:
    """Apply fitz text-block overlays for narrative pages."""

    info: PassInfo = PassInfo(
        name="prose_layout_bands",
        stage="synthesize",
        layer_flag="BLUE_TITLE",
        order=236,
        description=(
            "Narrative pages: fitz block-based blue big-title and red "
            "sub-header highlights, blue paragraph/group wrappers, and "
            "yellow bullet highlights."
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

        # The contour pass occasionally emits a BLUE outer wrapper whose
        # bbox is narrower than the union of its child ORANGE cells (e.g.
        # the wrapper covers the left column but not the right).  When
        # that wrapper is drawn as a blue outline, its right edge cuts
        # through cell text and looks broken.  Expand each wrapper to
        # actually enclose all of its children before prose detection or
        # rendering uses the bbox.  Universal — any document where the
        # contour pass under-sizes a wrapper benefits.
        result = _fit_outer_wrappers_to_children(result)
        state.result = result

        # Treat each ``vN`` contour box as a table region; prose detection
        # inside one of these is suppressed so the contour overlay owns the
        # tabular area.  Allows mixed pages (prose + an embedded table) to
        # still receive prose overlays for the surrounding text.  We use
        # ALL ``vN`` boxes (not just the BLUE outer wrapper) because the
        # outer wrapper sometimes covers a subset of the table while the
        # ORANGE cell rectangles extend further — text in those uncovered
        # cells would otherwise leak into adjacent paragraph wrappers.
        v_wrappers = [b for b in result.boxes if _is_v_wrapper(b)]
        v_boxes = [b for b in result.boxes if _is_v_box(b)]
        exclude_regions = [tuple(b.px_bbox) for b in v_boxes]

        synth_boxes, has_signal = detect_prose_layout_boxes(
            pdf_path=str(ctx.pdf_path),
            page_index=ctx.page_index,
            scale=scale,
            cw_quarter_turns=cw,
            exclude_regions=exclude_regions,
        )
        if not has_signal:
            return state

        # On mixed pages, drop the small-caps ``textsec_*`` heading rectangles
        # AND the right-margin ``tbstruct_*``/``tbtext_*`` sidebar phantoms
        # OUTSIDE the table regions — the prose overlay subsumes them
        # cleanly.  Keep textsec/tb boxes that sit inside a table region in
        # case the contour pass is using them as in-table titles.
        prose_bboxes = [tuple(b.px_bbox) for b in synth_boxes]
        # Cell rectangles from the authoritative contour pass (``v1..vN``)
        # take priority over the title-block right-sidebar ``tb*`` phantoms
        # whenever the two overlap — those phantoms otherwise draw
        # spurious blue/orange edges that cut through real cell content.
        v_cell_bboxes = [
            tuple(b.px_bbox) for b in result.boxes if _is_v_box(b)
        ]
        # ``mccol_*`` (multi-column contact bands) and synthetic prose
        # boxes that land entirely inside a real table region are noise:
        # the multi-col contact pass mis-fires when it sees evenly spaced
        # text in table cells, and the prose pass already excludes table
        # text but synth boxes from earlier runs may still carry over.
        # Drop them universally to stop the random tiny boxes inside
        # tabular rows.
        def _id_starts(b: VisibleBox, prefixes: tuple[str, ...]) -> bool:
            bid = b.box_id or ""
            return bid.startswith(prefixes)

        # Children of a ``v0``/``vN`` outer wrapper that are NOT real cell
        # rectangles (e.g. synthetic ``mccol_*`` from the multi-col pass)
        # also need dropping when they nest inside the wrapper.
        kept: list[VisibleBox] = []
        for b in result.boxes:
            bb = tuple(b.px_bbox)
            inside_table = any(_bbox_inside(bb, r) for r in exclude_regions)
            if _is_tb_noise(b):
                overlaps_cell = _bbox_overlaps_any(bb, v_cell_bboxes)
                # Drop tb phantoms outside any table region (existing
                # behaviour) AND tb phantoms that duplicate a real v* cell
                # rectangle (the visible blue/orange edge artefact).
                if not inside_table or overlaps_cell:
                    continue
            if _is_textsec(b):
                if _bbox_overlaps_any(bb, prose_bboxes) and not inside_table:
                    continue
            # Multi-column contact bands inside a detected table region
            # are spurious — the table contour pass already owns the area.
            if _id_starts(b, ("mccol_",)) and inside_table:
                continue
            # Generic catch-all: any synthetic non-``vN`` box inside a
            # table region is noise (prose synth strays, mccol bands,
            # etc.).  Real ``vN`` cells are kept (they're not synthetic).
            if (
                getattr(b, "synthetic", False)
                and inside_table
                and not _is_v_box(b)
            ):
                continue
            kept.append(b)
        new_result = VisibleBoxResult(
            boxes=[*kept, *synth_boxes],
            image_width=result.image_width,
            image_height=result.image_height,
            debug_stats={
                **dict(result.debug_stats or {}),
                "prose_layout_bands": len(synth_boxes),
            },
        )
        state.result = new_result
        state.artifacts.setdefault("stage_order", []).append(self.info.name)
        return state


__all__ = ["ProseLayoutBandsPass", "detect_prose_layout_boxes"]
