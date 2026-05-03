"""Color-driven structured document extractor.

The overlay pipeline tags every region with a color and a box id.  This
extractor uses ONLY those tags as the source of truth for what each
piece of content is — no per-document heuristics, no font heuristics,
no keyword matching.

Universal color → role rules:

    BLUE  prosehead_N_title               level-0 TITLE
    BLUE  prosesub_N_title (red band)     level-1+ SECTION HEADING
    BLUE  prosepara_N                     PARAGRAPH
    BLUE  prosebullet_N_title (yellow)    BULLET LIST
    BLUE  prosesubbul_N_title (green)     SUB-BULLET (o glyph)
    BLUE  prosesubsubbul_N_title (purple) SUB-SUB-BULLET (▪ glyph)
    BLUE  prosebulgrp_N                   INTRO+BULLET GROUP wrapper
    BLUE  vN (outer wrapper)              TABLE container
    ORANGE vN (children)                  TABLE cell
    CYAN  colhdr_*, mccol_*_hdr_*         TABLE column-header cell
    BLUE  toc_*                           TABLE-OF-CONTENTS entry
    BLUE  textsec_N_title / textsec_N_body  prose notes
    BLUE  rfpcover_main / rfpcover_program  cover title bands
    YELLOW rfpcover_footer                cover footer band

The output schema (one JSON per page):

    {
      "document": { source, page, title, metadata[] },
      "outline":  [{level, heading, block_count}],
      "sections": [{
          heading, level,
          blocks: [
              { kind: "paragraph", text },
              { kind: "bullet_list", intro?, items[] },
              { kind: "table", columns[], rows[{col: value}] },
              { kind: "note", text },
          ]
      }],
      "color_legend": { ... }
    }
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

try:
    import fitz as _fitz
except Exception:  # pragma: no cover
    _fitz = None


# ───────────────────── public API ────────────────────────────────────────

COLOR_LEGEND = {
    "BLUE big-title": "Top-level document or page title",
    "RED sub-header": "Section / sub-section heading (numbered or italic-bold)",
    "BLUE paragraph wrapper": "Body paragraph block",
    "YELLOW bullet block": "Bullet list (• or N. or A) markers)",
    "GREEN sub-bullet": "Second-level bullet (o glyph)",
    "PURPLE sub-sub-bullet": "Third-level bullet (▪ glyph)",
    "BLUE intro+bullet group": "A colon-intro line bound to the bullet list under it",
    "BLUE table outer wrapper": "Container of a structured table",
    "ORANGE table cell": "Single data or header cell inside a table",
    "CYAN column header": "Borderless column-header label (multi-col layouts)",
    "BLUE TOC entry": "Table-of-contents row (top-level)",
    "RED TOC sub-entry": "Table-of-contents row (sub-level)",
    "BLUE textsec / cover band": "General prose notes block / cover title band",
}


def extract_structured(payload: dict[str, Any], pdf_path: str | Path) -> dict[str, Any]:
    """Build a clean structured doc from an overlay JSON payload.

    ``payload`` is the dict written by ``detect_standalone --json-out``
    (must contain ``boxes``, ``image_width``, ``image_height``).
    ``pdf_path`` is the source PDF.  We open it to read the actual text
    inside each detected region.
    """
    if _fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for structured extraction")

    boxes = payload.get("boxes") or []
    image_w = float(payload.get("image_width") or 0)
    image_h = float(payload.get("image_height") or 0)
    page_idx = int(payload.get("page", 0))
    debug = payload.get("debug_stats", {}) or {}
    scale = float(debug.get("render_scale_used") or 1.0) or 1.0

    doc = _fitz.open(str(pdf_path))
    try:
        page = doc[page_idx]
        page_w_pt = float(page.rect.width)
        page_h_pt = float(page.rect.height)
        rotated_cw = bool(debug.get("rotated_cw", False))
        cw_quarter_turns = int(debug.get("rotated_cw_quarter_turns") or 0) % 4

        builder = _DocBuilder(
            page=page,
            scale=scale,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            image_w=image_w,
            image_h=image_h,
            rotated_cw=rotated_cw,
            cw_quarter_turns=cw_quarter_turns,
        )
        return builder.build(
            boxes=boxes,
            source=str(pdf_path),
            page_index=page_idx,
        )
    finally:
        doc.close()


def write_structured(out_path: str | Path, structured_doc: dict[str, Any]) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(structured_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


# ───────────────────── implementation ───────────────────────────────────

_NUM_PREFIX_RE = re.compile(r"^\s*(\d+)((?:\.\d+){0,3})\s+\S")
_NUM_LIST_RE = re.compile(r"^\s*\d{1,3}\.\s+\S")
_ALPHA_LIST_RE = re.compile(r"^\s*[A-Za-z][\.\)]\s+\S")
_BULLET_GLYPH_RE = re.compile(r"^[\s•\u2022▪\u25aa\u00ba]+")


def _id(b: dict) -> str:
    return str(b.get("box_id") or "")


def _color(b: dict) -> str:
    return str(b.get("color") or "").upper()


def _px(b: dict) -> tuple[int, int, int, int]:
    px = b.get("px_bbox") or [0, 0, 0, 0]
    return int(px[0]), int(px[1]), int(px[2]), int(px[3])


def _is_synthetic(b: dict) -> bool:
    return bool(b.get("synthetic", False))


def _bbox_inside(inner: tuple[int, int, int, int],
                 outer: tuple[int, int, int, int],
                 slack: int = 6) -> bool:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return (ix0 >= ox0 - slack and iy0 >= oy0 - slack
            and ix1 <= ox1 + slack and iy1 <= oy1 + slack)


def _bbox_overlaps(a: tuple[int, int, int, int],
                   b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0]
                or a[3] <= b[1] or b[3] <= a[1])


def _overlap_frac(a: tuple[int, int, int, int],
                  b: tuple[int, int, int, int]) -> float:
    """Fraction of ``a`` that lies inside ``b`` (0..1)."""
    if not _bbox_overlaps(a, b):
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    a_area = max(1, (ax1 - ax0) * (ay1 - ay0))
    return (iw * ih) / a_area


def _id_starts(box_id: str, prefixes: tuple[str, ...]) -> bool:
    return any(box_id.startswith(p) for p in prefixes)


def _numbered_depth(text: str) -> int | None:
    m = _NUM_PREFIX_RE.match(text or "")
    if not m:
        return None
    return m.group(2).count(".")


def _strip_bullet_glyph(text: str) -> str:
    return _BULLET_GLYPH_RE.sub("", text or "").strip()


class _DocBuilder:
    def __init__(self, *, page, scale, page_w_pt, page_h_pt,
                 image_w, image_h, rotated_cw, cw_quarter_turns):
        self.page = page
        self.scale = scale
        self.page_w_pt = page_w_pt
        self.page_h_pt = page_h_pt
        self.image_w = image_w
        self.image_h = image_h
        self.rotated_cw = rotated_cw
        self.cw_quarter_turns = cw_quarter_turns

    # ── geometry helpers ────────────────────────────────────────────────

    def px_to_pdf_rect(self, px_bbox: tuple[int, int, int, int]):
        sx = self.scale or 1.0
        x0, y0, x1, y1 = (v / sx for v in px_bbox)
        return _fitz.Rect(x0, y0, x1, y1)

    def text_in_px(self, px_bbox: tuple[int, int, int, int]) -> str:
        rect = self.px_to_pdf_rect(px_bbox)
        # Inflate slightly so punctuation hugging the edge isn't dropped.
        rect = _fitz.Rect(rect.x0 - 0.5, rect.y0 - 0.5,
                          rect.x1 + 0.5, rect.y1 + 0.5)
        try:
            txt = self.page.get_text("text", clip=rect) or ""
        except Exception:
            txt = ""
        return _strip_clip_noise(_normalize_ws(txt))

    def _lines_in_px(self, px_bbox: tuple[int, int, int, int]) -> list[str]:
        """Return cleaned visible-text lines inside the rect (in y-order).

        Each line is stripped of clip-noise and consecutive empty lines
        are collapsed.  Useful when we need to separate a stacked
        sub-heading from the colon-intro line just under it.
        """
        rect = self.px_to_pdf_rect(px_bbox)
        rect = _fitz.Rect(rect.x0 - 0.5, rect.y0 - 0.5,
                          rect.x1 + 0.5, rect.y1 + 0.5)
        try:
            txt = self.page.get_text("text", clip=rect) or ""
        except Exception:
            txt = ""
        out: list[str] = []
        for raw in txt.splitlines():
            s = _strip_clip_noise(" ".join(raw.split()))
            if s:
                out.append(s)
        return out

    def _all_pdf_lines(self) -> list[dict]:
        """All PDF text-layer lines with size + bold/italic flag + px bbox."""
        try:
            page_dict = self.page.get_text("dict")
        except Exception:
            return []
        lines: list[dict] = []
        for block in page_dict.get("blocks", []) or []:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []) or []:
                spans = line.get("spans", []) or []
                if not spans:
                    continue
                sizes = [float(s.get("size", 0)) for s in spans if s.get("text", "").strip()]
                flags = [int(s.get("flags", 0)) for s in spans if s.get("text", "").strip()]
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text or not sizes:
                    continue
                bbox = line.get("bbox") or [0, 0, 0, 0]
                # PyMuPDF font flags: 1=superscript, 2=italic, 4=serifed,
                # 8=monospaced, 16=bold.
                bold = any((f & 16) for f in flags)
                italic = any((f & 2) for f in flags)
                x0, y0, x1, y1 = bbox
                sx = self.scale or 1.0
                px = (int(x0 * sx), int(y0 * sx),
                      int(x1 * sx), int(y1 * sx))
                lines.append({
                    "text": text,
                    "size": max(sizes),
                    "bold": bold,
                    "italic": italic,
                    "px": px,
                })
        return lines

    def _find_caption_lines(self) -> list[dict]:
        """Universal: any bold/italic text line, larger than the median
        body size, that is not part of a known overlay box.  Returns a
        list of ``{"text", "px"}`` records in y-order.
        """
        lines = self._all_pdf_lines()
        if not lines:
            return []
        sizes_sorted = sorted([l["size"] for l in lines])
        median_body = sizes_sorted[len(sizes_sorted) // 2]
        out: list[dict] = []
        for l in lines:
            if not (l["bold"] or l["italic"]):
                continue
            if l["size"] < median_body + 1.0:
                continue
            if len(l["text"]) > 80:
                continue
            out.append({"text": l["text"], "px": l["px"]})
        out.sort(key=lambda r: r["px"][1])
        return out

    def _find_uncovered_notes(
        self,
        known_pxs: list[tuple[int, int, int, int]],
    ) -> list[dict]:
        """Every plain (non-bold) PDF line not already covered by a
        known box.  These become ``note`` blocks placed at their y.
        """
        lines = self._all_pdf_lines()
        out: list[dict] = []
        for l in lines:
            if any(_overlap_frac(l["px"], kp) >= 0.5 for kp in known_pxs):
                continue
            out.append({"text": l["text"], "px": l["px"]})
        out.sort(key=lambda r: r["px"][1])
        return out

    def words_in_px(self, px_bbox: tuple[int, int, int, int]) -> list[dict]:
        rect = self.px_to_pdf_rect(px_bbox)
        rect = _fitz.Rect(rect.x0 - 0.5, rect.y0 - 0.5,
                          rect.x1 + 0.5, rect.y1 + 0.5)
        try:
            raw = self.page.get_text("words", clip=rect) or []
        except Exception:
            raw = []
        out = []
        for w in raw:
            if len(w) < 5:
                continue
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            if not str(text).strip():
                continue
            out.append({
                "text": str(text),
                "x0": float(x0), "y0": float(y0),
                "x1": float(x1), "y1": float(y1),
            })
        return out

    # ── classification ──────────────────────────────────────────────────

    def classify_boxes(self, boxes: list[dict]) -> dict[str, list[dict]]:
        """Bucket every box by its semantic role using box_id + color
        + marker attributes set by the detection passes.
        """
        buckets: dict[str, list[dict]] = {
            "title": [],            # prosehead_*_title
            "section_heading": [],  # prosesub_*_title (red band)
            "paragraph": [],        # prosepara_*
            "bullet_block": [],     # prosebullet_*_title
            "sub_bullet": [],       # prosesubbul_*_title (green)
            "sub_sub_bullet": [],   # prosesubsubbul_*_title (purple)
            "intro_bullet_group": [],  # prosebulgrp_*
            "table_wrapper": [],    # vN where is_outer_wrapper=True
            "table_cell": [],       # vN children
            "column_header": [],    # CYAN colhdr_* / mccol_*_hdr_*
            "toc_heading": [],      # toc_heading_*
            "toc_entry_top": [],    # toc_top_*
            "toc_entry_sub": [],    # toc_sub_*
            "cover_main": [],       # rfpcover_*_title (no cover_footer_band)
            "cover_footer": [],     # rfpcover_*_title (cover_footer_band=True)
            "textsec_title": [],    # textsec_N_title
            "textsec_body": [],     # textsec_N_body
        }
        for b in boxes:
            bid = _id(b)
            if not bid:
                continue
            if bid.startswith("rfpcover_"):
                if b.get("cover_footer_band"):
                    buckets["cover_footer"].append(b)
                else:
                    buckets["cover_main"].append(b)
            elif bid.startswith("prosehead_"):
                buckets["title"].append(b)
            elif bid.startswith("prosesub_"):
                buckets["section_heading"].append(b)
            elif bid.startswith("prosepara_"):
                buckets["paragraph"].append(b)
            elif bid.startswith("prosebullet_"):
                buckets["bullet_block"].append(b)
            elif bid.startswith("prosesubsubbul_"):
                buckets["sub_sub_bullet"].append(b)
            elif bid.startswith("prosesubbul_"):
                buckets["sub_bullet"].append(b)
            elif bid.startswith("prosebulgrp_"):
                buckets["intro_bullet_group"].append(b)
            elif re.fullmatch(r"v\d+", bid) and not _is_synthetic(b):
                if b.get("is_outer_wrapper"):
                    buckets["table_wrapper"].append(b)
                else:
                    buckets["table_cell"].append(b)
            elif bid.startswith(("colhdr_",)) or "mccol_" in bid and "_hdr_" in bid:
                buckets["column_header"].append(b)
            elif bid.startswith("toc_heading"):
                buckets["toc_heading"].append(b)
            elif bid.startswith("toc_top_"):
                buckets["toc_entry_top"].append(b)
            elif bid.startswith("toc_sub_"):
                buckets["toc_entry_sub"].append(b)
            elif bid.startswith("textsec_") and bid.endswith("_title"):
                buckets["textsec_title"].append(b)
            elif bid.startswith("textsec_") and bid.endswith("_body"):
                buckets["textsec_body"].append(b)
        return buckets

    # ── table assembly ──────────────────────────────────────────────────

    def assemble_table(self, wrapper: dict,
                       cells: list[dict]) -> dict[str, Any]:
        wrap_id = _id(wrapper)
        wrap_px = _px(wrapper)
        my_cells = [c for c in cells if (c.get("parent_box_id") or "") == wrap_id]
        if not my_cells:
            my_cells = [c for c in cells if _bbox_inside(_px(c), wrap_px, slack=4)]
        rows = _cluster_rows(my_cells)
        if not rows:
            return {
                "kind": "table",
                "columns": [],
                "rows": [],
                "raw_text": self.text_in_px(wrap_px),
            }

        # Build a stable column model from the WIDEST row (typically the
        # header row).  Each column owns an x-span (left, right) and
        # a column-id integer.
        widest = max(rows, key=lambda r: len(r))
        widest_sorted = sorted(widest, key=lambda c: _px(c)[0])
        col_spans: list[tuple[int, int]] = []
        for c in widest_sorted:
            x0, _, x1, _ = _px(c)
            col_spans.append((x0, x1))

        # Universal "lost column" rescue: when the contour detector
        # missed an outer column (the wrapper bbox stops short of the
        # last visible column), there will be PDF text just to the
        # RIGHT of the wrapper that aligns vertically with the table
        # rows.  Scan past the wrapper's right edge using the row
        # heights and append any discovered columns.  Same trick on
        # the LEFT for cases where a leading column is cropped off.
        page_w = self.page.rect.width * self.scale
        wrap_right = wrap_px[2]
        wrap_left = wrap_px[0]
        # Use the header row to determine y-band(s) for sniffing.
        header_row = sorted(widest, key=lambda c: _px(c)[0])
        if header_row:
            hy0 = min(_px(c)[1] for c in header_row)
            hy1 = max(_px(c)[3] for c in header_row)
            # Sniff right side.
            right_extras = self._sniff_extra_columns(
                y_band=(hy0, hy1),
                x_range=(wrap_right + 2, int(page_w) - 1),
                wrapper_rows=rows,
            )
            for span in right_extras:
                col_spans.append(span)
            # Sniff left side.
            left_extras = self._sniff_extra_columns(
                y_band=(hy0, hy1),
                x_range=(0, wrap_left - 2),
                wrapper_rows=rows,
            )
            for span in left_extras:
                col_spans.insert(0, span)
        col_spans.sort(key=lambda s: s[0])

        def assign_col(cx0: int, cx1: int) -> int:
            mid = 0.5 * (cx0 + cx1)
            best_i = 0
            best_d = float("inf")
            for i, (sx0, sx1) in enumerate(col_spans):
                if sx0 <= mid <= sx1:
                    return i
                d = min(abs(mid - sx0), abs(mid - sx1))
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i

        n_cols = len(col_spans)
        text_rows: list[list[str]] = []
        for row in rows:
            row_cells = sorted(row, key=lambda c: _px(c)[0])
            cells_text = ["" for _ in range(n_cols)]
            for c in row_cells:
                cx0, cy0, cx1, cy1 = _px(c)
                ci = assign_col(cx0, cx1)
                t = self.text_in_px((cx0, cy0, cx1, cy1))
                if t:
                    if cells_text[ci]:
                        cells_text[ci] = cells_text[ci] + " " + t
                    else:
                        cells_text[ci] = t
            # For any column that has no contour cell on this row but
            # falls in the rescued left/right extras, clip text directly
            # from the PDF using the row's y-band and the column span.
            row_y0 = min(_px(c)[1] for c in row)
            row_y1 = max(_px(c)[3] for c in row)
            for ci, (sx0, sx1) in enumerate(col_spans):
                if cells_text[ci]:
                    continue
                # Only fill from PDF text if this column lies OUTSIDE
                # the original wrapper — those are the rescued columns.
                col_mid = 0.5 * (sx0 + sx1)
                if wrap_left <= col_mid <= wrap_right:
                    continue
                t = self.text_in_px((sx0, row_y0, sx1, row_y1))
                if t:
                    cells_text[ci] = t
            text_rows.append(cells_text)

        # First non-empty row is the header (if it looks like header-y).
        # Heuristic-free choice: take row 0 as headers.
        header_row_txt = text_rows[0] if text_rows else []
        body = text_rows[1:] if len(text_rows) > 1 else []
        columns = [h or f"col_{i+1}" for i, h in enumerate(header_row_txt)]
        rows_obj = [
            {columns[i]: cell for i, cell in enumerate(r)}
            for r in body
        ]
        return {
            "kind": "table",
            "columns": columns,
            "rows": rows_obj,
        }

    def _sniff_extra_columns(
        self,
        *,
        y_band: tuple[int, int],
        x_range: tuple[int, int],
        wrapper_rows: list[list[dict]],
    ) -> list[tuple[int, int]]:
        """Return ``[(x0, x1), ...]`` spans for columns that exist in
        the PDF text layer to the SIDE of a table wrapper but were
        missed by the contour detector.

        We use the same column-pitch as the wrapper (median width of
        existing cells) to bucket text into discrete columns.  Spans
        are de-duplicated and only kept if at least one wrapper row
        has matching text under them.  Universal — relies only on
        text-layer geometry, no font/keyword heuristics.
        """
        x0_lo, x0_hi = x_range
        if x0_hi - x0_lo < 20:
            return []
        y_lo, y_hi = y_band
        # Collect all text spans inside the side strip via the PDF
        # textpage's word layout.
        rect = _fitz.Rect(
            x0_lo / self.scale, y_lo / self.scale,
            x0_hi / self.scale, y_hi / self.scale,
        )
        try:
            words = self.page.get_text("words", clip=rect) or []
        except Exception:
            words = []
        if not words:
            return []
        # Each word is (x0, y0, x1, y1, text, block_no, line_no, word_no).
        # Project to image px and merge into x-bands.
        px_words = []
        for w in words:
            wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
            if not (text and text.strip()):
                continue
            px_words.append((
                int(round(wx0 * self.scale)),
                int(round(wy0 * self.scale)),
                int(round(wx1 * self.scale)),
                int(round(wy1 * self.scale)),
            ))
        if not px_words:
            return []
        # Greedy x-band cluster: merge words whose x-spans overlap or
        # are within a small gap (≈ row height).
        gap = max(8, int((y_hi - y_lo) * 0.4))
        px_words.sort(key=lambda w: w[0])
        bands: list[list[int]] = []  # [x0, x1]
        for wx0, _, wx1, _ in px_words:
            if not bands:
                bands.append([wx0, wx1])
                continue
            last = bands[-1]
            if wx0 <= last[1] + gap:
                last[1] = max(last[1], wx1)
            else:
                bands.append([wx0, wx1])
        # Pad bands a touch so the column's middle is comfortably inside.
        return [(b[0] - 4, b[1] + 4) for b in bands]

    # ── high-level build ────────────────────────────────────────────────

    def build(self, *, boxes: list[dict], source: str, page_index: int) -> dict[str, Any]:
        buckets = self.classify_boxes(boxes)

        # Cover-page priority: ONLY rfpcover_* bands (cover-page title
        # bands) become the document title.  Every other "big title"
        # box (prosehead_*) is treated as a section heading on the body
        # pages — that way an "Introduction" page-header doesn't get
        # promoted to the document title and lose its body content.
        title_text = ""
        title_b = None
        cover_meta: list[str] = []
        cover_main_sorted = sorted(buckets["cover_main"], key=lambda x: _px(x)[1])
        cover_footer_sorted = sorted(buckets["cover_footer"], key=lambda x: _px(x)[1])
        if cover_main_sorted:
            for cm in cover_main_sorted:
                t = self.text_in_px(_px(cm))
                if not t:
                    continue
                if not title_text:
                    title_text = t
                    title_b = cm
                else:
                    cover_meta.append(t)
            for cf in cover_footer_sorted:
                t = self.text_in_px(_px(cf))
                if t:
                    cover_meta.append(t)

        # Body pages: every prosehead_* becomes a section heading.
        title_sorted = sorted(buckets["title"], key=lambda x: _px(x)[1])

        # Build a flat ordered stream of "items" (heading + content).
        items: list[dict] = []

        # Drop a heading if it's a tiny noise box (e.g. one-char "x" cell
        # captured as a sub-header) — universal: any "heading" narrower
        # than ~40 px in image space is suspect and we ignore it.
        def _heading_too_small(b: dict) -> bool:
            x0, _, x1, _ = _px(b)
            return (x1 - x0) < 40

        # Section headings: ``prosesub_*`` (red) → level 2 by default,
        # bumped to deeper levels when the text begins with a numbered
        # prefix (1.1 → level 2, 1.1.1 → level 3, etc).
        # Long sentence-style red text (e.g. "Failure to report
        # discrepancies upon receipt transfers responsibility to the
        # partner") is a CALLOUT, not a section heading — emit it as a
        # ``note`` block instead so it attaches to the surrounding
        # section.  Universal: a label rarely runs past one short line.
        for b in buckets["section_heading"]:
            if _heading_too_small(b):
                continue
            t = self.text_in_px(_px(b))
            if not t:
                continue
            if _looks_like_callout(t):
                items.append({
                    "_kind": "note",
                    "text": t,
                    "y": _px(b)[1],
                    "px": _px(b),
                })
                continue
            depth = _numbered_depth(t)
            level = (depth + 1) if depth is not None else 2
            items.append({
                "_kind": "heading",
                "level": level,
                "text": t,
                "y": _px(b)[1],
                "px": _px(b),
            })
        # ``prosehead_*`` (blue big-title) → level 1 section headings.
        # On cover pages there are none — covers go through ``cover_main``.
        for b in title_sorted:
            if _heading_too_small(b):
                continue
            t = self.text_in_px(_px(b))
            if not t:
                continue
            items.append({
                "_kind": "heading",
                "level": 1,
                "text": t,
                "y": _px(b)[1],
                "px": _px(b),
            })

        # Universal caption detector: any short bold-or-italic text line
        # in the PDF text layer that is NOT inside any detected box and
        # is larger than the page's median body font becomes a level-1
        # section heading.  This catches italic section labels like
        # "Fiber backbone" / "Closet Build out" that don't trip the
        # prose pass's bold-only BIG_TITLE rule.  Universal — every PDF
        # follows the convention of "bigger / bolder / italic = label".
        all_known_pxs = [_px(b) for b in boxes]
        for cap in self._find_caption_lines():
            cap_px = cap["px"]
            # Skip if any known box already covers this line.
            if any(_overlap_frac(cap_px, kp) >= 0.5 for kp in all_known_pxs):
                continue
            items.append({
                "_kind": "heading",
                "level": 1,
                "text": cap["text"],
                "y": cap_px[1],
                "px": cap_px,
                "_caption": True,
            })

        # Cover footer band — surface as a metadata line
        for b in buckets["cover_footer"]:
            t = self.text_in_px(_px(b))
            if t:
                items.append({"_kind": "footer", "text": t, "y": _px(b)[1], "px": _px(b)})

        # Paragraph wrappers — each becomes its own block
        # (skip ones that fully contain other paragraph blocks to avoid dupes)
        para_pxs = [_px(b) for b in buckets["paragraph"]]
        for b in buckets["paragraph"]:
            bb = _px(b)
            t = self.text_in_px(bb)
            if not t:
                continue
            items.append({"_kind": "paragraph", "text": t, "y": bb[1], "px": bb})

        # Bullet blocks (yellow). Each carries a list of bullet items.
        # Items are filtered to ones that begin with a recognized bullet
        # marker — strips PyMuPDF clip-edge noise like "p g p q p".
        bullet_block_records: list[dict] = []
        for b in buckets["bullet_block"]:
            bb = _px(b)
            rec = {
                "_kind": "bullet_list",
                "items": _split_bullet_items(self.text_in_px(bb)),
                "y": bb[1],
                "px": bb,
            }
            bullet_block_records.append(rec)
            items.append(rec)

        # Intro+bullet groups (blue wrapper around colon-intro + bullets).
        # Two scenarios:
        #   1) Wrapper contains an inner prosesub heading + bullet block
        #      — the heading IS the section label.  We skip emitting a
        #      group item; the heading and bullet list flow naturally in
        #      y-order via the section builder below.
        #   2) Wrapper contains only a bullet block — there's an
        #      unboxed colon-intro line above it.  We extract that line
        #      from (wrapper_top..bullet_top) and pair it with the
        #      bullet list as ``intro``.
        for b in buckets["intro_bullet_group"]:
            gb = _px(b)
            inner_heading = None
            for sh in buckets["section_heading"]:
                if _bbox_inside(_px(sh), gb, slack=8):
                    inner_heading = sh
                    break
            inner_bullet = None
            for rec in bullet_block_records:
                if _bbox_inside(rec["px"], gb, slack=8):
                    if inner_bullet is None or rec["px"][1] < inner_bullet["px"][1]:
                        inner_bullet = rec
            if inner_heading is not None:
                continue
            intro_text = ""
            extra_label = ""
            if inner_bullet is not None:
                intro_px = (gb[0], gb[1], gb[2], inner_bullet["px"][1])
                # Extract WITH preserved line breaks so we can split a
                # heading + intro pair (e.g. "Kitting Requirements\n
                # Partner(s) must:") that the prose pass folded into a
                # single wrapper region.
                lines = self._lines_in_px(intro_px)
                if lines:
                    # Last line is the intro; anything before is an
                    # untagged sub-heading or label sitting on top of
                    # the bullet block.
                    intro_text = lines[-1]
                    if len(lines) > 1:
                        extra_label = " ".join(lines[:-1])
            items.append({
                "_kind": "group",
                "px": gb,
                "y": gb[1],
                "intro_text": intro_text,
                "extra_label": extra_label,
                "_inner_bullet": inner_bullet,
            })

        # Sub-bullets / sub-sub bullets — recorded with kind so the
        # bullet_list pass can incorporate them as nested children.
        for b in buckets["sub_bullet"]:
            bb = _px(b)
            t = _strip_bullet_glyph(self.text_in_px(bb))
            if t:
                items.append({"_kind": "sub_bullet", "text": t, "y": bb[1], "px": bb})
        for b in buckets["sub_sub_bullet"]:
            bb = _px(b)
            t = _strip_bullet_glyph(self.text_in_px(bb))
            if t:
                items.append({"_kind": "sub_sub_bullet", "text": t, "y": bb[1], "px": bb})

        # Tables.
        wrapper_pxs = [_px(w) for w in buckets["table_wrapper"]]
        table_items: list[dict] = []
        for w in buckets["table_wrapper"]:
            tbl = self.assemble_table(w, buckets["table_cell"])
            wbb = _px(w)
            ti = {"_kind": "table", "table": tbl, "y": wbb[1], "px": wbb}
            table_items.append(ti)
            items.append(ti)

        # Universal "lost notes" pass: every PDF text line not covered
        # by a known overlay box becomes a note.  Each note attaches to
        # the nearest preceding table (within 60 px) as ``table.notes``;
        # otherwise it floats as its own block in reading order.
        all_known_after = (
            [_px(b) for b in boxes if not _is_synthetic(b)]
            + [_px(b) for b in buckets["title"]]
            + [_px(b) for b in buckets["section_heading"]]
            + [_px(b) for b in buckets["paragraph"]]
            + [_px(b) for b in buckets["bullet_block"]]
            + [_px(b) for b in buckets["sub_bullet"]]
            + [_px(b) for b in buckets["sub_sub_bullet"]]
            + [_px(b) for b in buckets["intro_bullet_group"]]
            + [_px(b) for b in buckets["cover_main"]]
            + [_px(b) for b in buckets["cover_footer"]]
            + [_px(b) for b in buckets["textsec_title"]]
            + [_px(b) for b in buckets["textsec_body"]]
            + [_px(b) for b in buckets["toc_heading"]]
            + [_px(b) for b in buckets["toc_entry_top"]]
            + [_px(b) for b in buckets["toc_entry_sub"]]
            + [it["px"] for it in items if it.get("_caption")]
        )
        for note in self._find_uncovered_notes(all_known_after):
            text = note["text"].strip()
            # Skip single-word fragments — those are usually
            # mis-detected table header cells leaking out, not real
            # narrative notes.
            if len(text.split()) < 2:
                continue
            note_y = note["px"][1]
            note_bot = note["px"][3]
            # Allow a small negative gap to handle wrappers that
            # extend a few px past their last text row.
            best = None
            best_gap = 1e9
            for ti in table_items:
                tx0, ty0, tx1, ty1 = ti["px"]
                gap = note_y - ty1
                if -8 <= gap <= 60 and gap < best_gap:
                    best = ti
                    best_gap = gap
            if best is not None:
                tbl = best["table"]
                existing = tbl.get("notes")
                tbl["notes"] = (existing + " " + text) if existing else text
            else:
                items.append({
                    "_kind": "note",
                    "text": text,
                    "y": note_y,
                    "px": note["px"],
                })

        # Drop paragraph blocks that are **substantially** inside a table
        # (they're cell text duplicates).  A paragraph that just barely
        # touches the table edge (e.g. a notes line written immediately
        # below a table whose wrapper extends 5–10 px past its bottom)
        # is kept — that's the Units→notes case.
        items = [it for it in items
                 if not (it["_kind"] == "paragraph"
                         and any(_overlap_frac(it["px"], wb) >= 0.4
                                 for wb in wrapper_pxs))]

        # TOC.
        toc_blocks: list[dict] = []
        if buckets["toc_heading"] or buckets["toc_entry_top"]:
            toc_heading_text = ""
            if buckets["toc_heading"]:
                toc_heading_text = self.text_in_px(_px(buckets["toc_heading"][0]))
            entries = []
            for b in sorted(buckets["toc_entry_top"], key=lambda x: _px(x)[1]):
                entries.append({"level": 1, "text": self.text_in_px(_px(b))})
            for b in sorted(buckets["toc_entry_sub"], key=lambda x: _px(x)[1]):
                entries.append({"level": 2, "text": self.text_in_px(_px(b))})
            entries.sort(key=lambda e: e.get("text", ""))
            toc_blocks.append({
                "kind": "table_of_contents",
                "heading": toc_heading_text or "Contents",
                "entries": entries,
            })

        # Sort everything by y to get a reading-order stream.
        items.sort(key=lambda it: it["y"])

        # ── Assemble sections ───────────────────────────────────────────
        # A "section" starts at every heading (or at top-of-page if there
        # are pre-heading items).  Drop the synthetic intermediate kinds
        # (group, sub_bullet, sub_sub_bullet) into the bullet_list above.
        # Pull metadata off the front of the page (everything before the
        # first PARA/TABLE/BULLET and after the title).

        sections: list[dict] = []
        current_heading: dict | None = None
        current_blocks: list[dict] = []

        def flush():
            if current_heading is None and not current_blocks:
                return
            sections.append({
                "heading": current_heading["text"] if current_heading else None,
                "level": current_heading["level"] if current_heading else 1,
                "blocks": current_blocks.copy(),
            })

        # Universal partition: metadata = every heading that appears
        # BEFORE the first body block (paragraph / table / bullet list)
        # AND owns no body content of its own (no body block between it
        # and the next heading).  Everything else flows into the
        # section-builder below.  This works for any layout:
        #   - top-of-page label callouts (addresses, dates, totals)
        #   - cover-page bands
        #   - headers above the first table
        body_kinds = ("paragraph", "table", "bullet_list")
        first_body_y = next(
            (it["y"] for it in items if it["_kind"] in body_kinds),
            None,
        )
        meta_lines: list[str] = []
        body_items: list[dict] = []
        for i, it in enumerate(items):
            if it["_kind"] != "heading":
                body_items.append(it)
                continue
            # Find the y-ordered slice between this heading and the next.
            next_heading_y = None
            for j in range(i + 1, len(items)):
                if items[j]["_kind"] == "heading":
                    next_heading_y = items[j]["y"]
                    break
            owns_body = any(
                items[j]["_kind"] in body_kinds
                and items[j]["y"] > it["y"]
                and (next_heading_y is None or items[j]["y"] < next_heading_y)
                for j in range(i + 1, len(items))
            )
            before_first_body = (
                first_body_y is None or it["y"] < first_body_y
            )
            if (
                not owns_body
                and before_first_body
                and len(it["text"]) <= 100
            ):
                meta_lines.append(it["text"])
                continue
            body_items.append(it)

        # If no cover band claimed the title and the first metadata
        # line is a BIG_TITLE (prosehead) heading, promote it to the
        # document title.  Universal — header pages typically lead with
        # the document name above secondary callouts.
        if not title_text and meta_lines:
            first_meta = meta_lines[0]
            for b in title_sorted:
                t = self.text_in_px(_px(b))
                if t and t == first_meta:
                    title_text = first_meta
                    meta_lines = meta_lines[1:]
                    break

        # Stream into sections.  A `group` (intro+bullet wrapper) emits
        # a single bullet_list block with its intro_text on it; the
        # inner bullet_list is then suppressed via _consumed.
        intro_groups = [it for it in items if it["_kind"] == "group"]
        for it in body_items:
            kind = it["_kind"]
            if kind == "heading":
                flush()
                current_heading = {"text": it["text"], "level": it["level"]}
                current_blocks = []
                continue
            if kind == "paragraph":
                # If this paragraph is INSIDE an intro+bullet group AND
                # the next item is a bullet_list inside the same group,
                # emit a single combined block.
                paired = _find_grouped_bullet(it, body_items, intro_groups)
                if paired is not None:
                    current_blocks.append({
                        "kind": "bullet_list",
                        "intro": it["text"],
                        "items": _build_bullet_tree(paired["items"]),
                    })
                    paired["_consumed"] = True
                else:
                    current_blocks.append({"kind": "paragraph", "text": it["text"]})
                continue
            if kind == "group":
                inner = it.get("_inner_bullet")
                if inner is None or inner.get("_consumed"):
                    continue
                inner["_consumed"] = True
                # If the wrapper had stacked content (an unboxed
                # sub-heading on top of a colon-intro line), the
                # earlier line is exposed as a sub-section heading
                # and the bullets nest under it.
                extra = (it.get("extra_label") or "").strip()
                if extra:
                    flush()
                    current_heading = {"text": extra, "level": 2}
                    current_blocks = []
                current_blocks.append({
                    "kind": "bullet_list",
                    "intro": it.get("intro_text") or None,
                    "items": _build_bullet_tree(inner["items"]),
                })
                continue
            if kind == "bullet_list":
                if it.get("_consumed"):
                    continue
                current_blocks.append({
                    "kind": "bullet_list",
                    "items": _build_bullet_tree(it["items"]),
                })
                continue
            if kind == "table":
                current_blocks.append(it["table"])
                continue
            if kind == "note":
                current_blocks.append({"kind": "note", "text": it["text"]})
                continue
            # sub_bullet/sub_sub_bullet/footer handled elsewhere

        flush()

        # Drop empty sections.
        sections = [s for s in sections if s["blocks"] or s["heading"]]

        # Per-page level normalization: if a sub-section appears
        # BEFORE any level-1 section on the page (continuation from a
        # previous page), promote it to level 1 so it isn't
        # orphan-rooted in the nested tree.  Universal — every page
        # is treated as starting fresh.
        for i, s in enumerate(sections):
            if s.get("level", 1) == 1:
                break
            if any(prev.get("level", 1) == 1 for prev in sections[:i]):
                break
            s["level"] = 1

        # Nest sections into a tree by level — a section with level N
        # becomes a subsection of the most recent section at level < N.
        # Universal: works for any documents that follow the
        # bigger-heading-on-top convention.
        nested = _nest_sections(sections)

        outline = [
            {"level": s["level"], "heading": s["heading"], "block_count": len(s["blocks"])}
            for s in sections if s["heading"]
        ]

        out: dict[str, Any] = {
            "document": {
                "source": str(Path(source).name),
                "page": page_index,
                "title": title_text or None,
                "metadata": cover_meta + meta_lines,
            },
            "outline": outline,
            "sections": nested,
            "color_legend": COLOR_LEGEND,
        }
        if toc_blocks:
            out["table_of_contents"] = toc_blocks
        return out


# ───────────────────── helper utilities ─────────────────────────────────

def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


# Stray characters that PyMuPDF text clipping sometimes returns when
# the rect edge cuts through descenders of the line directly above the
# region (e.g. "p", "q", "g", "y", "j").  We remove a leading run of
# such SINGLE-LETTER tokens (each followed by whitespace) so paragraphs
# don't start with garbage like "p y q" or "q NOTE:".  Universal —
# real prose almost never opens with N consecutive single letters.
_DESC_CHARS = "pqgyj"
_LEADING_DESC_RUN_RE = re.compile(
    rf"^(?:[{_DESC_CHARS}]\s+){{1,6}}",
    re.IGNORECASE,
)


def _strip_clip_noise(text: str) -> str:
    """Strip leading runs of single-letter descender pixels that
    PyMuPDF clip extraction sometimes returns at the top edge of a
    region.  Has no effect on normal text.
    """
    if not text:
        return text
    cleaned = _LEADING_DESC_RUN_RE.sub("", text, count=1)
    return cleaned.strip()


def _nest_sections(flat: list[dict]) -> list[dict]:
    """Convert flat sections into a nested tree by ``level``.

    A section with ``level == N`` becomes a child of the most recent
    section with ``level < N``.  Headings at the same level are
    siblings.  Each output node has the same fields as the input plus
    a ``subsections`` list.
    """
    roots: list[dict] = []
    stack: list[dict] = []
    for sec in flat:
        node = {
            "heading": sec.get("heading"),
            "level": sec.get("level", 1),
            "blocks": sec.get("blocks", []),
            "subsections": [],
        }
        # Pop until top has strictly smaller level (or empty).
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if stack:
            stack[-1]["subsections"].append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _cluster_rows(cells: list[dict], y_slack: int = 8) -> list[list[dict]]:
    """Cluster cells into rows by y-midpoint."""
    if not cells:
        return []
    sorted_cells = sorted(cells, key=lambda c: (_px(c)[1], _px(c)[0]))
    rows: list[list[dict]] = []
    for c in sorted_cells:
        cy = 0.5 * (_px(c)[1] + _px(c)[3])
        placed = False
        for row in rows:
            ry0 = min(_px(rc)[1] for rc in row)
            ry1 = max(_px(rc)[3] for rc in row)
            if abs(cy - 0.5 * (ry0 + ry1)) <= y_slack:
                row.append(c)
                placed = True
                break
        if not placed:
            rows.append([c])
    rows.sort(key=lambda r: min(_px(c)[1] for c in r))
    return rows


_BULLET_LEAD_RE = re.compile(
    r"^(?:[•\u2022▪\u25aa]|o\s|\d{1,3}\.\s|[A-Za-z][\.\)]\s)"
)


def _bullet_depth(item: str) -> int:
    """Return the bullet depth: 1 = • / N. / A), 2 = o, 3 = ▪."""
    head = item.lstrip()
    if not head:
        return 1
    c0 = head[0]
    if c0 in ("\u25aa", "▪"):
        return 3
    if c0 == "o" and (len(head) == 1 or head[1] in (" ", "\t")):
        return 2
    return 1


def _strip_bullet_marker(item: str) -> str:
    """Remove the leading bullet glyph (and one space) from an item."""
    head = item.lstrip()
    if not head:
        return ""
    if head[0] in ("•", "\u2022", "▪", "\u25aa"):
        return head[1:].lstrip()
    if head[0] == "o" and (len(head) == 1 or head[1] in (" ", "\t")):
        return head[2:].lstrip()
    m = re.match(r"^\d{1,3}\.\s+", head)
    if m:
        return head[m.end():]
    m = re.match(r"^[A-Za-z][\.\)]\s+", head)
    if m:
        return head[m.end():]
    return head


def _split_bullet_items(text: str) -> list[str]:
    """Split a yellow bullet block's flat text into individual items.

    Drops fragments that don't actually start with a bullet marker
    (clip-edge noise — descender pixels of the line above leaking in
    as stray characters).
    """
    if not text:
        return []
    parts = re.split(
        r"(?:^|\s)(?=(?:•|\u2022|\u25aa|▪|o\s|\d{1,3}\.\s|[A-Za-z][\.\)]\s))",
        text.strip(),
    )
    items: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not _BULLET_LEAD_RE.match(p):
            continue
        items.append(p)
    return items


def _build_bullet_tree(flat_items: list[str]) -> list[dict]:
    """Convert a flat list of bullet strings into a nested tree.

    Each node is ``{"text", "children": [...]}``.  Depth is derived
    from the leading glyph (•=1, o=2, ▪=3).
    """
    root: list[dict] = []
    stack: list[tuple[int, dict]] = []
    for raw in flat_items:
        depth = _bullet_depth(raw)
        node = {"text": _strip_bullet_marker(raw), "children": []}
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            root.append(node)
        stack.append((depth, node))
    return root


def _has_subsequent_body_under_heading(items: list[dict], heading: dict) -> bool:
    """True if a paragraph/table/bullet appears after this heading on the page."""
    after_y = heading["y"]
    for it in items:
        if it.get("y", -1) <= after_y:
            continue
        if it["_kind"] in ("paragraph", "table", "bullet_list"):
            return True
    return False


def _looks_like_callout(text: str) -> bool:
    """A "section heading" that's actually a one-line callout sentence.

    True when the text reads like a complete sentence:
      * has 8+ words AND
      * ends in sentence-final punctuation (.?!) OR is long (>= 60 chars).
    A real label/heading tends to be short (< 8 words) and rarely ends
    with a period.
    """
    if not text:
        return False
    t = text.strip()
    n_words = len(t.split())
    ends_sentence = t.endswith((".", "?", "!"))
    if n_words >= 8 and (ends_sentence or len(t) >= 60):
        return True
    return False


def _looks_like_section_label(text: str) -> bool:
    """Generic structural cue (numbered prefix, ALL CAPS, ends with marker)."""
    t = text.strip()
    if not t:
        return False
    if _NUM_PREFIX_RE.match(t):
        return True
    # Mostly-uppercase short label e.g. "SCOPE OF WORK"
    letters = [c for c in t if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.85:
        return True
    return False


def _find_grouped_bullet(intro_item: dict, items: list[dict],
                         groups: list[dict]) -> dict | None:
    """If intro_item sits inside a group bbox AND a bullet_list also sits
    inside the same group bbox, return the bullet_list dict for pairing.
    """
    for g in groups:
        gx0, gy0, gx1, gy1 = g["px"]
        if not _bbox_inside(intro_item["px"], (gx0, gy0, gx1, gy1)):
            continue
        for it in items:
            if it["_kind"] != "bullet_list":
                continue
            if _bbox_inside(it["px"], (gx0, gy0, gx1, gy1)):
                return it
    return None


def _collect_nested(bullet_item: dict, items: list[dict]) -> list[dict[str, Any]]:
    """Return nested sub_bullet / sub_sub_bullet entries inside the bullet bbox."""
    bx0, by0, bx1, by1 = bullet_item["px"]
    nested: list[dict[str, Any]] = []
    for it in items:
        if it["_kind"] not in ("sub_bullet", "sub_sub_bullet"):
            continue
        if _bbox_inside(it["px"], (bx0, by0, bx1, by1)):
            nested.append({
                "level": 2 if it["_kind"] == "sub_bullet" else 3,
                "text": it["text"],
            })
    nested.sort(key=lambda n: n.get("level", 0))
    return nested


# ───────────────────── CLI shim ─────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Color-driven structured doc extraction.")
    ap.add_argument("--overlay-json", required=True,
                    help="Path to overlay JSON written by detect_standalone --json-out")
    ap.add_argument("--pdf", required=True,
                    help="Source PDF path")
    ap.add_argument("--out", required=True,
                    help="Output structured JSON path")
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.overlay_json).read_text())
    doc = extract_structured(payload, pdf_path=args.pdf)
    write_structured(args.out, doc)
    print(f"Structured doc -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLOR_LEGEND",
    "extract_structured",
    "write_structured",
    "main",
]
