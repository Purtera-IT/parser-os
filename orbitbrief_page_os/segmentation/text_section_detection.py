"""Font-aware text section title detection.

This is a complement to the geometry-based box detector in
``detect_standalone``: many engineering / structural-notes pages have no
real cell grid for the contour pipeline to grip on, but they always have
**bold ALL-CAPS section headers** above each block of text.  This module
finds those headers via PyMuPDF font information and returns synthetic
BLUE title bands that the standard overlay renderer paints on top of the
page image.

Public API
----------

- ``TextSectionConfig`` — tuning knobs (separate from the box detector's
  ``Cfg`` so this module can be reused / re-tuned independently).
- ``detect_text_sections(pdf_path, page_index, scale, cw_quarter_turns,
  visible_box_cls, rect_cls, cfg=None)`` — returns a list of synthetic
  ``VisibleBox`` instances ready to be appended to a detection result.

The function takes ``visible_box_cls`` / ``rect_cls`` so it stays
decoupled from a single ``VisibleBox`` definition; pass the same classes
your detector returns.

Algorithm
---------

1. Open the PDF with PyMuPDF and read every text span with its font name,
   bold flag, and bounding box (in PDF points).
2. Filter to candidate header spans:

   - bold (font flag bit 4, or font name contains ``Bold/Black/Heavy``),
   - all letters are uppercase,
   - at least ``min_header_chars`` letters,
   - letter height (``min(width, height)`` of the bbox) ≤ ``max_header_height_pt``.

   Using ``min(w, h)`` rather than just ``h`` makes this work for
   sideways-stored text where the span bbox is tall and narrow instead
   of wide and short.

3. Use the page's ``cw_quarter_turns`` (already inferred upstream from
   character reading-order) to know which axis is the column axis (the
   axis the headers stack along) vs the section axis (the axis a section
   body extends along).
4. Group headers into columns by snapping the column-axis position into
   buckets of ``column_x_tol_pt`` width.
5. For each header, build a "title band" PDF bbox that is

   - full-column-wide along the column axis (taken from the union of
     header bboxes in that column), and
   - header-tall along the section axis with a small ``band_pad_pt`` to
     include any underline below the bold text.

6. Transform every PDF bbox into image-pixel coords using
   ``_pdf_bbox_to_image_bbox`` (rotation- and scale-aware).
7. Emit one synthetic ``VisibleBox`` per header with ``box_id`` ending in
   ``"_title"`` so the standard overlay renderer paints it as a blue band.

Returns ``[]`` if PyMuPDF is unavailable or fewer than
``min_headers_per_page`` candidates are found (so it's a no-op on
tabular pages where the box detector already does the work).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:                            # PyMuPDF is the only hard dep here
    import fitz as _fitz
except Exception:               # pragma: no cover
    _fitz = None


# ─── public config ────────────────────────────────────────────────────────────

@dataclass
class TextSectionConfig:
    """Tunables for font-based text-section detection."""

    enabled: bool = True
    min_header_chars: int = 2              # short tokens like "C.I." / "P.I." count
    # When a header is INSIDE an ORANGE cell, keep it only if there are
    # ≥ this many OTHER bold/all-caps headers in the same image-y row.
    # Multiple aligned table-header cells (ITEM/C.I./P.I./...) → keep
    # all of them.  A lone bold word in a cell → drop.
    table_header_row_min_neighbors: int = 2
    table_header_row_y_tol_px: int = 10    # image-y tolerance for "same row"
    min_headers_per_page: int = 4          # need ≥ N to bother running
    max_header_height_pt: float = 20.0     # letter-height ceiling in points
    column_x_tol_pt: float = 14.0          # spans within this dx are "same column"
    band_pad_pt: float = 1.0               # PDF-pt padding around header bbox
                                           # (just enough to catch the underline
                                           # below the bold text without leaking
                                           # into the first line of body text)

    # Section body wrappers: when True, each detected header also gets a
    # BLUE outlined box around its body text — extending from the header
    # start in the section axis to the next header in the same column
    # (or the page edge for the last header).  Drawn as 3 px outlined
    # rectangles (no fill), so they don't visually compete with the
    # alpha-blended title bands.
    emit_body_wrappers: bool = True
    body_gap_pt: float = 1.0               # gap between body and next header
    body_min_extent_pt: float = 8.0        # don't bother emitting if body would be tiny


# ─── coord transforms (PDF point → rendered-image pixel) ──────────────────────

def _pdf_pt_to_image_xy(x_pt: float, y_pt: float,
                         page_w_pt: float, page_h_pt: float,
                         scale: float, cw_quarter_turns: int) -> tuple[float, float]:
    """Map a PyMuPDF point (origin top-left, y-down) to rendered-image
    pixel coords (also top-left, y-down) given render scale and number
    of CW quarter-turns applied via ``np.rot90(k=-cw_quarter_turns)``
    after rendering at pypdfium ``rotation=0``.

    NOTE: PyMuPDF returns rectangles already in top-left/y-down coords
    (the same convention as the rendered image), so no y-flip is needed.
    Both pypdfium2's render at rotation=0 and PyMuPDF's text bboxes
    agree that y=0 is the visual top of the page.
    """
    px = x_pt * scale
    py = y_pt * scale
    W_orig = page_w_pt * scale
    H_orig = page_h_pt * scale
    n = cw_quarter_turns % 4
    if n == 0:
        return (px, py)
    if n == 1:                              # CW 90°
        return (H_orig - py, px)
    if n == 2:                              # 180°
        return (W_orig - px, H_orig - py)
    return (py, W_orig - px)                # CW 270° (= CCW 90°)


def _pdf_bbox_to_image_bbox(bbox_pt: tuple[float, float, float, float],
                             page_w_pt: float, page_h_pt: float,
                             scale: float, cw_quarter_turns: int
                             ) -> tuple[int, int, int, int]:
    """PDF bbox ``(x0, y0, x1, y1)`` → image bbox ``(x0, y0, x1, y1)``,
    rotation-aware.  Returns int pixel coords."""
    x0p, y0p, x1p, y1p = bbox_pt
    pts = [
        _pdf_pt_to_image_xy(x0p, y0p, page_w_pt, page_h_pt, scale, cw_quarter_turns),
        _pdf_pt_to_image_xy(x1p, y0p, page_w_pt, page_h_pt, scale, cw_quarter_turns),
        _pdf_pt_to_image_xy(x0p, y1p, page_w_pt, page_h_pt, scale, cw_quarter_turns),
        _pdf_pt_to_image_xy(x1p, y1p, page_w_pt, page_h_pt, scale, cw_quarter_turns),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (int(round(min(xs))), int(round(min(ys))),
            int(round(max(xs))), int(round(max(ys))))


# ─── header detection ─────────────────────────────────────────────────────────

def _candidate_headers(pdf_path: str, page_index: int,
                        cfg: TextSectionConfig
                        ) -> tuple[list[dict[str, Any]],
                                   list[tuple[float, float, float, float]],
                                   dict[tuple[float, float, float, float], str],
                                   float, float]:
    """Return (headers, all_span_bboxes, span_text_by_bbox, page_w_pt, page_h_pt).

    ``all_span_bboxes`` is the bbox of every non-empty text span on the
    page (including non-header spans).  Used to compute the *actual*
    body extent of each section: the body box wraps all spans whose
    centre falls inside the section's region.

    ``span_text_by_bbox`` maps each bbox tuple to its text content.
    Used by ``rules/label_value_pairing.py`` to identify colon-ending
    labels that should pull in their right-hand values.
    """
    if _fitz is None:
        return [], [], {}, 0.0, 0.0

    fdoc = _fitz.open(pdf_path)
    try:
        page = fdoc[page_index]
        page_w_pt = float(page.rect.width)
        page_h_pt = float(page.rect.height)
        td = page.get_text("dict")
    finally:
        pass

    headers: list[dict[str, Any]] = []
    all_spans: list[tuple[float, float, float, float]] = []
    all_spans_text: dict[tuple[float, float, float, float], str] = {}
    for block in td.get("blocks", []):
        if block.get("type", 0) != 0:                # skip image blocks
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                bbox = span.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                bx0, by0, bx1, by1 = bbox
                if bx1 <= bx0 or by1 <= by0:
                    continue
                bbox_key = (bx0, by0, bx1, by1)
                all_spans.append(bbox_key)
                all_spans_text[bbox_key] = text
                w_pt, h_pt = bx1 - bx0, by1 - by0
                letter_h = min(w_pt, h_pt)
                if letter_h <= 0 or letter_h > cfg.max_header_height_pt:
                    continue
                font = span.get("font", "") or ""
                flags = int(span.get("flags", 0))
                is_bold = bool(flags & 16) or any(
                    tag in font for tag in ("Bold", "bold", "Black", "Heavy"))
                letters = [c for c in text if c.isalpha()]
                if len(letters) < cfg.min_header_chars:
                    continue
                # ALL letters uppercase is the strict canonical case.
                # Also accept "predominantly uppercase" titles (≥85%
                # uppercase) so unit-prefixed titles like
                # ``92mm INTERIOR PARTITION - TYPICAL`` (lowercase
                # letters only in the unit ``mm``) qualify.
                #
                # Parenthetical content is excluded from the dominant
                # caps test because it's commonly a Title-Case
                # descriptor on a CAPS title — e.g.
                # ``WALKWAY WALL (Exterior to Interior)`` or
                # ``NORTH WALL (STORAGE GARAGE)``.  The canonical title
                # is what's outside the parens; the parenthetical is
                # supplemental.
                #
                # Mixed-case prose (typical "Hello World" Title-Case)
                # has at most ~50% uppercase letters even after paren
                # stripping, so the 85% threshold still excludes it.
                import re as _re
                text_no_parens = _re.sub(r"\([^)]*\)", "", text)
                letters_no_parens = [c for c in text_no_parens if c.isalpha()]
                upper_count = sum(1 for c in letters if c.isupper())
                upper_count_np = sum(
                    1 for c in letters_no_parens if c.isupper())
                is_caps = (upper_count == len(letters))
                is_caps_dominant = (
                    (upper_count / len(letters)) >= 0.85
                    or (len(letters_no_parens) >= cfg.min_header_chars
                        and (upper_count_np / len(letters_no_parens)) >= 0.85))
                ends_colon = text.endswith(":")
                colon_title = is_caps and ends_colon and len(letters) >= 8
                if (is_bold and is_caps_dominant) or colon_title:
                    headers.append({
                        "text": text, "bbox": bbox, "font": font,
                        "is_bold": is_bold, "is_caps": is_caps,
                    })

    # ── Universal underline-titles rule (rules/underline_titles.py) ───────
    # Non-bold all-caps spans with a tight (~text-width) drawn underline
    # beneath them are section titles too — even when they aren't bold
    # and don't end with a colon.  Catches test7 sub-section headings
    # like ``EAST WALL`` / ``SOUTH WALL`` / ``WEST WALL`` / ``NORTH WALL
    # (STORAGE GARAGE)`` etc. that the original gate rejected.  The
    # width-ratio test (under_w / text_w <= 1.5) excludes table column
    # headers whose "underline" is actually the column divider.  The
    # isolation test additionally excludes any column header rows that
    # slip past the width test.
    from .rules import collect_underlined_caps_titles
    underline_titles = collect_underlined_caps_titles(page)
    # Dedupe: skip any already-known header bbox.
    existing_bboxes = {tuple(round(v, 1) for v in h["bbox"]) for h in headers}
    for ut in underline_titles:
        key = tuple(round(v, 1) for v in ut["bbox"])
        if key in existing_bboxes:
            continue
        headers.append(ut)

    fdoc.close()

    # ── Universal header-isolation rule (rules/header_isolation.py) ───────
    # Non-bold colon-uppercase spans are field labels (not titles) when
    # they appear in stacks.  A real section title that lacks the bold
    # weight is still solitary in its column.  This filters the 86
    # non-bold field labels in test7's BUILDING CODE SUMMARY zone while
    # preserving the 1 isolated colon-only title in test5
    # (``CERTIFICATE OF AUTHORIZATION NO:``) plus test7's real titles
    # (``ABBREVIATIONS:``, ``GENERAL NOTES:``, ``PROJECT NUMBER:``).
    from .rules import filter_isolated_colon_titles
    headers = filter_isolated_colon_titles(headers)

    # ── Universal fact-statement rule (rules/fact_statements.py) ─────────
    # A bold uppercase line containing ``=`` is a fact statement
    # (calculation result / summary), not a section title.  Drops e.g.
    # ``BUILDING TOTAL = 123 PERSONS PERMITTED`` on test7 — bold, all
    # caps, would otherwise pass the header gate and get its own title
    # band, but it's a summary fact that belongs INSIDE SECTION 3.1's
    # body, not above its own sub-section.
    from .rules import filter_fact_statements
    headers = filter_fact_statements(headers)

    return headers, all_spans, all_spans_text, page_w_pt, page_h_pt


# ─── main entry ───────────────────────────────────────────────────────────────

def detect_text_sections(pdf_path: str, page_index: int,
                          scale: float,
                          cw_quarter_turns: int,
                          visible_box_cls,
                          rect_cls,
                          cfg: TextSectionConfig | None = None,
                          existing_boxes: list | None = None) -> list:
    """Detect bold ALL-CAPS section headers and return synthetic
    ``VisibleBox`` title bands + body wrappers ready to be appended to
    the detector's output list.

    Pass the same ``VisibleBox`` and ``Rect`` classes the detector emits
    so the synthetic boxes integrate seamlessly into the rendering pass.

    ``existing_boxes`` (optional) is the list of already-detected boxes
    from the geometric pipeline.  When provided, each text-section
    body / title bbox is CLAMPED to the smallest non-synthetic BLUE
    wrapper that contains the header — so body wrappers can never spill
    into a sibling structural region (e.g. the title block on the right
    side of an engineering drawing).
    """
    cfg = cfg or TextSectionConfig()
    if not cfg.enabled or _fitz is None:
        return []

    headers, all_spans, all_spans_text, page_w_pt, page_h_pt = _candidate_headers(
        pdf_path, page_index, cfg)
    if len(headers) < cfg.min_headers_per_page:
        return []

    qt = cw_quarter_turns % 4
    sideways = qt in (1, 3)

    # Drop NON-BOLD colon-ending candidates whose IMMEDIATE next line
    # in the line-stack does NOT continue under the candidate (i.e.
    # the next line wraps back to a shallower indent).  This is the
    # universal "section heading" structural signal:
    #   • a real title       → next line(s) indented under it
    #     (next-line start ≥ candidate's reading-axis start)
    #   • body emphasis      → next line wraps to paragraph margin
    #     (next-line start < candidate's reading-axis start)
    # For sideways pages: line-stack axis = PDF x; reading axis = PDF y.
    # For upright pages: line-stack axis = PDF y; reading axis = PDF x.
    if all_spans and headers:
        # Cache spans by line-stack key for fast neighbor lookup.
        span_keys = []
        for sb in all_spans:
            ls = (sb[0] + sb[2]) / 2 if sideways else (sb[1] + sb[3]) / 2
            span_keys.append((ls, sb))
        span_keys.sort(key=lambda x: x[0])

        # Quick text-content guards for NON-BOLD candidates: drop
        # obvious sentence fragments before the structural check.
        SENTENCE_START_WORDS = {
            "MAINTAIN", "USE", "FOLLOW", "VERIFY", "DO", "MAKE", "KEEP", "SEE",
            "PROVIDE", "INSTALL", "SUBMIT", "REFER", "COORDINATE",
            "BUT", "IF", "AND", "OR", "IS", "ARE", "WAS", "WERE",
            "WILL", "SHALL", "MAY", "MUST", "CAN", "FOR", "TO", "OF", "AT",
            "WITH", "BY", "WHEN", "WHERE", "WHICH", "AS",
        }
        def _wcount(t: str) -> int:
            return sum(1 for w in t.replace(":", "").split() if w)
        def _firstw(t: str) -> str:
            for w in t.replace(":", "").split():
                return w.upper()
            return ""

        kept: list[dict[str, Any]] = []
        for h in headers:
            if h["is_bold"] or h.get("is_table_header"):
                kept.append(h)
                continue
            # Non-bold path: text content guards FIRST.
            t = h["text"]
            if _wcount(t) > 6 or _firstw(t) in SENTENCE_START_WORDS:
                continue
            hb = h["bbox"]
            # Header's line-stack center, line-stack letter height,
            # and reading-axis start.
            if sideways:
                hs_center = (hb[0] + hb[2]) / 2
                letter_h  = hb[2] - hb[0]
                read_start = hb[1]
            else:
                hs_center = (hb[1] + hb[3]) / 2
                letter_h  = hb[3] - hb[1]
                read_start = hb[0]
            # For sideways CCW the next line BELOW has a SMALLER PDF x
            # than the header (lines stack toward smaller PDF x).  For
            # CW, larger PDF x.  For upright, larger PDF y.
            line_step_dir = +1 if (not sideways or qt == 1) else -1
            # Search a half-window in the line-stack axis: 1× to ~3×
            # letter_h beyond the header — that's where child lines
            # live.  KEEP the candidate if ANY span in this window is
            # an INDENTED CHILD that begins close to the candidate's
            # reading-axis start (within ~2× letter_h).  This filters
            # out random spans from unrelated blocks that happen to
            # fall in the line-stack window.
            min_step  = letter_h * 0.5
            max_step  = letter_h * 5.0
            close_band = letter_h * 3.0     # how close child's start may be
            has_indented_child = False
            had_any_neighbor = False
            for ls, sb in span_keys:
                step = (ls - hs_center) * line_step_dir
                if step <= min_step or step > max_step:
                    continue
                had_any_neighbor = True
                sb_read_start = sb[1] if sideways else sb[0]
                # Indented child = starts AT or AFTER candidate AND
                # within close_band of candidate's start.
                if (read_start - 2.0) <= sb_read_start <= (read_start + close_band):
                    has_indented_child = True
                    break
            if not had_any_neighbor or has_indented_child:
                kept.append(h)
        headers = kept
        if len(headers) < cfg.min_headers_per_page:
            return []

    # Tag every header with whether it lives inside an existing ORANGE
    # cell.  We KEEP all bold/all-caps headers (every one is a real
    # title — column headers, section dividers, sub-section labels) but
    # the ``_in_orange`` flag tells the emit loop to skip the body
    # wrapper for in-cell headers (the cell already has its border —
    # adding a body rectangle would clutter).
    if existing_boxes:
        existing_orange_imgs: list[tuple[int, int, int, int]] = []
        for b in existing_boxes:
            if getattr(b, "synthetic", False):
                continue
            if getattr(b, "color", None) != "ORANGE":
                continue
            px = getattr(b, "px_bbox", None)
            if px and len(px) == 4:
                existing_orange_imgs.append(tuple(int(v) for v in px))

        def _inside_orange(ix: tuple[int, int, int, int]) -> bool:
            hx0, hy0, hx1, hy1 = ix
            m = 2
            for (ox0, oy0, ox1, oy1) in existing_orange_imgs:
                if (ox0 - m) <= hx0 and (oy0 - m) <= hy0 \
                        and hx1 <= (ox1 + m) and hy1 <= (oy1 + m):
                    return True
            return False

        for h in headers:
            ix = _pdf_bbox_to_image_bbox(
                h["bbox"], page_w_pt, page_h_pt, scale, cw_quarter_turns)
            h["_in_orange"] = _inside_orange(ix)
        if len(headers) < cfg.min_headers_per_page:
            return []

    # Reading-axis-aware grouping.  See module docstring step 3-4.
    def _col_key(b):
        bx0, by0, bx1, by1 = b
        return by0 if sideways else bx0

    def _sec_top(b):
        bx0, by0, bx1, by1 = b
        return bx1 if sideways else by0

    # Section-axis direction in PDF coords.  PyMuPDF uses top-left/y-down
    # coords (same as the rendered image), so for HORIZONTAL text the
    # body grows DOWN = PDF y INCREASING → sec_dir = +1.  For sideways
    # text, body grows along PDF x; CCW (qt=3) reads top→bottom in PDF y
    # which after rotation appears left→right in image — body in PDF x
    # extends in DECREASING direction (sec_dir = -1).  CW (qt=1) is the
    # mirror case.
    if not sideways:
        sec_dir = +1                         # horizontal y-down: body grows down
    else:
        sec_dir = -1 if qt == 3 else +1      # sideways: rotation-handed

    tol = cfg.column_x_tol_pt
    by_col: dict[float, list[dict[str, Any]]] = {}
    for h in headers:
        ck = _col_key(h["bbox"])
        snapped = round(ck / tol) * tol
        by_col.setdefault(snapped, []).append(h)
    # Sort each column in READING ORDER along the section axis.  When
    # sec_dir = +1 the first header has the SMALLEST sec_top; when
    # sec_dir = -1 it has the LARGEST.
    for ck in by_col:
        by_col[ck].sort(key=lambda h: _sec_top(h["bbox"]) * sec_dir)

    # Each column's extent in the COLUMN axis is the strip of page that
    # belongs to that column (= the body-text width).  Inner columns get
    # bounded by the next column's key (minus a small inter-column gap).
    # The LAST column previously ran to the page edge — but on pages
    # where the right side is occupied by a title block (ASPEN-style
    # sheets) that pushes the body box right through the title block.
    # Instead, for the last column we measure the actual text spans
    # within the column's section-axis range and bound col_hi by the
    # max span edge + a tiny pad, so the body box stops at the real
    # text and leaves the title block alone.
    sorted_keys = sorted(by_col.keys())
    col_axis_max = page_h_pt if sideways else page_w_pt
    inter_col_gap = 4.0
    last_col_text_pad = 6.0
    col_extents: dict[float, tuple[float, float]] = {}
    for i, ck in enumerate(sorted_keys):
        col_headers = by_col[ck]
        if sideways:
            min_edge = min(h["bbox"][1] for h in col_headers)
        else:
            min_edge = min(h["bbox"][0] for h in col_headers)
        col_lo = min_edge - 1.0
        if i + 1 < len(sorted_keys):
            col_hi = sorted_keys[i + 1] - inter_col_gap
        else:
            # Last column: bound by the actual rightmost (= biggest
            # column-axis coord) text span that lies in this column's
            # section-axis range.  Section-axis range = union of header
            # bbox section-axis extents in this column.
            if sideways:
                sec_axis_lo = min(h["bbox"][0] for h in col_headers)
                sec_axis_hi = max(h["bbox"][2] for h in col_headers)
                # Allow body to extend the section axis to include
                # everything up to / down to the page edge in the
                # body-grow direction (sec_dir).  We use the broader
                # cone of "any span whose section-axis centre is in
                # [0, sec_axis_hi]" so the body of headers further along
                # the section axis is included in the measurement too.
                if sec_dir < 0:
                    sec_lo, sec_hi = 0.0, sec_axis_hi
                else:
                    sec_lo, sec_hi = sec_axis_lo, page_w_pt
                cands = [s for s in all_spans
                         if sec_lo <= (s[0] + s[2]) * 0.5 <= sec_hi
                         and (s[1] + s[3]) * 0.5 >= col_lo - 5
                         and (s[1] + s[3]) * 0.5 <= col_axis_max]
                if cands:
                    col_hi = min(col_axis_max - 1.0,
                                 max(s[3] for s in cands) + last_col_text_pad)
                else:
                    col_hi = col_axis_max - 1.0
            else:
                sec_axis_lo = min(h["bbox"][1] for h in col_headers)
                sec_axis_hi = max(h["bbox"][3] for h in col_headers)
                if sec_dir < 0:
                    sec_lo, sec_hi = 0.0, sec_axis_hi
                else:
                    sec_lo, sec_hi = sec_axis_lo, page_h_pt
                cands = [s for s in all_spans
                         if sec_lo <= (s[1] + s[3]) * 0.5 <= sec_hi
                         and (s[0] + s[2]) * 0.5 >= col_lo - 5
                         and (s[0] + s[2]) * 0.5 <= col_axis_max]
                if cands:
                    col_hi = min(col_axis_max - 1.0,
                                 max(s[2] for s in cands) + last_col_text_pad)
                else:
                    col_hi = col_axis_max - 1.0
        if col_hi <= col_lo:
            if sideways:
                ys = [h["bbox"][1] for h in col_headers] + \
                     [h["bbox"][3] for h in col_headers]
            else:
                ys = [h["bbox"][0] for h in col_headers] + \
                     [h["bbox"][2] for h in col_headers]
            col_lo, col_hi = min(ys), max(ys)
        col_extents[ck] = (col_lo, col_hi)

    # Build clamp candidates from the geometric pipeline:
    #   * `clamp_wrappers` — non-synthetic BLUE wrappers; each text-
    #     section bbox is clamped to the smallest wrapper that
    #     contains the header (= must stay INSIDE).
    #   * `obstacle_boxes` — every other non-synthetic non-header box;
    #     body bboxes shrink so they never OVERLAP these (this is what
    #     keeps body wrappers from spilling into the title-block /
    #     side-panel cells that share the same outer wrapper).
    clamp_wrappers: list[tuple[int, int, int, int]] = []
    obstacle_boxes: list[tuple[int, int, int, int]] = []
    if existing_boxes:
        for b in existing_boxes:
            if getattr(b, "synthetic", False):
                continue
            px = getattr(b, "px_bbox", None)
            if not px or len(px) != 4:
                continue
            box = tuple(int(v) for v in px)
            color = getattr(b, "color", None)
            if color == "BLUE":
                clamp_wrappers.append(box)
            obstacle_boxes.append(box)
    clamp_inset = 2   # px — stay just inside the wrapper border

    def _smallest_containing_wrapper(hbox_img: tuple[int, int, int, int]):
        hx0, hy0, hx1, hy1 = hbox_img
        best = None
        best_area = None
        for (wx0, wy0, wx1, wy1) in clamp_wrappers:
            if wx0 <= hx0 and wy0 <= hy0 and wx1 >= hx1 and wy1 >= hy1:
                area = (wx1 - wx0) * (wy1 - wy0)
                if best_area is None or area < best_area:
                    best = (wx0, wy0, wx1, wy1)
                    best_area = area
        return best

    def _shrink_around_obstacles(bbox: tuple[int, int, int, int],
                                  hbox: tuple[int, int, int, int]
                                  ) -> tuple[int, int, int, int]:
        """Shrink ``bbox`` so it does not overlap any obstacle box that
        sits OUTSIDE the body.  If the obstacle is fully INSIDE the body
        bbox, it's a child cell of this section and we leave it alone —
        the body wrapper is supposed to enclose its children.

        This generalises an earlier exemption that only spared obstacles
        straddling the header.  Universal label-value pairing (Rule 7)
        widens bodies to enclose values, sub-blocks, and sub-section
        underlines (e.g. ``MAIN FLOOR OCCUPANT LOAD``'s blue underline
        on test7) that don't straddle the header but still belong to
        the section.
        """
        x0, y0, x1, y1 = bbox
        hx0, hy0, hx1, hy1 = hbox
        margin = 2
        for (ox0, oy0, ox1, oy1) in obstacle_boxes:
            # Skip the header span itself (or boxes that contain it).
            if ox0 <= hx0 and oy0 <= hy0 and ox1 >= hx1 and oy1 >= hy1:
                continue
            # Skip if no overlap at all.
            if ox1 <= x0 or ox0 >= x1 or oy1 <= y0 or oy0 >= y1:
                continue
            # Skip when the obstacle is entirely inside the body bbox.
            # The body wrapper is built to enclose the section's content;
            # obstacles fully contained in it are by definition children
            # of the section (this includes sub-section underlines that
            # don't straddle the header — see Rule 7 in RULES.md).
            if x0 <= ox0 and x1 >= ox1 and y0 <= oy0 and y1 >= oy1:
                continue
            # Compute how far each edge would have to retract to clear
            # the obstacle; pick the minimum (= least disruptive).
            shrink_right  = x1 - (ox0 - margin)        # move x1 leftward
            shrink_left   = (ox1 + margin) - x0        # move x0 rightward
            shrink_bottom = y1 - (oy0 - margin)        # move y1 upward
            shrink_top    = (oy1 + margin) - y0        # move y0 downward
            # Only allow edges that don't go through the header.
            cands = []
            if ox0 - margin >= hx1:                    # obstacle is to the right of header
                cands.append(("R", shrink_right))
            if ox1 + margin <= hx0:                    # obstacle is to the left of header
                cands.append(("L", shrink_left))
            if oy0 - margin >= hy1:                    # obstacle is below header
                cands.append(("B", shrink_bottom))
            if oy1 + margin <= hy0:                    # obstacle is above header
                cands.append(("T", shrink_top))
            if not cands:
                continue
            side, _ = min(cands, key=lambda c: c[1])
            if side == "R":
                x1 = max(x0, ox0 - margin)
            elif side == "L":
                x0 = min(x1, ox1 + margin)
            elif side == "B":
                y1 = max(y0, oy0 - margin)
            elif side == "T":
                y0 = min(y1, oy1 + margin)
        return (x0, y0, x1, y1)

    def _emit_box(box_id_suffix: str, pdf_bbox, parent_id=None,
                  clamp_to: tuple[int, int, int, int] | None = None,
                  hbox_img: tuple[int, int, int, int] | None = None):
        """Helper: build a synthetic VisibleBox from a PDF bbox.

        ``clamp_to``  — image bbox to clip to (the smallest containing
                        BLUE wrapper).
        ``hbox_img``  — header image bbox; when provided we additionally
                        shrink to avoid overlapping any non-header
                        obstacle box (e.g., title-block cells).
        """
        x0, y0, x1, y1 = _pdf_bbox_to_image_bbox(
            pdf_bbox, page_w_pt, page_h_pt, scale, cw_quarter_turns)
        if clamp_to is not None:
            cx0, cy0, cx1, cy1 = clamp_to
            x0 = max(x0, cx0 + clamp_inset)
            y0 = max(y0, cy0 + clamp_inset)
            x1 = min(x1, cx1 - clamp_inset)
            y1 = min(y1, cy1 - clamp_inset)
        if hbox_img is not None and obstacle_boxes:
            x0, y0, x1, y1 = _shrink_around_obstacles((x0, y0, x1, y1), hbox_img)
        if x1 <= x0 or y1 <= y0:
            return None
        return visible_box_cls(
            box_id=f"textsec_{sid}{box_id_suffix}",
            rect=rect_cls(x0 / scale, y0 / scale, x1 / scale, y1 / scale),
            area_pt2=(x1 - x0) * (y1 - y0) / (scale * scale),
            fill_ratio=1.0,
            nested_depth=3,
            is_outer_wrapper=False,
            parent_box_id=parent_id,
            color="BLUE",
            px_bbox=(x0, y0, x1, y1),
            children_count=0,
            synthetic=True,
        )

    synth: list = []
    pad = cfg.band_pad_pt
    body_gap = cfg.body_gap_pt
    sid = 0
    # Map from header dict id() → emitted sid, for post-processing
    # parent-child hierarchy linking.
    header_to_sid: dict[int, int] = {}

    # ── Build column-anchor → (first_y, last_y) map ──────────────────────
    # Used by the universal label-value-pairing rule (rules/label_value_pairing)
    # to compute "next section column whose vertical band actually overlaps
    # ours" — i.e. the rightmost a body wrapper may grow to.  An anchor
    # whose own section sits in a different y-range doesn't block our
    # sweep, even if its x-position is just past ours.
    column_anchors: list[tuple[float, float, float]] = []
    for ck, hs in by_col.items():
        if not hs:
            continue
        if sideways:
            ys = [h["bbox"][0] for h in hs] + [h["bbox"][2] for h in hs]
        else:
            ys = [h["bbox"][1] for h in hs] + [h["bbox"][3] for h in hs]
        column_anchors.append((ck, min(ys), max(ys)))
    page_extent_for_rule = page_h_pt if sideways else page_w_pt

    for col_key, col_headers in by_col.items():
        col_lo, col_hi = col_extents[col_key]
        n_in_col = len(col_headers)
        for i, h in enumerate(col_headers):
            hx0, hy0, hx1, hy1 = h["bbox"]
            sid += 1
            header_to_sid[id(h)] = sid

            # ── Title band (BLUE alpha-fill via "_title" suffix) ──────
            # Sized to the header text itself + a small pad — the band
            # should highlight ONLY the bold underlined title, not the
            # whole column width (the body wrapper below already shows
            # what the title belongs to).
            band_pdf = (hx0 - pad, hy0 - pad, hx1 + pad, hy1 + pad)
            # Find the smallest existing BLUE wrapper that contains
            # this header — used to clamp both the title band and body.
            hbox_img = _pdf_bbox_to_image_bbox(
                h["bbox"], page_w_pt, page_h_pt, scale, cw_quarter_turns)
            clamp = _smallest_containing_wrapper(hbox_img)
            band_box = _emit_box("_title", band_pdf, clamp_to=clamp,
                                  hbox_img=hbox_img)
            if band_box is None:
                continue
            synth.append(band_box)

            # ── Body wrapper (BLUE outlined "_body" — drawn as 3px frame) ──
            if not cfg.emit_body_wrappers:
                continue
            # In-cell headers (table column headers like ITEM/C.I./P.I. or
            # sub-section dividers like CONCRETE:) get only the title
            # band — no body wrapper.  The cell itself already has its
            # own border; a body wrapper would just look like a
            # redundant blue rectangle around the cell.
            if h.get("_in_orange"):
                continue
            next_h = col_headers[i + 1] if (i + 1 < n_in_col) else None

            # Compute the SECTION REGION in PDF coords.  This is the
            # search region inside which we collect actual text spans to
            # measure the true body extent.  Section axis range comes
            # from header start → next header start; column axis range
            # comes from the column extents we computed above.
            if sideways:
                # Section axis = x.  Range = from header → next header
                # in the sec_dir direction.
                if sec_dir > 0:
                    sec_lo, sec_hi = hx0, ((next_h["bbox"][0] - body_gap)
                                            if next_h else page_w_pt)
                else:
                    sec_lo, sec_hi = ((next_h["bbox"][2] + body_gap)
                                       if next_h else 0.0), hx1
                if (sec_hi - sec_lo) < cfg.body_min_extent_pt:
                    continue
                # Search region: this section's full PDF region.
                # Span belongs to this section if its CENTRE is inside.
                region = (sec_lo, col_lo, sec_hi, col_hi)
            else:
                # Horizontal text: section axis = y, body grows down.
                sec_lo, sec_hi = hy0, ((next_h["bbox"][1] - body_gap)
                                        if next_h else page_h_pt)
                if (sec_hi - sec_lo) < cfg.body_min_extent_pt:
                    continue
                region = (col_lo, sec_lo, col_hi, sec_hi)

            rx0, ry0, rx1, ry1 = region
            # A span belongs to the section when its centre lies inside
            # the region — but with two relaxations:
            #   • Column axis uses BBOX OVERLAP (not strict centre) so
            #     long lines whose midpoint falls 1-2 pt past col_hi/lo
            #     still count.
            #   • Section axis uses centre-in, BUT with a small
            #     half-letter-height pad so spans that share the same
            #     baseline as the next header (e.g. the last line of
            #     this section that visually sits just above the next
            #     header) aren't excluded.
            sec_pad = 4.0          # PDF pt
            col_pad = 6.0
            # Column-axis anchor: every body line should START near the
            # header's column-axis start (e.g. all bullet items aligned
            # with header begin at the same PDF y for a sideways page).
            # Spans whose column-axis start is more than this far from
            # the header's start belong to a different block.
            col_anchor_tol = 12.0
            if sideways:
                hdr_col_start = hy0
            else:
                hdr_col_start = hx0
            inside: list[tuple[float, float, float, float]] = []
            for (sx0, sy0, sx1, sy1) in all_spans:
                cx = (sx0 + sx1) * 0.5
                cy = (sy0 + sy1) * 0.5
                # Orientation filter: a sideways-page section's body
                # consists of VERTICAL text lines (PDF height >= width).
                # Reject horizontal spans (e.g. title-block text).
                w = sx1 - sx0
                h = sy1 - sy0
                if sideways:
                    if w >= h:                      # horizontal span — skip
                        continue
                    # Column-axis anchor — span must START near the
                    # header's column-axis start (same indent / column).
                    if abs(sy0 - hdr_col_start) > col_anchor_tol:
                        continue
                    if not (rx0 - sec_pad <= cx <= rx1 + sec_pad):
                        continue
                    if sy1 < ry0 - col_pad or sy0 > ry1 + col_pad:
                        continue
                else:
                    if h >= w:                      # vertical span — skip
                        continue
                    if abs(sx0 - hdr_col_start) > col_anchor_tol:
                        continue
                    if not (ry0 - sec_pad <= cy <= ry1 + sec_pad):
                        continue
                    if sx1 < rx0 - col_pad or sx0 > rx1 + col_pad:
                        continue
                inside.append((sx0, sy0, sx1, sy1))

            # ── Universal label-value pairing rule ─────────────────────────
            # (rules/label_value_pairing.py)
            # A section "owns" the horizontal band from its own column to
            # the start of the next column whose header sits in a DIFFERENT
            # vertical band.  All text within that expanded rectangle
            # belongs to the section, regardless of x-anchor.  Sub-headings
            # INSIDE the section's y-range (e.g. ``BUILDING TOTAL`` inside
            # SECTION 3.1) belong to the section even though they create
            # their own x-anchor in the body builder's pre-split.
            from .rules import add_value_spans_for_colon_labels
            inside = add_value_spans_for_colon_labels(
                inside, all_spans,
                col_lo=col_lo, col_hi=col_hi,
                sec_lo=sec_lo, sec_hi=sec_hi,
                sideways=sideways,
                other_column_anchors=column_anchors,
                page_extent_pt=page_extent_for_rule,
                inter_col_gap_pt=inter_col_gap,
            )

            # ── Trim spans separated by a large gap from the section ──────
            # ``inside`` may contain stray spans far away from the actual
            # body (e.g. title-block text that happens to share the
            # column-axis range).  Walk the spans outward from the
            # header in the body-grow direction (sec_dir) and stop
            # accumulating once a vertical gap exceeds ~2× letter
            # height.  Anything past that gap is a different block.
            if inside:
                # Sort spans along the SECTION axis in body-grow direction.
                if sideways:
                    # section axis = x.  sec_dir>0 → spans further along
                    # the section axis come last.
                    inside.sort(key=lambda s: s[0] * sec_dir)
                else:
                    inside.sort(key=lambda s: s[1] * sec_dir)
                # Estimate typical line gap: median letter height + gap.
                letter_hs = []
                for s in inside:
                    if sideways:
                        letter_hs.append(s[2] - s[0])
                    else:
                        letter_hs.append(s[3] - s[1])
                letter_hs.sort()
                med_h = letter_hs[len(letter_hs) // 2] if letter_hs else 8.0
                max_gap = max(15.0, med_h * 3.0)  # paragraph-tight
                # Track the EDGE of the previous span that's adjacent to
                # the next span in body-grow direction.  For sec_dir>0
                # that's s[0] of current vs s[2] of previous (we walked
                # forward past previous's right edge).  For sec_dir<0
                # that's s[2] of current vs s[0] of previous.
                trimmed = []
                last_far_edge = None      # edge of last kept span that the next span will hit
                for s in inside:
                    if sideways:
                        cur_near = s[0] if sec_dir > 0 else s[2]
                        cur_far  = s[2] if sec_dir > 0 else s[0]
                    else:
                        cur_near = s[1] if sec_dir > 0 else s[3]
                        cur_far  = s[3] if sec_dir > 0 else s[1]
                    if last_far_edge is not None:
                        # signed gap in body-grow direction: positive = real gap.
                        if sec_dir > 0:
                            gap = cur_near - last_far_edge
                        else:
                            gap = last_far_edge - cur_near
                        if gap > max_gap:
                            break
                    trimmed.append(s)
                    last_far_edge = cur_far
                inside = trimmed
            if inside:
                # Actual text bbox (union of span bboxes) plus pad.
                tx0 = min(s[0] for s in inside) - pad
                ty0 = min(s[1] for s in inside) - pad
                tx1 = max(s[2] for s in inside) + pad
                ty1 = max(s[3] for s in inside) + pad
                # Always include the header itself in the wrapper.
                tx0 = min(tx0, hx0)
                ty0 = min(ty0, hy0)
                tx1 = max(tx1, hx1)
                ty1 = max(ty1, hy1)
                # Clamp ONLY along the section axis (sec_hi/sec_lo).
                # On the column axis we let the body box hug the actual
                # text span union — DON'T force it to col_lo/col_hi
                # since that extends the box past where text ends.
                if sideways:
                    # section axis = x; column axis = y (text union already)
                    tx0 = max(tx0, rx0)
                    tx1 = min(tx1, rx1)
                else:
                    ty0 = max(ty0, ry0)
                    ty1 = min(ty1, ry1)
                body_pdf = (tx0, ty0, tx1, ty1)
            else:
                # No body text spans — fall back to the search region.
                body_pdf = region

            body_box = _emit_box("_body", body_pdf,
                                 parent_id=f"textsec_{sid}_title",
                                 clamp_to=clamp,
                                 hbox_img=hbox_img)
            if body_box is not None:
                synth.append(body_box)

    # ── Universal section-hierarchy rule (rules/section_hierarchy) ────────
    # Identify SECTION X.Y parent headers and their child sub-headings.
    # Two link types:
    #   - STRUCTURAL: same-column child between parent and next parent,
    #     OR a child whose y falls inside a SECTION parent's natural
    #     y-range (in any column).  Expand parent body to enclose
    #     same-column children; mark child as sub-header.
    #   - STYLE-ONLY: cross-column orphan child at top of next column.
    #     Mark as sub-header (green wash) but don't expand parent body
    #     (would create giant rectangles overlapping peer sections).
    #
    # CRITICAL: a child's OWN ``_body`` wrapper is REMOVED so the section
    # renders as a single overarching blue box with green sub-header
    # highlights inside, not nested boxes-inside-boxes.  The child's
    # content is already inside the parent's blue body wrapper; drawing
    # an additional inner blue wrapper around the child creates
    # confusing visual noise (lines splitting the section).
    #
    # See RULES.md Rule 12.
    from .rules import find_parent_child_links
    structural_links, style_links = find_parent_child_links(headers)
    all_links = structural_links + style_links
    if all_links:
        synth_by_id = {b.box_id: b for b in synth}
        # Track which child sids are sub-headers — their bodies will
        # be removed at the end.
        subheader_sids: set[int] = set()
        for parent_i, child_i in all_links:
            parent_h = headers[parent_i]
            child_h = headers[child_i]
            psid = header_to_sid.get(id(parent_h))
            csid = header_to_sid.get(id(child_h))
            if psid is None or csid is None:
                continue
            child_title = synth_by_id.get(f"textsec_{csid}_title")
            # Mark child title as a sub-header for the renderer (both
            # structural and style links get the green wash).
            if child_title is not None:
                try:
                    setattr(child_title, "is_subheader", True)
                except Exception:
                    object.__setattr__(child_title, "is_subheader", True)
                subheader_sids.add(csid)

        # Body-expansion ONLY for structural (same-column) links.
        for parent_i, child_i in structural_links:
            parent_h = headers[parent_i]
            child_h = headers[child_i]
            psid = header_to_sid.get(id(parent_h))
            csid = header_to_sid.get(id(child_h))
            if psid is None or csid is None:
                continue
            parent_body = synth_by_id.get(f"textsec_{psid}_body")
            child_title = synth_by_id.get(f"textsec_{csid}_title")
            child_body = synth_by_id.get(f"textsec_{csid}_body")
            if parent_body is None:
                continue
            px0, py0, px1, py1 = parent_body.px_bbox
            expansions = []
            if child_title is not None:
                expansions.append(child_title.px_bbox)
            if child_body is not None:
                expansions.append(child_body.px_bbox)
            for cx0, cy0, cx1, cy1 in expansions:
                px0 = min(px0, cx0)
                py0 = min(py0, cy0)
                px1 = max(px1, cx1)
                py1 = max(py1, cy1)
            try:
                object.__setattr__(parent_body, "px_bbox",
                                   (px0, py0, px1, py1))
            except Exception:
                pass

        # Remove sub-header bodies so each section renders as a single
        # overarching blue wrapper without inner blue boxes.
        if subheader_sids:
            removed_ids = {f"textsec_{sid}_body" for sid in subheader_sids}
            synth = [b for b in synth if b.box_id not in removed_ids]

    # ── Geometric body-containment cleanup (Rule 12 extension) ────────────
    # After parent body expansion, any text-section whose ``_title``
    # is geometrically CONTAINED inside another text-section's
    # ``_body`` is treated as a sub-section: its title is marked
    # ``is_subheader=True`` (green wash) and its inner ``_body`` wrapper
    # is removed, so the section renders as a single overarching blue
    # box without nested boxes-inside-boxes.  Title containment (vs
    # full-body containment) is used because a sub-section's own body
    # may extend slightly past its parent's body when the parent body
    # ends at the boundary of the next peer section but the sub-section's
    # content runs a few lines further.  The title's inclusion in the
    # parent body is the unambiguous signal that the section is nested.
    contain_tol_pt = 4.0
    textsec_titles = [
        b for b in synth
        if b.box_id.startswith("textsec_") and b.box_id.endswith("_title")
    ]
    textsec_bodies = [
        b for b in synth
        if b.box_id.startswith("textsec_") and b.box_id.endswith("_body")
    ]
    # Sort bodies by area descending — larger ones make more
    # plausible parents.
    textsec_bodies.sort(
        key=lambda b: -((b.px_bbox[2] - b.px_bbox[0])
                         * (b.px_bbox[3] - b.px_bbox[1])))
    contained_body_ids: set[str] = set()
    synth_by_id_geom = {b.box_id: b for b in synth}
    for inner_title in textsec_titles:
        # Identify the matching body for this title.
        body_id = inner_title.box_id[:-len("_title")] + "_body"
        if body_id in contained_body_ids:
            continue
        if body_id not in synth_by_id_geom:
            continue
        ix0, iy0, ix1, iy1 = inner_title.px_bbox
        for outer_body in textsec_bodies:
            if outer_body.box_id == body_id:
                continue
            if outer_body.box_id in contained_body_ids:
                continue
            ox0, oy0, ox1, oy1 = outer_body.px_bbox
            # Title fully inside outer body (with small tolerance) AND
            # outer body's title is NOT this title (don't match self
            # as parent).
            outer_title_id = outer_body.box_id[:-len("_body")] + "_title"
            if outer_title_id == inner_title.box_id:
                continue
            if (ox0 - contain_tol_pt <= ix0
                    and ix1 <= ox1 + contain_tol_pt
                    and oy0 - contain_tol_pt <= iy0
                    and iy1 <= oy1 + contain_tol_pt):
                # Inner title is contained in outer's body → sub-header.
                contained_body_ids.add(body_id)
                try:
                    setattr(inner_title, "is_subheader", True)
                except Exception:
                    object.__setattr__(inner_title, "is_subheader", True)
                break

    if contained_body_ids:
        synth = [b for b in synth if b.box_id not in contained_body_ids]

    # ── Bodyless-title-in-ORANGE-cell rule (Rule 12 extension) ────────────
    # A textsec_*_title without a corresponding _body wrapper that
    # sits inside a non-synthetic ORANGE cell is an in-cell sub-title
    # (e.g. ``EXTERIOR WALL`` text inside the WALL column data cell of
    # an EXTERIOR WALL ASSEMBLIES table on test7).  These should be
    # marked ``is_subheader=True`` so downstream rendering and
    # column-header detection treat them as sub-headings, not as
    # peer schedule titles that would seed colhdr_* emission.
    #
    # Discriminator (two gates):
    #   (a) Cell height ≥ 2× title height — title occupies a small
    #       fraction of the cell, leaving room for cell content below.
    #       Excludes test5 schedule title-strips (h ≈ title h).
    #   (b) Cell width ≥ 200 px — substantial data cell, not a tiny
    #       title-block element.  Excludes test5 title-block cells
    #       (~115 px wide) which have local sub-titles that the
    #       title-block synthesis pipeline handles separately.
    #
    # test7 WALL column cell: 679 wide × 627 tall vs 41 title (15x).
    # test7 FURRING WALL cell: 679 wide × 101 tall vs 41 title (2.46x).
    # test5 schedule title strip: 1209 wide × 17 tall vs 23 title (0.7x).
    # test5 title-block cell: 115 wide × 28 tall vs 11 title (2.55x).
    # The (height_ratio ≥ 2.0 AND width ≥ 200) AND-gate cleanly
    # selects only the test7 in-cell sub-titles.
    if existing_boxes:
        orange_leaves = [
            b for b in existing_boxes
            if (not getattr(b, "synthetic", False)
                and getattr(b, "color", None) == "ORANGE"
                and (getattr(b, "box_id", "") or "").startswith("v"))
        ]
        for inner_title in textsec_titles:
            if getattr(inner_title, "is_subheader", False):
                continue
            body_id = inner_title.box_id[:-len("_title")] + "_body"
            if body_id in synth_by_id_geom:
                continue   # has a body — not bodyless
            ix0, iy0, ix1, iy1 = inner_title.px_bbox
            title_h = iy1 - iy0
            if title_h <= 0:
                continue
            for ol in orange_leaves:
                ox0, oy0, ox1, oy1 = ol.px_bbox
                cell_h = oy1 - oy0
                cell_w = ox1 - ox0
                if (ox0 - contain_tol_pt <= ix0
                        and ix1 <= ox1 + contain_tol_pt
                        and oy0 - contain_tol_pt <= iy0
                        and iy1 <= oy1 + contain_tol_pt
                        and cell_h >= 2.0 * title_h
                        and cell_w >= 200):
                    # Cell is at least 2x taller than title AND wide
                    # enough to be a data cell (not a title-block
                    # sub-element).  Title is a sub-heading; mark it.
                    try:
                        setattr(inner_title, "is_subheader", True)
                    except Exception:
                        object.__setattr__(inner_title, "is_subheader", True)
                    break

    # ── Universal continuation-blocks rule (rules/continuation_blocks) ────
    # Detect orphan label-rows at the top of a column with no header
    # above them — these are content that flowed across columns from
    # a section that began at the bottom of the previous column.
    # Wrap them with a ``_continuation`` synthetic body so they're
    # visually marked as detected content rather than silently dropped.
    # See RULES.md Rule 11.
    from .rules import find_continuation_blocks
    continuation_blocks = find_continuation_blocks(headers, all_spans_text)
    for cb_pdf in continuation_blocks:
        sid += 1
        # Convert PDF bbox to image and emit as _continuation body.
        cx0, cy0, cx1, cy1 = cb_pdf
        # Pad slightly so the wrapper visually contains the text.
        cb_padded = (cx0 - pad, cy0 - pad, cx1 + pad, cy1 + pad)
        # No header to clamp around — pass clamp/hbox as None so the
        # body emits as-is (no obstacle shrink, no smaller-wrapper clip).
        cont_box = _emit_box("_continuation", cb_padded,
                             parent_id=None,
                             clamp_to=None,
                             hbox_img=None)
        if cont_box is not None:
            synth.append(cont_box)

    return synth
