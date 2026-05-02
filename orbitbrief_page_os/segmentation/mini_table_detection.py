"""Mini-table detection inside text bodies.

Body paragraphs on spec sheets and structural-notes pages often contain
small embedded tables — multiple rows of short text spans whose column
positions line up.  Plain text extraction loses the grid structure;
this module re-discovers it so downstream consumers know a block of
text is actually a table.

Detection algorithm (universal, no per-page tuning)
---------------------------------------------------

1. Pull every text span from the PDF (PyMuPDF) with its bbox.
2. Group spans into LINES along the line-axis (perpendicular to the
   page's reading direction — derived upstream from the same
   quarter-turns used by the box detector).
3. For each line, build a COLUMN SIGNATURE: the tuple of span starts
   along the reading axis, rounded to ``col_tol_pt`` to absorb jitter.
4. Find runs of ``≥ min_rows`` consecutive lines that share the same
   signature (or a signature close to it — same number of columns, each
   column within ``col_tol_pt`` of the corresponding column in the
   reference line).
5. Each run becomes a table: we emit one synthetic VisibleBox for the
   table frame and one per grid cell, with ``box_id`` suffix
   ``"_mtable"`` / ``"_mtcell"`` so the overlay renderer can style them
   distinctively.

Reading-direction-aware
-----------------------

For ``cw_quarter_turns == 0`` or ``2`` (horizontal text) the reading
axis is PDF x and the line axis is PDF y.  For ``1`` or ``3`` (sideways
text) they swap.  Column-signature matching is done entirely in the
reading-axis coordinate so it doesn't matter which orientation the
PDF was stored in.

Integration
-----------

Call ``detect_mini_tables(pdf_path, page_index, scale, cw_quarter_turns,
visible_box_cls, rect_cls, cfg=None, existing_boxes=None)``.  Pass the
same ``VisibleBox`` and ``Rect`` classes the standalone detector uses
so the synthetic entries append cleanly to the result list.
``existing_boxes`` (optional) lets the detector skip tables that land
inside a known structural-drawing cell (so we don't double-detect the
real tables the geometric pipeline already found).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import fitz as _fitz
except Exception:            # pragma: no cover
    _fitz = None


@dataclass
class MiniTableConfig:
    """Tunables for mini-table detection (all opt-in safe defaults)."""

    enabled: bool = True
    # A real table needs ≥ 3 rows (header + ≥ 2 data) AND ≥ 3 columns
    # (label-value pairs and numbered lists are filtered out by these).
    min_rows: int = 3
    min_cols: int = 3
    col_tol_pt: float = 10.0           # tolerance for matching column starts
    line_h_pt_max: float = 18.0        # max letter height for regular-line spans
    # "Cell-like" filter: every table cell must be SHORT in the reading
    # axis.  Real-table cells are typically a couple of words / a number
    # / a unit — never a paragraph line.  Spans longer than this are
    # excluded from the line-signature build entirely.
    cell_max_reading_len_pt: float = 60.0
    cell_pad_pt: float = 0.0           # pad around each synthesised cell bbox
    # Spans this close together in the reading axis are merged into one
    # logical cell.  This handles two-line headers like "HEAD" +
    # "DIAMETER" stacked on top of each other (~1 pt gap).  Must stay
    # < the typical inter-column gap (~3-5 pt) or unrelated header
    # tokens collapse into one cell.
    cell_merge_gap_pt: float = 1.5
    # ── Stricter table-likeness filters (post-run-build) ─────────────
    # Real-table cells average ~10-20 pt long; paragraph wrap averages
    # 40+ pt.  Drop any candidate whose AVERAGE cell length exceeds
    # this threshold.
    max_avg_cell_len_pt: float = 22.0
    # No single column-aligned cell may exceed this — a long span here
    # is almost certainly a paragraph line.
    max_cell_len_pt: float = 35.0
    # Real tables fill nearly every (row × col) intersection.
    min_fill_rate: float = 0.85
    # Skip candidate tables whose frame is entirely inside a detected
    # non-synthetic ORANGE box (= already represented by the box pipeline).
    skip_if_inside_detected: bool = True


# ─── coord transforms (shared style with text_section_detection) ─────────────

def _pdf_pt_to_image_xy(x_pt: float, y_pt: float,
                         page_w_pt: float, page_h_pt: float,
                         scale: float, cw_quarter_turns: int) -> tuple[float, float]:
    px = x_pt * scale
    py = y_pt * scale
    W_orig = page_w_pt * scale
    H_orig = page_h_pt * scale
    n = cw_quarter_turns % 4
    if n == 0:
        return (px, py)
    if n == 1:
        return (H_orig - py, px)
    if n == 2:
        return (W_orig - px, H_orig - py)
    return (py, W_orig - px)


def _pdf_bbox_to_image_bbox(bbox_pt: tuple[float, float, float, float],
                             page_w_pt: float, page_h_pt: float,
                             scale: float, cw_quarter_turns: int
                             ) -> tuple[int, int, int, int]:
    pts = [
        _pdf_pt_to_image_xy(bbox_pt[0], bbox_pt[1], page_w_pt, page_h_pt, scale, cw_quarter_turns),
        _pdf_pt_to_image_xy(bbox_pt[2], bbox_pt[1], page_w_pt, page_h_pt, scale, cw_quarter_turns),
        _pdf_pt_to_image_xy(bbox_pt[0], bbox_pt[3], page_w_pt, page_h_pt, scale, cw_quarter_turns),
        _pdf_pt_to_image_xy(bbox_pt[2], bbox_pt[3], page_w_pt, page_h_pt, scale, cw_quarter_turns),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (int(round(min(xs))), int(round(min(ys))),
            int(round(max(xs))), int(round(max(ys))))


# ─── span collection ──────────────────────────────────────────────────────────

def _collect_spans(pdf_path: str, page_index: int,
                    cfg: MiniTableConfig
                    ) -> tuple[list[dict[str, Any]], float, float]:
    if _fitz is None:
        return [], 0.0, 0.0
    fdoc = _fitz.open(pdf_path)
    try:
        page = fdoc[page_index]
        page_w_pt = float(page.rect.width)
        page_h_pt = float(page.rect.height)
        td = page.get_text("dict")
    finally:
        pass

    spans: list[dict[str, Any]] = []
    for block in td.get("blocks", []):
        if block.get("type", 0) != 0:
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
                letter_h = min(bx1 - bx0, by1 - by0)
                if letter_h > cfg.line_h_pt_max:
                    continue
                spans.append({
                    "text": text,
                    "bbox": bbox,
                    "letter_h": letter_h,
                })
    fdoc.close()
    return spans, page_w_pt, page_h_pt


# ─── main entry ───────────────────────────────────────────────────────────────

def detect_mini_tables(pdf_path: str, page_index: int,
                        scale: float,
                        cw_quarter_turns: int,
                        visible_box_cls,
                        rect_cls,
                        cfg: MiniTableConfig | None = None,
                        existing_boxes: list | None = None) -> list:
    """Detect mini-tables embedded in text bodies.  Returns synthetic
    VisibleBox entries (frame + cells) with distinctive box_id suffixes.
    """
    cfg = cfg or MiniTableConfig()
    if not cfg.enabled or _fitz is None:
        return []

    spans, page_w_pt, page_h_pt = _collect_spans(pdf_path, page_index, cfg)
    if len(spans) < cfg.min_rows * cfg.min_cols:
        return []

    qt = cw_quarter_turns % 4
    sideways = qt in (1, 3)

    # Reading axis = where characters of a single line advance.
    # Line axis = where lines stack.
    def _reading_start(bbox):    # x_start in reading axis
        bx0, by0, bx1, by1 = bbox
        return by0 if sideways else bx0

    def _reading_end(bbox):      # x_end in reading axis
        bx0, by0, bx1, by1 = bbox
        return by1 if sideways else bx1

    def _line_center(bbox):      # perpendicular to reading (= which "line")
        bx0, by0, bx1, by1 = bbox
        return (bx0 + bx1) * 0.5 if sideways else (by0 + by1) * 0.5

    # ── 1. Filter to CELL-LIKE spans (short in reading axis) ─────────────────
    # Paragraph text has reading-axis length >> a handful of characters.
    # Table cells are short, so this filter excludes body prose while
    # keeping every value-cell / header-cell in any embedded table.
    cell_spans: list[dict[str, Any]] = []
    for s in spans:
        bx0, by0, bx1, by1 = s["bbox"]
        read_len = (by1 - by0) if sideways else (bx1 - bx0)
        if read_len <= cfg.cell_max_reading_len_pt:
            cell_spans.append(s)
    if len(cell_spans) < cfg.min_rows * cfg.min_cols:
        return []

    # ── 2. Group cell-like spans into lines ──────────────────────────────────
    if cell_spans:
        sorted_h = sorted(s["letter_h"] for s in cell_spans)
        median_letter_h = sorted_h[len(sorted_h) // 2]
    else:
        median_letter_h = 8.0
    line_tol = max(1.2, median_letter_h * 0.4)

    lines_raw: dict[int, list[dict[str, Any]]] = {}
    for s in cell_spans:
        key = int(round(_line_center(s["bbox"]) / line_tol))
        lines_raw.setdefault(key, []).append(s)
    def _merge_close_spans(bucket: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge spans that are very close in the reading axis into a
        single logical cell.  Handles two-token headers like "HEAD" +
        "DIAMETER" stacked together (where the gap between them is
        ≤ cell_merge_gap_pt)."""
        if not bucket:
            return bucket
        bucket = sorted(bucket, key=lambda s: _reading_start(s["bbox"]))
        merged: list[dict[str, Any]] = []
        for s in bucket:
            if not merged:
                merged.append(dict(s))     # shallow copy so we can mutate bbox
                continue
            prev = merged[-1]
            gap = _reading_start(s["bbox"]) - _reading_end(prev["bbox"])
            if gap <= cfg.cell_merge_gap_pt:
                # Merge: union bboxes and join text with a space
                pb = prev["bbox"]
                sb = s["bbox"]
                prev["bbox"] = (min(pb[0], sb[0]), min(pb[1], sb[1]),
                                max(pb[2], sb[2]), max(pb[3], sb[3]))
                prev["text"] = (prev.get("text", "") + " "
                                 + s.get("text", "")).strip()
            else:
                merged.append(dict(s))
        return merged

    lines: list[list[dict[str, Any]]] = []
    line_keys: list[int] = []
    for key in sorted(lines_raw.keys()):
        bucket = lines_raw[key]
        bucket = _merge_close_spans(bucket)
        lines.append(bucket)
        line_keys.append(key)

    if len(lines) < cfg.min_rows:
        return []

    # ── 3. Find grids of aligned cells ──────────────────────────────────────
    # Two lines are "column-aligned" if they share ≥ min_cols matching
    # reading-axis starts (within col_tol_pt).  We slide over lines and
    # grow runs: a line joins the current run if it shares ≥ min_cols
    # aligned columns with the run's current set.
    def _aligned_cols(a_starts: list[float],
                     b_starts: list[float]) -> list[float]:
        """Return the SUBSET of a_starts that has a match (within
        col_tol_pt) in b_starts."""
        matched = []
        for ax in a_starts:
            if any(abs(ax - bx) <= cfg.col_tol_pt for bx in b_starts):
                matched.append(ax)
        return matched

    def _line_starts(line):
        return [_reading_start(s["bbox"]) for s in line]

    # ── 3. Line-by-line accumulator ─────────────────────────────────────────
    # Simple, local: walk lines in line-axis order; from each starting
    # line build a "run" of subsequent lines that share ≥ min_cols
    # column starts (within col_tol_pt) with the active set.  Allow
    # non-matching intervening lines as long as they're within
    # max_key_gap of the most recent successful match.  ``active_cols``
    # tightens (intersection) on each match — never grows.
    max_key_gap = 9
    tables: list[dict[str, Any]] = []
    used = [False] * len(lines)
    def _seed_active_cols(start_idx: int) -> list[float]:
        """Build a seed column set by looking at the first ~6 lines
        within ``max_key_gap`` of ``start_idx`` and keeping columns
        that appear in ≥ 2 of them.  This filters out paragraph-
        fragment noise that contaminates a single starting line."""
        seed_lines = []
        seed_key = line_keys[start_idx]
        for k in range(start_idx, min(start_idx + 8, len(lines))):
            if line_keys[k] - seed_key > max_key_gap * 2:
                break
            seed_lines.append(k)
        # Bucket every span start across these lines.
        from collections import Counter
        bc: Counter = Counter()
        sample_starts: dict[int, float] = {}
        # Use the starting line's reading-axis span range as a
        # filter: only include seed-window spans whose reading_start
        # lies within ±3× the starting line's reading extent.  This
        # prevents far-field spans from other tables (sharing the same
        # line-stack key) from polluting the anchor set.
        seed_line_starts = [_reading_start(s["bbox"])
                            for s in lines[start_idx]]
        if seed_line_starts:
            rs_min = min(seed_line_starts) - 3 * cfg.col_tol_pt
            rs_max = max(seed_line_starts) + 3 * cfg.col_tol_pt
        else:
            rs_min, rs_max = -1e9, 1e9
        for li in seed_lines:
            seen = set()
            for s in lines[li]:
                rs = _reading_start(s["bbox"])
                if not (rs_min <= rs <= rs_max):
                    continue     # far-field span from a different block
                b = int(round(rs / cfg.col_tol_pt))
                if b not in seen:
                    bc[b] += 1
                    sample_starts.setdefault(b, rs)
                    seen.add(b)
        # Keep buckets shared by ≥ 2 of the sampled lines.
        kept = sorted(b for b, c in bc.items() if c >= 2)
        return [sample_starts[b] for b in kept]

    import os as _dbg2_os
    _dbg2 = _dbg2_os.environ.get("DEBUG_MTBR2") == "1"
    for i in range(len(lines)):
        if used[i]:
            if _dbg2 and line_keys[i] in (273, 275, 278, 280):
                print(f"  i={i} key={line_keys[i]} USED, skip")
            continue
        # Skip line-stack groups that contain too many spans —
        # these are usually two or more unrelated table rows whose
        # content PDF lines happen to share the same line-stack key.
        # A real 2-column abbreviation row has 2 spans; a real 8-col
        # schedule row has 8-10.  Cap at 12 to exclude line-keys that
        # merge an abbreviation row WITH a pump-table row.
        if len(lines[i]) > 12:
            continue
        starts_i = _line_starts(lines[i])
        if _dbg2 and line_keys[i] in (273, 275, 278, 280):
            print(f"  i={i} key={line_keys[i]} starts={starts_i}")
        if len(starts_i) < cfg.min_cols:
            if _dbg2 and line_keys[i] in (273, 275, 278, 280):
                print(f"     starts<min_cols={cfg.min_cols} skip")
            continue
        anchors = _seed_active_cols(i)
        if _dbg2 and line_keys[i] in (273, 275, 278, 280):
            print(f"     anchors={anchors}")
        if len(anchors) < cfg.min_cols:
            if _dbg2 and line_keys[i] in (273, 275, 278, 280):
                print(f"     anchors<min_cols skip")
            continue
        if len(_aligned_cols(anchors, starts_i)) < cfg.min_cols:
            if _dbg2 and line_keys[i] in (273, 275, 278, 280):
                print(f"     line doesn't hit min_cols anchors skip")
            continue
        active_cols = list(anchors)
        run = [i]
        last_match_key = line_keys[i]
        for j in range(i + 1, len(lines)):
            if used[j]:
                break
            key_j = line_keys[j]
            if abs(key_j - last_match_key) > max_key_gap:
                break
            starts_j = _line_starts(lines[j])
            # A line counts as a table row if it hits ≥ min_cols anchors.
            shared = _aligned_cols(anchors, starts_j)
            if len(shared) < cfg.min_cols:
                continue
            run.append(j)
            last_match_key = key_j
        if len(run) < cfg.min_rows:
            continue
        # ── Gap-fill pass ─────────────────────────────────────────────────
        # After the initial accumulator, some lines between the first and
        # last run key may have been skipped because their description
        # column starts fall outside col_tol_pt of the seed anchor.  For
        # 2-column abbreviation lists the description column start varies
        # (text is right-aligned to a fixed END) while the code column
        # start is fixed.  Fill the gaps: any line between run[0] and
        # run[-1] that hits ≥ 1 anchor (has at least the code column) AND
        # has the right number of spans (= same as surrounding rows) is
        # interpolated into the run.
        if len(run) >= cfg.min_rows and len(run) < (run[-1] - run[0] + 1):
            expected_ncols = len(lines[run[0]])   # spans per line in the seed row
            all_keys_in_range = [i for i in range(run[0], run[-1] + 1)
                                  if i < len(lines) and not used[i]]
            filled_run = []
            for idx in all_keys_in_range:
                if idx in run:
                    filled_run.append(idx)
                elif not used[idx]:
                    sj = _line_starts(lines[idx])
                    n_spans = len(lines[idx])
                    # Include if it has the same span count AND hits ≥ 1 anchor.
                    if (abs(n_spans - expected_ncols) <= 1
                            and len(_aligned_cols(anchors, sj)) >= 1):
                        filled_run.append(idx)
            if len(filled_run) >= len(run):
                run = filled_run
        block_lines = [lines[k] for k in run]
        # Refine col_starts: only keep anchors that appear in ≥ half the
        # run rows.  This drops noise columns from the seed that happened
        # to satisfy the row-membership test but aren't truly part of
        # the table grid.
        from collections import Counter as _Cnt
        col_hits: _Cnt = _Cnt()
        for ln in block_lines:
            sj = _line_starts(ln)
            for c in anchors:
                if any(abs(c - x) <= cfg.col_tol_pt for x in sj):
                    col_hits[c] += 1
        # Keep only columns that appear in ≥ 70 % of the run's rows.
        # Paragraph noise typically clusters in only a handful of lines
        # (~30-50 %), so this drops it without losing real table columns
        # which fill nearly every row.
        threshold = max(2, int(round(len(run) * 0.70)))
        col_starts = sorted(c for c, h in col_hits.items() if h >= threshold)
        if len(col_starts) < cfg.min_cols:
            continue
        active_cols = col_starts
        # Re-validate: keep only run lines that hit ≥ min_cols of the
        # (narrowed) col_starts.  Drop the table if fewer than
        # min_rows lines remain — the run was held together by noise.
        valid_run = [k for k in run
                     if len(_aligned_cols(active_cols, _line_starts(lines[k]))) >= cfg.min_cols]
        if len(valid_run) < cfg.min_rows:
            continue
        run = valid_run
        block_lines = [lines[k] for k in run]
        # False-positive guard: paragraph-wrap fragments often produce
        # runs where every row's spans are roughly the same widths —
        # visually "table-shaped" but actually wrapped paragraph text.
        # Real tables have at least one row whose cell widths vary
        # noticeably (label + value).
        varied = False
        for ln in block_lines:
            lengths = [(_reading_end(s["bbox"]) - _reading_start(s["bbox"]))
                        for s in ln]
            if len(lengths) >= 2:
                spread = max(lengths) - min(lengths)
                if spread > 6.0:
                    varied = True
                    break
        import os as _dbg_os
        if _dbg_os.environ.get("DEBUG_MTBR") == "1":
            print(f"  RUN keys={[line_keys[k] for k in run]} cols={col_starts} varied={varied}")
        if not varied:
            continue
        # ── Stricter table-likeness filters ─────────────────────────────────
        # Only count spans whose reading-axis start is near one of the
        # active column anchors — those are the "real" table cells.
        # Loose paragraph spans that happen to share a line key but
        # don't sit in a true column shouldn't influence stats.
        col_cell_lengths: list[float] = []
        col_cell_count = 0
        possible_cells = len(block_lines) * len(active_cols)
        for ln in block_lines:
            for c in active_cols:
                # Best span = greatest bbox-overlap with the column's
                # tolerance window — same matching rule as cell emission.
                best = None
                best_ov = 0.0
                for s in ln:
                    sb_lo = _reading_start(s["bbox"])
                    sb_hi = _reading_end(s["bbox"])
                    ov = min(sb_hi, c + cfg.col_tol_pt) - max(sb_lo, c - cfg.col_tol_pt)
                    if ov > best_ov:
                        best_ov = ov
                        best = s
                if best is not None:
                    col_cell_lengths.append(
                        _reading_end(best["bbox"]) - _reading_start(best["bbox"]))
                    col_cell_count += 1
        if not col_cell_lengths:
            continue
        avg_len = sum(col_cell_lengths) / len(col_cell_lengths)
        max_len = max(col_cell_lengths)
        fill_rate = col_cell_count / max(1, possible_cells)
        if _dbg_os.environ.get("DEBUG_MTBR") == "1":
            print(f"     avg={avg_len:.1f} max={max_len:.1f} fill={fill_rate:.2f}")
        if avg_len > cfg.max_avg_cell_len_pt:
            continue
        if max_len > cfg.max_cell_len_pt:
            continue
        if fill_rate < cfg.min_fill_rate:
            continue
        # Bbox spans ONLY the column-aligned cells — not every span in
        # those lines.  Otherwise a single paragraph fragment far
        # outside the table's true columns balloons the frame to full
        # page width.
        col_aligned_bboxes = []
        for ln in block_lines:
            for s in ln:
                rs = _reading_start(s["bbox"])
                if any(abs(rs - c) <= cfg.col_tol_pt for c in active_cols):
                    col_aligned_bboxes.append(s["bbox"])
        if not col_aligned_bboxes:
            continue
        tx0 = min(b[0] for b in col_aligned_bboxes)
        ty0 = min(b[1] for b in col_aligned_bboxes)
        tx1 = max(b[2] for b in col_aligned_bboxes)
        ty1 = max(b[3] for b in col_aligned_bboxes)
        tables.append({
            "bbox_pt": (tx0, ty0, tx1, ty1),
            "lines": block_lines,
            "col_starts": active_cols,
        })
        for k in run:
            used[k] = True

    if not tables:
        return []

    # Collect every non-synthetic detected box.  We use them in two
    # ways below:
    #   • skip a candidate mini-table that's entirely inside an existing
    #     ORANGE cell (rare but cheap);
    #   • skip a candidate whose area is already COVERED by several
    #     existing non-synthetic ORANGE cells (= the geometric pipeline
    #     already found that table as a real grid; no point double-
    #     detecting it as a mini-table).
    existing_orange: list[tuple[int, int, int, int]] = []
    existing_blue:   list[tuple[int, int, int, int]] = []
    if existing_boxes:
        for b in existing_boxes:
            if getattr(b, "synthetic", False):
                continue
            px = getattr(b, "px_bbox", None)
            if not (px and len(px) == 4):
                continue
            tup = tuple(int(v) for v in px)
            color = getattr(b, "color", None)
            if color == "ORANGE":
                existing_orange.append(tup)
            elif color == "BLUE":
                existing_blue.append(tup)

    # ── 4. Emit synthetic VisibleBoxes ────────────────────────────────────────
    synth: list = []
    mt_id = 0
    for table in tables:
        tbl_img = _pdf_bbox_to_image_bbox(
            table["bbox_pt"], page_w_pt, page_h_pt, scale, cw_quarter_turns)
        x0, y0, x1, y1 = tbl_img
        if x1 <= x0 or y1 <= y0:
            continue
        # ── BLUE-wrapper containment ─────────────────────────────────
        # A real mini-table lives INSIDE one wrapper.  If the candidate
        # straddles 2+ BLUE wrappers (or none), it's almost certainly a
        # bogus run that picked up text from neighbouring sections.
        if existing_blue:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            tol = 6      # tolerate a few pixels of slop on each side
            # Wrappers that strictly contain the candidate (with tol).
            containing = [
                (bx0, by0, bx1, by1)
                for (bx0, by0, bx1, by1) in existing_blue
                if (bx0 - tol) <= x0 and (by0 - tol) <= y0
                and x1 <= (bx1 + tol) and y1 <= (by1 + tol)
            ]
            # Wrappers that the candidate's centre lies inside (less
            # strict — useful when the candidate slightly overshoots).
            centre_inside = [
                (bx0, by0, bx1, by1)
                for (bx0, by0, bx1, by1) in existing_blue
                if bx0 <= cx <= bx1 and by0 <= cy <= by1
            ]
            if not containing and not centre_inside:
                # Candidate centre lies in NO BLUE wrapper at all → it's
                # floating across page-level whitespace; reject.
                continue
            # Wrappers the candidate clearly overlaps (intersection
            # area > 25 % of either side).  If it crosses 2+ such
            # wrappers without being contained by any, reject.
            crossed = []
            cand_area = max(1, (x1 - x0) * (y1 - y0))
            for (bx0, by0, bx1, by1) in existing_blue:
                ix0 = max(x0, bx0); iy0 = max(y0, by0)
                ix1 = min(x1, bx1); iy1 = min(y1, by1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter / cand_area >= 0.10:
                    crossed.append((bx0, by0, bx1, by1))
            if not containing and len(crossed) >= 2:
                continue
        # Skip when the candidate is entirely inside an existing ORANGE
        # cell OR when several existing ORANGE cells already overlap its
        # area (= geometric pipeline already found this table).
        if existing_orange:
            inside_any = any(
                ox0 <= x0 and oy0 <= y0 and x1 <= ox1 and y1 <= oy1
                for (ox0, oy0, ox1, oy1) in existing_orange)
            if inside_any:
                continue
            # Only suppress when the existing ORANGE cells cover a
            # MEANINGFUL fraction of the candidate.  A handful of tiny
            # strips in one corner (e.g. a sidebar that happens to
            # share image y-range) shouldn't kill an unrelated table.
            cand_area = max(1, (x1 - x0) * (y1 - y0))
            cov_area = 0
            cov_count = 0
            for (ox0, oy0, ox1, oy1) in existing_orange:
                ix0 = max(x0, ox0); iy0 = max(y0, oy0)
                ix1 = min(x1, ox1); iy1 = min(y1, oy1)
                if ix1 > ix0 and iy1 > iy0:
                    cov_count += 1
                    cov_area += (ix1 - ix0) * (iy1 - iy0)
            cov_frac = cov_area / cand_area
            # Real detected grid: ≥ 3 cells inside AND ≥ 25 % coverage.
            if cov_count >= 3 and cov_frac >= 0.25:
                continue
        mt_id += 1
        # Frame box
        synth.append(visible_box_cls(
            box_id=f"minitable_{mt_id}_mtable",
            rect=rect_cls(x0 / scale, y0 / scale, x1 / scale, y1 / scale),
            area_pt2=(x1 - x0) * (y1 - y0) / (scale * scale),
            fill_ratio=1.0,
            nested_depth=4,
            is_outer_wrapper=False,
            parent_box_id=None,
            color="GREEN",
            px_bbox=(x0, y0, x1, y1),
            children_count=0,
            synthetic=True,
        ))
        # One synthetic cell per (row, column)
        block_lines = table["lines"]
        col_starts = table["col_starts"]
        # Compute column bounds (reading axis): boundary i = midpoint
        # between col_starts[i] and col_starts[i+1], with outer bounds
        # at the overall table reading-axis extent.
        if sideways:
            tbl_read_lo = table["bbox_pt"][1]    # PDF y low
            tbl_read_hi = table["bbox_pt"][3]
        else:
            tbl_read_lo = table["bbox_pt"][0]
            tbl_read_hi = table["bbox_pt"][2]
        boundaries = [tbl_read_lo]
        for k in range(1, len(col_starts)):
            boundaries.append((col_starts[k - 1] + col_starts[k]) * 0.5)
        boundaries.append(tbl_read_hi)

        # Match a span to a column by ANY overlap of its bbox with the
        # column-anchor's tolerance window — not just its reading_start.
        # This handles two-line headers like "HEAD DIAMETER" whose
        # merged bbox starts well above the data column anchor (the
        # data values cluster around the second line of the header).
        def _best_col(span_bbox: tuple[float, float, float, float]) -> int | None:
            """Return the col index whose anchor's tolerance window
            best overlaps the span's reading-axis bbox.

            Matching uses BOTH reading_start (normal left-aligned cells)
            and reading_end (right-aligned cells whose variable-length
            text ends at a fixed column boundary, e.g. abbreviation
            descriptions).  This means a description span like
            "AMERICANS WITH DISABILITIES ACT" (starts at y=605, ends
            at y=651) will match the description column anchor at ~623
            via its END (651 − anchor 623 = 28 pt overlap).
            """
            sb_lo = _reading_start(span_bbox)
            sb_hi = _reading_end(span_bbox)
            best_i = None
            best_score = -1.0
            for i, c in enumerate(col_starts):
                w_lo = c - cfg.col_tol_pt
                w_hi = c + cfg.col_tol_pt
                # Score 1: overlap of full span with the anchor window.
                ov = min(sb_hi, w_hi) - max(sb_lo, w_lo)
                if ov <= 0:
                    ov = 0.0
                # Score 2: proximity of reading_end to anchor (catches
                # right-aligned columns).  Bonus of up to col_tol_pt.
                end_bonus = max(0.0, cfg.col_tol_pt - abs(sb_hi - c))
                score = ov + end_bonus
                if score > best_score and (ov > 0 or end_bonus > 0):
                    best_score = score
                    best_i = i
            return best_i

        # Detect the header row by content: pick the row with the most
        # column-aligned cells whose text is alphabetic & non-numeric
        # (e.g. "NAIL SIZE", "LENGTH").  Data rows have numeric/unit
        # cells (e.g. "8d", "0.131\"").  This is rotation-agnostic.
        def _is_textual(text: str) -> bool:
            t = (text or "").strip()
            if len(t) < 2:
                return False
            has_alpha = any(c.isalpha() for c in t)
            # Pure numeric, decimal, or starts with digit → data
            if not has_alpha:
                return False
            if t[0].isdigit():
                return False
            return True

        def _span_in_any_col(sb: tuple[float, float, float, float]) -> bool:
            sb_lo, sb_hi = _reading_start(sb), _reading_end(sb)
            for c in col_starts:
                if min(sb_hi, c + cfg.col_tol_pt) > max(sb_lo, c - cfg.col_tol_pt):
                    return True
            return False
        text_per_row = []
        for ln in block_lines:
            cnt = 0
            for s in ln:
                if _span_in_any_col(s["bbox"]) and _is_textual(s.get("text", "")):
                    cnt += 1
            text_per_row.append(cnt)
        # Header = the *unique* row with the most textual cells, if any.
        # Abbreviation / implied lists have the same shape on every line
        # (short code + description) → every row ties on textual count.
        # Picking any single tied row (previously: last index) falsely
        # tagged e.g. the bottom line as a CSV "header".  When more than
        # one row shares max_t, there is no real header row — all cells
        # are data (_mtcelld).
        header_row_idx = -1
        if text_per_row:
            max_t = max(text_per_row)
            if max_t >= cfg.min_cols - 1:
                winners = [i for i, c in enumerate(text_per_row) if c == max_t]
                if len(winners) == 1:
                    header_row_idx = winners[0]

        for r, line in enumerate(block_lines):
            cell_spans: list[tuple[float, float, float, float] | None] = [None] * len(col_starts)
            for s in line:
                ci = _best_col(s["bbox"])
                if ci is None:
                    continue
                sb = s["bbox"]
                if cell_spans[ci] is None:
                    cell_spans[ci] = sb
                else:
                    cb = cell_spans[ci]
                    cell_spans[ci] = (
                        min(cb[0], sb[0]), min(cb[1], sb[1]),
                        max(cb[2], sb[2]), max(cb[3], sb[3]))

            is_header_row = (r == header_row_idx)
            cell_suffix = "_mtcellh" if is_header_row else "_mtcelld"
            row_cell_pxs: list[tuple[int, int, int, int]] = []

            for c, span_bbox in enumerate(cell_spans):
                if span_bbox is None:
                    continue
                sx0, sy0, sx1, sy1 = span_bbox
                pad = cfg.cell_pad_pt
                cell_pdf = (sx0 - pad, sy0 - pad, sx1 + pad, sy1 + pad)
                cx0, cy0, cx1, cy1 = _pdf_bbox_to_image_bbox(
                    cell_pdf, page_w_pt, page_h_pt, scale, cw_quarter_turns)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                # Overlap suppression vs existing ORANGE.
                cell_area = max(1, (cx1 - cx0) * (cy1 - cy0))
                skip = False
                for (ox0, oy0, ox1, oy1) in existing_orange:
                    ix0 = max(cx0, ox0); iy0 = max(cy0, oy0)
                    ix1 = min(cx1, ox1); iy1 = min(cy1, oy1)
                    if ix1 > ix0 and iy1 > iy0:
                        if (ix1 - ix0) * (iy1 - iy0) / cell_area >= 0.30:
                            skip = True
                            break
                if skip:
                    continue
                # Drop cells that cross a BLUE wrapper edge.
                if existing_blue:
                    inside_wrapper = False
                    crosses_edge = False
                    for (bx0, by0, bx1, by1) in existing_blue:
                        if bx0 - 2 <= cx0 and by0 - 2 <= cy0 \
                                and cx1 <= bx1 + 2 and cy1 <= by1 + 2:
                            inside_wrapper = True
                            break
                        ix0 = max(cx0, bx0); iy0 = max(cy0, by0)
                        ix1 = min(cx1, bx1); iy1 = min(cy1, by1)
                        if ix1 > ix0 and iy1 > iy0:
                            crosses_edge = True
                    if not inside_wrapper and crosses_edge:
                        continue
                row_cell_pxs.append((cx0, cy0, cx1, cy1))
                synth.append(visible_box_cls(
                    box_id=f"minitable_{mt_id}_r{r}c{c}{cell_suffix}",
                    rect=rect_cls(cx0 / scale, cy0 / scale, cx1 / scale, cy1 / scale),
                    area_pt2=(cx1 - cx0) * (cy1 - cy0) / (scale * scale),
                    fill_ratio=1.0,
                    nested_depth=5,
                    is_outer_wrapper=False,
                    parent_box_id=f"minitable_{mt_id}_mtable",
                    color="GREEN",
                    px_bbox=(cx0, cy0, cx1, cy1),
                    children_count=0,
                    synthetic=True,
                ))

            # Emit per-row wrapper for data rows only (r > 0) — groups
            # all cells in a row so the user can see which cells belong
            # together as one tuple of values.
            # Emit per-row wrapper for ANY data row with ≥ 2 cells —
            # even when the table itself only has 2 columns (e.g. an
            # abbreviation list: code | description).  The wrapper
            # makes it visually clear which cells belong together as
            # one tuple of values.
            # One surviving cell still deserves a row ring — overlap suppression
            # vs contour ORANGE can drop every other column on that PDF line.
            if not is_header_row and len(row_cell_pxs) >= 1:
                # Tight ring hugging the cell edges (1 px breathing room).
                rx0 = min(p[0] for p in row_cell_pxs) - 1
                ry0 = min(p[1] for p in row_cell_pxs) - 1
                rx1 = max(p[2] for p in row_cell_pxs) + 1
                ry1 = max(p[3] for p in row_cell_pxs) + 1
                synth.append(visible_box_cls(
                    box_id=f"minitable_{mt_id}_r{r}_mtrow",
                    rect=rect_cls(rx0 / scale, ry0 / scale, rx1 / scale, ry1 / scale),
                    area_pt2=(rx1 - rx0) * (ry1 - ry0) / (scale * scale),
                    fill_ratio=1.0,
                    nested_depth=4,
                    is_outer_wrapper=False,
                    parent_box_id=f"minitable_{mt_id}_mtable",
                    color="RED",
                    px_bbox=(rx0, ry0, rx1, ry1),
                    children_count=0,
                    synthetic=True,
                ))

    # ── Final pass: drop frames that overlap a larger frame heavily ──────────
    # Two adjacent line-key tables can produce visually overlapping
    # frames in image space; suppress smaller ones whose bbox is mostly
    # covered by a larger kept frame so the visual output shows clear
    # separation between distinct tables.
    frames = [b for b in synth if b.box_id.endswith("_mtable")]
    frames.sort(key=lambda b: -b.area_pt2)
    keep_ids: set[str] = set()
    for f in frames:
        fx0, fy0, fx1, fy1 = f.px_bbox
        f_area = max(1, (fx1 - fx0) * (fy1 - fy0))
        drop = False
        for kid in keep_ids:
            kf = next(b for b in frames if b.box_id == kid)
            kx0, ky0, kx1, ky1 = kf.px_bbox
            ix0, iy0 = max(fx0, kx0), max(fy0, ky0)
            ix1, iy1 = min(fx1, kx1), min(fy1, ky1)
            if ix1 > ix0 and iy1 > iy0:
                # ANY overlap is too much — mini-tables should never share
                # pixels with each other (the user wants clear separation
                # between distinct table blocks).
                drop = True
                break
        if not drop:
            keep_ids.add(f.box_id)

    if len(keep_ids) < len(frames):
        kept_table_prefixes = {kid.rsplit("_mtable", 1)[0] for kid in keep_ids}
        synth = [b for b in synth
                 if any(b.box_id.startswith(p + "_") or b.box_id == p + "_mtable"
                        for p in kept_table_prefixes)]
    return synth
