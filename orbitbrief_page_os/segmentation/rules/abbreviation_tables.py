"""Universal rule: dedicated abbreviation-table detection for sections
that are NOT caught by the generic mini_table_detection (because their
definition cells exceed the strict ``cell_max_reading_len_pt`` and
``max_cell_len_pt`` thresholds for paragraph-wrap exclusion).

Symptom that motivates this rule
--------------------------------
test7 has an ABBREVIATIONS section laid out as TWO PARALLEL pairs of
(abbr, defn) columns side-by-side (4 columns total).  Each line looks
like::

    ACT    ACOUSTIC CEILING TILE     N/A    NOT APPLICABLE
    ADJ    ADJUSTABLE                NBC    NATIONAL BUILDING CODE
    AFF    ABOVE FINISHED FLOOR      NIC    NOT IN CONTRACT

The definition cells (``ACOUSTIC CEILING TILE`` ~93pt, ``NATIONAL
BUILDING CODE`` ~104pt) exceed ``cell_max_reading_len_pt = 60``, and
their average run length far exceeds ``max_avg_cell_len_pt = 22``.
The generic mini_table detector excludes them entirely.

But these are CLEARLY tables — the column structure is unmistakable.
This dedicated detector targets exactly this pattern.

Two-stage approach
------------------
Stage 1: discover candidate "column anchors" — x-positions where many
spans start (≥ ``anchor_min_hits`` spans share that x-position within
``col_tol_pt``).

Stage 2: try every 4-column or 2-column anchor combination and apply
gates:
    - ≥ ``min_rows`` lines hit ALL anchors with a span
    - alternating short-long pattern (col 0/2 short ≤ abbr_max_med_pt,
      col 1/3 long ≥ defn_min_med_pt)
    - regular line pitch (variance ≤ ``line_pitch_var_max``)
    - definition cells ≤ ``defn_max_pt`` (excludes paragraph wrap)

This per-x-anchor approach avoids the trap of looking at all spans on
a line — irrelevant spans on the same y from other parts of the page
don't pollute the analysis.

Why universal & safe
--------------------
The combined "alternating short-long pattern at consistent column
anchors" + "regular line pitch" gate is highly specific.  Random
paragraph text fails the alternation gate; wide schedule tables fail
the median checks; narrow numeric tables fail the long-defn-median
gate.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List

from ..mini_table_detection import _collect_spans, MiniTableConfig


def find_abbreviation_tables(
    pdf_path: str,
    page_index: int,
    *,
    min_rows: int = 5,
    abbr_max_med_pt: float = 30.0,
    defn_min_med_pt: float = 25.0,
    defn_max_pt: float = 200.0,
    line_pitch_var_max: float = 0.30,
    col_tol_pt: float = 6.0,
    anchor_min_hits: int = 5,
    max_x_span_pt: float = 350.0,
) -> List[Dict[str, Any]]:
    """Find abbreviation-style tables with 2 or 4 columns.

    Returns list of dicts::

        {
          "bbox_pdf": (x0, y0, x1, y1),
          "rows": [[(span_bbox, span_text), ...], ...],
          "n_cols": 2 or 4,
        }
    """
    cfg = MiniTableConfig(cell_max_reading_len_pt=defn_max_pt)
    spans, page_w_pt, page_h_pt = _collect_spans(pdf_path, page_index, cfg)
    if not spans:
        return []

    # ── Stage 1: discover candidate column anchors ──
    x_buckets: Dict[int, int] = defaultdict(int)
    x_sample: Dict[int, float] = {}
    for s in spans:
        rs = s["bbox"][0]
        rl = s["bbox"][2] - s["bbox"][0]
        if rl > defn_max_pt:
            continue
        b = int(round(rs / col_tol_pt))
        x_buckets[b] += 1
        x_sample.setdefault(b, rs)
    anchor_x = sorted([x_sample[b] for b, c in x_buckets.items()
                       if c >= anchor_min_hits])
    if len(anchor_x) < 2:
        return []

    # ── Stage 2: try anchor combinations ──
    sorted_h = sorted(s["letter_h"] for s in spans)
    median_letter_h = sorted_h[len(sorted_h) // 2]
    line_tol = max(1.2, median_letter_h * 0.4)

    # Pre-compute: for each anchor x, the set of line-keys where a span
    # starts within col_tol_pt of that x.  Two anchors only co-occur in
    # an abbreviation table if their line-key sets overlap heavily — so
    # we can prune the combinatorial search using set intersection
    # before doing the expensive per-line picking.
    #
    # Without this prune, test5 (48 anchors → 194,580 4-combinations)
    # times out.  With it, almost all combinations are eliminated by
    # an O(1) intersection-size check.
    anchor_lines: Dict[float, set] = {}
    for a in anchor_x:
        keys: set = set()
        for s in spans:
            if abs(s["bbox"][0] - a) > col_tol_pt:
                continue
            bbox = s["bbox"]
            y_center = (bbox[1] + bbox[3]) / 2.0
            keys.add(int(round(y_center / line_tol)))
        anchor_lines[a] = keys

    results: List[Dict[str, Any]] = []
    used_x_ranges: List[tuple] = []

    def _overlaps_used(x0, x1, y0, y1) -> bool:
        for ux0, ux1, uy0, uy1 in used_x_ranges:
            if x1 <= ux0 or x0 >= ux1:
                continue
            if y1 <= uy0 or y0 >= uy1:
                continue
            return True
        return False

    # Try 4-col first (more specific), then 2-col.
    for n_cols in (4, 2):
        for anchors in combinations(anchor_x, n_cols):
            anchors = list(anchors)
            if anchors[-1] - anchors[0] < 30:
                continue
            # Reject combos that span too wide — prevents two separate discipline
            # columns (e.g. MECHANICAL + ELECTRICAL) from being merged into one
            # spurious 4-col abbreviation table.
            if anchors[-1] - anchors[0] > max_x_span_pt:
                continue
            # 4-col gate: between-pairs gap > each in-pair gap
            if n_cols == 4:
                gap_a = anchors[1] - anchors[0]
                gap_b = anchors[3] - anchors[2]
                gap_mid = anchors[2] - anchors[1]
                if not (gap_mid > gap_a and gap_mid > gap_b):
                    continue

            # FAST PRUNE: intersection of per-anchor line-key sets must
            # already have ≥ min_rows entries.  Anchors that don't
            # co-occur on enough lines can never form a table.
            shared_keys = anchor_lines[anchors[0]]
            for a in anchors[1:]:
                shared_keys = shared_keys & anchor_lines[a]
                if len(shared_keys) < min_rows:
                    break
            if len(shared_keys) < min_rows:
                continue

            # Find lines that hit every anchor
            by_line: Dict[int, List[dict]] = defaultdict(list)
            for s in spans:
                rs = s["bbox"][0]
                for a in anchors:
                    if abs(rs - a) <= col_tol_pt:
                        bbox = s["bbox"]
                        y_center = (bbox[1] + bbox[3]) / 2.0
                        key = int(round(y_center / line_tol))
                        by_line[key].append(s)
                        break

            qualifying_keys: List[int] = []
            qualifying_lines: List[List[dict]] = []
            for key in sorted(by_line.keys()):
                ln = sorted(by_line[key], key=lambda s: s["bbox"][0])
                picked: List[dict] = []
                used_idx = set()
                for a in anchors:
                    best_i = -1
                    best_d = col_tol_pt + 1
                    for i, s in enumerate(ln):
                        if i in used_idx:
                            continue
                        d = abs(s["bbox"][0] - a)
                        if d <= col_tol_pt and d < best_d:
                            best_d = d
                            best_i = i
                    if best_i >= 0:
                        picked.append(ln[best_i])
                        used_idx.add(best_i)
                if len(picked) == n_cols:
                    qualifying_keys.append(key)
                    qualifying_lines.append(picked)

            if len(qualifying_keys) < min_rows:
                continue

            # Find longest consecutive run (with stride ≤ 6)
            best_run_lines: List[List[dict]] = []
            best_run_keys: List[int] = []
            i = 0
            while i < len(qualifying_keys):
                rk = [qualifying_keys[i]]
                rl_ = [qualifying_lines[i]]
                j = i + 1
                while j < len(qualifying_keys):
                    if qualifying_keys[j] - rk[-1] > 6:
                        break
                    rk.append(qualifying_keys[j])
                    rl_.append(qualifying_lines[j])
                    j += 1
                if len(rk) > len(best_run_keys):
                    best_run_keys = rk
                    best_run_lines = rl_
                i = j if j > i else i + 1

            if len(best_run_keys) < min_rows:
                continue

            # Gate: alternating short-long pattern
            col_lens: List[List[float]] = [[] for _ in range(n_cols)]
            for ln in best_run_lines:
                for c, s in enumerate(ln):
                    col_lens[c].append(s["bbox"][2] - s["bbox"][0])
            col_meds = [sorted(cl_)[len(cl_) // 2] for cl_ in col_lens]
            short_cols = [0] + ([2] if n_cols == 4 else [])
            long_cols = [1] + ([3] if n_cols == 4 else [])
            pattern_ok = (
                all(col_meds[c] <= abbr_max_med_pt for c in short_cols)
                and all(col_meds[c] >= defn_min_med_pt for c in long_cols)
            )
            if not pattern_ok:
                continue

            # Gate: regular line pitch — at least 70% of inter-line
            # strides must be within ±1 of the mode stride.  This is
            # more robust than max-deviation because abbreviation
            # tables have small integer line_tol stride buckets where
            # typical strides are 2-3 with occasional 4 or 5 due to
            # letter-height rounding.  A pure "max_dev / median ≤ 30%"
            # gate would reject these by demanding stride_max ≤ 1.3 ×
            # stride_median (untenable when median is 3).
            if len(best_run_keys) >= 3:
                strides = [best_run_keys[k + 1] - best_run_keys[k]
                           for k in range(len(best_run_keys) - 1)]
                from collections import Counter as _Counter
                sc = _Counter(strides)
                mode_stride, _ = sc.most_common(1)[0]
                near_mode = sum(c for s, c in sc.items()
                                if abs(s - mode_stride) <= 1)
                pitch_ok = near_mode / len(strides) >= (
                    1.0 - line_pitch_var_max)
                if not pitch_ok:
                    continue

            # Build output
            all_x0 = min(s["bbox"][0] for ln in best_run_lines for s in ln)
            all_y0 = min(s["bbox"][1] for ln in best_run_lines for s in ln)
            all_x1 = max(s["bbox"][2] for ln in best_run_lines for s in ln)
            all_y1 = max(s["bbox"][3] for ln in best_run_lines for s in ln)
            if _overlaps_used(all_x0, all_x1, all_y0, all_y1):
                continue
            used_x_ranges.append((all_x0, all_x1, all_y0, all_y1))

            rows = [
                [(s["bbox"], s.get("text", "")) for s in ln]
                for ln in best_run_lines
            ]
            results.append({
                "bbox_pdf": (all_x0, all_y0, all_x1, all_y1),
                "rows": rows,
                "n_cols": n_cols,
            })

    return results
