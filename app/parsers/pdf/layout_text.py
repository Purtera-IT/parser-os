"""Layout-aware page text for the prose splitter.

``page.get_text("text")`` hands the prose splitter one line per text line and
NO paragraph breaks: PyMuPDF never emits a blank line between blocks. The
splitter only ends a paragraph at a blank line or a heading, so everything
between two headings became ONE paragraph, and therefore one atom. A CDW
"QUOTE CONFIRMATION" page glued the greeting, the eProcurement log-in
boilerplate, the "click below" call to action and the red "Convert Quote to
Order" button into a single 400-character atom.

It also reads the page as one column. Boxes standing side by side (shipping /
remit-to / sales contact; a two-column numbered install guide) were read row
by row across the boxes, fusing them or interleaving them.

This module rebuilds the page text from the layout instead:

* **Columns first.** Lines are grouped into rows; consecutive rows sharing a
  vertical gutter form a multi-column region. Each column of a region is read
  top to bottom (recursively, so a box inside a column is still a box). A
  region whose columns are really a table (a numeric column, or a narrow
  label column beside its values) is read row by row, cells joined by `` | ``.
* **Blank line at every visual-block boundary** inside a column: a vertical gap
  larger than the line pitch, a font-size change, a filled rectangle (a
  button / banner) entered or left, a region change, and (only when the line
  above is not a wrapped sentence) a weight / colour / link-run change or a
  salutation line ("MATTHEW BRUNTON,").

A wrapped line of one sentence is never broken: a soft (style / link) break is
skipped when the line above runs near the column's full width without
terminal punctuation.

Returns ``None`` on any failure so callers fall back to the plain text.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Iterable

_TERMINAL = re.compile(r"[.!?:;]\s*[\"”’')\]]*\s*$")
_NUMERICISH = re.compile(r"^[\s$€£#%()+\-.,/:x\d]*\d[\s$€£#%()+\-.,/:x\d]*(?:[A-Za-z]{0,3})$")
_SALUTATION = re.compile(r"^(?:(?:dear|hi|hello|greetings|attn:?)\s+)?[A-Z][\w.'’ -]{1,40},$", re.I)
# A list marker standing alone on its line: a step number ("6", "6.", "(6)",
# "Step 6") or a bullet glyph.
_ENUMERATOR = re.compile(r"^(?:(?:step\s+)?\(?\d{1,2}[.):]?|[•▪●◦‣■□➢►\-–—*])$", re.I)


class _Seg:
    """A run of spans on one text line with no wide horizontal gap inside."""

    __slots__ = ("x0", "y0", "x1", "y1", "text", "size", "bold", "color", "link", "fill", "boxes")

    def __init__(self, x0, y0, x1, y1, text, size, bold, color):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.text = text
        self.size = size
        self.bold = bold
        self.color = color
        self.link = False
        self.fill = -1
        #: Stroked frames drawn around this segment: ``((area, index), ...)``.
        self.boxes: tuple[tuple[float, int], ...] = ()

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def _dominant(spans: list[dict[str, Any]], key) -> Any:
    weights: dict[Any, int] = {}
    for s in spans:
        k = key(s)
        weights[k] = weights.get(k, 0) + len((s.get("text") or "").strip())
    return max(weights.items(), key=lambda kv: kv[1])[0] if weights else None


def _segments(page: Any, exclude: Iterable[Any]) -> list[_Seg]:
    data = page.get_text("dict") or {}
    excl = list(exclude or [])
    out: list[_Seg] = []
    for blk in data.get("blocks", []) or []:
        for ln in blk.get("lines", []) or []:
            spans = [s for s in (ln.get("spans") or []) if (s.get("text") or "")]
            if not spans:
                continue
            # Split the line wherever two spans sit further apart than a couple
            # of characters: that is a gutter between boxes, not a word space.
            groups: list[list[dict[str, Any]]] = [[spans[0]]]
            for s in spans[1:]:
                prev = groups[-1][-1]
                gap = float(s["bbox"][0]) - float(prev["bbox"][2])
                size = max(float(s.get("size") or 10), float(prev.get("size") or 10))
                if gap > 1.6 * size:
                    groups.append([s])
                else:
                    groups[-1].append(s)
            for g in groups:
                text = "".join(s.get("text", "") for s in g)
                if not text.strip():
                    continue
                inked = [s for s in g if (s.get("text") or "").strip()] or g
                x0 = min(float(s["bbox"][0]) for s in inked)
                y0 = min(float(s["bbox"][1]) for s in inked)
                x1 = max(float(s["bbox"][2]) for s in inked)
                y1 = max(float(s["bbox"][3]) for s in inked)
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                if any(_inside(cx, cy, r) for r in excl):
                    continue
                size = float(_dominant(inked, lambda s: round(float(s.get("size") or 0), 1)) or 0)
                bold = bool(_dominant(inked, lambda s: bool(int(s.get("flags") or 0) & 16)
                                      or "bold" in str(s.get("font") or "").lower()
                                      or str(s.get("font") or "").lower().endswith(("-bd", "bo"))))
                color = _dominant(inked, lambda s: int(s.get("color") or 0))
                out.append(_Seg(x0, y0, x1, y1, " ".join(text.split()), size, bold, color))
    return _attach_enumerators(out)


def _attach_enumerators(segs: list[_Seg]) -> list[_Seg]:
    """Glue a free-standing list marker to the step text beside it.

    A numbered instruction sheet sets the step number apart from its text
    ("6" in a large bold face, the sentence a few characters to its right).
    Left as its own segment the number became a column of figures, so a
    two-column step layout was read as a TABLE, row by row across both
    columns ("1 Flip the two breakers... | 2 | Loosen the turnbuckles..."),
    and the lone "6" reached the atomizer as a value. The marker belongs to
    the sentence that starts on its line, just to its right.

    Narrow: the marker must be alone in its segment, the text must start on
    the same line within a few characters, and the text must read as a
    sentence (a capitalised word followed by at least two more words), so a
    number cell in a grid of figures is never glued to its neighbour.
    """
    used: set[int] = set()
    by_id = list(segs)
    rail = _numbered_rail(by_id)
    for i, s in enumerate(by_id):
        if i in used or not _ENUMERATOR.match(s.text.strip()):
            continue
        h = max(1.0, s.y1 - s.y0)
        # A marker on a numbered RAIL (1, 2, 3 ... stacked at one x) may sit
        # well left of its text: a step guide set with a wide number column
        # was read as a four-column table, both columns of steps interleaved
        # row by row ("Flip the two breakers | Connect the cellular").
        reach = (12.0 if i in rail else 3.0) * max(s.size, 6.0)
        best = None
        for j, t in enumerate(by_id):
            if j == i or j in used:
                continue
            gap = t.x0 - s.x1
            if gap < -0.5 or gap > max(reach, 3.0 * max(s.size, t.size, 6.0)):
                continue
            ov = min(s.y1, t.y1) - max(s.y0, t.y0)
            if ov < 0.3 * min(h, max(1.0, t.y1 - t.y0)):
                continue
            # A marker set vertically centred on a step that wraps over three
            # or four lines sits beside a MIDDLE line: "1" beside "marked
            # PUMP to the off position", with "Flip the two breakers" a line
            # above. Gluing it there cut the step in two, mid-sentence. Walk
            # up to the first line of the block the marker heads.
            j = _block_top(by_id, j, s)
            t = by_id[j]
            if j in used or j == i:
                continue
            words = t.text.split()
            if len(words) < 3 or not words[0][:1].isupper() or _ENUMERATOR.match(t.text.strip()):
                continue
            if best is None or gap < best[0]:
                best = (gap, j)
        if best is None:
            continue
        t = by_id[best[1]]
        t.text = f"{s.text.strip()} {t.text}"
        t.x0 = min(t.x0, s.x0)
        t.y0 = min(t.y0, s.y0)
        t.y1 = max(t.y1, s.y1)
        used.add(i)
    return [s for i, s in enumerate(by_id) if i not in used]


def _numbered_rail(segs: list[_Seg]) -> set[int]:
    """Indices of bare step numbers stacked at one x that count up by one
    (1, 2, 3 ...; at least three of them). A column of figures in a table
    does not count 1, 2, 3 down a column with prose beside each."""
    nums: list[tuple[int, int]] = []
    for i, s in enumerate(segs):
        m = re.fullmatch(r"(?:step\s+)?\(?(\d{1,2})[.):]?", s.text.strip(), re.I)
        if m:
            nums.append((i, int(m.group(1))))
    out: set[int] = set()
    groups: list[list[tuple[int, int]]] = []
    for i, n in sorted(nums, key=lambda p: (round(segs[p[0]].x0 / 4.0), segs[p[0]].y0)):
        if groups and abs(segs[groups[-1][-1][0]].x0 - segs[i].x0) <= 4.0:
            groups[-1].append((i, n))
        else:
            groups.append([(i, n)])
    for g in groups:
        g.sort(key=lambda p: segs[p[0]].y0)
        run = [g[0]]
        for p in g[1:]:
            if p[1] == run[-1][1] + 1:
                run.append(p)
            else:
                if len(run) >= 3:
                    out.update(i for i, _ in run)
                run = [p]
        if len(run) >= 3:
            out.update(i for i, _ in run)
    return out


def _block_top(segs: list[_Seg], j: int, marker: _Seg) -> int:
    """Index of the first line of the left-aligned text block whose line
    ``segs[j]`` is, walking up only while the line above starts at the same
    x, sits one line pitch above, does not end a sentence, and still lies
    beside the marker (its centre no higher than one marker-height above
    the marker's top)."""
    cur = j
    for _ in range(6):
        t = segs[cur]
        lh = max(1.0, t.y1 - t.y0)
        above = None
        for k, u in enumerate(segs):
            if k == cur or abs(u.x0 - t.x0) > 2.0:
                continue
            gap = t.y0 - u.y1
            if gap < -0.3 * lh or gap > 0.8 * lh:
                continue
            if u.cy >= t.cy:
                continue
            if above is None or u.y1 > segs[above].y1:
                above = k
        if above is None:
            break
        u = segs[above]
        if _TERMINAL.search(u.text) or u.cy < marker.y0 - (marker.y1 - marker.y0):
            break
        cur = above
    return cur


def _inside(cx: float, cy: float, r: Any) -> bool:
    try:
        return r.x0 - 1 <= cx <= r.x1 + 1 and r.y0 - 1 <= cy <= r.y1 + 1
    except Exception:
        return False


def _mark_boxes(page: Any, segs: list[_Seg]) -> None:
    """Record the stroked frames around each segment (a table's border, a
    box's outline), so two framed tables set side by side are never read as
    one grid."""
    try:
        pr = page.rect
        page_area = float(pr.width * pr.height) or 1.0
        rects = []
        for d in page.get_drawings() or []:
            r = d.get("rect")
            if r is None or d.get("color") is None or "s" not in str(d.get("type") or ""):
                continue
            area = float(r.width * r.height)
            if area <= 0 or area > 0.6 * page_area or r.width < 20 or r.height < 8:
                continue
            rects.append((area, r))
    except Exception:
        return
    for sg in segs:
        sg.boxes = tuple((a, i) for i, (a, r) in enumerate(rects) if _inside(sg.x0 + 1, sg.cy, r)
                         and _inside(sg.x1 - 1, sg.cy, r))


def _split_by_frames(run: list[list[_Seg]]) -> list[list[_Seg]]:
    """The run's segments grouped by the largest frame that holds some but
    not all of them (left to right); one group when nothing frames them
    apart."""
    segs = [sg for r in run for sg in r]
    everywhere = set.intersection(*(set(sg.boxes) for sg in segs)) if segs else set()
    keyed: dict[int, list[_Seg]] = {}
    for sg in segs:
        own = [b for b in sg.boxes if b not in everywhere]
        keyed.setdefault(max(own)[1] if own else -1, []).append(sg)
    if len(keyed) < 2 or -1 in keyed:
        return [segs]
    return sorted(keyed.values(), key=lambda g: (min(sg.x0 for sg in g), min(sg.y0 for sg in g)))


def _mark_links_and_fills(page: Any, segs: list[_Seg]) -> None:
    try:
        links = [l.get("from") for l in (page.get_links() or []) if l.get("from") is not None]
    except Exception:
        links = []
    try:
        pr = page.rect
        page_area = float(pr.width * pr.height) or 1.0
        fills = []
        for d in page.get_drawings() or []:
            fill = d.get("fill")
            r = d.get("rect")
            if fill is None or r is None:
                continue
            # White / near-white fill is the page background, not a box.
            if all(float(c) >= 0.95 for c in fill):
                continue
            area = float(r.width * r.height)
            if area <= 0 or area > 0.25 * page_area or r.height < 6:
                continue
            fills.append(r)
    except Exception:
        fills = []
    for s in segs:
        w = max(1e-6, s.x1 - s.x0)
        for lr in links:
            ov = max(0.0, min(s.x1, lr.x1) - max(s.x0, lr.x0))
            if ov >= 0.5 * w and lr.y0 - 1 <= s.cy <= lr.y1 + 1:
                s.link = True
                break
        for i, fr in enumerate(fills):
            if _inside(s.x0 + 1, s.cy, fr) and _inside(s.x1 - 1, s.cy, fr):
                s.fill = i
                break


def _rows(segs: list[_Seg]) -> list[list[_Seg]]:
    """Group segments that share a text line (vertical-centre overlap)."""
    rows: list[list[_Seg]] = []
    for s in sorted(segs, key=lambda s: (s.cy, s.x0)):
        if rows:
            r = rows[-1]
            ry0 = min(t.y0 for t in r)
            ry1 = max(t.y1 for t in r)
            h = min(ry1 - ry0, s.y1 - s.y0)
            ov = min(ry1, s.y1) - max(ry0, s.y0)
            if h > 0 and ov >= 0.5 * h:
                r.append(s)
                continue
        rows.append([s])
    for r in rows:
        r.sort(key=lambda s: s.x0)
    return rows


def _gutters(rows: list[list[_Seg]], min_gap: float) -> list[tuple[float, float]]:
    spans = sorted((s.x0, s.x1) for r in rows for s in r)
    if not spans:
        return []
    out: list[tuple[float, float]] = []
    cur0, cur1 = spans[0]
    for a, b in spans[1:]:
        if a - cur1 >= min_gap:
            out.append((cur1, a))
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    return out


def _regions(rows: list[list[_Seg]]) -> list[tuple[list[list[_Seg]], list[tuple[float, float]]]]:
    """Consecutive rows that keep the same column structure form one region."""
    if not rows:
        return []
    sizes = [s.size for r in rows for s in r if s.size] or [10.0]
    min_gap = max(10.0, 1.6 * statistics.median(sizes))
    regions: list[tuple[list[list[_Seg]], list[tuple[float, float]]]] = []
    cur: list[list[_Seg]] = [rows[0]]
    cur_g = _gutters(cur, min_gap)
    med = statistics.median(sizes)
    for row in rows[1:]:
        row_g = _gutters([row], min_gap)
        trial_g = _gutters(cur + [row], min_gap)
        # A wide band of white space ends a column region when the row below
        # it does not keep the region's gutters: boxes set above a table share
        # its gutter COUNT by coincidence, and kept together the
        # table below was read column by column ("CDW# 7506872 5502114",
        # "Mfg# QM75C WMN6575SE") under the boxes' columns.
        if cur_g and min(s.y0 for s in row) - max(s.y1 for s in cur[-1]) > 2.5 * med \
                and not _gutters_kept(cur_g, trial_g):
            regions.append((cur, cur_g))
            cur, cur_g = [row], row_g
            continue
        if cur_g:
            # Same columns: the row sits inside the region's gutters.
            keep = len(trial_g) == len(cur_g)
        else:
            # A single-column run continues while rows stay single-column; it
            # also becomes the first rows of a column region when none of its
            # lines crosses the gutter the new row opens (a box that starts a
            # line higher than its neighbour).
            keep = (not row_g) or bool(trial_g)
        if keep:
            cur.append(row)
            cur_g = trial_g
        else:
            regions.append((cur, cur_g))
            cur, cur_g = [row], row_g
    regions.append((cur, cur_g))
    return _pull_box_heads(regions, min_gap)


def _gutters_kept(cur_g: list[tuple[float, float]], trial_g: list[tuple[float, float]]) -> bool:
    """Every gutter of the region survives, at least half its width, once the
    row is added."""
    for a, b in cur_g:
        w = max(1e-6, b - a)
        if not any(min(b, d) - max(a, c) >= 0.5 * w for c, d in trial_g):
            return False
    return True


def _pull_box_heads(
    regions: list[tuple[list[list[_Seg]], list[tuple[float, float]]]], min_gap: float
) -> list[tuple[list[list[_Seg]], list[tuple[float, float]]]]:
    """A box that starts higher than the box beside it begins in the
    single-column run above their shared rows: a ship-to box ("SHIP TO:",
    company, street ...) whose last lines sit beside a "Shipping Method" box.
    Cut off from its head, the column region held only the shared rows, two
    rows read as a table and fused "NEW YORK, NY 10014-1066 | Shipping
    Method: DROP SHIP-GROUND" (010003). Two or more trailing rows of the run
    above, stacked tight on the first column's left edge and leaving a gutter
    before the second column, move down into the column region."""
    out = list(regions)
    for i in range(1, len(out)):
        rows, gut = out[i]
        prev_rows, prev_gut = out[i - 1]
        if not gut or prev_gut or len(prev_rows) < 3:
            continue
        first = [s for s in rows[0] if s.x1 <= gut[0][0] + 0.5]
        if not first:
            continue
        x0 = min(s.x0 for s in first)
        lh = max(1.0, max(s.y1 - s.y0 for s in first))
        take = 0
        top = min(s.y0 for s in rows[0])
        for r in reversed(prev_rows):
            if top - max(s.y1 for s in r) > 0.9 * lh:
                break
            if abs(min(s.x0 for s in r) - x0) > 3.0 or max(s.x1 for s in r) > gut[0][1] - min_gap:
                break
            take += 1
            top = min(s.y0 for s in r)
            if take >= len(prev_rows) - 1:
                break
        if take < 2:
            continue
        moved = prev_rows[-take:]
        out[i - 1] = (prev_rows[:-take], prev_gut)
        new_rows = moved + rows
        out[i] = (new_rows, _gutters(new_rows, min_gap) or gut)
    return [r for r in out if r[0]]


def _split_columns(rows: list[list[_Seg]], gutters: list[tuple[float, float]]) -> list[list[_Seg]]:
    cuts = [(a + b) / 2.0 for a, b in gutters]
    cols: list[list[_Seg]] = [[] for _ in range(len(cuts) + 1)]
    for r in rows:
        for s in r:
            i = sum(1 for c in cuts if (s.x0 + s.x1) / 2.0 > c)
            cols[i].append(s)
    return cols


def _is_table(rows: list[list[_Seg]], cols: list[list[_Seg]]) -> bool:
    """A grid read row by row (cells of a row belong together) rather than as
    independent boxes read top to bottom."""
    if len(rows) <= 2:
        return True  # one row of cells, or a header over one row of values
    # A column of figures (prices, quantities, dates, ids) means rows.
    # The top cell of each column is its header, so it is not counted.
    for col in cols[1:]:
        cells = [s.text for s in sorted(col, key=lambda s: s.y0)[1:]]
        if len(cells) >= 2:
            num = sum(1 for c in cells if _NUMERICISH.match(c))
            if num >= 0.6 * len(cells):
                return True
    # A narrow label column beside its values ("Customer:" | "Acme") is a
    # key/value grid, not two boxes.
    if len(cols) == 2 and cols[0] and cols[1]:
        left = cols[0]
        colon = sum(1 for s in left if s.text.rstrip().endswith(":"))
        aligned = sum(1 for r in rows if len(r) >= 2)
        # A right cell wrapped onto one more line under a full row (a site's
        # "NEW YORK, NY 10014" under its street, 010003) is still that row.
        left_ids = {id(s) for s in left}
        aligned += sum(
            1 for prev, r in zip(rows, rows[1:])
            if len(prev) >= 2 and len(r) == 1 and id(r[0]) not in left_ids
        )
        lw = max(s.x1 for s in left) - min(s.x0 for s in left)
        rw = max(s.x1 for s in cols[1]) - min(s.x0 for s in cols[1])
        if colon >= 0.5 * len(left):
            return True
        if aligned >= 0.8 * len(rows) and lw < 0.5 * rw:
            return True
        # Every row is a full row or a right cell wrapped under one, and a
        # wrap is a row's own (one between two full rows, or an address's
        # "City, ST ZIP" line closing the last row): a grid whose left cells
        # name what the right cells hold. Its width ratio says nothing --
        # "New York, NY 10014" is narrower than "NEW YORK, NY 10014" and the
        # 010003 site table fell apart into two boxes on the case.
        wraps = [
            i for i in range(1, len(rows))
            if len(rows[i - 1]) >= 2 and len(rows[i]) == 1 and id(rows[i][0]) not in left_ids
        ]
        full = sum(1 for r in rows if len(r) >= 2)
        if wraps and full >= 2 and full + len(wraps) == len(rows) and lw < rw and all(
            (i + 1 < len(rows) and len(rows[i + 1]) >= 2) or _CITY_LINE.search(rows[i][0].text)
            for i in wraps
        ):
            return True
    return False


#: The city line of an address: "New York, NY 10014".
_CITY_LINE = re.compile(r"[A-Za-z .'-]+,\s*[A-Z]{2}\.?\s+\d{5}(?:-\d{4})?\s*$")


def _style_key(s: _Seg) -> tuple:
    return (s.bold, s.color, s.link)


def _leaf_lines(segs: list[_Seg]) -> list[str]:
    """One column box, top to bottom, with "" at every visual-block boundary."""
    rows = _rows(segs)
    if not rows:
        return []
    lines = [(r, " ".join(s.text for s in r)) for r in rows]
    width = max(max(s.x1 for s in r) for r, _ in lines) - min(min(s.x0 for s in r) for r, _ in lines)
    lefts = [min(s.x0 for s in r) for r, _ in lines]
    pitches = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        sz = max(max(s.size for s in a), max(s.size for s in b)) or 10.0
        p = min(s.y1 for s in b) - max(s.y1 for s in a)
        if 0 < p < 2.0 * sz:
            pitches.append(p)
    pitch = statistics.median(pitches) if pitches else None

    out: list[str] = []
    for i, (row, text) in enumerate(lines):
        if i == 0:
            out.append(text)
            continue
        prow, ptext = lines[i - 1]
        ps = max(prow, key=lambda s: len(s.text))
        cs = max(row, key=lambda s: len(s.text))
        psize = max(s.size for s in prow) or 10.0
        csize = max(s.size for s in row) or 10.0
        p = min(s.y1 for s in row) - max(s.y1 for s in prow)
        hard = False
        # Vertical gap: wider than the box's own line pitch.
        if pitch is not None:
            if p > 1.35 * pitch and p - pitch > 2.5:
                hard = True
        elif p > 1.7 * max(psize, csize):
            hard = True
        # Font size change (a heading or a fine-print band).
        if abs(csize - psize) > 0.15 * max(csize, psize):
            hard = True
        # Into / out of a filled box (a button or banner).
        if ps.fill != cs.fill:
            hard = True
        soft = False
        if not hard:
            if _style_key(ps) != _style_key(cs):
                soft = True
                # A short bold label over its box's body ("Shipping Method" /
                # "UPS Ground") is the box's caption, not a separate block.
                if ps.bold and not cs.bold and len(ptext.split()) <= 6 \
                        and not re.search(r"[.!?]$", ptext.strip()) \
                        and (ps.color, ps.link) == (cs.color, cs.link):
                    soft = False
            if _SALUTATION.match(ptext.strip()) and len(ptext.split()) <= 5:
                soft = True
            # A line that starts left of the line above's start (outdent) after
            # a terminal line is a new paragraph only with another signal, so
            # indentation alone is not used.
            if soft:
                wrapped = (
                    width > 0
                    and (max(s.x1 for s in prow) - lefts[i - 1]) >= 0.8 * width
                    and not _TERMINAL.search(ptext)
                )
                if wrapped:
                    soft = False
        if hard or soft:
            out.append("")
        out.append(text)
    return out


_SELF_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9 .#/&()'-]{0,30}?:(?:\s+\S|\s*$)")


def _labels_itself(cell: str) -> bool:
    m = _SELF_LABEL.match(cell)
    return bool(m) and len(cell.split(":", 1)[0].split()) <= 4


def _self_labelled_records(rows: list[list[_Seg]], gutters: list[tuple[float, float]]) -> list[str]:
    """A contact / form grid whose cells carry their own labels ("FULL NAME:
    Chase Smith", "JOB TITLE: Director of Operations", "EMAIL ADDRESS: ..."),
    one record per row, read as one line per record with its cells joined by
    " | ". A cell whose value wrapped onto the line below (a row with no
    label of its own) folds into the cell above it in the same column. Read
    column by column instead, every field was its own line and a wrapped
    value ("EMAIL ADDRESS:" / "john@...") read as a heading. ``[]`` when the
    region is not such a grid."""
    cuts = [(a + b) / 2.0 for a, b in gutters]
    if not cuts:
        return []
    records: list[list[str]] = []
    for r in rows:
        cells = [""] * (len(cuts) + 1)
        for sg in sorted(r, key=lambda x: x.x0):
            i = sum(1 for c in cuts if (sg.x0 + sg.x1) / 2.0 > c)
            cells[i] = f"{cells[i]} {sg.text}".strip()
        filled = [c for c in cells if c]
        if any(_labels_itself(c) for c in filled):
            records.append(cells)
        elif records:
            for i, c in enumerate(cells):
                if c:
                    records[-1][i] = f"{records[-1][i]} {c}".strip()
        else:
            return []
    full = [[c for c in rec if c] for rec in records]
    if not full or any(len(f) < 2 for f in full):
        return []
    cells_all = [c for f in full for c in f]
    if sum(1 for c in cells_all if _labels_itself(c)) < 0.8 * len(cells_all):
        return []
    # One label twice on a line ("Name: Jane Doe" | "Name: John Smith") is two
    # boxes side by side -- the two parties of a signature block -- whose
    # columns are read one box at a time, not a record per line.
    for f in full:
        labels = [c.split(":", 1)[0].strip().lower() for c in f if _labels_itself(c)]
        if len(labels) != len(set(labels)):
            return []
    return [" | ".join(f) for f in full]


def _record_runs(rows: list[list[_Seg]]) -> list[tuple[bool, list[list[_Seg]]]]:
    """Split ``rows`` into ``(is_record_grid, rows)`` runs: a run of rows of
    two or more self-labelled cells (plus the lines their values wrapped onto)
    is a record grid; everything else is read as before."""
    sizes = [sg.size for r in rows for sg in r if sg.size] or [10.0]
    min_gap = max(10.0, 1.6 * statistics.median(sizes))
    lh = statistics.median([max(1.0, sg.y1 - sg.y0) for r in rows for sg in r]) if rows else 10.0
    out: list[tuple[bool, list[list[_Seg]]]] = []
    buf: list[list[_Seg]] = []
    i = 0
    while i < len(rows):
        if len(rows[i]) >= 2 and any(_labels_itself(sg.text.strip()) for sg in rows[i]):
            j, wraps = i + 1, 0
            while j < len(rows):
                gap = min(sg.y0 for sg in rows[j]) - max(sg.y1 for sg in rows[j - 1])
                if gap > 1.6 * lh:
                    break
                labelled = any(_labels_itself(sg.text.strip()) for sg in rows[j])
                if labelled and len(rows[j]) >= 2:
                    wraps = 0
                elif not labelled and wraps < 2:
                    wraps += 1
                else:
                    break
                j += 1
            run = rows[i:j]
            while run and not any(_labels_itself(sg.text.strip()) for sg in run[-1]) and wraps > 0 \
                    and len(run) > 1 and len(run[-1]) == 1 and len(run[-2]) >= 2 \
                    and _TERMINAL.search(run[-1][0].text):
                run, wraps = run[:-1], wraps - 1  # a closing sentence, not a wrap
                j -= 1
            gutters = _gutters(run, min_gap) if len(run) >= 2 else []
            if gutters and _self_labelled_records(run, gutters):
                if buf:
                    out.append((False, buf))
                    buf = []
                out.append((True, run))
                i = j
                continue
        buf.append(rows[i])
        i += 1
    if buf:
        out.append((False, buf))
    return out


def _read(segs: list[_Seg], depth: int = 0) -> list[str]:
    out: list[str] = []
    for is_grid, run in _record_runs(_rows(segs)):
        if out and out[-1] != "":
            out.append("")
        if is_grid:
            sizes = [sg.size for r in run for sg in r if sg.size] or [10.0]
            min_gap = max(10.0, 1.6 * statistics.median(sizes))
            for group in _split_by_frames(run):
                rows = _rows(group)
                recs = _self_labelled_records(rows, _gutters(rows, min_gap)) if len(rows) >= 2 else []
                if out and out[-1] != "":
                    out.append("")
                out.extend(recs or _read_rows(rows, depth))
            continue
        out.extend(_read_rows(run, depth))
    return out


def _read_rows(rows: list[list[_Seg]], depth: int = 0) -> list[str]:
    out: list[str] = []
    for region_rows, gutters in _regions(rows):
        if out and out[-1] != "":
            out.append("")
        if not gutters or depth >= 3:
            out.extend(_leaf_lines([s for r in region_rows for s in r]))
            continue
        cols = _split_columns(region_rows, gutters)
        if _is_table(region_rows, cols):
            cuts = [(a + b) / 2.0 for a, b in gutters]
            grid: list[list[str]] = []
            for r in region_rows:
                cells = [""] * (len(cuts) + 1)
                for s in r:
                    i = sum(1 for c in cuts if (s.x0 + s.x1) / 2.0 > c)
                    cells[i] = f"{cells[i]} {s.text}".strip()
                grid.append(cells)
            if len(grid) == 2 and all(grid[0]) and all(grid[1]):
                # A header row over one value row ("QUOTE # | QUOTE DATE ..." /
                # "PSNV676 | 1/14/2026 ...") is a set of labelled values: keep
                # each value with its label on one line.
                out.append(" | ".join(f"{h}: {v}" for h, v in zip(grid[0], grid[1])))
                continue
            for cells in grid:
                filled = [c for c in cells if c]
                out.append(" | ".join(filled) if len(filled) > 1 else (filled[0] if filled else ""))
            continue
        for col in cols:
            if not col:
                continue
            if out and out[-1] != "":
                out.append("")
            out.extend(_read(col, depth + 1))
    return out


def layout_page_text(page: Any, exclude_bboxes: Iterable[Any] | None = None) -> str | None:
    """The page's text in column-aware reading order with a blank line between
    visual blocks, or ``None`` when the page has no usable text layer."""
    try:
        segs = _segments(page, exclude_bboxes or [])
        if not segs:
            return None
        _mark_links_and_fills(page, segs)
        _mark_boxes(page, segs)
        lines = _read(segs)
        # Collapse runs of blank lines.
        cleaned: list[str] = []
        for ln in lines:
            if ln == "" and (not cleaned or cleaned[-1] == ""):
                continue
            cleaned.append(ln)
        while cleaned and cleaned[-1] == "":
            cleaned.pop()
        return "\n".join(cleaned) + "\n" if cleaned else None
    except Exception:
        return None


def region_is_side_by_side_boxes(page: Any, bbox: Any) -> bool:
    """True when the text inside ``bbox`` lays out as independent boxes standing
    side by side (shipping | remit-to | contact; a two-column step list) rather
    than a grid whose rows belong together. A whitespace-column table extractor
    pairs such boxes line by line, fusing unrelated lines into "rows"."""
    try:
        segs = [s for s in _segments(page, []) if _inside((s.x0 + s.x1) / 2.0, s.cy, bbox)]
        if len(segs) < 4:
            return False
        for region_rows, gutters in _regions(_rows(segs)):
            if not gutters or len(region_rows) < 3:
                continue
            if not _is_table(region_rows, _split_columns(region_rows, gutters)):
                return True
        return False
    except Exception:
        return False


# ── Vendor quote / PO line-item grids ────────────────────────────────────
#
# A reseller quote (CDW, SHI, Insight ...) or a PO sets its items under a
# header row ("ITEM | QTY | CDW# | UNIT PRICE | EXT. PRICE"). Each item is an
# ANCHOR line carrying the figures, then lines under it in the text column
# only: the wrapped description, "Mfg. Part#: QM55C", "Contract: Standard
# Pricing". Read as text, the anchor line was one row and its tail glued to
# the NEXT item ("...Standard Pricing Samsung QM75C ..."); read by the
# whitespace-column extractor, a right-aligned quantity drifted out of the
# QTY column and the item number landed under QTY (010003). This reads the
# grid by its header: every cell is assigned to the header it sits under,
# every tail line folds into the item above it, and a "Label: value" tail
# line becomes a field of its own.

_LABEL_VALUE = re.compile(r"^([A-Za-z][\w .#/&'-]{0,30}?)\s*:\s*(\S.*)$")
_HEADER_WORD = re.compile(r"^[A-Za-z][A-Za-z .#/&'()-]*$")


def _cell_column(seg: _Seg, heads: list[_Seg]) -> int:
    best, best_ov = -1, 0.0
    for i, h in enumerate(heads):
        ov = min(seg.x1, h.x1) - max(seg.x0, h.x0)
        if ov > best_ov:
            best, best_ov = i, ov
    if best >= 0:
        return best
    cx = (seg.x0 + seg.x1) / 2.0
    return min(range(len(heads)), key=lambda i: abs(cx - (heads[i].x0 + heads[i].x1) / 2.0))


def _is_header_row(row: list[_Seg]) -> bool:
    if len(row) < 3:
        return False
    for s in row:
        t = s.text.strip()
        if not t or len(t.split()) > 3 or not _HEADER_WORD.match(t) or len(t) > 24:
            return False
    return True


def header_grid_tables(page: Any, exclude_bboxes: Iterable[Any] | None = None) -> list[tuple[dict[str, Any], Any]]:
    """``[(table block, fitz.Rect)]`` for each line-item grid on the page."""
    try:
        import fitz  # type: ignore[import-not-found]

        segs = _segments(page, exclude_bboxes or [])
        rows = _rows(segs)
    except Exception:
        return []
    out: list[tuple[dict[str, Any], Any]] = []
    taken: set[int] = set()
    i = 0
    while i < len(rows):
        head = rows[i]
        if not _is_header_row(head):
            i += 1
            continue
        heads = sorted(head, key=lambda s: s.x0)
        lh = statistics.median([max(1.0, s.y1 - s.y0) for r in rows for s in r]) if rows else 10.0
        records: list[dict[str, Any]] = []
        numeric_cols: set[int] = set()
        prev_bottom = max(s.y1 for s in head)
        used = [head]
        j = i + 1
        while j < len(rows):
            row = rows[j]
            gap = min(s.y0 for s in row) - prev_bottom
            if gap > (2.5 if not records else 1.6) * lh:
                break
            cells: dict[int, list[str]] = {}
            for s in row:
                cells.setdefault(_cell_column(s, heads), []).append(s.text.strip())
            nums = {c for c, ts in cells.items() if all(_NUMERICISH.match(t) for t in ts)}
            if 0 in cells and len(cells) >= 2 and nums:
                records.append({"cells": cells, "extra": []})
                numeric_cols |= nums
            elif records and not nums and not (set(cells) & numeric_cols) \
                    and not _is_header_row(row):
                rec = records[-1]
                for c, ts in cells.items():
                    for t in ts:
                        m = _LABEL_VALUE.match(t)
                        if m and c == 0:
                            rec["extra"].append((m.group(1).strip(), m.group(2).strip()))
                        else:
                            rec["cells"].setdefault(c, []).append(t)
            else:
                break
            used.append(row)
            prev_bottom = max(s.y1 for s in row)
            j += 1
        # One header over one value row ("QUOTE # | QUOTE DATE | ... |
        # GRAND TOTAL" over "PSNV676 | 1/14/2026 | ...") is a set of labelled
        # values: one row whose cells pair each header with its value.
        if numeric_cols and records:
            columns = [h.text.strip() for h in heads]
            out_rows = []
            for rec in records:
                d: dict[str, str] = {}
                for c, name in enumerate(columns):
                    d[name] = " ".join(rec["cells"].get(c, [])).strip()
                for k, v in rec["extra"]:
                    if k not in d:
                        d[k] = v
                out_rows.append(d)
            box = fitz.Rect(min(s.x0 for r in used for s in r), min(s.y0 for r in used for s in r),
                            max(s.x1 for r in used for s in r), max(s.y1 for r in used for s in r))
            out.append(({"kind": "table", "columns": columns, "rows": out_rows,
                         "extraction": "header_grid_v1"}, box))
            taken.update(id(r) for r in used)
            i = j
        else:
            i += 1
    # Totals: "SUBTOTAL  $5,694.50", "SALES TAX  $0.00", "GRAND TOTAL  $4,691.64"
    # set as a label with its amount to the right. Read as two boxes the
    # amount came away from its label; each is one labelled value.
    for row in rows:
        if id(row) in taken or len(row) < 2:
            continue
        segs = sorted(row, key=lambda s: s.x0)
        label = " ".join(s.text.strip() for s in segs[:-1]).strip().rstrip(":").strip()
        amount = segs[-1].text.strip()
        if not (_TOTAL_LABEL.match(label) and _MONEY.fullmatch(amount)):
            continue
        try:
            box = fitz.Rect(min(s.x0 for s in segs), min(s.y0 for s in segs),
                            max(s.x1 for s in segs), max(s.y1 for s in segs))
        except Exception:
            continue
        out.append(({"kind": "table", "columns": [label], "rows": [{label: amount}],
                     "extraction": "labelled_total_v1"}, box))
    return out


_TOTAL_LABEL = re.compile(
    r"^(?:(?:sub|grand|order|quote|estimated|est\.?)\s*)?total(?:\s+(?:amount|price|due|cost))?|"
    r"^(?:sales\s+|use\s+)?tax(?:es)?|^shipping(?:\s*(?:&|and)\s*handling)?|^freight|^handling|"
    r"^(?:environmental|recycling|ewaste|e-waste)\s+fees?|^discount|^balance\s+due|^amount\s+due",
    re.I)
_MONEY = re.compile(r"-?\(?\$?\s?\d[\d,]*(?:\.\d{2})?\)?|\$?0\.00|free|tbd|included", re.I)
