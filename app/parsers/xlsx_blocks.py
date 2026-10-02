"""Structure-faithful xlsx block detection.

Splits a sheet's rows into rectangular BLOCKS so each atom can carry a real
path — ``sheet > title > table > row {col_header: cell}`` — instead of the
single-header-per-sheet model that drops multi-block sheets (e.g. a Deal Kit
"Summary" tab holding a "Detailed Level of Effort" table AND a side "Key Unit
Metrics" table). Pure: takes already-read cell rows, returns block dicts.

Each block: {"title": str|None, "kind": "table"|"keyval"|"text",
             "header": list[str] (table only), "rows": list[list[str]] (table
             data rows), "pairs": list[(label,value)] (keyval), "text": str}.
"""
from __future__ import annotations
import re

_NUM = re.compile(r"^-?[\d,]+(\.\d+)?%?$")


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return str(round(v, 4))
    s = str(v).strip()
    return "" if s.lower() in ("none", "nan") else s


class _Row(list):
    """A sheet row that remembers which row of the sheet it is.

    Block detection slices rows by blank-row bands and blank-column groups, and
    the resulting `row` in an atom's locator was a running counter over those
    blocks -- not a worksheet row. Source replay reads absolute worksheet rows,
    so on deal 010215 every xlsx_block_row_v1 atom cited a row exactly two off,
    and the Deal Kit's priced service lines could not be verified at all.

    A list subclass so every existing check (_row_blank, _filled, indexing,
    len) behaves identically; only the origin travels with it.
    """

    __slots__ = ("sheet_row",)

    def __init__(self, cells, sheet_row=None):
        super().__init__(cells)
        self.sheet_row = sheet_row


def _grid(rows):
    return [_Row((_clean(c) for c in r), i + 1) for i, r in enumerate(rows)]


def _row_blank(r):
    return all(c == "" for c in r)


def _filled(r):
    return sum(1 for x in r if x != "")


def _is_num(s):
    return bool(_NUM.match(s.replace("$", "").strip())) if s else False


def _band_split(grid):
    bands, cur = [], []
    for r in grid:
        if _row_blank(r):
            if cur:
                bands.append(cur); cur = []
        else:
            cur.append(r)
    if cur:
        bands.append(cur)
    return bands


def _rejoin_split_table(band, groups):
    """Put back together one table that a blank column cut in two.

    "Item | Description | <blank> | Qty | Unit Price" is one table with a
    spacer column, not two tables: split there, each part number lost its
    prices to a separate block. Two neighbouring column groups are one table
    when they fill the same rows and either side fails to read as a table of
    its own -- its rows below the first do not carry both a label and a
    number. Side-by-side tables (a Level of Effort table beside a Key Unit
    Metrics box) each carry both, or fill different rows, and stay apart.
    """
    if len(groups) < 2:
        return groups

    def _cells(r, a, b):
        return [r[c] for c in range(a, b) if c < len(r) and r[c] != ""]

    def _rows(a, b):
        return {i for i, r in enumerate(band) if _cells(r, a, b)}

    def _self_contained(a, b):
        filled = [_cells(r, a, b) for r in band]
        filled = [f for f in filled if f][1:]
        if not filled:
            return False
        whole = sum(1 for f in filled
                    if any(_is_num(x) for x in f) and any(not _is_num(x) for x in f))
        return whole * 5 >= len(filled) * 4

    out = [groups[0]]
    for g in groups[1:]:
        a0, b0 = out[-1]
        a1, b1 = g
        if (_rows(a0, b0) == _rows(a1, b1)
                and not (_self_contained(a0, b0) and _self_contained(a1, b1))):
            out[-1] = (a0, b1)
        else:
            out.append(g)
    return out


def _col_split(band):
    width = max((len(r) for r in band), default=0)
    blank_col = [all((c >= len(r) or r[c] == "") for r in band) for c in range(width)]
    groups, start = [], None
    for c in range(width):
        if not blank_col[c] and start is None:
            start = c
        elif blank_col[c] and start is not None:
            groups.append((start, c)); start = None
    if start is not None:
        groups.append((start, width))
    groups = _rejoin_split_table(band, groups)
    out = []
    for (a, b) in groups:
        # Carry each row's sheet origin across the column slice, or the block
        # loses the only link back to where it came from.
        sub = [
            _Row(((r[c] if c < len(r) else "") for c in range(a, b)), getattr(r, "sheet_row", None))
            for r in band
        ]
        # KEEP blank rows: a column often stacks several sub-tables separated by
        # rows that are blank IN THIS column (but not across the sheet). Dropping
        # them here would erase those boundaries and collapse the sub-tables into
        # one blob — re-band-split downstream needs the blanks to separate them.
        if any(any(x != "" for x in r) for r in sub):
            out.append((a, b, sub))
    return out


def _overlap(r1, r2):
    return max(r1[0], r2[0]) < min(r1[1], r2[1])


def _segment_inline_titles(band):
    """Split a band at INLINE title rows — a lone text cell sitting directly
    under a prior block's data with no blank-row separator (e.g. a Deal Kit right
    rail stacking "Overall Deal Kit Summary" / "Deal Kit Excluding Expenses" /
    "Gross Margin Deal Kit" back-to-back). Each such row starts a new segment."""
    titles = [
        i for i, r in enumerate(band)
        if [x for x in r if x != ""] and len([x for x in r if x != ""]) == 1
        and not _is_num([x for x in r if x != ""][0]) and len([x for x in r if x != ""][0]) <= 70
    ]
    if len(titles) <= 1:
        return [band]
    segs = []
    if titles[0] > 0:
        segs.append(band[:titles[0]])
    for k, ti in enumerate(titles):
        end = titles[k + 1] if k + 1 < len(titles) else len(band)
        segs.append(band[ti:end])
    return [s for s in segs if s]


def _lone_label(r):
    """The text of a row holding a single short non-numeric cell, else None."""
    nz = [x for x in r if x != ""]
    if len(nz) == 1 and not _is_num(nz[0]) and len(nz[0]) <= 70:
        return nz[0]
    return None


def _rejoin_sections(segs):
    """Put a table's phase/section rows back into the table they divide.

    `_segment_inline_titles` cuts a band at every lone text row, which is
    right for a stack of separate boxes but wrong for one table whose rows
    are grouped under phase rows -- a Gantt's "Planning" / "Install" /
    "Closeout". Cut there, the header row was left alone in its own
    segment, each phase became a headerless table, and its first task row
    was promoted to the header the rest were bound to.

    A segment is rejoined to the headed table above it when it reads as
    more rows of that table: every row has >=2 cells, all of them under
    the header's columns, and at least one row has >=3. A stacked box of
    label/value pairs (2 cells a row) never qualifies, so it still splits.
    """
    if len(segs) < 2:
        return segs
    base = segs[0]
    hi = _best_header_idx(base)
    if hi is None:
        return segs
    hdr = base[hi]
    hcols = {c for c, x in enumerate(hdr) if x != ""}
    if len(hcols) < 3:
        return segs
    out = [list(base)]
    k = 1
    while k < len(segs):
        seg = segs[k]
        if not seg or _lone_label(seg[0]) is None:
            break
        rows = [r for r in seg[1:] if _filled(r)]
        if not rows:
            break
        fits = all(
            _filled(r) >= 2
            and {c for c, x in enumerate(r) if x != ""} <= hcols
            for r in rows
        ) and any(_filled(r) >= 3 for r in rows)
        if not fits:
            break
        out[0].extend(seg)
        k += 1
    return out + segs[k:]


def _header_score(r):
    cells = [x for x in r if x != ""]
    if len(cells) < 2:
        return -1
    nonnum = sum(1 for x in cells if not _is_num(x) and len(x) <= 45)
    return len(cells) + nonnum


_DATE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}/\d{1,2}/\d{2,4})$"
)


def _is_date(s):
    return bool(_DATE.match(s.strip())) if s else False


def _looks_header(r):
    """Does this row name the columns, rather than fill them?

    A date is a VALUE in a data row and a column name only in a timeline's
    week strip, so dates are judged apart from the other cells: a row of
    dates with no number beside them is a week strip; otherwise they are left
    out of the vote. A one-character cell (a Gantt "x" bar mark) names
    nothing either. Without this, a Gantt task row -- "Cable pulls | 120 |
    2025-03-10 | 2025-03-28" -- read as three labels out of four, became the
    table's header, and every task under it was glued to it column by column
    ("Cable pulls: Rack and stack | 120: 40 | ...").
    """
    cells = [x for x in r if x != ""]
    if len(cells) < 2:
        return False
    dates = [x for x in cells if _is_date(x)]
    rest = [x for x in cells if not _is_date(x)]
    labels = sum(1 for x in rest if not _is_num(x) and 2 <= len(x) <= 40)
    if len(dates) >= 3 and not any(_is_num(x) for x in rest):
        return True                         # a week / date strip
    return labels >= max(2, int(0.6 * len(rest)))


_YEAR = re.compile(r"^(19|20)\d{2}$")


def _is_data_row_over_data(body, i, look=3):
    """Is row ``i`` a data row, judged by the column under it?

    A column's NAME is not a number. A candidate holding a plain number
    (not a year, which does head columns) above a column that carries
    numbers too is the first record of the table, not its header --
    "Planning | Kickoff meeting | 4" over "Planning | Site survey | 16".
    """
    row = body[i]
    below = [r for r in body[i + 1:i + 1 + look] if _filled(r)]
    if len(below) < 2:
        return False
    for c, x in enumerate(row):
        if x == "" or not _is_num(x) or _YEAR.match(x.strip()):
            continue
        under = sum(1 for r in below if c < len(r) and _is_num(r[c]))
        if under >= 2:
            return True
    return False


def _best_header_idx(body, scan=5):
    best_i, best_s = None, -1
    for i in range(min(scan, len(body))):
        if _looks_header(body[i]) and not _is_data_row_over_data(body, i):
            s = _header_score(body[i])
            if s > best_s:
                best_i, best_s = i, s
    return best_i


_BOX_LABEL_END = re.compile(r"[:#?]\s*$")


def _is_label_value_box(body):
    """Is this a two-column info box -- a label per row, its value beside it?

    "Customer | OxBlue" over "OPPTY # | 010246" passed the header test (two
    words), so the box was read as a table and its second row bound to the
    first: "Customer: OPPTY # | OxBlue: 010246". A box like that is pairs, not
    columns: every row holds a short text label in the same left column and at
    most one value beside it, and the labels say so -- one ends in ":", "#" or
    "?", or the values under the would-be header are a mix of numbers and
    words (a column of one kind of thing is a real two-column table, e.g.
    "Item | Qty" over quantities).
    """
    rows = [r for r in body if _filled(r)]
    if not rows:
        return False
    if len(rows) == 1:
        # One "Customer: | OxBlue" pair: a label that says it is a label.
        nz = [x for x in rows[0] if x != ""]
        return (len(nz) == 2 and nz[0].rstrip().endswith(":")
                and not _is_num(nz[0]) and len(nz[0]) <= 40)
    cols = set()
    for r in rows:
        cols |= {c for c, x in enumerate(r) if x != ""}
    if len(cols) != 2:
        return False
    left, right = sorted(cols)
    labels = [r[left] if left < len(r) else "" for r in rows]
    if any(not lab or _is_num(lab) or _is_date(lab) or len(lab) > 40 for lab in labels):
        return False
    # A label that ends in ":" anywhere, or in "#" / "?" below the first row
    # (a header's own "Part #" names a column; "OPPTY #" under "Customer"
    # names a field).
    if labels[0].rstrip().endswith(":") or any(_BOX_LABEL_END.search(lab) for lab in labels[1:]):
        return True
    vals = [r[right] for r in rows[1:] if right < len(r) and r[right] != ""]
    nums = sum(1 for v in vals if _is_num(v) or _is_date(v))
    # Values of different kinds under one would-be header are fields, not a
    # column ("Customer | OxBlue" / "Date | 3/1/2025" / "Sites | 12").
    return len(vals) >= 2 and 0 < nums < len(vals)


def _classify_block(block):
    """-> (title, header_idx_into_block, kind). kind in {table, keyval, text}."""
    if sum(_filled(r) for r in block) <= 1:
        return None, None, "text"          # bare title / caption -> carryable
    title = None
    i = 0
    width = max((_filled(r) for r in block), default=0)
    if width >= 2:
        while i < len(block) and _filled(block[i]) == 1 and i < 2:
            t = next(x for x in block[i] if x != "")
            if len(t) > 70:
                break
            title = t if title is None else f"{title} — {t}"
            i += 1
    body = block[i:]
    if not body:
        return title, None, "text"
    if _is_label_value_box(body):
        return title, i, "labelbox"
    hb = _best_header_idx(body)
    if hb is not None:
        return title, i + hb, "table"
    kv = sum(1 for r in body if _filled(r) == 2 and not _is_num([x for x in r if x != ""][0]))
    if kv >= max(2, int(0.6 * len(body))):
        return title, i, "keyval"
    return title, i, "keyval"


def _clean_title(t):
    """Tidy a block title for the section path: collapse whitespace, strip
    trailing punctuation, and trim a long clarifier clause (after ' - ' / ' — ')
    or cap length — so a section reads 'Deal Kit Excluding Expenses' not
    'Deal Kit Excluding Expenses - Materials, Lift, Travel, etc. Removed'."""
    if not t:
        return t
    t = re.sub(r"\s+", " ", str(t)).strip().rstrip(":;,.-").strip()
    for sep in (" - ", " — ", " – "):
        if sep in t and len(t) > 45:
            head = t.split(sep, 1)[0].strip()
            if len(head) >= 6:
                t = head
                break
    return (t[:57].rstrip() + "…") if len(t) > 60 else t


def _synth_keyval_title(pairs):
    """Synthesize a section heading for a TITLED-less key-value box.

    A highlighted box of label:value rows with no caption still "obviously
    belongs together" — so when the document gave no title, recover one from
    the labels' shared leading stem (e.g. "Expected internal cost target, low"
    + "...target, high" -> "Expected internal cost target"). Honest: only
    fires when ≥2 labels share a real multi-character word stem; otherwise
    returns None and the rows stay grouped by section_path + group metadata
    alone, never by invented text.
    """
    # Index rather than unpack: pairs carry a third element (the worksheet row
    # each came from) inside sheet_blocks, and unpacking two silently raised
    # here -- swallowed upstream, so a sheet produced ZERO block atoms and
    # simply looked empty.
    labels = [str(p[0]).strip() for p in pairs if str(p[0]).strip()]
    if len(labels) < 2:
        return None
    toks = [re.split(r"\s+", l) for l in labels]
    common = []
    for i in range(min(len(t) for t in toks)):
        w = toks[0][i]
        if all(len(t) > i and t[i].lower() == w.lower() for t in toks):
            common.append(w)
        else:
            break
    stem = " ".join(common).strip(" ,:;-")
    # Need a substantive stem (a real word, not just "the"/"a") shared by the box.
    return stem if len(stem) >= 4 else None


def _lum(rgb):
    """Perceived luminance (0-255) of an ARGB/RGB hex fill, 255 if unparseable."""
    try:
        s = str(rgb)[-6:]
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return 0.299 * r + 0.587 * g + 0.114 * b
    except Exception:
        return 255.0


def _is_dark_fill(rgb):
    """A dark fill = a banner header bar (white text on a deep color), the
    universal styling for a section title. Pastel data-box fills are light."""
    return rgb is not None and _lum(rgb) < 140


def _row_first(grid_row, style_row):
    """First non-empty cell of a row as (text, fill, bold)."""
    for c, val in enumerate(grid_row):
        if val != "":
            if style_row and c < len(style_row):
                return val, style_row[c][0], style_row[c][1]
            return val, None, False
    return None, None, False


def _fill_runs(pairs, label_fill):
    """Split a key-value box into contiguous same-fill runs, so two differently
    highlighted boxes stacked with NO blank row between them separate into their
    own groups. Guarded against zebra striping (alternating row colors inside ONE
    box): if colors change on more than half the row boundaries it is decorative
    striping, not grouping, so the box stays whole."""
    if not label_fill or len(pairs) < 2:
        return [(label_fill.get(pairs[0][0]) if (label_fill and pairs) else None, pairs)]
    fills = [label_fill.get(p[0]) for p in pairs]
    changes = sum(1 for i in range(1, len(fills)) if fills[i] != fills[i - 1])
    if changes > len(pairs) / 2:            # zebra / decorative — do not split
        return [(fills[0], pairs)]
    runs, cur, cur_fill = [], [], object()
    for pair, f in zip(pairs, fills):
        if cur and f != cur_fill:
            runs.append((cur_fill, cur)); cur = []
        cur_fill = f; cur.append(pair)
    if cur:
        runs.append((cur_fill, cur))
    return runs


def _style_index(grid, styles):
    """From the cell-style grid, derive (banner_titles, label_fill):

    * banner_titles — texts of lone-text rows wearing a *header* style (a dark
      banner fill, or the same fill the sheet's table-header rows use). These
      are section headers even when they sit flush against the body they title
      (e.g. "Customer Facing Quote Language" directly above its paragraph).
    * label_fill — first-column text -> its fill, for same-fill box grouping.
    """
    banner_titles, label_fill = set(), {}
    if not styles:
        return banner_titles, label_fill
    header_fills = set()
    for r, gr in enumerate(grid):
        if _looks_header(gr):
            sr = styles[r] if r < len(styles) else None
            if sr:
                for c, val in enumerate(gr):
                    if val != "" and c < len(sr) and sr[c][0]:
                        header_fills.add(sr[c][0])
    for r, gr in enumerate(grid):
        sr = styles[r] if r < len(styles) else None
        text, fill, bold = _row_first(gr, sr)
        if text is None:
            continue
        label_fill.setdefault(text, fill)
        nz = [x for x in gr if x != ""]
        if (len(nz) == 1 and len(text) <= 70 and not _is_num(text)
                and (_is_dark_fill(fill) or (fill is not None and fill in header_fills))):
            banner_titles.add(text)
    return banner_titles, label_fill



def _split_at_reheaders(body):
    """Cut a headed table where a NEW header row starts another table.

    A rate sheet stacks a service-rate block (``Service | Sell | Cost``) flush
    under its per-country matrix (``Country | Request | ...``) with no blank
    row between. Read as one table, the service header and its rows were bound
    to the matrix's columns -- "Country: PC | Request: 50". A row is a new
    header when it is all labels (no numbers, no dates), it is not shaped like
    a row of the current table (it fills fewer than half its named columns),
    and the next row's cells all sit under its labels and carry a number.
    """
    if len(body) < 3:
        return [body]
    parts, cur = [], [body[0]]
    hdr = body[0]
    for k in range(1, len(body)):
        r = body[k]
        if any(_filled(x) for x in cur[1:]) and _is_reheader(cur[1:], r, body[k + 1:k + 4]):
            parts.append(cur)
            cur, hdr = [r], r
            continue
        cur.append(r)
    parts.append(cur)
    return parts


def _is_reheader(data, r, after):
    cells = [x for x in r if x != ""]
    if len(cells) < 2 or len(set(cells)) < 2:
        return False
    if any(_is_num(x) or _is_date(x) or len(x) > 40 for x in cells):
        return False
    cols = {c for c, x in enumerate(r) if x != ""}
    # A word where this table keeps numbers: the row names that column
    # rather than filling it ("Sell" over a column of rates).
    def _numeric_col(c):
        vals = [x[c] for x in data if c < len(x) and x[c] != ""]
        return len(vals) >= 2 and sum(1 for v in vals if _is_num(v)) * 2 > len(vals)
    if not any(_numeric_col(c) for c in cols):
        return False
    nxt = next((x for x in after if _filled(x)), None)
    if nxt is None:
        return False
    nf = {c for c, x in enumerate(nxt) if x != ""}
    return nf <= cols and any(_is_num(x) for x in nxt if x != "")


def _emit_table(out, title, body):
    """Append one table block (header row + its data rows) to ``out``."""
    header = [h if h != "" else f"col{j+1}" for j, h in enumerate(body[0])]
    filled = [r for r in body[1:] if _filled(r)]
    # A lone label row inside a table of >=3 named columns, with a real
    # row (>=2 cells) under it, is a phase / section heading for the
    # rows beneath it (a Gantt's "Install" over its tasks) -- context
    # for those rows, not a row of its own. A lone row with nothing
    # under it stays a row, so nothing is dropped.
    wide = sum(1 for h in body[0] if h != "") >= 3
    data, sections, section = [], [], None
    for i, r in enumerate(filled):
        lab = _lone_label(r) if wide else None
        nxt = filled[i + 1] if i + 1 < len(filled) else None
        if lab is not None and nxt is not None and _filled(nxt) >= 2:
            section = lab
            continue
        data.append(r)
        sections.append(section)
    if data:
        out.append({
            "title": title, "kind": "table", "header": header, "rows": data,
            # 1-based worksheet row for each data row, so an atom can cite
            # where it actually came from and source replay can find it.
            "row_indices": [getattr(r, "sheet_row", None) for r in data],
            # The phase/section heading each row sits under (None if none).
            "row_sections": sections,
        })


def sheet_blocks(rows, styles=None):
    """Detect blocks in a sheet's rows. Returns a list of block dicts in reading
    order, with titles carried onto the table/keyval block they head.

    ``styles`` (optional) is a per-cell ``(fill_rgb, bold)`` grid aligned to
    ``rows``; when present, cell styling is used to title style-banner section
    headers and to keep same-fill highlighted boxes grouped. Absent it, the
    detector falls back to pure geometry (blank-row / blank-column banding)."""
    grid = _grid(rows)
    banner_titles, label_fill = _style_index(grid, styles)
    items = []                              # [band_idx, a, b, block, title, hidx, kind]
    gi = 0
    for band in _band_split(grid):
        for (a, b, colblock) in _col_split(band):
            # Re-band-split WITHIN the column: a single column often stacks
            # several tables separated by blank rows that aren't blank across the
            # whole sheet (e.g. a Deal Kit right rail = "Overall Deal Kit
            # Summary" + "Deal Kit Excluding Expenses" + "Gross Margin Deal Kit").
            # Without this they collapse into one block and the lower ones vanish.
            for band2 in _band_split(colblock):
                for subband in _rejoin_sections(_segment_inline_titles(band2)):
                    title, hidx, kind = _classify_block(subband)
                    items.append([gi, a, b, subband, title, hidx, kind])
                    gi += 1

    # TITLE-CARRY: a bare-title text block attaches to the next column-overlapping block.
    for k, it in enumerate(items):
        if it[6] != "text":
            continue
        ttl = it[3][0][0] if it[3] and it[3][0] else (it[4] or "")
        ttl = ttl.strip() if isinstance(ttl, str) else ""
        if not ttl or len(ttl) > 70:
            continue
        for j in range(k + 1, len(items)):
            if items[j][0] > it[0] and items[j][6] != "text" and _overlap((it[1], it[2]), (items[j][1], items[j][2])):
                items[j][4] = ttl if not items[j][4] else f"{ttl} — {items[j][4]}"
                it[6] = "_consumed"
                break

    out = []
    for (bi, a, b, block, title, hidx, kind) in items:
        if kind == "_consumed":
            continue
        title = _clean_title(title)
        if kind == "table":
            for sub in _split_at_reheaders(block[hidx:]):
                _emit_table(out, title, sub)
        elif kind in ("keyval", "labelbox"):
            body = block if hidx is None else block[hidx:]
            # Same reason as the table branch: an atom's `row` must be a
            # worksheet row, and pairs alone lose which row each came from.
            pairs = []
            for r in body:
                nz = [x for x in r if x != ""]
                if len(nz) >= 2:
                    pairs.append((nz[0], " ".join(nz[1:]), getattr(r, "sheet_row", None)))
                elif len(nz) == 1:
                    pairs.append((nz[0], "", getattr(r, "sheet_row", None)))
            if not pairs:
                continue
            # A styled banner header sitting INSIDE the block (a lone dark-fill
            # row, e.g. "Customer Facing Quote Language" flush above its
            # paragraph) is a section title, not a key-value label — split there
            # so the rows under it group beneath that heading.
            groups = []                      # (group_title, group_pairs)
            cur_title, cur = title, []
            for pair in pairs:
                kk, vv = pair[0], pair[1]
                if banner_titles and vv == "" and kk in banner_titles:
                    if cur:
                        groups.append((cur_title, cur))
                    cur_title, cur = kk, []
                else:
                    cur.append(pair)
            if cur:
                groups.append((cur_title, cur))
            for gt, gp in groups:
                # Same highlight color = one box: split a group into contiguous
                # same-fill runs so stacked, differently-colored boxes separate.
                for run_fill, run_pairs in _fill_runs(gp, label_fill):
                    rt = gt
                    if not rt:
                        # No caption anywhere — recover a heading from the
                        # labels' shared stem so the box still reads as a group.
                        rt = _synth_keyval_title(run_pairs)
                    out.append({
                        "title": _clean_title(rt), "kind": "keyval",
                        # 2-tuples for every existing consumer; the worksheet row
                        # each pair came from travels alongside.
                        "pairs": [(p[0], p[1]) for p in run_pairs],
                        "pair_rows": [p[2] if len(p) > 2 else None for p in run_pairs],
                        "fill": run_fill,
                        # A two-column label/value box recognised as such
                        # (not a headerless table that fell through).
                        "label_box": kind == "labelbox",
                    })
        else:
            txt = " ".join(x for r in block for x in r if x != "")
            if txt.strip():
                out.append({"title": title, "kind": "text", "text": txt})
    return out
