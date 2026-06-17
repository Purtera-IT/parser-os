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


def _grid(rows):
    return [[_clean(c) for c in r] for r in rows]


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
    out = []
    for (a, b) in groups:
        sub = [[(r[c] if c < len(r) else "") for c in range(a, b)] for r in band]
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


def _header_score(r):
    cells = [x for x in r if x != ""]
    if len(cells) < 2:
        return -1
    nonnum = sum(1 for x in cells if not _is_num(x) and len(x) <= 45)
    return len(cells) + nonnum


def _looks_header(r):
    cells = [x for x in r if x != ""]
    if len(cells) < 2:
        return False
    nonnum = sum(1 for x in cells if not _is_num(x) and len(x) <= 40)
    return nonnum >= max(2, int(0.6 * len(cells)))


def _best_header_idx(body, scan=5):
    best_i, best_s = None, -1
    for i in range(min(scan, len(body))):
        if _looks_header(body[i]):
            s = _header_score(body[i])
            if s > best_s:
                best_i, best_s = i, s
    return best_i


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
    labels = [str(k).strip() for k, _ in pairs if str(k).strip()]
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
    fills = [label_fill.get(k) for k, _ in pairs]
    changes = sum(1 for i in range(1, len(fills)) if fills[i] != fills[i - 1])
    if changes > len(pairs) / 2:            # zebra / decorative — do not split
        return [(fills[0], pairs)]
    runs, cur, cur_fill = [], [], object()
    for (k, v), f in zip(pairs, fills):
        if cur and f != cur_fill:
            runs.append((cur_fill, cur)); cur = []
        cur_fill = f; cur.append((k, v))
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
                for subband in _segment_inline_titles(band2):
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
            body = block[hidx:]
            header = [h if h != "" else f"col{j+1}" for j, h in enumerate(body[0])]
            data = [r for r in body[1:] if _filled(r)]
            if data:
                out.append({"title": title, "kind": "table", "header": header, "rows": data})
        elif kind == "keyval":
            body = block if hidx is None else block[hidx:]
            pairs = []
            for r in body:
                nz = [x for x in r if x != ""]
                if len(nz) >= 2:
                    pairs.append((nz[0], " ".join(nz[1:])))
                elif len(nz) == 1:
                    pairs.append((nz[0], ""))
            if not pairs:
                continue
            # A styled banner header sitting INSIDE the block (a lone dark-fill
            # row, e.g. "Customer Facing Quote Language" flush above its
            # paragraph) is a section title, not a key-value label — split there
            # so the rows under it group beneath that heading.
            groups = []                      # (group_title, group_pairs)
            cur_title, cur = title, []
            for (kk, vv) in pairs:
                if banner_titles and vv == "" and kk in banner_titles:
                    if cur:
                        groups.append((cur_title, cur))
                    cur_title, cur = kk, []
                else:
                    cur.append((kk, vv))
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
                    out.append({"title": _clean_title(rt), "kind": "keyval",
                                "pairs": run_pairs, "fill": run_fill})
        else:
            txt = " ".join(x for r in block for x in r if x != "")
            if txt.strip():
                out.append({"title": title, "kind": "text", "text": txt})
    return out
