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


def sheet_blocks(rows):
    """Detect blocks in a sheet's rows. Returns a list of block dicts in reading
    order, with titles carried onto the table/keyval block they head."""
    grid = _grid(rows)
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
            if pairs:
                out.append({"title": title, "kind": "keyval", "pairs": pairs})
        else:
            txt = " ".join(x for r in block for x in r if x != "")
            if txt.strip():
                out.append({"title": title, "kind": "text", "text": txt})
    return out
