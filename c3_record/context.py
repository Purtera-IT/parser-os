"""Where a line sits, as extraction facts (C3_CONTEXT_TAGS): its table's
header cells, its sheet, its section heading.

Two halves, one rule on both sides:

* ``table_fields`` / ``table_headers`` stamp ``table_id``, ``table_row_index``,
  ``sheet`` and ``table_header`` on rows, by the rule run E's training build
  (``runE/build.py`` step (d)) wrote them into the training views: the table
  is the line's table ref ("<table> r<N>"), a spreadsheet's table is its
  sheet, and a row's header is its own ``Key: value`` cell keys, else the
  table's top row when that row is bare, non-numeric text of two or more
  cells. Serving calls these on the compile's rows; training rows already
  carry the fields.
* ``context_tags`` turns one row's fields into hashed tags (``hdr:``,
  ``sheet:``, ``sec:``). ``ml.c3.data.featurize`` calls it for training and
  serving alike, so a row with the same fields gets the same tags.

Quoted-email depth is not here: the training views carry no quoted field
(``locator.quoted`` / ``quoted_old_email`` never reached export), and the
sheet's classified role (``sheet_role``) is not in them either, so serving
must not send either until the views are rebuilt with them.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

#: "<table id> r<row>" (c3_record._TABLE_REF: a float row "r3.0" is row 3).
_TABLE_REF = re.compile(r"^(.*?)\s+r(\d+)(?:\.0+)?$")
_SHEET_FILES = (".xlsx", ".xls", ".xlsm", ".csv")
#: build.py's NUMERIC: a cell that is only a number, money, date or code.
_NUMERIC = re.compile(r"^[\s$€£(),.%#:+\-\d/x]*\d[\s$€£(),.%#:+\-\d/x]*$")
_KEYED = re.compile(r"^([A-Za-z][\w ./()#&-]{0,40}):\s+(.*)$")

#: Caps: header cells per line, words per header cell / sheet name, heading words.
MAX_HEADER_CELLS = 8
MAX_NAME_WORDS = 3
MAX_HEADING_WORDS = 4
_STOP = frozenset({"the", "and", "of", "to", "for", "in", "on", "an", "at", "by", "or", "with"})
#: Section values that name a speaker, not a heading: a meeting line's section is its speaker.
_SPEAKER_KINDS = frozenset({"meeting"})


def cells_of(text: str) -> list[tuple[str | None, str]]:
    """(key or None, value) per cell; tabs or pipes (build.py ``cells_of``)."""
    t = str(text or "").strip()
    parts = [p.strip() for p in (t.split("\t") if "\t" in t and "|" not in t else t.split("|"))]
    out: list[tuple[str | None, str]] = []
    for p in parts:
        if not p:
            continue
        m = _KEYED.match(p)
        out.append((m.group(1).strip(), m.group(2).strip()) if m else (None, p))
    return out


def table_fields(doc_id: str, table_ref: Any, filename: Any = "") -> dict[str, Any]:
    """``table_id`` / ``table_row_index`` (and ``sheet`` for a spreadsheet)
    for a table-row line, else {}: build.py's (d), from the same table ref."""
    m = _TABLE_REF.match(str(table_ref or "").strip())
    if not m:
        return {}
    out: dict[str, Any] = {"table_id": f"{doc_id}:{m.group(1)}", "table_row_index": int(m.group(2))}
    if str(filename or "").lower().endswith(_SHEET_FILES):
        out["sheet"] = m.group(1)
    return out


def table_headers(rows: list[dict[str, Any]]) -> None:
    """Stamp ``table_header`` on table rows in place (build.py's (d)): a row
    whose cells are all ``Key: value`` takes its keys; any other row of a
    table takes the table's top row when that row is bare text cells (two or
    more, none keyed, none numeric). Only a row index's first line counts,
    as in the build."""
    tables: dict[str, dict[int, int]] = defaultdict(dict)
    for i, r in enumerate(rows):
        if r.get("table_id") and r.get("table_row_index") is not None:
            tables[str(r["table_id"])].setdefault(int(r["table_row_index"]), i)
    for idx in tables.values():
        top = idx[min(idx)]
        head = cells_of(rows[top].get("text") or "")
        bare = ([v for _, v in head] if len(head) >= 2 and not any(k for k, _ in head)
                and not any(_NUMERIC.match(v) for _, v in head) else None)
        for i in idx.values():
            cells = cells_of(rows[i].get("text") or "")
            if cells and all(k for k, _ in cells):
                rows[i]["table_header"] = [k for k, _ in cells]
            elif bare and i != top:
                rows[i]["table_header"] = bare


def _words(text: Any) -> list[str]:
    """Lower-case letter words of two or more letters, digits dropped
    ("12E Closet" -> closet, "Sheet1" -> sheet), stop words out."""
    out = []
    for w in re.findall(r"[a-z]+", str(text or "").lower()):
        if len(w) >= 2 and w not in _STOP:
            out.append(w)
    return out


def _name(text: Any) -> str:
    return "_".join(_words(text)[:MAX_NAME_WORDS])[:32]


def quoted_depth(quoted: Any = False, message_index: Any = None, authority_class: Any = "") -> int:
    """How deep in quoted reply history a line sits: 0 for the message's own
    text; else its message's place in the thread (``message_index``, newest
    message 0), at least 1. A line is quoted when the parser marked it so
    (``locator.quoted`` / ``value.quoted``) or classed it ``quoted_old_email``.
    An extraction fact, the same rule for a training row and a served one."""
    auth = str(getattr(authority_class, "value", authority_class) or "").rsplit(".", 1)[-1]
    if not (quoted is True or str(quoted).strip().lower() in ("true", "1") or auth == "quoted_old_email"):
        return 0
    try:
        idx = int(message_index)
    except (TypeError, ValueError):
        idx = 0
    return max(1, idx)


def quoted_tags(depth: Any) -> list[str]:
    """``quoted:1`` / ``quoted:2+`` for a line in quoted history, else none."""
    try:
        d = int(depth or 0)
    except (TypeError, ValueError):
        d = 0
    return [] if d <= 0 else ["quoted:1"] if d == 1 else ["quoted:2+"]


def context_tags(section: Any = "", sheet: Any = "", table_header: Iterable[Any] = (),
                 doc_kind: Any = "", quoted_depth: Any = 0) -> list[str]:
    """The line's context tags, in a fixed order, no repeats:

    * ``hdr:<cell>`` per header cell (first MAX_HEADER_CELLS cells, words joined by _);
    * ``sheet:<name>`` the sheet's name words;
    * ``sec:<word>`` per word of the nearest heading (the section path's last
      part), capped; not for a meeting line (its section is the speaker) or a
      section that is the sheet itself;
    * ``quoted:1`` / ``quoted:2+`` for a line in quoted reply history
      (``quoted_tags``)."""
    out: list[str] = []
    for h in list(table_header or ())[:MAX_HEADER_CELLS]:
        n = _name(h)
        if n:
            out.append(f"hdr:{n}")
    sh = _name(sheet)
    if sh:
        out.append(f"sheet:{sh}")
    sec = str(section or "").strip()
    last = sec.split(" > ")[-1]
    if sec and str(doc_kind or "").strip().lower() not in _SPEAKER_KINDS and not (sh and _name(last) == sh):
        for w in _words(last)[:MAX_HEADING_WORDS]:
            out.append(f"sec:{w}")
    out.extend(quoted_tags(quoted_depth))
    seen: set[str] = set()
    return [t for t in out if not (t in seen or seen.add(t))]


#: Row keys these facts travel in (training views and the serving payload).
CONTEXT_KEYS = ("sheet", "table_header", "quoted_depth")

__all__ = ["cells_of", "table_fields", "table_headers", "context_tags", "quoted_depth", "quoted_tags",
           "CONTEXT_KEYS",
           "MAX_HEADER_CELLS", "MAX_HEADING_WORDS"]
