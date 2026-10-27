"""One standard form for a table row before the model reads it (C3_CANON_ROWS=1).

Training text puts a row's cells on one line, "key: value | key: value | value". The same row can reach us laid out
another way: tab-separated, one "key: value" per line, a markdown table, or a "key = value; ..." list. Each becomes
the training form here, so every layout of a row reads the same. Anything that is not a row (fewer than three cells,
or fewer than two "key: value" cells) is returned unchanged, and so is a row already in the training form, so turning this on
leaves every training line as it was.
"""
from __future__ import annotations

import os
import re

_KV = re.compile(r"^\s*([A-Za-z][\w ./()#&-]{0,40}?)\s*(?::|=)\s+(.*\S)\s*$")
_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def enabled() -> bool:
    return os.environ.get("C3_CANON_ROWS", "0") == "1"


def _cells(parts: list[str]) -> list[str]:
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = _KV.match(p)
        out.append(f"{m.group(1).strip()}: {m.group(2).strip()}" if m else p)
    return out


def _pipe_cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _short(cell: str) -> bool:
    """A table cell, not a sentence: at most 6 words and 60 characters, not ending like a sentence."""
    c = cell.strip()
    return bool(c) and len(c) <= 60 and len(c.split()) <= 6 and not c.endswith((".", "?", "!"))


def _cells_like(cells: list[str]) -> bool:
    """Bare values of one record: none reads as a sentence, at most 100 characters each, most of them short."""
    cs = [c.strip() for c in cells if c.strip()]
    return (len(cs) >= 3 and all(len(c) <= 100 and not (len(c.split()) > 6 and c.endswith((".", "?", "!")))
                                 for c in cs)
            and sum(_short(c) for c in cs) * 2 >= len(cs))


def canonical_row(text: str) -> str:
    t = str(text or "")
    s = t.strip()
    if not s:
        return t
    lines = [ln for ln in s.split("\n") if ln.strip()]
    # A markdown table: header, rule, one or more data rows.
    if len(lines) >= 3 and _RULE.match(lines[1]) and lines[0].lstrip().startswith("|"):
        head = _pipe_cells(lines[0])
        rows = []
        for ln in lines[2:]:
            vals = _pipe_cells(ln)
            if len(vals) == len(head):
                rows.append(" | ".join(f"{h}: {v}" if h and v else v for h, v in zip(head, vals) if v))
            else:
                rows.append(" | ".join(v for v in vals if v))
        return "\n".join(r for r in rows if r) or t
    # One markdown row with outer pipes and no rule (a cell may hold line breaks).
    if s.startswith("|") and s.endswith("|") and s.count("|") >= 3 and not any(_RULE.match(ln) for ln in lines):
        return " | ".join(c for c in _cells(_pipe_cells(s)))
    # One "key: value" (or "key = value") per line.
    if len(lines) >= 2 and sum(bool(_KV.match(ln)) for ln in lines) >= 2 and not any("|" in ln for ln in lines) \
            and all(len(ln) <= 160 for ln in lines):
        return " | ".join(_cells(lines))
    # Bare values, one per line (a row with no keys): short cells, no sentences.
    if len(lines) >= 3 and not any("|" in ln for ln in lines) and _cells_like(lines):
        return " | ".join(_cells(lines))
    if len(lines) == 1:
        # Tabs.
        if "\t" in s and "|" not in s:
            cells = _cells(s.split("\t"))
            if len(cells) >= 3:
                return " | ".join(cells)
    # A "key = value; key = value" list (a cell may hold line breaks).
    if ";" in s and "|" not in s and "\t" not in s:
        parts = [p for p in s.split(";") if p.strip()]
        if len(parts) >= 3 and (sum(bool(_KV.match(p)) for p in parts) >= 2 or _cells_like(parts)):
            return " | ".join(_cells(parts))
    return t


__all__ = ["canonical_row", "enabled"]
