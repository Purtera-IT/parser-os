"""Table header rows name fields; they are not facts.

Deal 010353 SOW rate / contact tables: the header cells "Stated Rate",
"Business Hours", "After Hours" and a bare "Chase Smith" name cell each came
out as their own atom. A header row only says what the cells below it are, so
it must never be emitted; its labels are the field names of the rows under it.
A row holding ONE value ("Chase Smith" with the rest of the row blank) is
meaningless without that field name, so it reads "Name: Chase Smith".

Shared by the docx and xlsx table paths. Structural only -- cell shape and
column alignment, never a vocabulary of header words.
"""

from __future__ import annotations

import re
from typing import Sequence

_VALUE_CHARS = re.compile(r"[\d@$%]")


def is_label_cell(text: str) -> bool:
    """A short, value-free cell: a column name, never a value."""
    t = (text or "").strip()
    if not t or len(t) > 30 or _VALUE_CHARS.search(t):
        return False
    return bool(re.search(r"[A-Za-z]", t)) and len(t.split()) <= 4


def _has_value(text: str) -> bool:
    return bool(_VALUE_CHARS.search(text or ""))


def is_header_only_row(rows: Sequence[Sequence[str]], i: int, lookahead: int = 5) -> bool:
    """Row ``i`` holds only column labels for the rows beneath it.

    ``rows`` are positional cell texts (blank = ""). True when every filled
    cell of row ``i`` is a short label (and there are at least two distinct
    labels, or exactly one), and the rows after it -- up to a blank row or
    ``lookahead`` rows -- fill only columns row ``i`` names, at least one of
    them with a value (a number, amount, email). A row of names or other short
    words over data that spills into columns it does not name is data, not a
    header.
    """
    if i < 0 or i >= len(rows):
        return False
    cells = [str(c or "").strip() for c in rows[i]]
    filled = [c for c in cells if c]
    if not filled or not all(is_label_cell(c) for c in filled):
        return False
    named = {k for k, c in enumerate(cells) if c}
    labels = set(filled)
    if len(labels) == 1 and len(named) > 1:
        return False  # one title merged across the row: a caption, not names
    seen_value = False
    seen_rows = 0
    for j in range(i + 1, min(len(rows), i + 1 + lookahead)):
        nxt = [str(c or "").strip() for c in rows[j]]
        nf = {k for k, c in enumerate(nxt) if c}
        if not nf:
            if seen_rows:
                break
            continue
        if not nf <= named:
            return False
        if seen_rows == 0 and [c for c in nxt if c] == filled:
            return False  # the same words again: a repeated row, not a header
        seen_rows += 1
        if any(_has_value(nxt[k]) for k in nf):
            seen_value = True
    return seen_value


def is_banner_row(rows: Sequence[Sequence[str]], i: int) -> bool:
    """Row ``i`` is one label ("Stated Rate", merged or not) heading the
    header row directly beneath it: a group caption, not a fact."""
    if i < 0 or i >= len(rows):
        return False
    filled = {str(c or "").strip() for c in rows[i]} - {""}
    if len(filled) != 1 or not is_label_cell(next(iter(filled))):
        return False
    for j in range(i + 1, len(rows)):
        if any(str(c or "").strip() for c in rows[j]):
            return is_header_only_row(rows, j)
    return False


def is_form_label(text: str) -> bool:
    """A form field's label: "Name:", "Signature:", "Date:"."""
    t = (text or "").strip()
    return t.endswith(":") and is_label_cell(t.rstrip(":").strip())


def is_caption_row(rows: Sequence[Sequence[str]], i: int) -> bool:
    """Row ``i`` is one label ("The Buyer") captioning the form fields beneath
    it ("Signature: | ____", "Name: | Chase Smith"): the block's name, which
    belongs on its fields, not a fact of its own (010003 / 010353 e-sign)."""
    if i < 0 or i >= len(rows):
        return False
    filled = {str(c or "").strip() for c in rows[i]} - {""}
    if len(filled) != 1:
        return False
    label = next(iter(filled))
    if not is_label_cell(label) or label.endswith(":"):
        return False
    for j in range(i + 1, len(rows)):
        nxt = [str(c or "").strip() for c in rows[j] if str(c or "").strip()]
        if nxt:
            return is_form_label(nxt[0])
    return False


def merge_header(previous: Sequence[str], header: Sequence[str]) -> list[str]:
    """The new header row, falling back to ``previous`` where it is blank."""
    width = max(len(previous), len(header))
    out = []
    for k in range(width):
        h = str(header[k]).strip() if k < len(header) and header[k] else ""
        p = str(previous[k]).strip() if k < len(previous) and previous[k] else ""
        out.append(h or p)
    return out


def bind_lone_cell(cells: Sequence[str], header: Sequence[str]) -> str | None:
    """``"Label: value"`` for a row whose only content is one cell.

    ``cells`` are positional (merged cells repeat their text). None when the
    row holds more than one value, or no header names its column, or the
    header is the value itself.
    """
    pos = [(k, str(c or "").strip()) for k, c in enumerate(cells) if str(c or "").strip()]
    if not pos or len({v for _k, v in pos}) != 1:
        return None
    k, value = pos[0]
    if value.endswith(":"):
        return None  # an unfilled form field, not a value
    label = str(header[k]).strip() if k < len(header) and header[k] else ""
    if not label or label == value or re.fullmatch(r"col_?\d+", label, re.I):
        return None
    return f"{label.rstrip(':').strip()}: {value}"


def bind_row(cells: Sequence[str], header: Sequence[str]) -> str:
    """``"Label: value | Label: value"`` -- each filled cell under its column's
    name (a cell with no name, or that IS its name, stays bare)."""
    out: list[str] = []
    for k, c in enumerate(cells):
        v = str(c or "").strip()
        if not v:
            continue
        label = str(header[k]).strip().rstrip(":").strip() if k < len(header) and header[k] else ""
        item = f"{label}: {v}" if label and label != v and not v.endswith(":") else v
        if item not in out:
            out.append(item)
    return " | ".join(out)
