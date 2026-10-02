"""Atoms whose text the parser composed, and the source lines behind them.

A message header atom reads "From: Jane <j@x.com> | To: ... | Subject: ...",
a HubSpot note's reads "note_id=123 | author=AJ Evans | date=...". Neither
string exists in the file, so a reviewer's UI cannot highlight it and a label
on it points nowhere.

Their text is NOT rewritten to a source line: the composed string is what
downstream readers key on (header/threading heuristics, dedup, and the atom id
itself, so every label already given to these atoms would be orphaned). They
are marked instead, so they stay labelable and the UI knows what to do:

* ``value["synthetic_text"] = True`` and review flag ``synthetic_text`` --
  the UI skips highlighting ``raw_text``;
* ``value["source_lines"]`` -- the verbatim lines the atom was built from
  (``{"line": 1-based line number, "text": the line as written}``), which the
  UI highlights instead.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

SYNTHETIC_FLAG = "synthetic_text"


def find_source_lines(
    lines: Iterable[str], labels: Iterable[str], *, stop_at_blank: bool = False,
    start: int = 0, limit: int | None = None,
) -> list[dict[str, Any]]:
    """The first line starting with each ``label`` ("From", "HubSpot Note ID"),
    verbatim, in file order. ``stop_at_blank`` confines the search to a header
    block (an .eml's headers end at the first blank line)."""
    wanted = [l.lower() for l in labels]
    found: dict[str, dict[str, Any]] = {}
    seq = list(lines)
    end = len(seq) if limit is None else min(len(seq), start + limit)
    for i in range(start, end):
        raw = seq[i].rstrip("\r\n")
        if stop_at_blank and not raw.strip():
            break
        s = raw.strip()
        for lab in wanted:
            if lab in found:
                continue
            if re.match(rf"^{re.escape(lab)}\s*:", s, re.I):
                found[lab] = {"line": i + 1, "text": s}
                break
    return sorted(found.values(), key=lambda d: d["line"])


def mark_synthetic(atom: Any, source_lines: list[dict[str, Any]] | None = None) -> Any:
    """Flag a parser-composed atom and attach the source lines behind it."""
    val = dict(atom.value) if isinstance(getattr(atom, "value", None), dict) else {}
    val["synthetic_text"] = True
    val["source_lines"] = list(source_lines or [])
    atom.value = val
    flags = list(getattr(atom, "review_flags", None) or [])
    if SYNTHETIC_FLAG not in flags:
        flags.append(SYNTHETIC_FLAG)
    atom.review_flags = flags
    return atom


__all__ = ["SYNTHETIC_FLAG", "find_source_lines", "mark_synthetic"]
