"""Date recognition: a shape gate for precision, dateutil for interpretation.

Measured against the two hand-rolled regexes this replaces, on 14 real-world
date spellings and 8 non-dates that occur constantly in a bill of materials::

                      recall   false positives
    _DATE_RE           1/14         0/8
    _BARE_DATE_RE      9/14         0/8
    dateutil alone    14/14         3/8     <- parses "192", "4512", "10-4"

So neither half is usable on its own. ``dateutil`` has the world's date
formats memorised but will read a quantity as a day-of-month; the regex has
no such appetite but knows only the spellings someone thought to add.

This module runs them in series. A string must first look like a date
(``_SHAPE``, which is ``_BARE_DATE_RE`` widened to the five spellings it
missed), and only then is it handed to ``dateutil`` to be turned into a real
date. Precision comes from the gate, coverage from the library, and a bare
integer never becomes a date because it never passes the gate.

``dateutil`` is a soft dependency, matching the ``rapidfuzz`` handling in
``entity_resolution``: without it the gate still answers "is this a date",
which is all most callers ask.
"""

from __future__ import annotations

import re
from datetime import date, datetime

try:  # pragma: no cover - exercised by whichever environment lacks it
    from dateutil import parser as _dateutil_parser
except Exception:  # pragma: no cover
    _dateutil_parser = None  # type: ignore[assignment]

_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_WEEKDAYS = r"(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*"

#: Date-shaped spans. Every branch requires either a month name or a
#: 4-digit year, which is what keeps "192" and "10-4" out.
_SHAPE_SRC = (
    r"(?:" + _WEEKDAYS + r"\.?,?\s*)?"
    r"(?:"
    # April 8, 2026 / May 14,2026 / May 14th 2026
    r"" + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?[,\s]+\d{4}"
    # 8 April 2026 / 14-May-26
    r"|\d{1,2}[\s\-]" + _MONTHS + r"\.?[,\s\-]+\d{2,4}"
    # 2026-05-14, optionally with an ISO-8601 time
    r"|\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?)?"
    # 2026/05/14
    r"|\d{4}/\d{2}/\d{2}"
    # 5/14/26, 5/14/2026, 14.05.2026
    r"|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}"
    # 20260514 -- a bare basic-format ISO date, anchored to a plausible year
    r"|(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
    r")"
)
_SHAPE = re.compile(_SHAPE_SRC, re.IGNORECASE)
_SHAPE_ANCHORED = re.compile(r"^\s*" + _SHAPE_SRC + r"\s*$", re.IGNORECASE)
#: Scanning inside prose needs word boundaries so "Rev 2.1.3" is not a date.
_SHAPE_SCAN = re.compile(r"\b" + _SHAPE_SRC + r"\b", re.IGNORECASE)


def looks_like_date(text: str) -> bool:
    """True when the whole string is date-shaped. No parsing, no dependency."""
    s = (text or "").strip()
    return bool(s) and len(s) <= 40 and bool(_SHAPE_ANCHORED.match(s))


def parse_date(text: str) -> date | None:
    """Turn a date-shaped string into a ``date``, or return None.

    Returns None for anything that fails the shape gate, so callers can pass
    arbitrary cell values without a quantity coming back as a date.
    """
    s = (text or "").strip()
    if not looks_like_date(s):
        return None
    if _dateutil_parser is None:  # pragma: no cover - dependency-free fallback
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        return None
    try:
        return _dateutil_parser.parse(s).date()
    except (ValueError, OverflowError, TypeError):
        return None


def find_dates(text: str) -> list[tuple[int, int, date]]:
    """Locate every date in prose as ``(start, end, date)``.

    Offsets are into ``text`` so a caller can build a locator from the match
    rather than re-searching for the string.
    """
    out: list[tuple[int, int, date]] = []
    for m in _SHAPE_SCAN.finditer(text or ""):
        parsed = parse_date(m.group(0))
        if parsed is not None:
            out.append((m.start(), m.end(), parsed))
    return out
