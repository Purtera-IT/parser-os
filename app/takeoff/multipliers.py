"""Floor / level multiplier inference from sheet titles.

A single "LEVEL 5-12 AND LEVEL 15 FLOOR PLAN" sheet represents the same
typical-floor wireless-node layout across 9 floors. Detecting how many
floors a sheet stands in for is what turns a 12-device sheet base
count into a 108-device extended count.

Inputs are plain sheet titles. Outputs are ``(levels_represented,
multiplier)`` tuples where the multiplier is ``len(levels_represented)``.

The implementation is deliberately list-of-regexes rather than a single
clever regex — drawing-set titles vary enough that explicit cases beat
a regex that handles every case poorly.
"""
from __future__ import annotations

import re

# ─── Range matchers ───
#
# We accept "LEVEL", "LEVELS", "FLOOR", and "FLOORS" interchangeably —
# different firms / disciplines use different terms. The separator can
# be a dash, en-dash, "TO", "THROUGH", or a slash ("LEVELS 5/8/12").

_FLOOR_TOKEN = r"(?:LEVELS?|FLOORS?)"

# "LEVEL 5-12", "FLOORS 17-18", "LEVELS 5 THROUGH 12", "LEVEL 5 TO 12"
_LEVEL_RANGE_RE = re.compile(
    rf"{_FLOOR_TOKEN}\s*(\d+)\s*(?:-|–|TO|THROUGH)\s*(\d+)",
    re.IGNORECASE,
)
# "AND LEVEL 15", "AND LEVELS 14 & 15", "AND FLOORS 8 / 12"
_AND_LEVEL_RE = re.compile(
    rf"AND\s+{_FLOOR_TOKEN}\s*((?:\d+(?:\s*(?:,|&|AND|/|\\)\s*)?)+)",
    re.IGNORECASE,
)
# Single-level "LEVEL 24", "FLOOR 24"
_SINGLE_LEVEL_RE = re.compile(rf"{_FLOOR_TOKEN}\s+(\d+)\b", re.IGNORECASE)
# Slash-separated list "LEVELS 5/8/12"
_SLASH_LIST_RE = re.compile(
    rf"{_FLOOR_TOKEN}\s+(\d+(?:\s*/\s*\d+){{1,}})",
    re.IGNORECASE,
)

# ─── Named-floor matchers ───
#
# Common named-floor conventions across hospitality / commercial /
# residential projects. Order matters only when one phrase is a
# substring of another (LOWER LOBBY before LOBBY).
_NAMED_FLOOR_MAP: tuple[tuple[re.Pattern[str], list[str]], ...] = (
    # Hospitality (Marriott, Hilton, Hyatt, ...).
    (re.compile(r"LOWER\s+LOBBY", re.IGNORECASE), ["Lower Lobby"]),
    (re.compile(r"LOBBY\s+LEVEL", re.IGNORECASE), ["Lobby"]),
    (re.compile(r"\bROOF\s+PLAN\b", re.IGNORECASE), ["Roof"]),
    (re.compile(r"SERVICE\s+LEVEL", re.IGNORECASE), ["Service"]),
    # Generic / residential / commercial.
    (re.compile(r"\bMEZZANINE\b", re.IGNORECASE), ["Mezzanine"]),
    (re.compile(r"\bPENTHOUSE\b", re.IGNORECASE), ["Penthouse"]),
    (re.compile(r"\bBASEMENT\b", re.IGNORECASE), ["Basement"]),
    (re.compile(r"\bGROUND\s+(?:FLOOR|LEVEL)\b", re.IGNORECASE), ["Ground"]),
    (re.compile(r"\bMAIN\s+(?:FLOOR|LEVEL)\b", re.IGNORECASE), ["Main"]),
    (re.compile(r"\bGARAGE\b", re.IGNORECASE), ["Garage"]),
    (re.compile(r"\bATTIC\b", re.IGNORECASE), ["Attic"]),
)


def _parse_int_list(blob: str) -> list[int]:
    """Pull integers out of a comma / & / 'and' separated list."""
    nums = re.findall(r"\d+", blob or "")
    return [int(n) for n in nums]


def levels_from_title(title: str) -> list[str]:
    """Return a list of level labels represented by the sheet title.

    Ordering follows the natural reading order — for "LEVEL 5-12 AND
    LEVEL 15" that's ``["5","6","7","8","9","10","11","12","15"]``.

    Returns an empty list when no level cue can be parsed.
    """
    if not title:
        return []
    t = title.strip()

    # Named floors first (Lower Lobby / Lobby / Roof / Service) — these
    # are mutually exclusive with numbered ranges in practice.
    for pat, labels in _NAMED_FLOOR_MAP:
        if pat.search(t):
            return list(labels)

    levels: list[str] = []
    seen: set[str] = set()

    def _push(level: int) -> None:
        label = str(level)
        if label not in seen:
            seen.add(label)
            levels.append(label)

    # Numeric range — may appear multiple times in title.
    for m in _LEVEL_RANGE_RE.finditer(t):
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        for n in range(lo, hi + 1):
            _push(n)

    # "AND LEVEL[S] N (& N ...)" extras after a range.
    for m in _AND_LEVEL_RE.finditer(t):
        for n in _parse_int_list(m.group(1)):
            _push(n)

    # Slash-separated list — "LEVELS 5/8/12".
    for m in _SLASH_LIST_RE.finditer(t):
        for n in _parse_int_list(m.group(1)):
            _push(n)

    if not levels:
        # Fall back to all single-level mentions (e.g. "LEVEL 24").
        for m in _SINGLE_LEVEL_RE.finditer(t):
            _push(int(m.group(1)))

    return levels


def multiplier_for_title(title: str) -> tuple[list[str], int]:
    """Return ``(levels_represented, multiplier)`` for a sheet title.

    When no level cue can be parsed the multiplier defaults to 1 with an
    empty level list — the caller may interpret that as "single floor,
    name not yet inferred".
    """
    levels = levels_from_title(title or "")
    return (levels, max(1, len(levels)))


def floor_label_for_title(title: str) -> str | None:
    """Best-effort short floor label for a sheet title.

    Used as the ``floor_label`` field on devices for human-readable
    rollups. For multi-floor sheets we return the title's natural label
    (e.g. ``"Lower Lobby"``, ``"Level 5-12 and 15"``).
    """
    if not title:
        return None
    t = title.strip()
    for pat, labels in _NAMED_FLOOR_MAP:
        if pat.search(t):
            return labels[0]
    # Reduce "... FLOOR PLAN" / "... PLAN" suffix for readability.
    short = re.sub(r"\s+(FLOOR\s+)?PLAN\b.*$", "", t, flags=re.IGNORECASE).strip()
    return short or None


__all__ = [
    "levels_from_title",
    "multiplier_for_title",
    "floor_label_for_title",
]
