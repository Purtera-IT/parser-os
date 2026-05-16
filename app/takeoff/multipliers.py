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
# "LEVEL 5-12 AND LEVEL 15", "LEVELS 17-18", "LEVEL 19-23"
_LEVEL_RANGE_RE = re.compile(
    r"LEVELS?\s*(\d+)\s*(?:-|–|TO|THROUGH)\s*(\d+)",
    re.IGNORECASE,
)
# "AND LEVEL 15", "AND LEVELS 14 & 15"
_AND_LEVEL_RE = re.compile(
    r"AND\s+LEVELS?\s*((?:\d+(?:\s*(?:,|&|AND)\s*)?)+)",
    re.IGNORECASE,
)
# Single-level "LEVEL 24"
_SINGLE_LEVEL_RE = re.compile(r"LEVEL\s+(\d+)\b", re.IGNORECASE)

# ─── Named-floor matchers ───
_NAMED_FLOOR_MAP: tuple[tuple[re.Pattern[str], list[str]], ...] = (
    (re.compile(r"LOWER\s+LOBBY", re.IGNORECASE), ["Lower Lobby"]),
    (re.compile(r"LOBBY\s+LEVEL", re.IGNORECASE), ["Lobby"]),
    (re.compile(r"\bROOF\s+PLAN\b", re.IGNORECASE), ["Roof"]),
    (re.compile(r"SERVICE\s+LEVEL", re.IGNORECASE), ["Service"]),
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
