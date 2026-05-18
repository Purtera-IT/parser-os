"""Home-run zone note parsing for floor-plan sheets.

Drawing sets annotate each floor-plan with one or more "HOMERUN ALL
CABLES ON ... TO IDF-N, ON LEVEL N." sentences. Each sentence binds a
list of levels to a single closet (MDF / IDF-N). This module parses
those sentences and exposes a small structured representation that the
fusion stage uses to assign ``home_run_to`` to each device.

The parser is regex-driven and intentionally tolerant — drawing-set
language varies enough that a too-strict parser would miss legitimate
zones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class HomeRunZone:
    """A single parsed home-run zone note from a sheet.

    Attributes:
        raw_text: the full sentence as extracted from the page text.
        target: the IDF / MDF closet name (``"IDF-5"``, ``"MDF ROOM"``).
        target_level: the level the closet lives on (``"5"``, ``"Lower
            Lobby"``).
        levels: the levels whose cables home-run to ``target``.
        applies_to_all_levels: True for "THIS LEVEL" style notes — the
            zone applies to whatever level(s) the sheet represents.
    """

    raw_text: str
    target: str | None = None
    target_level: str | None = None
    levels: list[str] = field(default_factory=list)
    applies_to_all_levels: bool = False


# ─── Patterns ───
#
# We accept many phrasings for "route cables to closet X" — each project
# / engineer / firm uses slightly different language. Patterns are
# evaluated in order; the first one that matches a sentence wins.

_HOMERUN_SENTENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Cooper Carry / NTI: "HOMERUN ALL CABLES ON THIS LEVEL TO IDF-5, ..."
    re.compile(r"\bHOMERUN\s+ALL\s+CABLES[^.]+?\.", re.IGNORECASE | re.DOTALL),
    # "HOME RUN ALL CABLES ..." (two-word variant — only matches when
    # the words are actually separated, so it doesn't double-count
    # HOMERUN-as-one-word sentences).
    re.compile(r"\bHOME\s+RUN\s+ALL\s+CABLES[^.]+?\.", re.IGNORECASE | re.DOTALL),
    # "RUN ALL CABLES ON LEVELS X-Y TO TR-3 ..." — word-boundary at
    # the start so this doesn't fire inside HOMERUN sentences.
    re.compile(r"\bRUN\s+(?:ALL\s+)?CABLES?\s+(?:ON\s+[^.]+?\s+)?TO\s+(?:MDF|IDF|TR|ER|BDF)[^.]+?\.", re.IGNORECASE | re.DOTALL),
    # "ALL CABLES BACK TO MDF ROOM ..." — anchored on the "ALL CABLES"
    # / "CABLES" + "BACK TO" combo so it can't match a generic
    # "cables back to" mid-sentence.
    re.compile(r"\b(?:ALL\s+)?CABLES?\s+BACK\s+TO\s+(?:MDF|IDF|TR|ER|BDF)[^.]+?\.", re.IGNORECASE | re.DOTALL),
    # "ROUTE ALL CABLES TO IDF-A ..."
    re.compile(r"\bROUTE\s+(?:ALL\s+)?CABLES?\s+TO\s+(?:MDF|IDF|TR|ER|BDF)[^.]+?\.", re.IGNORECASE | re.DOTALL),
    # "TERMINATE ALL CABLES AT IDF-12 ..."
    re.compile(r"\bTERMINATE\s+(?:ALL\s+)?CABLES?\s+(?:AT|IN)\s+(?:MDF|IDF|TR|ER|BDF)[^.]+?\.", re.IGNORECASE | re.DOTALL),
)

# Legacy alias so existing tests still import the symbol.
_HOMERUN_SENTENCE_RE = _HOMERUN_SENTENCE_PATTERNS[0]

# Closet target — covers "MDF ROOM", "IDF-5", "IDF 5", "IDF-21", and
# also "TR-3", "TR-A", "ER-1", "BDF-B" (other firms' room conventions).
_TARGET_RE = re.compile(
    r"\b(?:TO|AT|IN)\s+((?:MDF(?:\s+ROOM)?|IDF[-\s]?[A-Z0-9]+|TR[-\s]?[A-Z0-9]+|ER[-\s]?[A-Z0-9]+|BDF[-\s]?[A-Z0-9]+))",
    re.IGNORECASE,
)

# Target-level expression — "THIS LEVEL", "ON LEVEL 5", "ON THE LOWER
# LOBBY LEVEL".
_TARGET_LEVEL_RE = re.compile(
    r",\s*(?:ON\s+)?(THIS\s+LEVEL|ON\s+LEVEL\s+\d+|ON\s+THE\s+[A-Z ]+?\s+LEVEL)",
    re.IGNORECASE,
)

# Source levels — "ON THIS LEVEL", "ON LEVELS 5 & 6", "ON LEVEL 19",
# "ON LEVELS 7, 8 & 9".
_SOURCE_THIS_LEVEL_RE = re.compile(r"\bON\s+THIS\s+LEVEL\b", re.IGNORECASE)
_SOURCE_LEVELS_RE = re.compile(
    r"\bON\s+LEVELS?\s+([0-9,\s&]+?)(?=\s+TO\b)",
    re.IGNORECASE,
)


def _normalize_target(raw: str) -> str:
    """Normalize a target name for comparison."""
    t = re.sub(r"\s+", " ", raw or "").strip().upper()
    # Standardize MDF spelling.
    if t.startswith("MDF"):
        return "MDF ROOM"
    # Standardize IDF-NN form.
    m = re.match(r"IDF[-\s]?([A-Z0-9]+)", t)
    if m:
        return f"IDF-{m.group(1)}"
    # TR-N / ER-N / BDF-N — common alternatives to IDF in non-Marriott
    # firms (NTI / Newcomb & Boyd / CMTA, …).
    for prefix in ("TR", "ER", "BDF"):
        m = re.match(rf"{prefix}[-\s]?([A-Z0-9]+)", t)
        if m:
            return f"{prefix}-{m.group(1)}"
    return t


def _normalize_target_level(raw: str | None) -> str | None:
    """Derive a level label from a target-level phrase."""
    if not raw:
        return None
    s = raw.strip().upper()
    if s == "THIS LEVEL":
        return None  # caller resolves against sheet levels
    m = re.search(r"ON\s+LEVEL\s+(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"ON\s+THE\s+([A-Z ]+?)\s+LEVEL", s)
    if m:
        return m.group(1).strip().title()
    return None


def _parse_source_levels(sentence: str) -> tuple[list[str], bool]:
    """Return (level_list, applies_to_all_levels) for a homerun sentence."""
    if _SOURCE_THIS_LEVEL_RE.search(sentence):
        return ([], True)
    m = _SOURCE_LEVELS_RE.search(sentence)
    if not m:
        return ([], False)
    blob = m.group(1)
    nums = re.findall(r"\d+", blob)
    return ([n for n in nums], False)


def parse_zones(page_text: str) -> list[HomeRunZone]:
    """Pull all HOMERUN zone notes from a page's plain text.

    Returns a list in document order. Sentences without a recognizable
    target are dropped (they wouldn't help fusion anyway).
    """
    if not page_text:
        return []
    # Collapse whitespace so multi-line sentences match.
    normalized = " ".join(page_text.split())
    zones: list[HomeRunZone] = []
    # Collect every match from every pattern (different firms phrase
    # cable-routing notes differently). De-dupe identical sentences so
    # overlapping patterns don't double-count.
    seen_sentences: set[str] = set()
    matches: list = []
    for pat in _HOMERUN_SENTENCE_PATTERNS:
        for m in pat.finditer(normalized):
            key = m.group(0).strip().lower()
            if key in seen_sentences:
                continue
            seen_sentences.add(key)
            matches.append(m)
    # Stable order by position in document.
    matches.sort(key=lambda m: m.start())
    for m in matches:
        sentence = m.group(0).strip()
        target_m = _TARGET_RE.search(sentence)
        if not target_m:
            continue
        target = _normalize_target(target_m.group(1))
        # Target-level phrase comes after the target.
        tail = sentence[target_m.end():]
        tl_m = _TARGET_LEVEL_RE.search(tail)
        target_level = _normalize_target_level(tl_m.group(1) if tl_m else None)
        levels, applies_all = _parse_source_levels(sentence)
        zones.append(
            HomeRunZone(
                raw_text=sentence,
                target=target,
                target_level=target_level,
                levels=levels,
                applies_to_all_levels=applies_all,
            )
        )
    return zones


def assign_home_run(
    *,
    zones: list[HomeRunZone],
    sheet_levels: list[str],
    sheet_floor_label: str | None,
    sheet_number: str | None = None,
    device_level: str | None = None,
) -> tuple[str | None, str | None, list[str], list[str]]:
    """Pick a home-run target for a device.

    Returns ``(home_run_to, home_run_level, zone_notes, review_flags)``.

    Rules (in order of preference):

    1. **Exactly one zone on the sheet:** that zone wins, no flags.
    2. **Device level matches a zone's level list:** that zone wins.
    3. **Multiple zones, but the device cannot be resolved by level:**
       leave ``home_run_to=None`` and add ``ambiguous_homerun_zone``.

    The ``zone_notes`` list always reflects every zone seen on the
    sheet so reviewers see the full context.
    """
    if not zones:
        return (None, None, [], [])

    raw_notes = [z.raw_text for z in zones]

    if len(zones) == 1:
        z = zones[0]
        return (z.target, z.target_level, raw_notes, [])

    # Multiple zones — try to disambiguate by device_level.
    if device_level is not None:
        for z in zones:
            if device_level in z.levels:
                return (z.target, z.target_level, raw_notes, [])

    # Fallback: ambiguous. Add review flag but still surface zones.
    return (None, None, raw_notes, ["ambiguous_homerun_zone"])


def collect_zone_warnings(
    *,
    sheet_number: str | None,
    sheet_name: str | None,
    sheet_levels: list[str],
    zones: list[HomeRunZone],
) -> list[str]:
    """Return takeoff-document warnings for a sheet's zone coverage.

    Covers two specific situations called out in the spec:

    * **T1.06-style:** the sheet represents many levels but no zone
      explicitly mentions one of them — flag the unmentioned levels.
    * **T1.10-style typo:** zone text mentions a level the sheet
      doesn't represent (e.g. "LEVELS 10, 21, 22, & 23" on a 19-23
      sheet).
    """
    warnings: list[str] = []
    if not zones or not sheet_levels:
        return warnings

    # Levels mentioned across all zones (only numeric ones — named
    # zones like THIS LEVEL inherit the sheet's whole level list).
    mentioned: set[str] = set()
    inherits_all = any(z.applies_to_all_levels for z in zones)
    for z in zones:
        mentioned.update(z.levels)

    if inherits_all:
        return warnings

    # Missing-level warning.
    missing = [lvl for lvl in sheet_levels if lvl not in mentioned]
    if missing and len(zones) >= 2:
        warnings.append(
            f"missing_homerun_zone_for_levels: {sheet_number or '?'} "
            f"represents {sheet_levels} but zones only cover {sorted(mentioned)}; "
            f"no zone parsed for level(s) {missing}"
        )

    # Typo / mismatch warning — zone references levels not on the
    # sheet. T1.10 specifically has "10" referenced on a 19-23 sheet.
    extra = [lvl for lvl in sorted(mentioned) if lvl not in sheet_levels]
    if extra:
        warnings.append(
            f"possible_zone_note_ocr_or_design_typo: {sheet_number or '?'} "
            f"references LEVELS {extra} on a {sheet_name or 'unknown'} sheet."
        )

    return warnings


__all__ = [
    "HomeRunZone",
    "parse_zones",
    "assign_home_run",
    "collect_zone_warnings",
]
