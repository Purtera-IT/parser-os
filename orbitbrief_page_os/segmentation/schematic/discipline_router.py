"""Multi-discipline router for schematic sheets.

Real DD/CD sets mix disciplines: T (telecom), E (electrical),
M (mechanical), FA (fire alarm), A (architectural), P (plumbing),
S (structural), CO/CE (civil), L (landscape), FP (fire protection).

Each discipline has its own symbol vocabulary. Today's
LegendResolver assigns one legend to all pages sharing the same
discipline letter. That's correct for single-discipline DD sets but
breaks when:

* A consultant-set has both T and E legends on T0.01 +
  E0.01 — the resolver wrongly picks one for both
* Per-discipline cross-pollution: an electrical symbol legend
  shouldn't match symbols on T sheets

This module classifies sheets by **discipline prefix** and gives
the orchestrator a per-discipline scope so legends route correctly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# Canonical discipline codes used in US AEC drawing conventions.
# Order matters: longest prefix first so "FA" doesn't get matched
# as "F" + "A".
KNOWN_DISCIPLINES: tuple[tuple[str, str], ...] = (
    ("FA", "fire_alarm"),
    ("FP", "fire_protection"),
    ("CO", "civil"),
    ("CE", "civil"),
    ("PD", "plumbing_demolition"),
    ("HD", "hvac_demolition"),
    ("ED", "electrical_demolition"),
    ("AD", "architectural_demolition"),
    ("CD", "civil_demolition"),
    ("LD", "landscape_demolition"),
    ("T", "telecom"),
    ("E", "electrical"),
    ("M", "mechanical"),
    ("P", "plumbing"),
    ("A", "architectural"),
    ("S", "structural"),
    ("L", "landscape"),
    ("G", "general"),
    ("D", "demolition"),
    ("C", "civil"),
    ("H", "hvac"),
    ("Q", "equipment"),
)


# Sheet-number pattern. Tolerates "T0.01" / "T-101" / "T001" / "T1" /
# "T1.04A" / "T-1.04A". The discipline is the alpha prefix.
_SHEET_NUMBER_PATTERN = re.compile(
    r"\b([A-Z]{1,3})(?:[-_]?)(\d{1,3}(?:\.\d{1,2})?[A-Z]?)\b"
)


@dataclass(frozen=True)
class DisciplineAssignment:
    """Per-page discipline classification."""

    page_index: int
    sheet_number: str | None                          # e.g., "T0.01"
    discipline_code: str | None                       # e.g., "T"
    discipline_label: str | None                      # e.g., "telecom"
    confidence: float                                  # 0.0 - 1.0
    rationale: str


def parse_discipline_from_sheet_number(sheet_number: str | None) -> tuple[str | None, str | None]:
    """Extract (discipline_code, discipline_label) from a sheet number.

    Returns (None, None) when the sheet number doesn't look like a
    drawing sheet (e.g., "App. A" or "Page 5").
    """
    if not sheet_number:
        return None, None
    m = _SHEET_NUMBER_PATTERN.search(sheet_number.upper())
    if not m:
        return None, None
    prefix = m.group(1)
    # Longest-prefix match against KNOWN_DISCIPLINES
    for code, label in KNOWN_DISCIPLINES:
        if prefix == code:
            return code, label
        if prefix.startswith(code) and len(prefix) > len(code):
            # Mixed prefixes like "TA" (telecom-architectural) — uncommon
            # but treat as the longer match if it's also known
            continue
    # Fallback: single-letter prefix not in our table → use as-is
    if len(prefix) >= 1 and prefix[0] in [c[0] for c in KNOWN_DISCIPLINES if len(c[0]) == 1]:
        single = prefix[0]
        for code, label in KNOWN_DISCIPLINES:
            if code == single:
                return single, label
    return None, None


def assign_disciplines(
    sheet_numbers_by_page: dict[int, str | None],
) -> dict[int, DisciplineAssignment]:
    """Classify every page in a document by discipline.

    Input: {page_index: sheet_number_or_None}
    Output: {page_index: DisciplineAssignment}
    """
    out: dict[int, DisciplineAssignment] = {}
    for page_index, sheet in sorted(sheet_numbers_by_page.items()):
        code, label = parse_discipline_from_sheet_number(sheet)
        if code is None:
            out[page_index] = DisciplineAssignment(
                page_index=page_index,
                sheet_number=sheet,
                discipline_code=None,
                discipline_label=None,
                confidence=0.0,
                rationale="no_recognizable_sheet_number",
            )
        else:
            out[page_index] = DisciplineAssignment(
                page_index=page_index,
                sheet_number=sheet,
                discipline_code=code,
                discipline_label=label,
                confidence=1.0,
                rationale=f"sheet_prefix:{code}",
            )
    return out


def legend_scope_for_discipline(
    *,
    legend_page_discipline: str | None,
    target_page_discipline: str | None,
) -> str | None:
    """Return the LegendResolver scope/rationale string for a legend
    discovered on ``legend_page_discipline`` being applied to a target
    page on ``target_page_discipline``.

    None when the legend should NOT apply (cross-discipline mismatch).
    """
    if legend_page_discipline is None and target_page_discipline is None:
        return "no_discipline_either_side"
    if legend_page_discipline is None or target_page_discipline is None:
        return "discipline_unknown_one_side"
    if legend_page_discipline == target_page_discipline:
        return f"same_discipline:{target_page_discipline}"
    # Disciplines mismatch — don't apply
    return None


__all__ = [
    "KNOWN_DISCIPLINES",
    "DisciplineAssignment",
    "assign_disciplines",
    "legend_scope_for_discipline",
    "parse_discipline_from_sheet_number",
]
