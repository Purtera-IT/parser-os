"""Sheet classification — turn raw page text into a :class:`SheetRecord`.

The classifier is purely text-driven and deterministic:

1. Pull ``SHEET NUMBER: <num> - <name>`` (if present).
2. Apply page-type rules in priority order:
   a. Strong keywords always win ("SYMBOLS & LEGENDS" → legend,
      "RISER DIAGRAM" → riser, ...).
   b. Number prefix rules (T0.00 → spec, T1.xx → floor_plan, ...).
3. Apply scope rules — sheet text containing "NOT IN SCOPE" in a
   sheet-scope context marks the sheet ``in_scope=false``.

Nothing about plan viewports or multipliers lives in this module —
``plan_regions`` and ``multipliers`` own those concerns.
"""
from __future__ import annotations

import re

from app.takeoff.schemas import SheetPageType, SheetRecord

# ─── Sheet number / name parsing ───
# Adobe text extraction can put the number and name on adjacent lines
# instead of a single line, so we allow whitespace including newlines
# between "SHEET NUMBER:" and the value.
_SHEET_NUMBER_RE = re.compile(
    r"SHEET\s+NUMBER\s*[:\-]?\s*([A-Z]{1,3}\d+\.\d+)"
    r"(?:\s*[-–:]?\s*([A-Z0-9 &,\-/]{2,}?))?\s*(?:\n|$)",
    re.IGNORECASE,
)
_SHEET_NAME_FOLLOWUP_RE = re.compile(
    r"SHEET\s+TITLE\s*[:\-]?\s*([A-Z0-9 &,\-/]{2,})",
    re.IGNORECASE,
)

# ─── Strong keyword rules ───
_STRONG_KEYWORDS: tuple[tuple[str, SheetPageType], ...] = (
    ("SYMBOLS & LEGENDS", "legend"),
    ("SYMBOLS AND LEGENDS", "legend"),
    ("KEYED NOTE LEGEND", "legend"),
    ("RISER DIAGRAM", "riser"),
    ("EQUIPMENT ROOM", "equipment_room"),
    ("INSTALLATION DETAILS", "detail"),
    ("SECURITY DETAILS", "detail"),
    ("EQUIPMENT RACK DETAILS", "detail"),
    ("COMPONENT SPECIFICATIONS", "component_schedule"),
    ("COMPONENT SCHEDULE", "component_schedule"),
)

# ─── Prefix rules ───
#
# We treat the *digit family* (the number after the letter prefix) as
# the universal signal, then layer specific .00/.01/.02 rules on top.
# This makes T1.05 / LV1.05 / E1.05 / IT1.05 / TC1.05 all classify as
# floor_plan with no project-specific code.

_DIGIT_FAMILY_RE = re.compile(r"^[A-Z]+(\d+)(?:\.(\d+))?", re.IGNORECASE)

# (digit_family_pattern, page_type). The pattern is matched against the
# numeric portion only (letter prefix stripped). ``family_pattern`` of
# ``"1."`` matches any sheet whose digit portion starts with ``"1."``,
# e.g. ``T1.03`` or ``E1.0`` or ``LV1.05``.
_DIGIT_FAMILY_RULES: tuple[tuple[str, SheetPageType], ...] = (
    # Intro / setup pages.
    ("0.00", "spec"),
    ("0.01", "legend"),
    ("0.02", "component_schedule"),
    ("0.03", "component_schedule"),
    # Plan series — most firms use 1.xx, some spill into 2.xx / 3.xx.
    ("1.", "floor_plan"),
    ("2.", "floor_plan"),
    ("3.", "floor_plan"),
    # Typical / enlarged plans.
    ("4.", "typical_plan"),
    # Riser diagrams.
    ("5.", "riser"),  # some firms use 5.x for risers
    ("6.", "riser"),  # ditto
    ("7.", "riser"),
    # Equipment rooms.
    ("8.", "equipment_room"),
    # Installation details.
    ("9.", "detail"),
)


def _digit_family(sheet_number: str | None) -> str | None:
    """Extract the digit family (e.g. ``"1.05"``) from a sheet number.

    ``T1.05`` → ``"1.05"``. ``LV1.05`` → ``"1.05"``. ``E1.0`` →
    ``"1.0"``. ``TC-101`` → ``None`` (non-standard format).
    """
    if not sheet_number:
        return None
    s = sheet_number.strip().upper()
    m = _DIGIT_FAMILY_RE.match(s)
    if not m:
        return None
    major = m.group(1)
    minor = m.group(2)
    if minor is not None:
        return f"{major}.{minor}"
    return f"{major}."

# Scope rules — "NOT IN SCOPE" / "LEVEL NOT IN SCOPE" anywhere in the
# sheet text marks it out of scope. Keep simple; the spec asks for it.
_NOT_IN_SCOPE_RE = re.compile(r"\bNOT\s+IN\s+SCOPE\b", re.IGNORECASE)


def parse_sheet_number_and_name(page_text: str) -> tuple[str | None, str | None]:
    """Extract (sheet_number, sheet_name) from a page's raw text.

    Returns ``(None, None)`` if no canonical sheet-number line is
    present. The sheet_name may still be ``None`` even when the number
    was found.
    """
    if not page_text:
        return (None, None)
    m = _SHEET_NUMBER_RE.search(page_text)
    if m is None:
        return (None, None)
    number = m.group(1).strip().upper()
    name = (m.group(2) or "").strip()
    if not name:
        nm = _SHEET_NAME_FOLLOWUP_RE.search(page_text)
        if nm:
            name = nm.group(1).strip()
    if name:
        # Single-space and trim trailing punctuation.
        name = re.sub(r"\s+", " ", name).strip(" -–")
    return (number, name or None)


def classify_page_type(
    sheet_number: str | None,
    sheet_name: str | None,
    page_text: str,
) -> SheetPageType:
    """Return the page type for a given (number, name, text) tuple.

    Priority order:

    1. **Sheet name** keywords (most reliable — titles like
       "INSTALLATION DETAILS" or "SYMBOLS & LEGENDS" override
       everything).
    2. **Sheet number prefix** — T1.xx → floor_plan, T7.xx → riser, etc.
    3. **Full-text keyword** — only used as a last resort when neither
       a name nor a recognizable number is available (e.g. a page with
       no titleblock).

    The full-text scan deliberately does NOT win over a sheet number,
    because every page in a typical drawing set mentions terms like
    "EQUIPMENT ROOM" or "INSTALLATION DETAILS" somewhere in notes
    without that page itself being a detail or equipment-room sheet.
    """
    name_upper = (sheet_name or "").upper()

    # 1. Sheet-name keyword rules — these always win.
    for needle, page_type in _STRONG_KEYWORDS:
        if needle in name_upper:
            return page_type

    # Floor-plan name shortcuts (cover ROOF PLAN, FLOOR PLAN suffix).
    if "FLOOR PLAN" in name_upper or "ROOF PLAN" in name_upper:
        return "floor_plan"

    # 2. Sheet-number prefix rules — universal across any letter prefix.
    # Match by the digit family (e.g. "1.05" → floor_plan) rather than
    # the full "T1.05" string. Works for T-set / E-set / LV-set / etc.
    family = _digit_family(sheet_number)
    if family:
        # Exact matches first (e.g. "0.01" → legend beats "0." → spec).
        for pattern, page_type in _DIGIT_FAMILY_RULES:
            if not pattern.endswith(".") and family == pattern:
                return page_type
        # Then prefix matches (e.g. "1." matches "1.05").
        for pattern, page_type in _DIGIT_FAMILY_RULES:
            if pattern.endswith(".") and family.startswith(pattern):
                return page_type

    # 3. Full-text keyword rules — only when nothing else fired.
    text_upper = (page_text or "").upper()
    for needle, page_type in _STRONG_KEYWORDS:
        if needle in text_upper:
            return page_type

    return "unknown"


def is_sheet_in_scope(page_text: str, page_type: SheetPageType) -> tuple[bool, str | None]:
    """Apply scope rules — return (in_scope, scope_reason).

    Only floor-plan-style pages get scope-checked; legends / specs /
    details default to in-scope (they don't carry device counts).
    """
    if page_type not in {"floor_plan", "typical_plan"}:
        return (True, None)
    if _NOT_IN_SCOPE_RE.search(page_text or ""):
        return (False, "sheet text contains NOT IN SCOPE")
    return (True, None)


def classify_sheet(
    page_index: int,
    page_text: str,
) -> SheetRecord:
    """Top-level entry — build a :class:`SheetRecord` from raw page text.

    Floor-multiplier and plan-viewport fields are left at their defaults
    — the caller (the pipeline) fills those in after classification.
    """
    number, name = parse_sheet_number_and_name(page_text)
    page_type = classify_page_type(number, name, page_text)
    in_scope, scope_reason = is_sheet_in_scope(page_text, page_type)
    return SheetRecord(
        page_index=page_index,
        sheet_number=number,
        sheet_name=name,
        page_type=page_type,
        in_scope=in_scope,
        scope_reason=scope_reason,
    )


__all__ = [
    "classify_sheet",
    "classify_page_type",
    "is_sheet_in_scope",
    "parse_sheet_number_and_name",
]
