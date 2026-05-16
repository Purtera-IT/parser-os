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
_PREFIX_RULES: tuple[tuple[str, SheetPageType], ...] = (
    ("T0.00", "spec"),
    ("T0.01", "legend"),
    ("T0.02", "component_schedule"),
    ("T1.", "floor_plan"),
    ("T4.", "typical_plan"),
    ("T7.", "riser"),
    ("T8.", "equipment_room"),
    ("T9.", "detail"),
)

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

    # 2. Sheet-number prefix rules.
    if sheet_number:
        upper = sheet_number.upper()
        for prefix, page_type in _PREFIX_RULES:
            if upper == prefix or (
                prefix.endswith(".") and upper.startswith(prefix)
            ):
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
