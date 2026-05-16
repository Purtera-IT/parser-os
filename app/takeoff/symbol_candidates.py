"""Detect low-voltage symbol candidates from native PDF text.

This module produces :class:`SymbolCandidate` objects directly from
PyMuPDF native words. It applies the rejection rules required by the
spec:

* page_type not in ``{floor_plan, typical_plan}`` → reject
* sheet ``in_scope=False`` → reject
* token outside ``plan_viewport`` → reject
* token inside any excluded region → reject

Rejected candidates are still emitted (with ``rejection_reason``) so
reviewers can audit the false positives.
"""
from __future__ import annotations

from typing import Any

from app.core.ids import stable_id
from app.takeoff.legend_extractor import load_default_legend_rules, rules_by_symbol
from app.takeoff.pdf_native import PdfWord, dedupe_words, extract_page_words
from app.takeoff.plan_regions import is_excluded, is_inside
from app.takeoff.schemas import BBox, LegendRule, SheetRecord, SymbolCandidate

# Page types that may legitimately carry device symbols.
_DEVICE_BEARING_PAGE_TYPES = frozenset({"floor_plan", "typical_plan"})


def _candidate_id(page_index: int, raw_symbol: str, bbox: BBox) -> str:
    """Deterministic candidate id based on page + symbol + center."""
    cx, cy = bbox.center()
    return stable_id("cand", page_index, raw_symbol, round(cx, 1), round(cy, 1))


def _bbox_from_word(w: PdfWord) -> BBox:
    return BBox(
        x0=w.x0,
        y0=w.y0,
        x1=w.x1,
        y1=w.y1,
        coord_space="pdf_pt",
    )


def detect_symbol_candidates(
    *,
    page: Any,
    sheet: SheetRecord,
    legend_rules: list[LegendRule] | None = None,
) -> list[SymbolCandidate]:
    """Detect every native-text symbol candidate on a single page.

    Acceptance is determined by sheet metadata + viewport geometry. The
    method emits BOTH accepted (``rejection_reason=None``) and rejected
    (``rejection_reason="..."``) candidates.
    """
    rules = legend_rules if legend_rules is not None else load_default_legend_rules()
    symbol_index = rules_by_symbol(rules)
    raw_symbols = set(symbol_index.keys())

    words = extract_page_words(page)
    # Per-symbol dedupe at the spec's 0.5pt tolerance.
    matching = [w for w in words if w.text in raw_symbols]
    deduped = dedupe_words(matching, tolerance_pt=0.5)

    # Track which (text, key) pairs were duplicates — we don't emit them
    # but we annotate the surviving candidate with ``native_text_deduped``.
    seen_keys: set[tuple[str, int, int]] = set()
    duplicates_collapsed: dict[tuple[str, int, int], int] = {}
    for w in matching:
        cx, cy = w.center()
        key = (w.text, int(round(cx / 0.5)), int(round(cy / 0.5)))
        if key in seen_keys:
            duplicates_collapsed[key] = duplicates_collapsed.get(key, 1) + 1
        else:
            seen_keys.add(key)

    candidates: list[SymbolCandidate] = []
    page_type = sheet.page_type
    viewport = sheet.plan_viewport
    excluded = sheet.excluded_regions

    for w in deduped:
        bbox = _bbox_from_word(w)
        rule = symbol_index.get(w.text)
        normalized = rule.normalized_class if rule else None
        cx, cy = w.center()
        key = (w.text, int(round(cx / 0.5)), int(round(cy / 0.5)))
        source_methods = ["pdf_native_text"]
        if duplicates_collapsed.get(key, 0) > 1:
            source_methods.append("native_text_deduped")

        rejection_reason: str | None = None
        if page_type not in _DEVICE_BEARING_PAGE_TYPES:
            rejection_reason = f"page_type={page_type} is not device-bearing"
        elif not sheet.in_scope:
            rejection_reason = (
                f"sheet {sheet.sheet_number or sheet.page_index} is not in scope"
            )
        elif viewport is not None and not is_inside(bbox, viewport):
            rejection_reason = "outside plan_viewport"
        elif excluded and is_excluded(bbox, excluded):
            rejection_reason = "inside excluded_region (titleblock)"

        candidate = SymbolCandidate(
            id=_candidate_id(sheet.page_index, w.text, bbox),
            page_index=sheet.page_index,
            raw_symbol=w.text,
            normalized_class=normalized,
            bbox=bbox,
            source_methods=source_methods,
            confidence=0.94 if rejection_reason is None else 0.6,
            rejection_reason=rejection_reason,
            needs_review=rejection_reason is not None,
            nearby_text=[],
        )
        candidates.append(candidate)

    return candidates


def is_accepted(candidate: SymbolCandidate) -> bool:
    """True iff the candidate has no rejection reason."""
    return candidate.rejection_reason is None


__all__ = [
    "detect_symbol_candidates",
    "is_accepted",
]
