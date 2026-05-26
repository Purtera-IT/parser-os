"""Per-page kind classifier — routes each PDF page to the right pipeline.

Today, ``_run_schematic_pre_pass`` runs ``locate_legend_candidates`` +
``detect_symbols`` on every page of a PDF. That made sense for the
single-page SCHEMATIC_* test corpus where every page IS a schematic.
On real multi-page DD/CD sets (Marriott Atlanta, 25 pages), the same
flow chokes:

* Page 0-2 are SPEC PROSE (Bid Form responsibilities, copper specs).
  Running ``locate_legend_candidates`` on a 3-column spec list invents
  bogus legends.

* Page 1 is a LEGEND TABLE SHEET (T0.01) with FOUR distinct legends
  (Structured Cabling / Intrusion / Access Control / CCTV). The
  current locator returns the FIRST candidate above 0.45 and bails;
  on Marriott it picked a dimension callout ("1-1/4∅") and missed all
  four real legends.

* Pages 3-24 are SCHEMATIC DRAWINGS (floor plans, ceiling plans).
  These are where ``detect_symbols`` should actually fire, against
  the legend vocabulary extracted from the legend-table sheet.

The fix: classify each page into one of five kinds and route it to
the right specialized extractor:

* ``legend_table``     → multi-legend extractor (one symbol/label table per legend header)
* ``schematic_drawing``→ ``detect_symbols`` (look up symbols against the legend vocabulary)
* ``schedule_bom``     → schedule/BOM table extractor (existing PDF parser handles)
* ``spec_prose``       → skip schematic flow entirely (the generic PDF parser handles prose)
* ``cover_title``      → metadata-only extraction
* ``unknown``          → conservative default; run legend locator + detect_symbols (current behavior)

Determinism contract preserved: classifier is a pure function over
``(page, blocks)``. No LLM, no I/O, no global state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


# ── Page kinds ────────────────────────────────────────────────────

PageKind = str  # Literal["legend_table" | "schematic_drawing" | "schedule_bom" | "spec_prose" | "cover_title" | "unknown"]

LEGEND_TABLE = "legend_table"
SCHEMATIC_DRAWING = "schematic_drawing"
SCHEDULE_BOM = "schedule_bom"
SPEC_PROSE = "spec_prose"
COVER_TITLE = "cover_title"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class PageKindClassification:
    """Result of classifying one page.

    ``kind`` is the routing decision.

    ``signals`` is a flat dict of every signal the classifier saw
    (header matches, text-density estimate, vector-stroke estimate).
    Surfaced for debugging + for the schematic_warning emitter so a
    misclassification leaves a forensic trail.
    """

    page_index: int
    kind: PageKind
    confidence: float                                  # 0.0-1.0
    rationale: str
    signals: dict[str, Any]


# ── Pre-compiled detectors ────────────────────────────────────────

# Legend headers — case-insensitive. Marriott T0.01 has four:
# "STRUCTURED CABLING SYMBOLS LEGEND" / "INTRUSION DETECTION SYMBOLS LEGEND" /
# "ACCESS CONTROL AND INTERCOM SYMBOLS LEGEND" / "CCTV SYMBOLS LEGEND".
# The general pattern is `<system> [SYMBOLS] LEGEND`.
_LEGEND_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:symbols?\s+)?(?:legend|key)\b", re.IGNORECASE),
    re.compile(r"\bdrawing\s+(?:notes?|index)\b", re.IGNORECASE),
    re.compile(r"\bgraphic\s+symbols?\b", re.IGNORECASE),
)

# Schedule/BOM headers — also tables but with quantities, not symbols.
_SCHEDULE_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:device|equipment|fixture|panel|cable)\s+schedule\b", re.IGNORECASE),
    re.compile(r"\b(?:components?\s+)?specifications?\s+list\b", re.IGNORECASE),
    re.compile(r"\bbill\s+of\s+materials?\b", re.IGNORECASE),
    re.compile(r"\bresponsibility\s+matrix\b", re.IGNORECASE),
    re.compile(r"\bpart\s*number\b", re.IGNORECASE),
)

# Title/cover indicators.
_COVER_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcover\s+sheet\b", re.IGNORECASE),
    re.compile(r"\bdrawing\s+index\b", re.IGNORECASE),
)

# Sheet-number patterns — every drawing page has one. Markers like
# T0.01, E1.04, T1.12, A001, etc. Presence indicates a drawing-style
# title block. Used as a tie-breaker, not a sole signal.
_SHEET_NUMBER_PATTERN = re.compile(
    r"\b[A-Z]{1,3}\d{1,2}(?:\.\d{1,2})?\b"
)

# Tokens that look like dimensions / units — used to reject bogus
# "legend" rows where the label column is actually a conduit size.
_UNIT_TOKEN_PATTERN = re.compile(
    r"[Ø∅ø]|\"|'|\b(?:in|mm|cm|ft|m|kg|lb|oz|VAC|VDC|AWG|MHz|GHz)\b",
    re.IGNORECASE,
)


# ── Signal extractors ─────────────────────────────────────────────


def _page_text(blocks: Sequence[Any]) -> str:
    """Concatenate text from blocks. Tolerant of whatever block
    shape the locator hands us (object with ``.text`` or dict)."""
    parts: list[str] = []
    for b in blocks:
        text = getattr(b, "text", None)
        if text is None and isinstance(b, dict):
            text = b.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _count_header_matches(
    text: str, patterns: Sequence[re.Pattern[str]]
) -> int:
    """How many distinct header lines match any pattern. Counts
    *line-level* matches — a single doc may have multiple legends."""
    if not text:
        return 0
    n = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) > 100:               # headers are short
            continue
        for p in patterns:
            if p.search(line):
                n += 1
                break
    return n


def _vector_stroke_density(page: Any) -> float:
    """Estimate fraction of page area occupied by vector strokes.

    Used to distinguish schematic drawings (high stroke density →
    rooms, walls, callouts) from text-only pages (near-zero strokes).

    Returns 0.0-1.0; degrades to 0.0 on any pymupdf error so a
    misbehaving page never crashes the classifier."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return 0.0
    if not drawings:
        return 0.0
    try:
        page_rect = page.rect
        page_area = max(page_rect.width * page_rect.height, 1.0)
    except Exception:
        return 0.0
    stroke_bbox_area = 0.0
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        try:
            stroke_bbox_area += abs(rect.width) * abs(rect.height)
        except Exception:
            continue
    return min(1.0, stroke_bbox_area / page_area)


def _text_density(blocks: Sequence[Any], page: Any) -> float:
    """Estimate text density as chars per unit page area.

    Useful negative signal for schematic drawings (which have
    sparse, short text — room labels, callouts) vs spec prose
    (which has dense paragraph text). Returns a normalized
    ``chars / area`` ratio.
    """
    text = _page_text(blocks)
    try:
        r = page.rect
        area = max(r.width * r.height, 1.0)
    except Exception:
        area = 1.0
    return len(text) / area


def _avg_block_text_length(blocks: Sequence[Any]) -> float:
    """Average chars-per-block. Schematic drawings have short labels
    (~6-20 chars); spec prose has long paragraphs (~80-200 chars)."""
    lengths: list[int] = []
    for b in blocks:
        text = getattr(b, "text", None)
        if text is None and isinstance(b, dict):
            text = b.get("text")
        if text:
            lengths.append(len(str(text)))
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


# ── Classifier ────────────────────────────────────────────────────


def classify_page_kind(
    *,
    page_index: int,
    page: Any,
    blocks: Sequence[Any],
) -> PageKindClassification:
    """Deterministically classify a single PDF page into one routing kind.

    The classifier returns ``UNKNOWN`` when signals don't clearly
    point to one kind — the caller should treat ``UNKNOWN`` as
    "run the full schematic flow conservatively" to preserve
    existing behavior on cases the classifier hasn't been tuned on.
    """
    text = _page_text(blocks)
    legend_headers = _count_header_matches(text, _LEGEND_HEADER_PATTERNS)
    schedule_headers = _count_header_matches(text, _SCHEDULE_HEADER_PATTERNS)
    cover_headers = _count_header_matches(text, _COVER_HEADER_PATTERNS)
    stroke_density = _vector_stroke_density(page) if page is not None else 0.0
    text_density = _text_density(blocks, page) if page is not None else 0.0
    avg_block_len = _avg_block_text_length(blocks)
    n_blocks = len(blocks)
    sheet_number_present = bool(_SHEET_NUMBER_PATTERN.search(text or ""))

    signals = {
        "legend_headers": legend_headers,
        "schedule_headers": schedule_headers,
        "cover_headers": cover_headers,
        "stroke_density": round(stroke_density, 4),
        "text_density": round(text_density, 6),
        "avg_block_len": round(avg_block_len, 2),
        "n_blocks": n_blocks,
        "sheet_number_present": sheet_number_present,
    }

    # ── Rule order: most-specific first ──────────────────────────

    # 1. LEGEND_TABLE — at least one "LEGEND" header AND tabular
    #    content (many short text blocks) AND low-to-medium stroke
    #    density. Multiple legend headers strongly indicate this.
    if legend_headers >= 1 and avg_block_len < 60 and n_blocks >= 20:
        confidence = min(
            1.0, 0.55 + 0.15 * legend_headers + (0.10 if stroke_density < 0.30 else 0.0)
        )
        return PageKindClassification(
            page_index=page_index,
            kind=LEGEND_TABLE,
            confidence=confidence,
            rationale=(
                f"{legend_headers} legend header(s) + "
                f"{n_blocks} short text blocks (avg {avg_block_len:.0f} chars) → multi-legend table"
            ),
            signals=signals,
        )

    # 2. SCHEDULE_BOM — schedule-style header AND tabular content.
    #    Distinct from legend because the columns are
    #    qty/part/description, not symbol/label.
    if schedule_headers >= 1 and avg_block_len < 80 and n_blocks >= 15:
        return PageKindClassification(
            page_index=page_index,
            kind=SCHEDULE_BOM,
            confidence=min(1.0, 0.60 + 0.10 * schedule_headers),
            rationale=(
                f"{schedule_headers} schedule/BOM header(s) + tabular layout "
                f"({n_blocks} blocks, avg {avg_block_len:.0f} chars)"
            ),
            signals=signals,
        )

    # 3. SCHEMATIC_DRAWING — high vector stroke density AND short
    #    text blocks AND sheet number present. The drawing pages
    #    on Marriott score around 0.35-0.55 stroke density with
    #    very short callout text.
    if stroke_density >= 0.20 and avg_block_len < 40 and sheet_number_present:
        return PageKindClassification(
            page_index=page_index,
            kind=SCHEMATIC_DRAWING,
            confidence=min(1.0, 0.50 + stroke_density),
            rationale=(
                f"stroke density {stroke_density:.2f} + short callouts "
                f"(avg {avg_block_len:.0f} chars) + sheet number present"
            ),
            signals=signals,
        )

    # 4. COVER_TITLE — cover headers OR very few blocks with a sheet number.
    if cover_headers >= 1 or (n_blocks < 10 and sheet_number_present):
        return PageKindClassification(
            page_index=page_index,
            kind=COVER_TITLE,
            confidence=0.55 + 0.20 * cover_headers,
            rationale=(
                f"cover indicators ({cover_headers}) + {n_blocks} blocks"
            ),
            signals=signals,
        )

    # 5. SPEC_PROSE — long paragraph-shaped blocks, no legend/schedule
    #    headers, low stroke density.
    if avg_block_len >= 60 and stroke_density < 0.15 and legend_headers == 0:
        return PageKindClassification(
            page_index=page_index,
            kind=SPEC_PROSE,
            confidence=0.65,
            rationale=(
                f"paragraph blocks (avg {avg_block_len:.0f} chars) + "
                f"low stroke density {stroke_density:.2f}"
            ),
            signals=signals,
        )

    # Fallback: don't second-guess; run the conservative full flow.
    return PageKindClassification(
        page_index=page_index,
        kind=UNKNOWN,
        confidence=0.0,
        rationale="no rule fired; treat as schematic_drawing conservatively",
        signals=signals,
    )


__all__ = [
    "PageKindClassification",
    "PageKind",
    "LEGEND_TABLE",
    "SCHEMATIC_DRAWING",
    "SCHEDULE_BOM",
    "SPEC_PROSE",
    "COVER_TITLE",
    "UNKNOWN",
    "classify_page_kind",
]
