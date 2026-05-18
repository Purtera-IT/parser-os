"""Capture context text near each device candidate.

The takeoff schema has a ``nearby_text`` list on every :class:`SymbolCandidate`
(and the device fuser carries it through to :class:`DeviceInstance`). This
module populates it.

What we capture
---------------

For every accepted candidate we look at the PDF native words on the same
page and return the words whose bbox center is within a radius (default
60 pt ≈ 0.83 inch at 72 dpi) of the candidate's bbox center. Words on the
same baseline are joined into phrases first so multi-word labels like
``EXISTING MDF ROOM`` stay together instead of splitting into three hits.

Filters applied to keep noise out:

* Drop the candidate's own raw symbol token (you don't want ``WN`` to
  show up as nearby text for a ``WN`` candidate)
* Drop pure single-character / single-digit tokens (likely column-grid
  labels or keynote refs — handled separately by ``keynotes.py``)
* Drop legend-page header strings (``CABLE``, ``COUNT``, ``TYPE``, …)

The output is sorted by distance ascending; the caller usually keeps the
first N strings for display.

This module is pure / deterministic — same page words + same bbox →
same ``nearby_text`` list every time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.takeoff.pdf_native import PdfWord
from app.takeoff.schemas import BBox


# Words we always drop — table column headers, common short noise.
_HEADER_NOISE = frozenset({
    "CABLE", "COUNT", "TYPE", "DESC", "DESCRIPTION", "QTY", "REMARKS",
    "SYMBOL", "SYMBOLS", "LEGEND", "RUN", "MOUNT", "MOUNTING", "WORK",
    "AREA", "CLOSET", "TERMINATION", "ELECTRICAL", "POWER", "REQUIREMENT",
    "REQUIREMENTS", "FROM", "TO", "ROUGH-IN", "ROUGH", "IN", "AFF", "SEE",
    "AND", "OF", "OR", "WITH", "BY", "PER", "FOR", "AT", "ON",
})

# Single-token "address" patterns to keep — even though they're short
# they're meaningful (room IDs, IDF tags, MDF marker).
_ALWAYS_KEEP_PATTERNS = (
    "MDF", "IDF", "TR-", "ER-", "BDF",  # equipment room markers
)


@dataclass(frozen=True)
class _WordRect:
    """Internal: a word's text + bbox center + size."""
    text: str
    cx: float
    cy: float
    height: float


def _word_centers(words: Iterable[PdfWord]) -> list[_WordRect]:
    out: list[_WordRect] = []
    for w in words:
        cx = (w.x0 + w.x1) / 2.0
        cy = (w.y0 + w.y1) / 2.0
        h = max(1.0, w.y1 - w.y0)
        out.append(_WordRect(text=w.text, cx=cx, cy=cy, height=h))
    return out


def _is_noise(text: str) -> bool:
    """True when this token is structurally useless context."""
    t = text.strip()
    if not t:
        return True
    if t.upper() in _HEADER_NOISE:
        return True
    # Pure short numeric / single-char tokens are keynotes/grid labels —
    # they belong to ``keynotes.py``, not here.
    stripped = t.replace(".", "").replace(",", "")
    if stripped.isdigit() and len(stripped) <= 2:
        return True
    if len(t) == 1:
        return True
    return False


def _group_into_phrases(words: list[_WordRect], y_tolerance: float) -> list[_WordRect]:
    """Merge horizontally-adjacent words on the same baseline into one phrase.

    ``y_tolerance`` is the maximum y-center distance for two words to count
    as the same baseline. Phrases inherit the leftmost word's center for
    distance calculations.
    """
    if not words:
        return []
    # Sort by (y, x) so adjacent same-line words are consecutive.
    sorted_w = sorted(words, key=lambda w: (w.cy, w.cx))
    phrases: list[_WordRect] = []
    cur_text = sorted_w[0].text
    cur_cx, cur_cy, cur_h = sorted_w[0].cx, sorted_w[0].cy, sorted_w[0].height
    prev_right = sorted_w[0].cx
    for w in sorted_w[1:]:
        same_line = abs(w.cy - cur_cy) <= y_tolerance
        # Horizontal gap heuristic: ~1.6 × word height — typical inter-word spacing
        max_gap = 1.6 * cur_h
        if same_line and (w.cx - prev_right) <= max_gap and w.cx >= prev_right:
            cur_text = f"{cur_text} {w.text}"
            prev_right = w.cx
            continue
        phrases.append(_WordRect(text=cur_text, cx=cur_cx, cy=cur_cy, height=cur_h))
        cur_text, cur_cx, cur_cy, cur_h = w.text, w.cx, w.cy, w.height
        prev_right = w.cx
    phrases.append(_WordRect(text=cur_text, cx=cur_cx, cy=cur_cy, height=cur_h))
    return phrases


def collect_nearby_text(
    *,
    bbox: BBox,
    page_words: list[PdfWord],
    radius_pt: float = 60.0,
    own_symbol: str | None = None,
    max_results: int = 6,
) -> list[str]:
    """Return up to ``max_results`` phrases nearest ``bbox`` within ``radius_pt``.

    Phrases are reconstructed by joining horizontally-adjacent words on the
    same baseline so multi-word labels stay intact.

    The candidate's own symbol token is filtered out via ``own_symbol``.
    """
    cx = (bbox.x0 + bbox.x1) / 2.0
    cy = (bbox.y0 + bbox.y1) / 2.0
    words = _word_centers(page_words)
    # Estimate a typical word height for the y-tolerance — use the median.
    if words:
        heights = sorted(w.height for w in words)
        median_h = heights[len(heights) // 2]
    else:
        median_h = 8.0
    phrases = _group_into_phrases(words, y_tolerance=median_h * 0.5)

    own_upper = (own_symbol or "").upper()
    candidates: list[tuple[float, str]] = []
    for ph in phrases:
        text = ph.text.strip()
        if not text:
            continue
        upper_text = text.upper()
        # Filter out the candidate's own raw symbol (and any phrase that
        # is ONLY the symbol token, possibly repeated like ``"WN WN"``
        # which happens when phrase reconstruction joins adjacent
        # duplicate-symbol tokens on the same plan baseline).
        if own_upper:
            phrase_tokens = upper_text.split()
            if phrase_tokens and all(tok == own_upper for tok in phrase_tokens):
                continue
        # Keep equipment-room style markers even if short.
        keep_always = any(p in upper_text for p in _ALWAYS_KEEP_PATTERNS)
        if not keep_always and _is_noise(text):
            continue
        # Distance from candidate center.
        dx = ph.cx - cx
        dy = ph.cy - cy
        d = (dx * dx + dy * dy) ** 0.5
        if d > radius_pt:
            continue
        candidates.append((d, text))

    candidates.sort(key=lambda row: row[0])
    seen: set[str] = set()
    out: list[str] = []
    for _, text in candidates:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_results:
            break
    return out


__all__ = ["collect_nearby_text"]
