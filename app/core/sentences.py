r"""Sentence segmentation that survives abbreviations.

The regex this replaces -- ``(?<=[.!?])\s+(?=[A-Z0-9"'])`` -- splits on any
period followed by a capital, which in procurement prose is wrong constantly.
On one paragraph of ordinary SOW text it found ten sentences where there are
six, and severed a part number from its own label::

    | Part No.
    | 77-K298 ships from St.
    | Louis.

``pysbd`` is the Golden Rules segmenter: a rule set built against a published
test suite of exactly these cases (titles, initials, geographic abbreviations,
enumerations, decimals). It is pure Python, has no model to download, and
costs microseconds.

Soft dependency, as with ``rapidfuzz`` in ``entity_resolution``: absent the
library the old regex still runs, so segmentation degrades rather than fails.
"""

from __future__ import annotations

import re

try:  # pragma: no cover - exercised by whichever environment lacks it
    import pysbd as _pysbd
except Exception:  # pragma: no cover
    _pysbd = None  # type: ignore[assignment]

#: The previous behaviour, kept as the fallback so a missing dependency is a
#: quality regression and never an exception.
_NAIVE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

_segmenter = None


def _get_segmenter():
    """One reusable segmenter. Construction dominates the cost of a split."""
    global _segmenter
    if _segmenter is None and _pysbd is not None:
        # clean=False keeps the text byte-identical to the input, which the
        # callers that build character offsets depend on.
        _segmenter = _pysbd.Segmenter(language="en", clean=False)
    return _segmenter


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping abbreviations intact."""
    if not text or not text.strip():
        return []
    seg = _get_segmenter()
    if seg is None:  # pragma: no cover - dependency-free fallback
        return [s for s in _NAIVE_SPLIT.split(text) if s.strip()]
    try:
        return [s for s in seg.segment(text) if s and s.strip()]
    except Exception:  # pragma: no cover - never fail a parse over segmentation
        return [s for s in _NAIVE_SPLIT.split(text) if s.strip()]


def count_sentences(text: str) -> int:
    """Sentence count, used as the denominator in coverage ratios."""
    return len(split_sentences(text))
