"""Universal rule: no orange grid edge cuts through a PDF text word.

Symptom that motivates this rule
--------------------------------
Merged column headers like ``ELECTRICAL DATA`` span two sub-columns
(``AMPS/WATTS`` | ``VOLTS/PH``).  The sub-column separator's vertical
raster line extends UP into the parent header band, and the naive grid
splitter would split the parent into two cells whose shared edge cuts
straight through the parent's text.  This rule blocks that emission.

Inputs
------
A candidate cell bbox in image pixels and an iterable of PDF word bboxes
(also in image pixels).  Word bboxes typically come from ``fitz`` and are
mapped through the detector's image rotation.

Outputs
-------
A boolean: True when the candidate's left or right edge would bisect a
word's horizontal span (with the word's vertical span overlapping the
candidate's vertical span).  Callers should drop the candidate when True.

Why universal
-------------
The discriminator is purely structural ("does an edge sit inside a word's
horizontal extent?"). It does not depend on table type, schedule kind, or
which column.  Any merged column header on any drawing is automatically
covered.
"""
from __future__ import annotations

from typing import Iterable, Tuple

WordBBox = Tuple[int, int, int, int]   # (x0, y0, x1, y1) in image pixels
CellBBox = Tuple[int, int, int, int]


def edge_bisects_word(
    edge_x: int,
    cand_y0: int,
    cand_y1: int,
    word: WordBBox,
    tol: int = 1,
) -> bool:
    """True when ``edge_x`` sits strictly inside ``word``'s x-span and the
    word's y-span overlaps the candidate's y-span.

    A small ``tol`` (default 1 px) ignores words that just touch the edge
    at a column boundary.
    """
    wx0, wy0, wx1, wy1 = word
    if wy1 <= cand_y0 or wy0 >= cand_y1:
        return False
    return (wx0 + tol) < edge_x < (wx1 - tol)


def candidate_bisects_any_word(
    candidate: CellBBox,
    words: Iterable[WordBBox],
    tol: int = 1,
) -> bool:
    """True when the candidate cell's left OR right edge bisects any word.

    Use this from cell-emission code: when True, drop the candidate.
    """
    cx0, cy0, cx1, cy1 = candidate
    for w in words:
        if edge_bisects_word(cx0, cy0, cy1, w, tol):
            return True
        if edge_bisects_word(cx1, cy0, cy1, w, tol):
            return True
    return False
