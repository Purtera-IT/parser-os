"""Universal rule: a non-bold colon-uppercase span is a section title
ONLY if it stands alone in its column. Stacked colon-uppercase spans
are field labels, not titles.

Symptom that motivates this rule
--------------------------------
On test7 the BUILDING CODE SUMMARY zone has columns of field labels
like ``CONSTRUCTION CLASSIFICATION:``, ``LIMITING DISTANCE:``,
``EXPOSING BUILDING FACE:``, ``CLADDING:`` — every one all-uppercase,
ends with a colon, ≥8 letters, but **none of them is bold**.  The
earlier classifier accepted any all-caps colon-ending span ≥8 letters
as a "header candidate" and 86 of them got blue title washes,
flooding the bottom-left quadrant with title boxes that aren't titles.

The same earlier classifier was also doing useful work on test5: the
solitary ``CERTIFICATE OF AUTHORIZATION NO:`` span on the right side
of the title block IS a section title (no other colon-uppercase peers
nearby).

Inputs
------
A list of header-candidate dicts each with at least ``"bbox"`` and
``"is_bold"``.  Bold candidates pass through unchanged — bold means
title.  Non-bold colon-uppercase candidates are subject to the
isolation test.

Outputs
-------
A filtered list of headers, with non-bold candidates that have any
similar-shape neighbour within the isolation radius removed.

Why universal
-------------
The discriminator is purely structural:

- Field labels in well-typeset drawings come in stacks (one per row,
  same x-anchor, ~one line-height apart).  N >= 2 in a column = field
  labels.
- Real section titles stand alone in their column.

This is independent of which drawing, which schedule, which language —
the typographic convention "title is solitary, label is in a stack"
is universal.

Tunable parameters
------------------
``y_isolation_pt``:  vertical radius around the candidate.  If any
                    OTHER non-bold candidate sits within this radius
                    AND has a similar x-anchor, the candidate is a
                    field label, not a title.  Default 80 pt covers
                    ~6 lines at 11.4 pt line-height (the test7 case).
``x_anchor_tol_pt``: horizontal tolerance for "same x-anchor."  Default
                    8 pt covers small typesetter drift.
"""
from __future__ import annotations

from typing import Any, List


def filter_isolated_colon_titles(
    headers: List[dict],
    *,
    y_isolation_pt: float = 80.0,
    x_anchor_tol_pt: float = 8.0,
) -> List[dict]:
    """Drop non-bold COLON-ENDING uppercase candidates that have
    similar-shape peers nearby.  Bold candidates and non-colon
    candidates pass through unchanged.

    The original problem this rule solves is "field labels that look
    like titles" — colon-ending all-caps spans that come in stacks
    (``LIMITING DISTANCE:``, ``EXPOSING BUILDING FACE:``, etc.) and
    aren't section titles.  The discriminator is the colon, not just
    "non-bold."  Without the colon restriction, this rule incorrectly
    filtered legitimate non-bold title stacks like ``NORTH WALL
    (STORAGE GARAGE)`` / ``NORTH WALL (OFFICE)`` / ``SOUTH WALL
    (EXISTING BUILDING)`` on test7 — those are real underlined-title
    sub-sections, identified by the underline rule, and should
    survive.

    Each header dict must have:
      - ``"text"``: str
      - ``"bbox"``: (x0, y0, x1, y1) in PDF points
      - ``"is_bold"``: bool

    Returns a NEW list with the same dicts (no mutation).
    """
    # Only non-bold colon-ending candidates are subject to the
    # isolation test.  Everything else (bold candidates, non-colon
    # underline-rule additions, etc.) passes through untouched.
    def _is_subject(h):
        if h.get("is_bold"):
            return False
        text = (h.get("text", "") or "").rstrip()
        return text.endswith(":")

    subject_candidates = [h for h in headers if _is_subject(h)]
    survivors: List[dict] = []
    for h in headers:
        if not _is_subject(h):
            survivors.append(h)
            continue
        bx0, by0, bx1, by1 = h["bbox"]
        cy = 0.5 * (by0 + by1)
        # Peer test: is there ANY OTHER subject candidate whose y-centre
        # falls within y_isolation_pt of this one AND whose x0 is within
        # x_anchor_tol_pt of this one's x0?
        has_peer = False
        for other in subject_candidates:
            if other is h:
                continue
            ox0, oy0, ox1, oy1 = other["bbox"]
            if abs(ox0 - bx0) > x_anchor_tol_pt:
                continue
            ocy = 0.5 * (oy0 + oy1)
            if abs(ocy - cy) <= y_isolation_pt:
                has_peer = True
                break
        if not has_peer:
            survivors.append(h)
    return survivors
