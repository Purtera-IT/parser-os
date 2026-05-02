"""Universal rule: a section "owns" the horizontal band from its own
column to the start of the next column whose header sits in a
DIFFERENT vertical band.  All text within that expanded rectangle
belongs to the section, regardless of x-anchor.

Symptom that motivates this rule
--------------------------------
On test7, SECTION 3.1 - GENERAL is structured as:

  - a left column of bold/labelled keys (MAJOR OCCUPANCY..., GROSS
    BUILDING AREA..., BUILDING HEIGHT..., ...) at x≈154,
  - a right column of values (GROUP F3 - WAREHOUSE, 1,358 SM,
    2 STOREYS, FACING 3 STREETS, ...) at x≈316,
  - indented sub-blocks (MAIN FLOOR OCCUPANT LOAD, OFFICE: 9.3
    SM/PERSON, 450 SM / 9.3 = 48 PERSONS PERMITTED, ...) also at
    x≈316,
  - a bold sub-heading (BUILDING TOTAL = 123 PERSONS PERMITTED) at
    x≈316.

The body builder collects only spans matching the header's x-start
(x≈154), so just the left column lands in the body wrapper.  The
right column and all the sub-blocks are left out — naked text on
the page.

Sub-headings inside a section (BUILDING TOTAL... at x=316, y≈1186)
get their own column anchor in the body builder's pre-split, which
chops the section's reported ``col_hi`` down to ~164 pt.  Without an
override, the rule can't escape that tight bound.

The user's principle: *"this is section 1 — this should all be
boxed."*  Everything inside the section's vertical extent is part of
the section, whether it sits in the label column or to its right.

Inputs
------
- ``inside``: spans the body builder has already accepted (left
  column).
- ``all_spans``: every text span on the page.
- ``col_lo``, ``col_hi``: the section's column-axis extent AS
  REPORTED BY THE body builder.  ``col_lo`` is the left bound;
  ``col_hi`` is replaced when ``other_column_anchors`` is provided.
- ``sec_lo``, ``sec_hi``: section-axis extent (PDF pt).
- ``other_column_anchors``: list of ``(anchor_x, first_header_y)``
  for every OTHER section column on the page.  The effective
  ``col_hi`` is the leftmost anchor strictly greater than
  ``col_lo`` whose first_y is OUTSIDE ``[sec_lo, sec_hi]``.  When
  no qualifying anchor exists, falls back to ``page_extent_pt``.
- ``page_extent_pt``: page width (height when sideways) — final
  fallback.
- ``inter_col_gap_pt``: gap subtracted from the next-column anchor
  when computing eff_col_hi (default 4 pt).
- ``sideways``: orientation flag (matches the body builder's flag).

Outputs
-------
A new list with the original spans plus every span that falls inside
the expanded rectangle.  No mutation of inputs.

Why universal
-------------
The boundary is purely structural:

  - A sibling section blocks our sweep only if its header first
    appears OUTSIDE our y-range.
  - A sub-heading INSIDE our y-range belongs to us, even if it
    creates its own x-anchor.

Independent of which drawing or which language — this is the
universal convention for multi-column technical documents.

The earlier (now removed) ``line_tolerance_pt`` and
``max_value_sweep_pt`` parameters tried to add only "values on label
lines."  That missed sub-blocks like ``OFFICE: 9.3 SM/PERSON`` which
have no colon-label on their own line.  The current approach sweeps
the whole expanded rectangle — simpler and more complete.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

Span = Tuple[float, float, float, float]   # (x0, y0, x1, y1) in PDF pt


def _y_centre(s: Span) -> float:
    return 0.5 * (s[1] + s[3])


def _x_centre(s: Span) -> float:
    return 0.5 * (s[0] + s[2])


def add_value_spans_for_colon_labels(
    inside: List[Span],
    all_spans: Iterable[Span],
    *,
    col_lo: float,
    col_hi: float,
    sec_lo: float,
    sec_hi: float,
    sideways: bool = False,
    other_column_anchors: List[Tuple[float, float]] | None = None,
    page_extent_pt: float | None = None,
    inter_col_gap_pt: float = 4.0,
    max_horizontal_gap_pt: float = 80.0,
    # Reserved for backward-compat with earlier rule signature; unused.
    span_text_lookup=None,
    line_tolerance_pt: float = 2.5,
    max_value_sweep_pt: float = 350.0,
) -> List[Span]:
    """Return ``inside`` plus all section-content spans within the
    section's expanded rectangle.  See module docstring.

    ``max_horizontal_gap_pt`` (default 80 pt ~= 1.1 inch): when sweeping
    additions outward from the label column, stop accumulating once a
    horizontal gap of pure whitespace exceeds this threshold.  Catches
    the case where eff_col_hi is technically wide enough but our actual
    content ends well before another unrelated content cluster begins.
    """
    # Compute the effective right bound for the section sweep.
    eff_col_hi = col_hi
    if other_column_anchors:
        # Three cases for an anchor's vertical band relative to ours:
        #
        #   1. Fully OUTSIDE ours (last_y < sec_lo or first_y > sec_hi)
        #      → dormant; an unrelated section in a different vertical
        #      band, doesn't block us.
        #
        #   2. Fully INSIDE ours → SUB-SECTION of ours.  Doesn't block.
        #
        #   3. Otherwise (partial overlap or extends beyond ours) →
        #      peer/parent section.  BLOCKS.
        blocking = []
        for entry in other_column_anchors:
            if len(entry) == 3:
                a, fy, ly = entry
            else:
                a, fy = entry
                ly = fy
            if a <= col_lo + 1.0:
                continue
            # Case 1: dormant
            if ly < sec_lo - 0.5 or fy > sec_hi + 0.5:
                continue
            # Case 2: sub-section
            if sec_lo - 0.5 <= fy and ly <= sec_hi + 0.5:
                continue
            # Case 3: peer/parent → blocks
            blocking.append(a)
        if blocking:
            eff_col_hi = min(blocking) - inter_col_gap_pt
    # Never let the effective bound be tighter than what the body builder
    # gave us — the builder already chose col_hi >= label_far_edge.
    eff_col_hi = max(eff_col_hi, col_hi)

    # Existing in-body span IDs — don't re-add anything already inside.
    existing = {id(s) for s in inside}

    additions: List[Span] = []
    for s in all_spans:
        if id(s) in existing:
            continue
        sx0, sy0, sx1, sy1 = s
        cx, cy = _x_centre(s), _y_centre(s)
        if sideways:
            # column axis = y, section axis = x
            if not (col_lo <= cy <= eff_col_hi):
                continue
            if not (sec_lo <= cx <= sec_hi):
                continue
        else:
            # column axis = x, section axis = y (horizontal text)
            if not (col_lo <= cx <= eff_col_hi):
                continue
            if not (sec_lo <= cy <= sec_hi):
                continue
        additions.append(s)

    # ── Trim additions separated by a large horizontal gap ─────────────
    # The eff_col_hi sweep can over-reach when an unrelated content
    # column happens to fall in our y-range but is separated from our
    # actual content by visual whitespace (e.g. on test7 SECTION 3.4 -
    # EXITS at x=506..850 and the INTERIOR PARTITIONS table at
    # x=1230..1245 share the y-range 1459..1594; without this trim, we
    # pull ``S10`` and ``S11`` wall codes into SECTION 3.4's body).
    #
    # Build the rightmost edge of OUR known content (label column +
    # original inside spans).  Then walk additions left-to-right; stop
    # accumulating once we cross a horizontal gap exceeding
    # ``max_horizontal_gap_pt``.
    if additions:
        # Anchor: rightmost edge of label-column (inside) spans.
        # These came from the body builder's anchor-aligned filter.
        if sideways:
            base_far = max((s[3] for s in inside), default=col_lo)
        else:
            base_far = max((s[2] for s in inside), default=col_lo)
        # Sort additions outward (along the column axis).
        if sideways:
            additions.sort(key=lambda s: s[1])
        else:
            additions.sort(key=lambda s: s[0])
        kept: List[Span] = []
        last_far_edge = base_far
        for s in additions:
            if sideways:
                near = s[1]; far = s[3]
            else:
                near = s[0]; far = s[2]
            gap = near - last_far_edge
            if gap > max_horizontal_gap_pt:
                break    # whitespace gap — stop here
            kept.append(s)
            if far > last_far_edge:
                last_far_edge = far
        additions = kept

    return list(inside) + additions
