"""Universal rule: a contour-detected leaf box (id starting with ``v``)
with extreme flat aspect ratio is a thin-stroke artifact (drawn
underline, divider, decoration) — pure typography, not detected
structure.  Drop it from the box list entirely so the overlay doesn't
paint any color rectangle on top of it.

Symptom that motivates this rule
--------------------------------
On test7 the page contains thin black-stroke horizontal lines drawn
beneath certain inline headings (``MAIN FLOOR OCCUPANT LOAD``,
``SECOND FLOOR OCCUPANT LOAD``, ``NORTH WALL (...)``, ``EXTERIOR
WALL``, etc.).  The contour detector picks each up as a 5-10 pixel
strip with extreme flat aspect ratio.  These are pure typographic
underlines — part of how the heading text was set, not detected
content the overlay should annotate.

The classifier in core_002 promotes parentless contours to BLUE and
nests parented contours to ORANGE; in either case, painting a
colored rectangle on top of the underline is just visual noise.

The PDF text and its underline are already visible to the user under
the overlay, so we shouldn't paint anything on top of them
regardless of the box's parent state.

Inputs
------
A list of detected boxes (each with ``box_id``, ``px_bbox``,
``color``, ``synthetic``).

Outputs
-------
A new list with thin-stroke contour boxes removed entirely.
No mutation of inputs.

Why universal
-------------
The discriminator is purely geometric + structural-prefix:

- Real structural wrappers have substantial height (test5's smallest
  contour with low aspect is h=16; test5's gridcell_/tbcell_ thin
  strips are not contour artifacts and are exempt by id prefix).
- Thin stroke artifacts are h ≤ 10 with aspect ratio ≤ 0.1.

The ``v*`` id prefix restriction is structural — the contour
detector emits ``v0``, ``v1``, ... boxes for stroke-derived
geometry; OTHER detection passes (title-block ``tbcell_``,
grid-completion ``gridcell_``, mini-table ``mt*_``) emit boxes from
their own sources that may legitimately be thin (table dividers,
small numeric cells inside title blocks).  Restricting the rule to
``v*`` ids targets only the contour pass's stroke artifacts without
affecting other passes' outputs.

Tunable parameters
------------------
``max_h_px``: maximum image-pixel height for a stroke artifact.
              Default 10.
``max_aspect_ratio``: maximum height/width ratio for a stroke
              artifact.  Default 0.1.
"""
from __future__ import annotations

from typing import List


def reclassify_thin_stroke_artifacts(
    boxes: List,
    *,
    max_h_px: int = 6,
    max_aspect_ratio: float = 0.1,
) -> List:
    """Return a new list of boxes with contour-detected thin strokes
    REMOVED entirely.

    A box is removed if ALL of:
      - synthetic == False
      - box_id starts with ``v`` (contour-detector source)
      - color is BLUE or ORANGE
      - px_bbox height <= max_h_px (default 6 — clean gap below
        test5's smallest table cell at h=10 and just above the
        observed PDF underline rendering at h=5-6)
      - height/width aspect ratio <= max_aspect_ratio

    The rule applies regardless of whether the box has a parent —
    nested contour underlines (e.g. ORANGE underlines inside a larger
    BLUE wrapper of the wall-assemblies table on test7) are also
    artifacts and should be dropped.

    The function name is kept for backward compatibility; behaviour is
    drop-not-recolor.
    """
    out = []
    for b in boxes:
        bid = getattr(b, "box_id", "") or ""
        if (not getattr(b, "synthetic", False)
                and bid.startswith("v")
                and getattr(b, "color", None) in ("BLUE", "ORANGE")):
            x0, y0, x1, y1 = b.px_bbox
            w = x1 - x0
            h = y1 - y0
            ar = (h / w) if w > 0 else 1.0
            if h <= max_h_px and ar <= max_aspect_ratio:
                # Drop — don't paint over text underlines.
                continue
        out.append(b)
    return out


