"""Universal rule: a title-washed cell is purely a title — no cyan ring,
no green ring inside it.

Symptom that motivates this rule
--------------------------------
``ABBREVIATIONS`` is a labelled section title.  The renderer paints a
blue title wash over its cell.  Without this rule, cyan ``colhdr_*`` and
green ``subhdr_*`` rings emitted upstream still draw on top of the wash,
making the title cell look like a column-header row.  The legend says
title-washed cells should read as titles only.

Inputs
------
The full set of detected boxes plus the ``_skip_title_alpha`` gate (so we
only consider title bands that *will actually be drawn*, not every
synthetic ``*_title`` box).

Outputs
-------
A list of "drawn title" bboxes and a centroid-in helper.  Renderers call
the helper before painting cyan or green and skip the draw on a hit.

Why universal
-------------
The discriminator is purely positional: "ring centroid sits inside a
drawn title band."  No per-table rule, no text inspection.  Any future
title-detection improvements automatically extend the suppression.
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

BBox = Tuple[int, int, int, int]


def drawn_title_bboxes(
    all_boxes: List[Any],
    skip_title_alpha: Callable[[Any, dict], bool],
) -> List[BBox]:
    """Return bboxes for every BLUE ``*_title`` synthetic that *will draw*.

    The same ``skip_title_alpha`` gate used by the title-wash renderer is
    applied here so that only titles which actually receive a wash are
    considered for ring suppression.

    ``box_by_id`` covers BOTH synthetic and regular boxes — the parent
    look-up inside ``skip_title_alpha`` checks whether the title's parent
    wrapper extends meaningfully below the title.
    """
    box_by_id = {b.box_id: b for b in all_boxes}
    out: List[BBox] = []
    for tb in all_boxes:
        if not getattr(tb, "synthetic", False):
            continue
        if not str(getattr(tb, "box_id", "")).endswith("_title"):
            continue
        if getattr(tb, "color", None) != "BLUE":
            continue
        try:
            if not skip_title_alpha(tb, box_by_id):
                out.append(tb.px_bbox)
        except Exception:
            # Defensive: if the gate raises, don't suppress (no worse than
            # behaviour before this rule existed).
            pass
    return out


def centroid_in_drawn_title(
    bbox: BBox,
    drawn_titles: List[BBox],
) -> bool:
    """True when ``bbox``'s centroid lies inside any drawn title band.

    Use this immediately before painting a cyan ``colhdr_*`` or green
    ``subhdr_*`` ring; skip the draw on a True return.
    """
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    for (tx0, ty0, tx1, ty1) in drawn_titles:
        if tx0 < cx < tx1 and ty0 < cy < ty1:
            return True
    return False
