"""Universal rule: a narrow caption pinned to the far-right title block
must search for column-header cells within its own x-range only — not
across half the page.

Symptom that motivates this rule
--------------------------------
``textsec_9`` in test5 is the small ``CODE / 2021 IMC / 2021 IPC`` strip
on the far right of the title block (image x=1794..1906, side fraction
0.93).  The detection's "sheet-reach wing" widened the colhdr-search
``_hx0/_hx1`` window LEFT by ~1089 px, putting it inside the pump
schedule's x-range.  The downstream colhdr emitter then matched pump
data-row cells (``BUILDING HEAT``, ``SNOWMELT INJECTION``) at the same
y-band and tagged them as cyan column headers.  The original code's own
comment said "never harvest from unrelated schedules" — but the
implementation did exactly that.

Inputs
------
Title geometry (image px) plus the page width.

Outputs
-------
The horizontal scope ``(_hx0, _hx1)`` to use when searching for column
header cells under this title.

Why universal
-------------
The discriminator is purely positional: a narrow caption's centre
fraction.  ``side >= 0.85`` (far right margin) is unambiguously inside
the title-block strip on any drawing — no real schedule caption sits
there.  Real schedule titles on the left/centre keep their original
sheet-reach.
"""
from __future__ import annotations

from typing import Tuple


def x_scope_for_textsec_caption(
    *,
    tx0: int,
    tx1: int,
    sx0: int,
    sx1: int,
    page_w: int,
    sheet_reach_min: int = 260,
    sheet_reach_max: int = 1500,
    sheet_reach_frac: float = 0.55,
    far_right_threshold: float = 0.85,
    right_side_threshold: float = 0.55,
    left_side_threshold: float = 0.45,
) -> Tuple[int, int]:
    """Return ``(hx0, hx1)`` — the horizontal scope to search under a
    narrow ``textsec_*_title`` caption sitting on a page-wide shell.

    The caller has already established that:

    - ``tx1 - tx0 <= 220``           (the caption is narrow), and
    - ``sx1 - sx0 > min(980, 0.72 * page_w)``  (the only containing
                                                 wrapper is page-wide).

    For those captions, the answer depends on horizontal position:

    - far right (side >= ``far_right_threshold``):
          confine strictly to the caption's own x-range.  This is a
          title-block sidebar/caption; widening would let it claim
          unrelated cells.
    - right (side >= ``right_side_threshold``):
          extend left by ``sheet_reach``, right by a small wing.
    - left (side <= ``left_side_threshold``):
          extend right by ``sheet_reach``, left by a small wing.
    - centred (otherwise):
          symmetric narrow wings around the caption's centre.
    """
    sheet_reach = max(sheet_reach_min,
                      min(int(page_w * sheet_reach_frac), sheet_reach_max))
    tw_t = tx1 - tx0
    tcx = 0.5 * (tx0 + tx1)
    side = tcx / max(float(page_w), 1.0)

    if side >= far_right_threshold:
        # Far-right title-block caption — strict scope.
        return tx0, tx1
    if side >= right_side_threshold:
        hx0 = max(sx0, int(tx0 - sheet_reach))
        hx1 = min(sx1, int(tx1 + max(120, int(tw_t * 3.2))))
        return hx0, hx1
    if side <= left_side_threshold:
        hx0 = max(sx0, int(tx0 - max(120, int(tw_t * 3.2))))
        hx1 = min(sx1, int(tx1 + sheet_reach))
        return hx0, hx1
    # centred
    wing = int(min(sheet_reach * 0.48, 0.42 * float(page_w)))
    hx0 = max(sx0, int(tcx - wing))
    hx1 = min(sx1, int(tcx + wing))
    return hx0, hx1
