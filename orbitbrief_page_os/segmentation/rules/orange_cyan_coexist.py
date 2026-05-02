"""Universal rule: orange cell outline and cyan column-header ring
coexist on column-header cells — never suppress orange because cyan
exists.

Symptom that motivates this rule
--------------------------------
The earlier renderer suppressed the orange cell outline whenever a
``colhdr_*`` ring landed inside it ("headers read as CYAN only").  The
legend says otherwise: ORANGE = cell outline, CYAN = column-header word
ring — they describe DIFFERENT things on the SAME cell.  Without both
layers, header cells looked visually inconsistent with non-header cells
(only headers had no orange ring).  Per the user's principle, all cells
get the orange outline; cyan rings additionally mark column keys.

Inputs
------
None — the rule is "always render orange, regardless of cyan."

Outputs
-------
A constant True.  Kept as a function so that any future need to gate
this universally has one place to edit.

Why universal
-------------
Render order naturally handles the layering: orange paints in the
orange pass, cyan paints later in the colhdr pass on top.  Both visible.
Same treatment everywhere.
"""
from __future__ import annotations


def orange_should_render_with_colhdr(*_unused) -> bool:
    """Always True — orange outlines and cyan rings coexist universally.

    Accepts (and ignores) any positional arguments so it slots in cleanly
    where the earlier ``_orange_suppressed_by_colhdr`` predicate took the
    candidate's bbox.  Returning True means "orange should render" — i.e.
    the suppression returns ``not orange_should_render_with_colhdr(...)``,
    which is ``False`` (don't suppress).
    """
    return True
