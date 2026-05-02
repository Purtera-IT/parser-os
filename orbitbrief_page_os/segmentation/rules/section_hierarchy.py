"""Universal rule: a header matching the ``SECTION X.Y - NAME`` pattern
is a SECTION PARENT.  Any titles in the same column whose y falls
between the parent and the next section parent are CHILD titles
nested inside the parent.  The parent's body wrapper must enclose
all child titles + child bodies, and child titles render with a
distinct visual style so they read as sub-headers, not peer titles.

Symptom that motivates this rule
--------------------------------
On test7 ``SECTION 3.3 - SPATIAL SEPARATION`` (PDF y=1402) is
followed in the same column by 3 child sub-headings:

  - ``NORTH WALL (STORAGE GARAGE)`` (y=1413)
  - ``NORTH WALL (OFFICE)`` (y=1515)
  - ``SOUTH WALL (EXISTING BUILDING)`` (y=1618)

Each child has its own 7-row label/value content block.

The user's principle: *"all of this is SECTION 3.3, those titles are
sub-headers under it.  Make a universal rule about that.  Use a
different color (e.g. cyan) for sub-header titles so they're
visually distinguishable, and the whole thing should be in a single
SECTION 3.3 box."*

Currently SECTION 3.3's body wrapper (``textsec_N_body``) is sized
only to its own title (~26 px tall) because each wall sub-section
immediately claims its own column-anchor body.  The parent never
"sees" the children as inside-content, so the visual hierarchy is
flat: SECTION 3.3 looks like a peer of NORTH WALL (...) etc., when
in fact it should be their parent.

Discriminator
-------------
A header is a **section parent** iff its text matches the regex
``^SECTION\\s+\\d+(\\.\\d+)*\\b`` (case-insensitive).  This is the
universal numbered-section convention: ``SECTION 3``, ``SECTION 3.3``,
``SECTION 3.2.4``, etc.  Plain descriptive titles like ``EAST WALL``
or ``ABBREVIATIONS:`` do NOT match.

A header is a **child of a parent P** iff:

1. P is a section parent.
2. The header is at the same x-anchor as P (within ``x_anchor_tol_pt``,
   default 8 pt).
3. The header's y is strictly greater than P's y AND strictly less
   than the next section parent in the same column (or page bottom
   if P is the last parent in its column).
4. The header itself is NOT a section parent (otherwise it's a peer,
   not a child).

Outputs
-------
A list of (parent_idx, child_idx) tuples giving the parent→child
links between header indices in the input list.  The caller uses
these to:

- Resize the parent's body wrapper to enclose all child titles
  and child bodies.
- Mark child title boxes with a flag that the renderer uses to
  draw them in a sub-header style (different color or stroke).

Why universal
-------------
The "SECTION X.Y" pattern is a well-established universal numbering
convention used in technical documents, codes, and specs across
domains and languages (the keyword "SECTION" might vary across
languages but the structural pattern of a numbered prefix followed
by a name is universal; this rule explicitly targets the English
"SECTION" but can be extended trivially with locale-specific
keywords without changing the discriminator's logic).

Verification on existing PDFs
-----------------------------
- test5: zero candidates match ``^SECTION\\s+\\d`` — no section-style
  headers exist on test5's mechanical schedules page.  Rule has zero
  effect; test5 byte-identical.
- test7 left column: SECTION 3.3 - SPATIAL SEPARATION at y=1402 owns
  3 children: NORTH WALL (STORAGE GARAGE), NORTH WALL (OFFICE),
  SOUTH WALL (EXISTING BUILDING).
- test7 right column: SECTION 3.4 - EXITS: at y=1459 owns 1 child:
  EXIT CAPACITY at y=1527.
"""
from __future__ import annotations

import re
from typing import List, Tuple

# Match "SECTION 3", "SECTION 3.3", "SECTION 3.2.4", etc.
_SECTION_PARENT_RE = re.compile(r"^SECTION\s+\d+(\.\d+)*\b", re.IGNORECASE)


def is_section_parent(text: str) -> bool:
    """Return True if ``text`` matches the universal section-parent
    pattern ``SECTION <number>[.<sub>]*``.
    """
    if not text:
        return False
    return bool(_SECTION_PARENT_RE.match(text.strip()))


def find_parent_child_links(
    headers: List[dict],
    *,
    x_anchor_tol_pt: float = 8.0,
) -> tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return ``(structural_links, style_links)`` parent-child link lists.

    Each header dict must have ``"text"`` and ``"bbox"``.

    **Structural links** (``structural_links``): same-column children
    whose y is between the parent and the next section parent.  Used
    to (1) expand the parent's body to enclose them and (2) style the
    child title as a sub-header.

    **Style links** (``style_links``): cross-column orphan children at
    the TOP of a column with no SECTION parent above them.  Linked to
    the LAST section parent of the previous column.  Used ONLY for
    sub-header styling — NOT for body expansion, because expanding the
    parent body to enclose cross-column children would create a giant
    rectangle that overlaps with peer sections in the next column's
    middle.  Cross-column children get the same green wash so the
    visual linkage is clear, but each retains its own body wrapper at
    its actual location.

    On test7:
    - Structural: SECTION 3.3 → 3 NORTH wall children (left col),
      SECTION 3.2.5 → WATER SUPPLY:, SECTION 3.4 → EXIT CAPACITY
    - Style: SECTION 3.3 → EAST/SOUTH/WEST WALL (right col top)

    Note: additional sub-headers may be marked post-emission via
    geometric body-containment (any text-section whose ``_body`` is
    fully inside another text-section's ``_body`` is treated as a
    sub-header — this catches cases like OCCUPANT LOAD entries that
    are inside SECTION 3.1's expanded body but aren't structural
    children of any SECTION parent because their column has no
    SECTION X.Y header).  See `text_section_detection.py`.
    """
    indexed = [(i, h) for i, h in enumerate(headers)]

    def _col_key(h: dict) -> float:
        bx0 = h["bbox"][0]
        return round(bx0 / x_anchor_tol_pt) * x_anchor_tol_pt

    cols: dict[float, list[tuple[int, dict]]] = {}
    for i, h in indexed:
        cols.setdefault(_col_key(h), []).append((i, h))

    for k in cols:
        cols[k].sort(key=lambda ih: ih[1]["bbox"][1])

    structural_links: List[Tuple[int, int]] = []
    style_links: List[Tuple[int, int]] = []

    # --- Same-column structural links (Rule 12 base case) ---
    for k, entries in cols.items():
        for parent_pos, (parent_i, parent_h) in enumerate(entries):
            if not is_section_parent(parent_h["text"]):
                continue
            next_parent_pos = len(entries)
            for j in range(parent_pos + 1, len(entries)):
                if is_section_parent(entries[j][1]["text"]):
                    next_parent_pos = j
                    break
            for j in range(parent_pos + 1, next_parent_pos):
                child_i, child_h = entries[j]
                if is_section_parent(child_h["text"]):
                    continue
                structural_links.append((parent_i, child_i))

    # --- Cross-column style-only links (Rule 12 extension) ---
    sorted_col_keys = sorted(cols.keys())
    for col_idx, k in enumerate(sorted_col_keys):
        entries = cols[k]
        first_section_pos = None
        for j, (_i, h) in enumerate(entries):
            if is_section_parent(h["text"]):
                first_section_pos = j
                break
        if first_section_pos is None or first_section_pos == 0:
            continue
        orphan_prefix = entries[:first_section_pos]
        prev_parent_i: int | None = None
        for prev_idx in range(col_idx - 1, -1, -1):
            prev_k = sorted_col_keys[prev_idx]
            prev_parents = [
                (i, h) for i, h in cols[prev_k]
                if is_section_parent(h["text"])
            ]
            if prev_parents:
                prev_parents.sort(key=lambda ih: ih[1]["bbox"][1])
                prev_parent_i = prev_parents[-1][0]
                break
        if prev_parent_i is None:
            continue
        for child_i, _h in orphan_prefix:
            link = (prev_parent_i, child_i)
            if (link not in structural_links
                    and link not in style_links):
                style_links.append(link)

    return structural_links, style_links
