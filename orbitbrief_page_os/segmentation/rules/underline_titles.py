"""Universal rule: a non-bold all-uppercase span with a drawn underline
beneath it is a section title — even if it's not bold and doesn't end
in a colon.

Symptom that motivates this rule
--------------------------------
On test7 the SECTION 3.3 - SPATIAL SEPARATION block has many
sub-sections each headed by an underlined wall-name label:

  - ``EAST WALL`` / ``SOUTH WALL`` / ``WEST WALL`` (right column)
  - ``NORTH WALL (STORAGE GARAGE)`` / ``NORTH WALL (OFFICE)`` /
    ``SOUTH WALL (EXISTING BUILDING)`` (left column)
  - ``EXIT CAPACITY`` (in SECTION 3.4)

None of these are bold.  None end in a colon.  Each is followed by 7
rows of label/value content (LIMITING DISTANCE, EXPOSING BUILDING
FACE, etc.) that semantically belongs to it.  The earlier classifier
rejected them all (they fail ``(is_bold and is_caps)`` and
``(is_caps and ends_colon...)``), so their content is left as naked
unboxed text on the page.

The visual cue used by the typesetter is a **drawn horizontal
underline** beneath the title text.  This is the universal convention
for "I'm a heading; I'm not bold but I'm structural" in technical
documents.

Discriminator
-------------
A non-bold uppercase span is a section title (via underline) iff:

1. It has ≥5 alphabetic letters.
2. A horizontal stroke drawing exists in the PDF whose y is within
   3 pt of the span's bottom edge AND whose x range overlaps the
   span's x range.
3. It is NOT italic — italic underlined caps is typically a NOTE or
   emphasised sentence, not a heading (e.g. ``REFER TO EXTERIOR
   ELEVATIONS FOR SIDING ORIENTATION.`` on test7).
4. It is the ONLY such underlined span on its y-line (within 3 pt
   tolerance).  Multiple underlined caps spans on the same line are
   table column headers (e.g. ``TYPE | WALL | HEIGHT | FRR | STC |
   LB`` on test7's wall-assemblies tables), not section titles.

Why universal
-------------
Each gate is a structural / typographic property:

- Underlined caps + isolation = canonical section heading typography.
- Italic exclusion catches NOTE lines that wrap an underlined caps
  emphasis but aren't structural headings.
- Multi-span clusters identify table-column-header rows by their
  geometry, not by content.

Independent of which drawing or which language.

Verification on existing PDFs
-----------------------------
- test5: zero drawn underlines beneath caps spans on the relevant
  page — rule fires zero times.  Test5 byte-identical.
- test7: 11 candidates are added (all 6 wall labels + 2 OCCUPANT
  LOAD labels + EXIT CAPACITY + ABBREVIATIONS: + GENERAL NOTES:).
  The 3 colon-ending ones (ABBREVIATIONS:, GENERAL NOTES:,
  ABBREVIATIONS:) were already candidates via the colon-title gate;
  the 8 others are new.
"""
from __future__ import annotations

from typing import Any, List, Tuple


def collect_underlined_caps_titles(
    page,
    *,
    min_letters: int = 5,
    underline_y_tol_pt: float = 3.0,
    underline_width_ratio_max: float = 1.5,
    isolation_y_tol_pt: float = 3.0,
) -> List[dict]:
    """Return a list of header-candidate dicts for non-bold uppercase
    spans that have a drawn underline beneath them and pass the
    isolation test.

    Each returned dict has the same shape as ``_candidate_headers``
    produces:
        {"text": str, "bbox": (x0,y0,x1,y1),
         "font": str, "is_bold": bool, "is_caps": bool}

    Page must be a PyMuPDF ``Page`` (so we can call ``get_drawings()``
    and ``get_text("dict")``).

    The ``underline_width_ratio_max`` discriminator (default 1.5) is
    the key gate that distinguishes section titles from table column
    headers.  A real section title's underline is sized to the title
    text (ratio ≈ 1.00).  A table column header's "underline" is
    actually the column divider drawn across the full column width,
    which is much wider than the text — ratio 4-14× on test7.
    """
    # Collect thin horizontal stroke drawings (PDF-drawn underlines).
    strokes: List[Tuple[float, float, float, float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    for d in drawings:
        if d.get("type") != "s":   # stroked
            continue
        rect = d.get("rect")
        if rect is None:
            continue
        rx0, ry0, rx1, ry1 = rect.x0, rect.y0, rect.x1, rect.y1
        if (rx1 - rx0) < 30:
            continue
        if (ry1 - ry0) > 3:
            continue
        strokes.append((rx0, ry0, rx1, ry1))

    if not strokes:
        return []

    # First pass: collect every non-bold uppercase span that has a
    # text-tight underline beneath it.
    td = page.get_text("dict")
    candidates: List[dict] = []
    for block in td.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text", "") or "").strip()
                if not text:
                    continue
                bbox = span.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                bx0, by0, bx1, by1 = bbox
                if bx1 <= bx0 or by1 <= by0:
                    continue
                font = span.get("font", "") or ""
                flags = int(span.get("flags", 0))
                is_bold = bool(flags & 16) or any(
                    tag in font for tag in ("Bold", "bold", "Black", "Heavy"))
                is_italic = bool(flags & 2) or "Italic" in font or "italic" in font
                if is_bold or is_italic:
                    continue
                letters = [c for c in text if c.isalpha()]
                if len(letters) < min_letters:
                    continue
                if not all(c.isupper() for c in letters):
                    continue
                # Find an underline beneath this span whose width is
                # tight to the text (ratio ≤ underline_width_ratio_max).
                # Wider underlines are column dividers, not title
                # underlines.
                text_w = bx1 - bx0
                has_title_underline = False
                for sx0, sy0, sx1, sy1 in strokes:
                    if abs(by1 - sy0) > underline_y_tol_pt:
                        continue
                    if sx1 <= bx0 or sx0 >= bx1:
                        continue
                    under_w = sx1 - sx0
                    if text_w <= 0:
                        continue
                    ratio = under_w / text_w
                    if ratio > underline_width_ratio_max:
                        continue
                    has_title_underline = True
                    break
                if not has_title_underline:
                    continue
                candidates.append({
                    "text": text,
                    "bbox": (bx0, by0, bx1, by1),
                    "font": font,
                    "is_bold": False,
                    "is_caps": True,
                })

    if not candidates:
        return []

    # Second pass: isolation test — keep only candidates that are alone
    # on their y-line.  Multiple candidates on the same line (within
    # isolation_y_tol_pt) are table column headers, not titles.
    survivors: List[dict] = []
    for c in candidates:
        cy = 0.5 * (c["bbox"][1] + c["bbox"][3])
        peers = 0
        for o in candidates:
            if o is c:
                continue
            oy = 0.5 * (o["bbox"][1] + o["bbox"][3])
            if abs(cy - oy) <= isolation_y_tol_pt:
                peers += 1
                if peers > 0:
                    break
        if peers == 0:
            survivors.append(c)
    return survivors
