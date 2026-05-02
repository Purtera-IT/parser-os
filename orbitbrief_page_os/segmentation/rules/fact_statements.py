"""Universal rule: a bold uppercase line containing an equals sign is a
fact statement (a calculation result, summary, or assertion), not a
section title.  Drop it from the header candidates so it doesn't get
its own title-wash band.

Symptom that motivates this rule
--------------------------------
On test7 inside SECTION 3.1 - GENERAL there's a line:

    BUILDING TOTAL = 123 PERSONS PERMITTED

It's bold and uppercase like real section titles, so it passed the
``(is_bold and is_caps)`` gate in ``_candidate_headers`` and got its
own ``textsec_*_title`` blue band.  But it's not a section heading —
it's a summary fact line that belongs INSIDE SECTION 3.1's body
wrapper, not above its own sub-section.

The user's principle: *"this actually belongs in the box; this is not
its own title even though it's bolded — it's more a fact."*

Inputs
------
A list of header-candidate dicts each with at least ``"text"`` and
``"is_bold"``.  Non-bold candidates pass through unchanged
(orthogonal to Rule 6's isolation test for non-bold colon-titles).

Outputs
-------
A filtered list of headers, with bold candidates whose text contains
``=`` removed.

Why universal
-------------
The discriminator is purely typographic / lexical:

- Section titles are descriptive labels: ``MAIN FLOOR ASSEMBLY``,
  ``BOILER SCHEDULE``, ``SECTION 3.1 - GENERAL``, ``WATER SUPPLY:``,
  etc.  None contain ``=``.
- Calculation results / fact lines are formula-shaped: ``BUILDING
  TOTAL = 123 PERSONS PERMITTED``, ``MAIN FLOOR TOTAL = 75 PERSONS
  PERMITTED``, ``165 SM / 28 = 6 PERSONS PERMITTED`` etc.  All
  contain ``=``.

The ``=`` sign is structural notation: "X equals Y."  It's never used
as a section heading device — only to assert a value.  Independent of
which drawing or which language.

Verification on existing PDFs:
- test5: zero bold candidates contain ``=`` — rule has zero effect.
- test7: only ``BUILDING TOTAL = 123 PERSONS PERMITTED`` contains
  ``=``; correctly dropped.  Other 25 bold candidates preserved.
"""
from __future__ import annotations

from typing import List


def filter_fact_statements(headers: List[dict]) -> List[dict]:
    """Drop bold header candidates that contain ``=`` — these are
    fact statements (calculations, summaries), not section titles.

    Each header dict must have:
      - ``"text"``: str
      - ``"is_bold"``: bool

    Returns a NEW list with the same dicts (no mutation).
    """
    out = []
    for h in headers:
        text = h.get("text", "") or ""
        is_bold = bool(h.get("is_bold", False))
        if is_bold and "=" in text:
            # Fact statement, not a section title — drop.
            continue
        out.append(h)
    return out
