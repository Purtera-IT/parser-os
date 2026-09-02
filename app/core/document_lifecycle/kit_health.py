"""Is this Deal Kit actually this deal's kit?

Deal 010215's kit is a copy of another deal's. Its header still says:

    OPPTY #        131          <- deal 000131, not 010215
    QTY of Sites   TBD
    Date           2026-06-03   <- ten weeks before 010215 existed

and every summary formula reads ``#REF!``, because whoever repurposed it deleted
the line-items table those SUMIFs pointed at and never repopulated it. The
Summary sheet's QTY column is empty; Materials has 186 catalogue rows with no
Order QTY.

None of that is a parser problem -- the parser reported the cells honestly. It
is a defect in the document, and the kind worth telling a PM about: small,
specific, and fixable in minutes. A kit that silently belongs to another deal is
how one deal's pricing walks into another's quote.

Deliberately narrow. It reports what the file SAYS against what the deal IS. It
does not guess whether a kit is "finished" or grade its quality -- those need
judgement this cannot supply, and a noisy checker is one people learn to ignore.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

#: "OPPTY #", "Oppty#", "Opportunity Number" -- the field naming the deal.
_OPPTY_LABEL_RE = re.compile(r"^\s*oppty\s*#?|^\s*opportunity\s*(number|#)", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")
#: Excel's own error values, read back as text by the parser.
_EXCEL_ERROR_RE = re.compile(r"#(REF|DIV/0|VALUE|NAME\?|N/A|NULL)!?", re.IGNORECASE)


def _norm_deal_key(value: str | None) -> str:
    """010215, 10215 and 'OPPTY # 010215' all reduce to the same key."""
    digits = _DIGITS_RE.findall(str(value or ""))
    return digits[0].lstrip("0") if digits else ""


def check_deal_kit(
    *,
    atoms: Iterable[Mapping[str, Any]],
    deal_number: str | None,
) -> dict[str, Any]:
    """Findings for one Deal Kit. Empty ``findings`` means nothing to say.

    ``atoms`` are the kit's atoms as ``{atom_type, text}``.
    """
    mine = _norm_deal_key(deal_number)
    findings: list[dict[str, str]] = []
    claimed: str | None = None
    broken: list[str] = []

    for atom in atoms:
        text = str((atom or {}).get("text") or "")
        label, _, value = text.partition(":")
        if claimed is None and _OPPTY_LABEL_RE.match(label):
            key = _norm_deal_key(value)
            if key:
                claimed = key
        if _EXCEL_ERROR_RE.search(text):
            head = text.split("|", 1)[0].strip()[:60]
            if head and head not in broken:
                broken.append(head)

    # The kit names a different deal. This is the one that matters: a kit
    # belonging to another deal is how that deal's pricing walks into this quote.
    if claimed and mine and claimed != mine:
        findings.append({
            "kind": "kit_names_another_deal",
            "detail": f"The kit's OPPTY field says {claimed}, but this deal is {mine}.",
            "fix": "Rebuild the kit from the template rather than copying another deal's.",
        })

    # Broken formulas: the totals are unreadable, so the kit cannot state a
    # value even to a person.
    if broken:
        findings.append({
            "kind": "broken_formulas",
            "detail": f"{len(broken)} summary field(s) read as an Excel error: {', '.join(broken[:4])}.",
            "fix": "The referenced range was deleted; repoint the SUMIFs at the line-item table.",
        })

    return {
        "claimed_deal": claimed,
        "deal": mine or None,
        "findings": findings,
        "healthy": not findings,
    }
