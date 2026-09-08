"""Pairs of site rows that look like one place, for a person to judge.

Three separate attempts to merge sites automatically have been wrong often
enough to be dangerous. Merging is destructive: it deletes a location and
changes the project tier, which is scored on site count. Measured on a
140-envelope corpus sample (2026-09-07), a distinctive-token rule agreed with
a human about half the time — `ACC OP Lane` matched `50 W Lane Avenue` on the
word "Lane".

Half right is useless for an automatic merge and perfectly good for a
SHORTLIST. So this proposes candidates and decides nothing. A PM answers once,
`same_site` learns the shape, and the next deal's identical pair resolves
without asking.

The exemplar is the point of this module. Whatever asks the head and whatever
teaches it must use the SAME string, or a lesson is banked under a key nothing
ever looks up — which is the state the site head was already in: the Deal
Artifacts chip taught on "Palo Alto Office (alias, alias)" while the fusion
pass asked with "symphonyai hillview office || palo alto ca 94304".
"""

from __future__ import annotations

import re
from typing import Any

#: A pair is only worth a person's attention when one side has no location of
#: its own. Two anchored sites with different addresses are two sites, and
#: asking about them trains people to click through.
_MAX_CANDIDATES = 12


def _norm_tokens(text: Any) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if len(t) > 2}


def _row_phrase(row: dict[str, Any]) -> str:
    """One site as a person would describe it: name, street, then city/state.

    The slug alone ("palo alto ca 94304") hides the very evidence the judgement
    turns on — that 3300 Hillview Ave is this site's address, and the other row
    is called "Hillview Office".
    """
    name = str(row.get("facility_name") or row.get("display_name") or "").strip()
    if not name:
        name = str(row.get("site") or "").split(":", 1)[-1].replace("_", " ").strip()
    street = str(row.get("street_address") or row.get("address") or "").strip()
    locality = ", ".join(
        p for p in (str(row.get("city") or "").strip(), str(row.get("state") or "").strip()) if p
    )
    return " — ".join(p for p in (name, street, locality) if p)


def pair_exemplar(a: dict[str, Any], b: dict[str, Any]) -> str:
    """The one string this pair is asked and taught with.

    Ordered by site key so the same two rows produce the same exemplar however
    they were enumerated — an unordered pair that embeds two different ways is
    two lessons about one judgement.
    """
    first, second = sorted((a, b), key=lambda r: str(r.get("site") or ""))
    return f"{_row_phrase(first)} || {_row_phrase(second)}"


def _is_located(row: dict[str, Any]) -> bool:
    return any(
        str(row.get(f) or "").strip()
        for f in ("address", "street_address", "city", "state", "zip", "postal_code")
    )


def site_duplicate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pairs worth asking a person about, best evidence first.

    A candidate is an UNLOCATED row — no address, no city, not anchored, so it
    is a name that matched nothing — paired with the one located row whose
    address contains a token from that name. Requiring exactly one owner is
    what keeps it a question rather than a guess: a name that could belong to
    three sites is not evidence of anything.
    """
    located = [r for r in rows if isinstance(r, dict) and _is_located(r)]
    unlocated = [
        r for r in rows
        if isinstance(r, dict) and not _is_located(r) and not r.get("anchored")
    ]
    if not located or not unlocated:
        return []

    # A token is distinctive when exactly one located row's address carries it.
    owners: dict[str, list[dict[str, Any]]] = {}
    for row in located:
        for token in _norm_tokens(row.get("address") or row.get("street_address")):
            owners.setdefault(token, []).append(row)

    out: list[dict[str, Any]] = []
    for orphan in unlocated:
        name_tokens = _norm_tokens(orphan.get("facility_name") or orphan.get("site"))
        matched: dict[str, tuple[dict[str, Any], str]] = {}
        for token in name_tokens:
            candidates = owners.get(token) or []
            if len(candidates) != 1:
                continue
            row = candidates[0]
            matched[str(row.get("site"))] = (row, token)
        if len(matched) != 1:
            continue
        row, token = next(iter(matched.values()))
        out.append({
            "unlocated": str(orphan.get("site") or ""),
            "located": str(row.get("site") or ""),
            "exemplar": pair_exemplar(orphan, row),
            "shared_token": token,
            "why": (
                f"{_row_phrase(orphan)!r} has no address of its own, and "
                f"{token!r} appears in exactly one site's address on this deal."
            ),
        })
    # Most evidence first: a longer shared token is a stronger coincidence to
    # rule out than a short one.
    out.sort(key=lambda c: (-len(c["shared_token"]), c["unlocated"]))
    return out[:_MAX_CANDIDATES]
