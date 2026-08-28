"""Is this thing a place? Two deterministic rejections, shared by every minter.

A ``physical_site`` atom is the most trust-destroying thing this pipeline can
get wrong. A PM who opens a brief and finds ``1 16u wall mounted`` listed as a
job site concludes the software does not know what a site is, and stops using
it — which also ends the correction loop that would have taught it better.

Both rules were found in production on 2026-08-27 (deal 222b2173):

* **Equipment is not a place.** The scope line "install 1 vendor supplied 16u
  wall mounted hinged data rack with fan in the manager office" minted a site
  named ``1 16u wall mounted``. Alongside it: ``12 pair of pendant speakers``,
  ``3 sonance pendant subwoofer``.
* **Recycled pipeline output is not evidence of a place.** Every one of those
  atoms traced to ``context.prior_scope_process_v1...`` — the compile's own
  prior handoff, carried in the manifest and re-read as if a customer wrote it.
  ``atom_type_sanity.cap_authority_to_source`` demotes such atoms to rank 40,
  but capping authority never stopped them existing as places.

These live here, not inside one extractor, because sites are minted from at
least two independent paths (``entity_extraction``'s site_clusters bridge and
``site_atom_backfill``'s entity backfill) and a rule that guards only one of
them is a rule that does not hold. The first version of this fix guarded only
the first path and the bad sites kept appearing.

Both rules refuse to over-reject: a place noun redeems an equipment-adjacent
phrase, so "Data Center 3", "Cabinet Room" and "Server Room" survive.
"""
from __future__ import annotations

import re
from typing import Any

#: A rack-unit measurement ("16u", "12 U") is a hardware dimension.
_RACK_UNIT = re.compile(r"\b\d+\s*u\b", re.IGNORECASE)

#: Hardware nouns. Deliberately narrow: only things that are unambiguously
#: equipment, never words that could name a facility on their own.
_EQUIPMENT = re.compile(
    r"\b(rack|switch|router|firewall|patch\s*panel|cable|conduit|raceway|"
    r"ups|pdu|access\s*point|camera|display|monitor|kiosk|printer|"
    r"workstation|laptop|tablet|server|appliance|enclosure|cabinet|"
    r"speaker|subwoofer|amplifier|projector|antenna|sensor)s?\b",
    re.IGNORECASE,
)

#: Place nouns that redeem an equipment-adjacent phrase. A real facility may
#: legitimately be called "Server Room" or "Data Center 3".
_PLACE = re.compile(
    r"\b(office|store|center|centre|campus|plant|warehouse|branch|"
    r"school|hospital|clinic|hotel|room|floor|suite|hq|headquarters|"
    r"terminal|depot|yard|lab|studio|dealership|showroom|building|"
    r"facility|site|location|park|tower|annex|pavilion)s?\b",
    re.IGNORECASE,
)

#: The manifest ``context`` payload flattened into atom text. Anchored, so
#: prose that merely contains the word "context" is untouched.
_SERIALIZED_SOURCE = re.compile(r"^context\.[A-Za-z_]")


def _clean(s: Any) -> str:
    return str(s or "").replace(" ", " ")


def has_equipment_token(name: Any) -> bool:
    """True when the phrase names hardware at all (before redemption)."""
    s = _clean(name)
    return bool(_RACK_UNIT.search(s) or _EQUIPMENT.search(s))


def is_equipment_shaped(name: Any) -> bool:
    """True when the phrase is equipment and nothing redeems it as a place."""
    s = _clean(name)
    return bool(s) and not _PLACE.search(s) and has_equipment_token(s)


def is_serialized_source(text: Any) -> bool:
    """True when the text is the pipeline's own serialized prior output."""
    return bool(_SERIALIZED_SOURCE.match(_clean(text).lstrip()))


def rejects_as_site(name: Any = "", *texts: Any) -> str:
    """The one call every site minter should make.

    Returns a short reason string when this must NOT become a
    ``physical_site``, or ``""`` when it may. Returning the reason rather
    than a bool keeps the rejection recordable — a silently dropped
    candidate and a candidate that was never seen must not look the same.
    """
    if is_equipment_shaped(name):
        return "equipment_not_a_place"
    for t in (name, *texts):
        if is_serialized_source(t):
            return "recycled_pipeline_output"
    return ""
