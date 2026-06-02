"""Geographic fallback site extractor.

Some deals never name a street address or a facility ("ATL-HQ-01",
"Memorial Hospital") — the only locational anchor is a bare
``City, ST ZIP`` buried in a notes file. The Yonah deal is the canonical
case: ``location Santa Fe, NM 87506`` sits in Notes.pdf, no street
address anywhere, so the regular site detectors find nothing, zero
``physical_site`` atoms are emitted, ``site_readiness`` is empty, and the
brief goes RED with "no confirmed physical site" while the 15%
site-readiness score component sits at 0.

This module is a *fallback*: it runs only when no real ``physical_site``
atom exists, scans every atom for a ``City, ST ZIP`` anchor, and emits a
single low-confidence ``physical_site`` atom (flagged ``geo_fallback_site``,
``needs_review``) per distinct ZIP so the deal has a locational anchor
the PM can confirm — instead of a blank RED. Pure function, no I/O, no
LLM.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

_US_STATES: frozenset[str] = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
})

# "Santa Fe, NM 87506" / "Santa Fe NM 87506" — multiword title-case city,
# 2-letter state, 5(+4) ZIP. City is 1-4 capitalized tokens.
_CITY_STATE_ZIP_RE = re.compile(
    r"\b([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3}),?\s+"
    r"([A-Z]{2})\s+(\d{5})(?:-\d{4})?\b"
)

_MAX_FALLBACK_SITES = 5


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _has_real_site(atoms: list[Any]) -> bool:
    """A ``physical_site`` atom carrying an explicit id/site_id already
    anchors the deal — don't second-guess it with a geo guess."""
    for a in atoms:
        if _atom_type_str(a) != "physical_site":
            continue
        val = getattr(a, "value", None) or {}
        if isinstance(val, dict) and (val.get("id") or val.get("site_id")):
            return True
    return False


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def geo_fallback_sites(
    atoms: list[Any], *, project_id: str
) -> list[EvidenceAtom]:
    """Emit fallback ``physical_site`` atoms from ``City, ST ZIP`` anchors.

    Returns an empty list when a real site already exists or when no
    valid geographic anchor is found, so it never competes with genuine
    site detection.
    """
    if not atoms or _has_real_site(atoms):
        return []

    seen_zip: set[str] = set()
    out: list[EvidenceAtom] = []
    for atom in atoms:
        text = getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
        if not text:
            continue
        for m in _CITY_STATE_ZIP_RE.finditer(str(text)):
            city, state, zipc = m.group(1).strip(), m.group(2).upper(), m.group(3)
            if state not in _US_STATES or zipc in seen_zip:
                continue
            seen_zip.add(zipc)
            slug = f"{_slug(city)}_{zipc}"
            name = f"{city}, {state} {zipc}"
            artifact_id = getattr(atom, "artifact_id", "") or ""
            atom_id = stable_id("atm", artifact_id, "physical_site", slug)
            # Borrow the anchoring atom's provenance; synthesize a minimal
            # ref if it carries none (every EvidenceAtom needs ≥1 SourceRef).
            src_refs = list(getattr(atom, "source_refs", None) or [])
            if not src_refs:
                src_refs = [
                    SourceRef(
                        id=stable_id("src", atom_id),
                        artifact_id=artifact_id,
                        artifact_type=ArtifactType.txt,
                        filename=getattr(atom, "artifact_id", "") or "geo_fallback",
                        locator={"extraction": "site_geo_fallback"},
                        extraction_method="site_geo_fallback",
                        parser_version="site_geo_fallback_v1",
                    )
                ]
            out.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.physical_site,
                    raw_text=name,
                    normalized_text=name.lower(),
                    value={
                        "kind": "physical_site",
                        "id": slug,
                        "site_id": slug,
                        "name": name,
                        "names": [name, city],
                        "city": city,
                        "state": state,
                        "zip": zipc,
                        "inferred": True,
                    },
                    entity_keys=[f"site:{slug}"],
                    source_refs=src_refs,
                    receipts=[],
                    authority_class=AuthorityClass.machine_extractor,
                    confidence=0.5,
                    confidence_raw=0.5,
                    calibrated_confidence=0.5,
                    review_status=ReviewStatus.needs_review,
                    review_flags=["geo_fallback_site"],
                    parser_version="site_geo_fallback_v1",
                )
            )
            if len(out) >= _MAX_FALLBACK_SITES:
                return out
    return out


__all__ = ["geo_fallback_sites"]
