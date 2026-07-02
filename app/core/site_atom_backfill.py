"""Backfill physical_site atoms when the pipeline has site entities but no roster atoms.

Full-ML runs can promote ghost physical_site rows that semantic_dedup later
drops, while entity extraction still mints ``site:*`` records from prose. The
envelope then shows site entities with zero ``physical_site`` atoms — grade and
prefill see one ghost cluster instead of structured anchors. This pass is a
deterministic safety net: when no physical_site survives dedup, mint one atom per
distinct site entity (or per ``site:`` key on scope atoms) so downstream
projection and deal-kit prefill have real anchors.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.address_parse import US_STATES, find_us_addresses_in_text
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

_CITY_STATE_ZIP = re.compile(
    r"\b([A-Za-z][A-Za-z .'-]{1,40}),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b"
)


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _roster_location_buckets(atoms: list[Any]) -> set[str]:
    """Location buckets for all surviving physical_site roster atoms."""
    from app.core.semantic_dedup import _site_location_buckets

    buckets: set[str] = set()
    for atom in atoms:
        if _atom_type_str(atom) != "physical_site":
            continue
        val = getattr(atom, "value", None) or {}
        sid = ""
        if isinstance(val, dict):
            sid = str(val.get("site_id") or val.get("id") or "")
            buckets |= _site_location_buckets(val, sid)
        if sid:
            buckets |= _site_location_buckets({}, sid)
    return buckets


def _site_key_location_buckets(site_key: str) -> set[str]:
    from app.core.semantic_dedup import _bucket_from_site_id_slug, _site_location_buckets

    slug = site_key.split(":", 1)[-1] if ":" in site_key else site_key
    buckets = _site_location_buckets({}, slug)
    slug_bucket = _bucket_from_site_id_slug(slug)
    if slug_bucket:
        buckets.add(slug_bucket)
    return buckets


def _site_key_on_roster(site_key: str, atoms: list[Any]) -> bool:
    slug = site_key.split(":", 1)[-1] if ":" in site_key else site_key
    slug_norm = _slug(slug)
    for atom in atoms:
        if _atom_type_str(atom) != "physical_site":
            continue
        keys = getattr(atom, "entity_keys", None) or []
        if site_key in keys:
            return True
        val = getattr(atom, "value", None) or {}
        if isinstance(val, dict):
            sid = str(val.get("site_id") or val.get("id") or "")
            if sid and _slug(sid) == slug_norm:
                return True
    return False


def _slug_matches_roster_bucket(site_key: str, roster_buckets: set[str]) -> bool:
    """Match street-slug entity keys (``12575_oakland_park_blvd``) to roster address buckets."""
    slug = site_key.split(":", 1)[-1] if ":" in site_key else site_key
    slug_street = slug.replace("_", " ").lower().strip()
    if not slug_street or not slug_street[0].isdigit():
        return False
    for bucket in roster_buckets:
        head = bucket.split("|", 1)[0].lower()
        if slug_street == head or slug_street in head or head in slug_street:
            return True
    return False


def _physical_site_covers_location(atoms: list[Any], site_key: str, geo: dict[str, str]) -> bool:
    """True when any surviving physical_site shares a location bucket with this key/geo."""
    from app.core.semantic_dedup import _site_location_buckets

    pending = _site_key_location_buckets(site_key)
    if geo:
        pending |= _site_location_buckets(geo, "")
    if not pending:
        return False
    for atom in atoms:
        if _atom_type_str(atom) != "physical_site":
            continue
        val = getattr(atom, "value", None) or {}
        sid = ""
        if isinstance(val, dict):
            sid = str(val.get("site_id") or val.get("id") or "")
            if pending & _site_location_buckets(val, sid):
                return True
        if sid and pending & _site_location_buckets({}, sid):
            return True
    return False


def _site_key_covered(site_key: str, roster_buckets: set[str], atoms: list[Any]) -> bool:
    """True when an existing physical_site already anchors this site:* key."""
    if _site_key_on_roster(site_key, atoms):
        return True
    if roster_buckets:
        if _site_key_location_buckets(site_key) & roster_buckets:
            return True
        if _slug_matches_roster_bucket(site_key, roster_buckets):
            return True
    anchor = _anchor_atom_for_site_key(atoms, site_key)
    if anchor is None:
        return False
    geo = _parse_geo_from_text(
        str(getattr(anchor, "raw_text", None) or getattr(anchor, "text", None) or "")
    )
    if geo and roster_buckets:
        from app.core.semantic_dedup import _site_location_buckets

        return bool(_site_location_buckets(geo, "") & roster_buckets)
    return False


def _roster_is_sufficient(atoms: list[Any], entities: list[Any]) -> bool:
    """True when every distinct site:* key is anchored by a physical_site location."""
    phys = [a for a in atoms if _atom_type_str(a) == "physical_site"]
    if not phys:
        return False
    roster_buckets = _roster_location_buckets(atoms)
    site_keys: set[str] = set()
    for ent in entities or []:
        sk = _site_key_from_entity(ent)
        if sk:
            site_keys.add(sk)
    for atom in atoms:
        for key in getattr(atom, "entity_keys", None) or []:
            if str(key).startswith("site:"):
                site_keys.add(str(key))
    if not site_keys:
        return True
    if not roster_buckets:
        return len(phys) >= len(site_keys)
    return all(_site_key_covered(sk, roster_buckets, atoms) for sk in site_keys)


def _site_key_from_entity(entity: Any) -> str | None:
    key = str(getattr(entity, "canonical_key", "") or "").strip()
    if key.startswith("site:"):
        return key
    et = str(getattr(entity, "entity_type", "") or "").lower()
    if et == "site" and key:
        return key if key.startswith("site:") else f"site:{_slug(key)}"
    return None


def _parse_geo_from_text(text: str) -> dict[str, str]:
    text = (text or "").strip()
    if not text:
        return {}
    for item in find_us_addresses_in_text(text):
        if item.city and item.state in US_STATES:
            out: dict[str, str] = {"city": item.city, "state": item.state}
            if item.zip:
                out["zip"] = item.zip
            if item.street_address:
                out["street_address"] = item.street_address
                out["address"] = item.street_address
            return out
    m = _CITY_STATE_ZIP.search(text)
    if m:
        return {"city": m.group(1).strip(), "state": m.group(2), "zip": m.group(3)}
    return {}


def _mint_physical_site(
    *,
    project_id: str,
    site_key: str,
    display_name: str,
    geo: dict[str, str],
    source_atom: Any | None,
    reason: str,
) -> EvidenceAtom:
    slug = site_key.split(":", 1)[-1] if ":" in site_key else _slug(site_key)
    city = geo.get("city", "")
    state = geo.get("state", "")
    zipc = geo.get("zip", "")
    street = geo.get("street_address") or geo.get("address") or ""
    if street and city and state:
        name = f"{street}, {city}, {state} {zipc}".strip()
    elif city and state:
        name = f"{city}, {state} {zipc}".strip()
    else:
        name = display_name or slug.replace("_", " ").title()

    artifact_id = getattr(source_atom, "artifact_id", "") if source_atom else ""
    anchor_text = (
        getattr(source_atom, "raw_text", None)
        or getattr(source_atom, "text", None)
        or display_name
        or name
    )
    atom_id = stable_id("atm", artifact_id or project_id, "physical_site", slug, reason)
    src_refs = list(getattr(source_atom, "source_refs", None) or []) if source_atom else []
    if not src_refs:
        src_refs = [
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id=artifact_id or "site_backfill",
                artifact_type=ArtifactType.txt,
                filename=artifact_id or "site_backfill",
                locator={"extraction": "site_atom_backfill"},
                extraction_method="site_atom_backfill",
                parser_version="site_atom_backfill_v1",
            )
        ]

    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id or "site_backfill",
        atom_type=AtomType.physical_site,
        raw_text=str(anchor_text)[:2000],
        normalized_text=str(name).lower(),
        value={
            "kind": "physical_site",
            "id": slug,
            "site_id": slug,
            "name": name,
            "names": [display_name, name] if display_name else [name],
            "street_address": street or None,
            "address": street or None,
            "city": city or None,
            "state": state or None,
            "zip": zipc or None,
            "inferred": True,
            "backfill": reason,
        },
        entity_keys=[site_key if site_key.startswith("site:") else f"site:{slug}"],
        source_refs=src_refs,
        receipts=list(getattr(source_atom, "receipts", None) or []) if source_atom else [],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.55,
        confidence_raw=0.55,
        calibrated_confidence=0.55,
        review_status=ReviewStatus.needs_review,
        review_flags=["site_entity_backfill"],
        parser_version="site_atom_backfill_v1",
    )


def _anchor_atom_for_site_key(atoms: list[Any], site_key: str) -> Any | None:
    slug = site_key.split(":", 1)[-1] if ":" in site_key else site_key
    for atom in atoms:
        keys = getattr(atom, "entity_keys", None) or []
        if site_key in keys or f"site:{slug}" in keys:
            return atom
        text = str(getattr(atom, "raw_text", None) or getattr(atom, "text", None) or "")
        if slug.replace("_", " ") in text.lower():
            return atom
    return None


def backfill_physical_sites_from_entities(
    atoms: list[Any],
    entities: list[Any],
    *,
    project_id: str,
) -> tuple[list[Any], int]:
    """Mint physical_site atoms when entity extraction found sites but roster is empty."""
    if _roster_is_sufficient(atoms, entities):
        return atoms, 0

    site_entities = []
    seen_keys: set[str] = set()
    for ent in entities or []:
        sk = _site_key_from_entity(ent)
        if not sk or sk in seen_keys:
            continue
        seen_keys.add(sk)
        site_entities.append((sk, ent))

    # Also collect orphan site: keys from atoms when entities list is thin.
    for atom in atoms:
        for key in getattr(atom, "entity_keys", None) or []:
            if not str(key).startswith("site:") or key in seen_keys:
                continue
            seen_keys.add(str(key))
            site_entities.append((str(key), None))

    if not site_entities:
        return atoms, 0

    roster_buckets = _roster_location_buckets(atoms)
    added: list[Any] = []
    for site_key, ent in site_entities:
        if roster_buckets and _site_key_covered(site_key, roster_buckets, atoms):
            continue
        display = ""
        if ent is not None:
            display = str(getattr(ent, "canonical_name", "") or "").strip()
        anchor = _anchor_atom_for_site_key(atoms, site_key)
        geo = _parse_geo_from_text(display)
        if not geo and anchor is not None:
            geo = _parse_geo_from_text(
                str(getattr(anchor, "raw_text", None) or getattr(anchor, "text", None) or "")
            )
        if not geo and display:
            geo = _parse_geo_from_text(display.replace("_", " "))
        if _physical_site_covers_location(atoms + added, site_key, geo):
            continue
        try:
            from app.core.vendor_site_ban import is_purtera_vendor_address

            anchor_text = ""
            if anchor is not None:
                anchor_text = str(
                    getattr(anchor, "raw_text", None) or getattr(anchor, "text", None) or ""
                )
            if is_purtera_vendor_address(
                text=" ".join(filter(None, [display, anchor_text, site_key])),
                value=geo,
            ):
                continue
        except Exception:  # pragma: no cover
            pass
        minted = _mint_physical_site(
            project_id=project_id,
            site_key=site_key,
            display_name=display,
            geo=geo,
            source_atom=anchor,
            reason="entity_backfill",
        )
        added.append(minted)
        val = getattr(minted, "value", None) or {}
        sid = str(val.get("site_id") or val.get("id") or "") if isinstance(val, dict) else ""
        if isinstance(val, dict):
            from app.core.semantic_dedup import _site_location_buckets

            roster_buckets |= _site_location_buckets(val, sid)
        if sid:
            from app.core.semantic_dedup import _site_location_buckets

            roster_buckets |= _site_location_buckets({}, sid)

    return atoms + added, len(added)


__all__ = ["backfill_physical_sites_from_entities"]
