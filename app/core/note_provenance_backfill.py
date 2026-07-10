"""Ensure HubSpot note artifacts retain atoms after semantic dedup.

When a note body duplicates PDF/email facts, dedup collapses the note atoms
into winners from other artifacts — the Files tab then shows ``ok_empty`` for
that note. This stage re-mints lightweight provenance atoms on the note
``artifact_id`` with a ``duplicate_of`` / ``source_reference`` pointer.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.core.address_parse import US_STATES, find_us_addresses_in_text
from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.parsers.hubspot_note_parser import is_hubspot_note_path, parse_hubspot_note_text

_HS_NOTE_FILENAME_RE = re.compile(r"-hs-note-", re.I)
_SIMILARITY_THRESHOLD = 0.72


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_text(atom: Any) -> str:
    raw = getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
    if str(raw).strip():
        return str(raw).strip()
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        return str(val.get("text") or val.get("description") or val.get("rom_text") or "").strip()
    return ""


def _normalize_note_corpus(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _best_duplicate_match(note_body: str, atoms: list[Any], *, exclude_artifact_id: str) -> Any | None:
    note_norm = _normalize_note_corpus(note_body)
    if not note_norm:
        return None
    best: tuple[float, Any] | None = None
    for atom in atoms:
        aid = str(getattr(atom, "artifact_id", "") or "")
        if aid == exclude_artifact_id:
            continue
        other = _normalize_note_corpus(_atom_text(atom))
        if not other or len(other) < 24:
            continue
        ratio = SequenceMatcher(None, note_norm, other).ratio()
        if note_norm in other or other in note_norm:
            ratio = max(ratio, 0.85)
        if ratio >= _SIMILARITY_THRESHOLD and (best is None or ratio > best[0]):
            best = (ratio, atom)
    return best[1] if best else None


def _mint_provenance_atom(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    note_body: str,
    parsed: dict[str, Any],
    duplicate_atom: Any | None,
) -> Any:
    from app.core.schemas import (
        ArtifactType,
        AuthorityClass,
        EvidenceAtom,
        ReviewStatus,
        SourceRef,
    )

    dup_id = str(getattr(duplicate_atom, "id", "") or "") if duplicate_atom else ""
    dup_artifact = str(getattr(duplicate_atom, "artifact_id", "") or "") if duplicate_atom else ""
    dup_type = _atom_type_str(duplicate_atom) if duplicate_atom else ""
    text = note_body[:4000]
    atom_id = stable_id("atm", project_id, artifact_id, "note_provenance", text[:120])
    src = SourceRef(
        id=stable_id("src", atom_id),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.txt,
        filename=filename,
        locator={"kind": "hubspot_note_provenance"},
        extraction_method="note_provenance_backfill",
        parser_version="note_provenance_backfill_v1",
    )
    value: dict[str, Any] = {
        "field_name": "hubspot_note_provenance",
        "text": text,
        "hubspot_note_id": parsed.get("note_id"),
        "title": parsed.get("title"),
        "source": "hubspot_note",
        "source_reference": dup_artifact or None,
        "duplicate_of": dup_id or None,
        "duplicate_atom_type": dup_type or None,
    }
    atom_type = AtomType.deal_metadata
    entity_keys = list(getattr(duplicate_atom, "entity_keys", None) or [])
    review_flags = [
        "note_provenance_backfill",
        "hubspot_note_duplicate_pointer",
        "note_provenance_v2",
    ]
    confidence = 0.68
    review_status = ReviewStatus.needs_review

    # Address-only HubSpot notes (Trent-style street/city/ST/ZIP) must remint
    # as physical_site when dedup wiped the note — a deal_metadata pointer
    # leaves the Files tab looking "parsed" while sites stay empty.
    title = str(parsed.get("title") or "")
    corpus = f"{title}\n{text}".strip()
    # Also try the body alone and a title-cased variant — HubSpot notes are
    # often all-lowercase and may already be title-duplicated in the corpus.
    search_blobs = [corpus, text, title, text.title(), corpus.title()]
    site_addr = None
    for blob in search_blobs:
        if not blob or site_addr is not None:
            continue
        for parsed_addr in find_us_addresses_in_text(blob):
            if (
                parsed_addr.city
                and parsed_addr.state
                and parsed_addr.state in US_STATES
                and parsed_addr.street_address
            ):
                site_addr = parsed_addr
                break
    if site_addr is not None:
        slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            f"{site_addr.city}_{site_addr.state}_{site_addr.zip or site_addr.street_address}".lower(),
        ).strip("_")
        display = (
            f"{site_addr.street_address}, {site_addr.city}, "
            f"{site_addr.state} {site_addr.zip or ''}"
        ).strip()
        atom_type = AtomType.physical_site
        atom_id = stable_id("atm", project_id, artifact_id, "physical_site", slug)
        value = {
            "kind": "physical_site",
            "id": slug,
            "site_id": slug,
            "name": display,
            "names": list(dict.fromkeys([display, site_addr.city])),
            "aliases": [],
            "street_address": site_addr.street_address,
            "address": site_addr.street_address,
            "city": site_addr.city,
            "state": site_addr.state,
            "zip": site_addr.zip,
            "inferred": True,
            "source_context": corpus[:600],
            "hubspot_note_id": parsed.get("note_id"),
            "source": "hubspot_note",
            "source_reference": dup_artifact or None,
            "duplicate_of": dup_id or None,
        }
        entity_keys = [f"site:{slug}"]
        review_flags = [
            "note_provenance_backfill",
            "hubspot_note_physical_site",
            "hubspot_note_parser",
            "note_provenance_v2",
        ]
        confidence = 0.76
        review_status = ReviewStatus.auto_accepted
        text = display
    elif re.search(r"\b(?:ROM|\$\s*\d|2k|1500)\b", text, re.I):
        atom_type = AtomType.commercial_total
        value["category"] = "ROM"
    elif duplicate_atom and dup_type in {"scope_item", "customer_instruction", "constraint"}:
        atom_type = AtomType(dup_type) if dup_type in {t.value for t in AtomType} else AtomType.deal_metadata

    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=value,
        entity_keys=entity_keys,
        source_refs=[src],
        authority_class=AuthorityClass.meeting_note,
        confidence=confidence,
        review_status=review_status,
        review_flags=review_flags,
        parser_version="note_provenance_backfill_v1",
    )


def _note_has_physical_site(atoms: list[Any], artifact_id: str) -> bool:
    for atom in atoms:
        if str(getattr(atom, "artifact_id", "") or "") != artifact_id:
            continue
        if _atom_type_str(atom) == "physical_site":
            return True
        flags = list(getattr(atom, "review_flags", None) or [])
        if "hubspot_note_physical_site" in flags or "email_note_physical_site" in flags:
            return True
    return False


def _note_body_has_us_address(parsed: dict[str, Any], body: str) -> bool:
    """True when the HubSpot note is (or contains) a street/city/ST/ZIP site."""
    title = str(parsed.get("title") or "")
    corpus = f"{title}\n{body}".strip()
    for blob in (corpus, body, title, body.title(), corpus.title()):
        if not blob:
            continue
        for parsed_addr in find_us_addresses_in_text(blob):
            if (
                parsed_addr.city
                and parsed_addr.state
                and parsed_addr.state in US_STATES
                and parsed_addr.street_address
            ):
                return True
    return False


def ensure_hubspot_note_provenance(
    atoms: list[Any],
    *,
    project_id: str,
    artifact_paths: dict[str, Path],
) -> tuple[list[Any], int]:
    """Mint provenance atoms for hs-note files that lost atoms in dedup.

    Address-only notes are a special case: dedup often leaves a thin
    ``deal_metadata`` pointer on the note while wiping ``physical_site``.
    Remint those as ``physical_site`` even when *some* note atoms remain.
    """
    if not artifact_paths:
        return atoms, 0

    by_artifact: dict[str, int] = {}
    for atom in atoms:
        aid = str(getattr(atom, "artifact_id", "") or "")
        if aid:
            by_artifact[aid] = by_artifact.get(aid, 0) + 1

    minted = 0
    for artifact_id, path in artifact_paths.items():
        if not _HS_NOTE_FILENAME_RE.search(path.name) and not is_hubspot_note_path(path):
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        parsed = parse_hubspot_note_text(raw)
        body = str(parsed.get("body") or "").strip()
        if not body:
            continue

        has_any = by_artifact.get(artifact_id, 0) > 0
        needs_site = (
            _note_body_has_us_address(parsed, body)
            and not _note_has_physical_site(atoms, artifact_id)
        )
        # Skip notes that still have atoms *unless* this is an address note
        # whose physical_site was wiped (Stinson / Trent pattern).
        if has_any and not needs_site:
            continue

        duplicate = _best_duplicate_match(body, atoms, exclude_artifact_id=artifact_id)
        atoms.append(
            _mint_provenance_atom(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                note_body=body,
                parsed=parsed,
                duplicate_atom=duplicate,
            )
        )
        by_artifact[artifact_id] = by_artifact.get(artifact_id, 0) + 1
        minted += 1
    return atoms, minted


__all__ = ["ensure_hubspot_note_provenance"]

