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
    if re.search(r"\b(?:ROM|\$\s*\d|2k|1500)\b", text, re.I):
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
        entity_keys=list(getattr(duplicate_atom, "entity_keys", None) or []),
        source_refs=[src],
        authority_class=AuthorityClass.meeting_note,
        confidence=0.68,
        review_status=ReviewStatus.needs_review,
        review_flags=["note_provenance_backfill", "hubspot_note_duplicate_pointer"],
        parser_version="note_provenance_backfill_v1",
    )


def ensure_hubspot_note_provenance(
    atoms: list[Any],
    *,
    project_id: str,
    artifact_paths: dict[str, Path],
) -> tuple[list[Any], int]:
    """Mint provenance atoms for hs-note files that lost all atoms in dedup."""
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
        if by_artifact.get(artifact_id, 0) > 0:
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        parsed = parse_hubspot_note_text(raw)
        body = str(parsed.get("body") or "").strip()
        if not body:
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
