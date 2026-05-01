from __future__ import annotations

from difflib import SequenceMatcher

from app.core.ids import stable_id
from app.core.normalizers import normalize_entity_key
from app.core.schemas import EntityRecord, EvidenceAtom, ReviewStatus

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback path
    fuzz = None

def _fuzzy_score(a: str, b: str) -> float:
    if fuzz is not None:
        return float(fuzz.ratio(a, b))
    return SequenceMatcher(a=a, b=b).ratio() * 100.0


def extract_entity_records(project_id: str, atoms: list[EvidenceAtom]) -> list[EntityRecord]:
    grouped: dict[str, dict] = {}
    for atom in atoms:
        for key in atom.entity_keys:
            if ":" not in key:
                continue
            entity_type, raw_value = key.split(":", 1)
            canonical_key = normalize_entity_key(entity_type, raw_value)
            canonical_name = canonical_key.split(":", 1)[1].replace("_", " ")
            if canonical_key not in grouped:
                grouped[canonical_key] = {
                    "entity_type": entity_type,
                    "aliases": set(),
                    "source_atom_ids": set(),
                }
            grouped[canonical_key]["aliases"].add(key)
            grouped[canonical_key]["source_atom_ids"].add(atom.id)

    records: list[EntityRecord] = []
    for canonical_key in sorted(grouped):
        info = grouped[canonical_key]
        records.append(
            EntityRecord(
                id=stable_id("ent", project_id, canonical_key),
                project_id=project_id,
                entity_type=info["entity_type"],
                canonical_key=canonical_key,
                canonical_name=canonical_key.split(":", 1)[1].replace("_", " "),
                aliases=sorted(info["aliases"]),
                source_atom_ids=sorted(info["source_atom_ids"]),
                confidence=1.0,
                review_status=ReviewStatus.auto_accepted,
            )
        )
    return records


def resolve_aliases(records: list[EntityRecord]) -> list[EntityRecord]:
    if not records:
        return []

    ordered = sorted(records, key=lambda r: (r.entity_type, r.canonical_key, r.id))
    consumed: set[str] = set()
    resolved: list[EntityRecord] = []

    for record in ordered:
        if record.id in consumed:
            continue
        merged = record.model_copy(deep=True)
        consumed.add(record.id)

        for other in ordered:
            if other.id in consumed or other.entity_type != merged.entity_type:
                continue
            if other.canonical_key == merged.canonical_key:
                score = 100.0
            else:
                score = _fuzzy_score(merged.canonical_name, other.canonical_name)

            if score >= 92:
                merged.aliases = sorted(set(merged.aliases) | set(other.aliases) | {other.canonical_key})
                merged.source_atom_ids = sorted(set(merged.source_atom_ids) | set(other.source_atom_ids))
                consumed.add(other.id)
            elif 82 <= score < 92:
                merged.aliases = sorted(set(merged.aliases) | set(other.aliases) | {other.canonical_key})
                merged.source_atom_ids = sorted(set(merged.source_atom_ids) | set(other.source_atom_ids))
                merged.review_status = ReviewStatus.needs_review
                merged.confidence = min(merged.confidence, 0.82)
                consumed.add(other.id)

        resolved.append(merged)

    resolved.sort(key=lambda r: (r.entity_type, r.canonical_key))
    return resolved
