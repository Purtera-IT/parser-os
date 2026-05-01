from __future__ import annotations

import hashlib
import re

from app.core.normalizers import normalize_text
from app.core.schemas import AnchorSignature, AtomType, EvidenceAtom, PacketFamily


def _topic_slug(text: str) -> str:
    normalized = normalize_text(text)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "unknown"


def _best_site_key(atoms: list[EvidenceAtom], prioritize_exclusion_text: bool = False) -> str:
    site_keys = sorted({key for atom in atoms for key in atom.entity_keys if key.startswith("site:")})
    if not site_keys:
        return "site:unknown"
    if prioritize_exclusion_text:
        text_blob = " ".join(normalize_text(atom.raw_text) for atom in atoms if atom.atom_type == AtomType.exclusion)
        if "west wing" in text_blob:
            return "site:west_wing"
        if "main campus" in text_blob:
            return "site:main_campus"
        for atom in atoms:
            if atom.atom_type != AtomType.exclusion:
                continue
            text = normalize_text(atom.raw_text)
            for key in site_keys:
                label = key.split(":", 1)[1].replace("_", " ")
                if label in text:
                    return key
    return site_keys[0]


def _best_device_key(atoms: list[EvidenceAtom]) -> str:
    device_keys = sorted({key for atom in atoms for key in atom.entity_keys if key.startswith("device:")})
    return device_keys[0] if device_keys else "device:unknown"


def _first_atom_by_type(atoms: list[EvidenceAtom], atom_types: set[AtomType]) -> EvidenceAtom | None:
    filtered = [atom for atom in atoms if atom.atom_type in atom_types]
    if not filtered:
        return None
    return sorted(filtered, key=lambda atom: atom.id)[0]


def _signature_hash(anchor_type: str, canonical_key: str, entity_keys: list[str], normalized_topic: str, scope_dimension: str | None) -> str:
    payload = f"{anchor_type}|{canonical_key}|{'|'.join(sorted(entity_keys))}|{normalized_topic}|{scope_dimension or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_anchor_signature(
    family: PacketFamily,
    atoms: list[EvidenceAtom],
    *,
    owner: str | None = None,
    material_identity: str | None = None,
) -> AnchorSignature:
    entity_keys = sorted({key for atom in atoms for key in atom.entity_keys})

    if family in {PacketFamily.quantity_conflict, PacketFamily.vendor_mismatch} and material_identity:
        site_key = _best_site_key(atoms)
        canonical_key = f"material:{material_identity}"
        anchor_type = "material"
        normalized_topic = material_identity
        scope_dimension = "quantity"
        entity_keys = sorted({site_key, canonical_key})
    elif family in {PacketFamily.quantity_conflict, PacketFamily.vendor_mismatch}:
        canonical_key = _best_device_key(atoms)
        anchor_type = "device"
        normalized_topic = canonical_key.split(":", 1)[1]
        scope_dimension = "quantity"
        entity_keys = [canonical_key]
    elif family == PacketFamily.scope_exclusion:
        site_key = _best_site_key(atoms, prioritize_exclusion_text=True)
        canonical_key = site_key
        if material_identity:
            canonical_key = f"{site_key}|{material_identity}"
        anchor_type = "site"
        normalized_topic = site_key.split(":", 1)[1] if ":" in site_key else site_key
        scope_dimension = "exclusion"
        entity_keys = sorted({site_key, material_identity} if material_identity else {site_key})
    elif family == PacketFamily.site_access:
        canonical_key = _best_site_key(atoms)
        anchor_type = "site"
        normalized_topic = canonical_key.split(":", 1)[1]
        scope_dimension = "access"
        entity_keys = [canonical_key]
    elif family == PacketFamily.action_item:
        action_atom = _first_atom_by_type(atoms, {AtomType.action_item})
        action_text = action_atom.raw_text if action_atom else "unknown_action"
        owner_key = (owner or (action_atom.value.get("owner") if action_atom else "unknown") or "unknown").strip().lower()
        action_slug = _topic_slug(action_text)
        canonical_key = f"action_item:{owner_key}:{action_slug}"
        anchor_type = "action_item"
        normalized_topic = action_slug
        scope_dimension = "action"
        entity_keys = [f"action_item:{owner_key}"]
    elif family == PacketFamily.missing_info:
        if material_identity == "raceway_conduit":
            canonical_key = "missing_info:raceway_conduit"
            anchor_type = "missing_info"
            normalized_topic = "raceway_conduit"
            scope_dimension = "pathway"
            entity_keys = sorted({canonical_key, "pathway:raceway_conduit"})
        elif material_identity == "certification":
            canonical_key = "missing_info:requirement:certification"
            anchor_type = "missing_info"
            normalized_topic = "certification"
            scope_dimension = "requirement"
            entity_keys = sorted({canonical_key, "requirement:certification"})
        elif material_identity == "site_access_gate":
            canonical_key = "missing_info:access:site_gate"
            anchor_type = "missing_info"
            normalized_topic = "site_access_gate"
            scope_dimension = "access"
            entity_keys = sorted({canonical_key, "access:site_gate"})
        else:
            q_atom = _first_atom_by_type(atoms, {AtomType.open_question})
            topic = q_atom.raw_text if q_atom else "unknown_question"
            canonical_key = f"missing_info:{_topic_slug(topic)}"
            anchor_type = "missing_info"
            normalized_topic = _topic_slug(topic)
            scope_dimension = "question"
            entity_keys = [canonical_key]
    elif family == PacketFamily.meeting_decision:
        d_atom = _first_atom_by_type(atoms, {AtomType.decision, AtomType.meeting_commitment})
        topic = d_atom.raw_text if d_atom else "unknown_decision"
        canonical_key = f"meeting_decision:{_topic_slug(topic)}"
        anchor_type = "meeting_decision"
        normalized_topic = _topic_slug(topic)
        scope_dimension = "decision"
        entity_keys = [canonical_key]
    else:
        site_key = _best_site_key(atoms)
        canonical_key = site_key if site_key != "site:unknown" else _best_device_key(atoms)
        anchor_type = canonical_key.split(":", 1)[0] if ":" in canonical_key else "entity"
        normalized_topic = canonical_key.split(":", 1)[1] if ":" in canonical_key else canonical_key
        scope_dimension = None

    return AnchorSignature(
        anchor_type=anchor_type,
        canonical_key=canonical_key,
        entity_keys=entity_keys,
        normalized_topic=normalized_topic,
        scope_dimension=scope_dimension,
        hash=_signature_hash(anchor_type, canonical_key, entity_keys, normalized_topic, scope_dimension),
    )
