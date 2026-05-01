from __future__ import annotations

from collections import Counter
from typing import Any

from app.core.authority import authority_rank
from app.core.schemas import EvidenceAtom, EvidencePacket


def _parser_name(atom: EvidenceAtom) -> str:
    if atom.source_refs:
        return atom.source_refs[0].parser or atom.source_refs[0].artifact_type.value
    return "unknown"


def _artifact_type(atom: EvidenceAtom) -> str:
    if atom.source_refs:
        return atom.source_refs[0].artifact_type.value
    return "txt"


def build_atom_feature_row(atom: EvidenceAtom) -> dict[str, Any]:
    receipt_verified_count = sum(1 for receipt in atom.receipts if receipt.replay_status == "verified")
    has_quantity = bool(isinstance(atom.value, dict) and atom.value.get("quantity") is not None)
    return {
        "atom_type": atom.atom_type.value,
        "authority_class": atom.authority_class.value,
        "parser_name": _parser_name(atom),
        "confidence_raw": float(atom.confidence_raw if atom.confidence_raw is not None else atom.confidence),
        "source_ref_count": len(atom.source_refs),
        "receipt_verified_count": receipt_verified_count,
        "has_quantity": int(has_quantity),
        "entity_key_count": len(atom.entity_keys),
        "review_flag_count": len(atom.review_flags),
        "artifact_type": _artifact_type(atom),
    }


def _packet_atoms(packet: EvidencePacket, atoms: list[EvidenceAtom]) -> list[EvidenceAtom]:
    atom_by_id = {atom.id: atom for atom in atoms}
    ids = packet.governing_atom_ids + packet.supporting_atom_ids + packet.contradicting_atom_ids
    return [atom_by_id[atom_id] for atom_id in ids if atom_id in atom_by_id]


def build_packet_feature_row(packet: EvidencePacket, atoms: list[EvidenceAtom]) -> dict[str, Any]:
    packet_atoms = _packet_atoms(packet, atoms)
    governing_atoms = [atom for atom in packet_atoms if atom.id in packet.governing_atom_ids]
    receipt_verified = sum(
        1 for atom in packet_atoms for receipt in atom.receipts if receipt.replay_status == "verified"
    )
    receipt_total = sum(len(atom.receipts) for atom in packet_atoms)
    receipt_verified_rate = (receipt_verified / receipt_total) if receipt_total else 0.0
    authority_rank_max = max((authority_rank(atom.authority_class) for atom in governing_atoms), default=0)
    return {
        "family": packet.family.value,
        "raw_confidence": float(packet.confidence_raw if packet.confidence_raw is not None else packet.confidence),
        "governing_atom_count": len(packet.governing_atom_ids),
        "supporting_atom_count": len(packet.supporting_atom_ids),
        "contradicting_atom_count": len(packet.contradicting_atom_ids),
        "authority_rank_max": authority_rank_max,
        "ambiguity_score": float(packet.certificate.ambiguity_score if packet.certificate else 0.0),
        "evidence_completeness_score": float(
            packet.certificate.evidence_completeness_score if packet.certificate else 0.0
        ),
        "risk_score": float(packet.risk.risk_score if packet.risk else 0.0),
        "receipt_verified_rate": float(receipt_verified_rate),
        "review_flag_count": len(packet.review_flags),
    }


def summarize_label_balance(labels: list[int]) -> dict[str, int]:
    counts = Counter(labels)
    return {"negative": int(counts.get(0, 0)), "positive": int(counts.get(1, 0))}
