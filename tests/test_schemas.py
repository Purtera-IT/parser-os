from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    CompileResult,
    EvidenceAtom,
    EvidencePacket,
    PacketFamily,
    PacketStatus,
    ReviewStatus,
    SourceRef,
)


def _source_ref() -> SourceRef:
    return SourceRef(
        id="src_1",
        artifact_id="artifact_1",
        artifact_type=ArtifactType.email,
        filename="mail.eml",
        locator={"line": 1},
        extraction_method="rule_based",
        parser_version="0.1.0",
    )


def _atom(**overrides) -> EvidenceAtom:
    data = {
        "id": "atom_1",
        "project_id": "proj_1",
        "artifact_id": "artifact_1",
        "atom_type": AtomType.quantity,
        "raw_text": "Quantity: 42",
        "normalized_text": "quantity: 42",
        "value": {"quantity": 42},
        "entity_keys": ["site:nyc"],
        "source_refs": [_source_ref()],
        "authority_class": AuthorityClass.machine_extractor,
        "confidence": 0.7,
        "review_status": ReviewStatus.needs_review,
        "review_flags": [],
        "parser_version": "0.1.0",
    }
    data.update(overrides)
    return EvidenceAtom(**data)


def _packet(**overrides) -> EvidencePacket:
    data = {
        "id": "packet_1",
        "project_id": "proj_1",
        "family": PacketFamily.quantity_claim,
        "anchor_type": "site",
        "anchor_key": "site:nyc",
        "governing_atom_ids": ["atom_1"],
        "supporting_atom_ids": [],
        "contradicting_atom_ids": [],
        "related_edge_ids": [],
        "confidence": 0.8,
        "status": PacketStatus.active,
        "reason": "Primary quantity signal",
        "review_flags": [],
    }
    data.update(overrides)
    return EvidencePacket(**data)


def test_atom_requires_source_ref() -> None:
    with pytest.raises(ValidationError):
        _atom(source_refs=[])


def test_packet_requires_governing_atom_when_active() -> None:
    with pytest.raises(ValidationError):
        _packet(governing_atom_ids=[], status=PacketStatus.active)


def test_stable_id_deterministic() -> None:
    left = stable_id("atom", "Acme Corp", {"qty": 2})
    right = stable_id("atom", "  acme   corp ", {"qty": 2})
    assert left == right
    assert left.startswith("atom_")
    assert len(left.split("_", 1)[1]) == 16


def test_json_serialization_roundtrip() -> None:
    atom = _atom()
    packet = _packet()
    result = CompileResult(
        project_id="proj_1",
        atoms=[atom],
        entities=[],
        edges=[],
        packets=[packet],
        warnings=["demo warning"],
    )
    payload = result.model_dump_json()
    rebuilt = CompileResult.model_validate_json(payload)
    assert rebuilt.project_id == "proj_1"
    assert rebuilt.atoms[0].id == "atom_1"
    assert rebuilt.packets[0].id == "packet_1"


def test_confidence_validation() -> None:
    with pytest.raises(ValidationError):
        _atom(confidence=1.2)
