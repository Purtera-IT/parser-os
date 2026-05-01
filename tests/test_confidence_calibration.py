from __future__ import annotations

import json
from pathlib import Path

from app.core.compiler import compile_project
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    CompileResult,
    EdgeType,
    EvidenceAtom,
    EvidenceEdge,
    EvidencePacket,
    PacketCertificate,
    PacketFamily,
    PacketRisk,
    PacketStatus,
    ReviewStatus,
    SourceRef,
)
from app.learning.calibration import apply_calibration, load_calibrator, train_calibrator
from app.learning.features import build_atom_feature_row, build_packet_feature_row


def _atom(atom_id: str, *, atom_type: AtomType, authority: AuthorityClass, confidence: float, text: str) -> EvidenceAtom:
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id="art_1",
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value={"text": text, "quantity": 5 if atom_type == AtomType.quantity else None},
        entity_keys=["site:main_campus", "device:ip_camera"],
        source_refs=[
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id="art_1",
                artifact_type=ArtifactType.txt,
                filename="fixture.txt",
                locator={"line_start": 1, "line_end": 1},
                extraction_method="test",
                parser_version="test_v1",
                parser="test_parser",
            )
        ],
        receipts=[],
        authority_class=authority,
        confidence=confidence,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test_v1",
    )


def _packet(packet_id: str, atom_ids: list[str], *, family: PacketFamily, confidence: float, risk_score: float) -> EvidencePacket:
    return EvidencePacket(
        id=packet_id,
        project_id="proj_1",
        family=family,
        anchor_type="site",
        anchor_key="site:main_campus",
        governing_atom_ids=[atom_ids[0]],
        supporting_atom_ids=list(atom_ids),
        contradicting_atom_ids=[],
        related_edge_ids=[],
        confidence=confidence,
        status=PacketStatus.active,
        reason="test packet",
        review_flags=[],
        certificate=PacketCertificate(
            packet_id=packet_id,
            certificate_version="test",
            existence_reason="exists",
            governing_rationale="governs",
            minimal_sufficient_atom_ids=[atom_ids[0]],
            evidence_completeness_score=0.9,
            ambiguity_score=0.1,
        ),
        risk=PacketRisk(
            risk_score=risk_score,
            severity="medium",
            review_priority=3,
            risk_reasons=[],
        ),
    )


def _compile_result() -> CompileResult:
    good_atom = _atom(
        "atm_good",
        atom_type=AtomType.constraint,
        authority=AuthorityClass.customer_current_authored,
        confidence=0.92,
        text="Escort required after 5pm",
    )
    bad_atom = _atom(
        "atm_bad",
        atom_type=AtomType.open_question,
        authority=AuthorityClass.meeting_note,
        confidence=0.55,
        text="Need to confirm access",
    )
    packets = [
        _packet("pkt_good", [good_atom.id], family=PacketFamily.site_access, confidence=0.9, risk_score=0.4),
        _packet("pkt_bad", [bad_atom.id], family=PacketFamily.missing_info, confidence=0.55, risk_score=0.2),
    ]
    return CompileResult(
        project_id="proj_1",
        atoms=[good_atom, bad_atom],
        entities=[],
        edges=[],
        packets=packets,
        warnings=[],
    )


def test_feature_extraction_deterministic() -> None:
    result = _compile_result()
    atom_row_1 = build_atom_feature_row(result.atoms[0])
    atom_row_2 = build_atom_feature_row(result.atoms[0])
    packet_row_1 = build_packet_feature_row(result.packets[0], result.atoms)
    packet_row_2 = build_packet_feature_row(result.packets[0], result.atoms)
    assert atom_row_1 == atom_row_2
    assert packet_row_1 == packet_row_2


def test_calibrator_trains_on_fake_labels(tmp_path: Path) -> None:
    result = _compile_result()
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "reviews": [
                    {"packet_id": "pkt_good", "correct_packet": True},
                    {"packet_id": "pkt_bad", "correct_packet": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "calibrator.joblib"
    report = train_calibrator(labels_path, [result], model_path)
    assert report["packet_samples"] == 2
    assert model_path.exists()


def test_calibrated_score_in_range(tmp_path: Path) -> None:
    result = _compile_result()
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "reviews": [
                    {"packet_id": "pkt_good", "correct_packet": True},
                    {"packet_id": "pkt_bad", "correct_packet": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "calibrator.joblib"
    train_calibrator(labels_path, [result], model_path)
    calibrated = apply_calibration(result, model_path, abstain_threshold=0.7)
    assert all(0.0 <= packet.confidence <= 1.0 for packet in calibrated.packets)
    assert all(packet.calibrated_confidence is not None for packet in calibrated.packets)


def test_low_calibrated_packet_gets_abstain_flag(tmp_path: Path) -> None:
    result = _compile_result()
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "reviews": [
                    {"packet_id": "pkt_good", "correct_packet": True},
                    {"packet_id": "pkt_bad", "correct_packet": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "calibrator.joblib"
    train_calibrator(labels_path, [result], model_path)
    calibrated = apply_calibration(result, model_path, abstain_threshold=0.90)
    assert any("calibration_abstain" in packet.review_flags for packet in calibrated.packets)
    abstained = [packet for packet in calibrated.packets if "calibration_abstain" in packet.review_flags]
    assert all(packet.status == PacketStatus.needs_review for packet in abstained)


def test_no_calibrator_path_keeps_existing_behavior(tmp_path: Path) -> None:
    artifact = tmp_path / "customer_email.txt"
    artifact.write_text("From: customer@example.com\nSent: now\nSubject: test\nPlease remove west wing", encoding="utf-8")
    result = compile_project(tmp_path, allow_errors=True)
    assert all(packet.calibrated_confidence is None for packet in result.packets)
    assert all("calibration_abstain" not in packet.review_flags for packet in result.packets)


def test_model_artifact_load_save_roundtrip(tmp_path: Path) -> None:
    result = _compile_result()
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "reviews": [
                    {"packet_id": "pkt_good", "correct_packet": True},
                    {"packet_id": "pkt_bad", "correct_packet": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "calibrator.joblib"
    train_calibrator(labels_path, [result], model_path)
    loaded = load_calibrator(model_path)
    assert loaded["version"] == "calibrator_v1"
    assert loaded["packet_model"] is not None
