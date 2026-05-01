from __future__ import annotations

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
from app.core.validators import validation_failure_records
from app.eval.failure_taxonomy import (
    FailureCategory,
    failure_records_from_expected_label_mismatches,
    make_failure_record,
    summarize_failure_records,
)
from app.eval.gold import GoldExpectedPacket, GoldScenario


def _atom(atom_id: str, authority: AuthorityClass, *, atom_type: AtomType = AtomType.scope_item) -> EvidenceAtom:
    return EvidenceAtom(
        id=atom_id,
        project_id="scenario_x",
        artifact_id="art_1",
        atom_type=atom_type,
        raw_text="raw text",
        normalized_text="raw text",
        value={"text": "raw text"},
        entity_keys=["site:west_wing"],
        source_refs=[
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id="art_1",
                artifact_type=ArtifactType.txt,
                filename="fixture.txt",
                locator={},
                extraction_method="test",
                parser_version="test",
            )
        ],
        authority_class=authority,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def test_missing_expected_packet_creates_packet_missing_expected() -> None:
    result = CompileResult(
        project_id="scenario_x",
        compile_id="cmp_1",
        atoms=[],
        entities=[],
        edges=[],
        packets=[],
        warnings=[],
    )
    gold = GoldScenario(
        scenario_id="scenario_x",
        expected_packets=[GoldExpectedPacket(family="vendor_mismatch", anchor_key_contains="device:ip_camera")],
    )
    failures = failure_records_from_expected_label_mismatches(result, gold, scenario_id="scenario_x")
    assert failures
    assert any(failure.category == FailureCategory.PACKET_MISSING_EXPECTED for failure in failures)


def test_deleted_text_governing_creates_invalid_deleted_text_failure() -> None:
    deleted = _atom("atm_deleted", AuthorityClass.deleted_text)
    packet = EvidencePacket(
        id="pkt_1",
        project_id="scenario_x",
        family=PacketFamily.scope_exclusion,
        anchor_type="site",
        anchor_key="site:west_wing",
        governing_atom_ids=["atm_deleted"],
        supporting_atom_ids=["atm_deleted"],
        contradicting_atom_ids=[],
        related_edge_ids=[],
        confidence=0.9,
        status=PacketStatus.active,
        reason="reason",
    )
    result = CompileResult(
        project_id="scenario_x",
        compile_id="cmp_1",
        atoms=[deleted],
        entities=[],
        edges=[],
        packets=[packet],
        warnings=[],
    )
    failures = validation_failure_records(result, source_files_available=False)
    assert any(failure.category == FailureCategory.INVALID_DELETED_TEXT_GOVERNANCE for failure in failures)


def test_non_deterministic_signatures_create_failure_record() -> None:
    failure = make_failure_record(
        category=FailureCategory.NON_DETERMINISTIC_OUTPUT,
        severity="critical",
        scenario_id="scenario_x",
        message="Determinism check failed",
    )
    assert failure.category == FailureCategory.NON_DETERMINISTIC_OUTPUT


def test_summarize_groups_counts_by_category() -> None:
    failures = [
        make_failure_record(
            category=FailureCategory.PACKET_MISSING_EXPECTED,
            severity="high",
            scenario_id="s1",
            message="missing packet one",
        ),
        make_failure_record(
            category=FailureCategory.PACKET_MISSING_EXPECTED,
            severity="high",
            scenario_id="s2",
            message="missing packet two",
        ),
        make_failure_record(
            category=FailureCategory.NON_DETERMINISTIC_OUTPUT,
            severity="critical",
            scenario_id="s1",
            message="determinism failed",
        ),
    ]
    summary = summarize_failure_records(failures)
    categories = {row["category"]: row["count"] for row in summary["by_category"]}
    assert categories["PACKET_MISSING_EXPECTED"] == 2
    assert categories["NON_DETERMINISTIC_OUTPUT"] == 1
