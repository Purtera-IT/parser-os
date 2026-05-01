from __future__ import annotations

from app.core.authority import (
    authority_rank,
    choose_governing_atoms,
    compare_atoms,
    is_governing_candidate,
)
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(
    atom_id: str,
    *,
    authority: AuthorityClass,
    atom_type: AtomType = AtomType.scope_item,
    confidence: float = 0.9,
    entity_keys: list[str] | None = None,
    raw_text: str = "scope includes cameras",
    timestamp: str | None = None,
    value: dict | None = None,
) -> EvidenceAtom:
    locator = {}
    if timestamp is not None:
        locator["timestamp"] = timestamp
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id="art_1",
        atom_type=atom_type,
        raw_text=raw_text,
        normalized_text=raw_text.lower(),
        value=value or {},
        entity_keys=entity_keys or ["site:west_wing"],
        source_refs=[
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id="art_1",
                artifact_type=ArtifactType.txt,
                filename="x.txt",
                locator=locator,
                extraction_method="test",
                parser_version="test",
            )
        ],
        authority_class=authority,
        confidence=confidence,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def test_deleted_text_cannot_govern() -> None:
    deleted = _atom("a_deleted", authority=AuthorityClass.deleted_text)
    active = _atom("a_active", authority=AuthorityClass.approved_site_roster)
    assert not is_governing_candidate(deleted)
    winner = choose_governing_atoms([deleted, active])
    assert [a.id for a in winner] == ["a_active"]


def test_customer_current_beats_quoted_old_email() -> None:
    quoted = _atom("a_quoted", authority=AuthorityClass.quoted_old_email)
    current = _atom("a_current", authority=AuthorityClass.customer_current_authored)
    decision = compare_atoms(quoted, current)
    assert decision.governing_atom_id == "a_current"


def test_customer_exclusion_beats_spreadsheet_scope_inclusion() -> None:
    exclusion = _atom(
        "a_customer_exclusion",
        authority=AuthorityClass.customer_current_authored,
        atom_type=AtomType.exclusion,
        raw_text="Please remove west wing from scope",
    )
    roster_scope = _atom(
        "a_roster_scope",
        authority=AuthorityClass.approved_site_roster,
        atom_type=AtomType.scope_item,
        raw_text="West wing in site roster",
    )
    decision = compare_atoms(exclusion, roster_scope)
    assert decision.governing_atom_id == exclusion.id


def test_approved_site_roster_beats_vendor_quote_for_scope() -> None:
    roster_scope = _atom("a_roster", authority=AuthorityClass.approved_site_roster, atom_type=AtomType.scope_item)
    vendor_scope = _atom("a_vendor", authority=AuthorityClass.vendor_quote, atom_type=AtomType.scope_item)
    decision = compare_atoms(roster_scope, vendor_scope)
    assert decision.governing_atom_id == roster_scope.id


def test_roster_quantity_governs_vendor_in_quantity_conflict_context() -> None:
    roster = _atom(
        "roster_q",
        authority=AuthorityClass.approved_site_roster,
        atom_type=AtomType.quantity,
        entity_keys=["connector:rj45"],
        value={"quantity": 72, "normalized_item": "rj45", "aggregate": True},
        raw_text="rj45 72",
    )
    vendor = _atom(
        "vendor_q",
        authority=AuthorityClass.vendor_quote,
        atom_type=AtomType.quantity,
        entity_keys=["part:x"],
        value={"quantity": 68, "normalized_item": "rj45"},
        raw_text="rj45 68",
    )
    from app.core.schemas import PacketFamily

    decision = compare_atoms(roster, vendor, context={"packet_family": PacketFamily.quantity_conflict})
    assert decision.governing_atom_id == roster.id
    assert decision.losing_atom_id == vendor.id


def test_roster_quantity_governs_vendor_in_vendor_mismatch_context() -> None:
    roster = _atom(
        "roster_utp",
        authority=AuthorityClass.approved_site_roster,
        atom_type=AtomType.quantity,
        entity_keys=["material:cat6_utp"],
        value={"quantity": 66, "normalized_item": "cat6_utp", "aggregate": True},
        raw_text="cat6 utp 66",
    )
    vendor = _atom(
        "vendor_utp",
        authority=AuthorityClass.vendor_quote,
        atom_type=AtomType.quantity,
        entity_keys=["part:utp"],
        value={"quantity": 60, "normalized_item": "cat6_utp"},
        raw_text="cat6 utp 60",
    )
    from app.core.schemas import PacketFamily

    decision = compare_atoms(roster, vendor, context={"packet_family": PacketFamily.vendor_mismatch})
    assert decision.governing_atom_id == roster.id
    assert decision.losing_atom_id == vendor.id


def test_vendor_quote_governs_only_in_vendor_context_without_scope() -> None:
    vendor_qty = _atom(
        "a_vendor_qty",
        authority=AuthorityClass.vendor_quote,
        atom_type=AtomType.quantity,
        entity_keys=["part:cam_ip_001"],
        value={"context": "vendor_mismatch", "part_number": "CAM-IP-001", "quantity": 72},
        raw_text="CAM-IP-001 qty 72",
    )
    machine_qty = _atom(
        "a_machine_qty",
        authority=AuthorityClass.machine_extractor,
        atom_type=AtomType.quantity,
        entity_keys=["part:cam_ip_001"],
        raw_text="Extracted quantity 70",
    )
    governing = choose_governing_atoms([vendor_qty, machine_qty])
    assert [a.id for a in governing] == ["a_vendor_qty"]


def test_tie_breaks_by_confidence() -> None:
    a = _atom("a1", authority=AuthorityClass.meeting_note, confidence=0.81)
    b = _atom("a2", authority=AuthorityClass.meeting_note, confidence=0.88)
    decision = compare_atoms(a, b)
    assert decision.governing_atom_id == "a2"


def test_deterministic_ordering() -> None:
    a = _atom("a2", authority=AuthorityClass.machine_extractor, confidence=0.8, timestamp="2026-04-10T10:00:00")
    b = _atom("a1", authority=AuthorityClass.machine_extractor, confidence=0.8, timestamp="2026-04-10T10:00:00")
    result_1 = [x.id for x in choose_governing_atoms([a, b])]
    result_2 = [x.id for x in choose_governing_atoms([b, a])]
    assert result_1 == result_2
    assert result_1 == ["a1"]


def test_authority_rank_values() -> None:
    assert authority_rank(AuthorityClass.contractual_scope) > authority_rank(AuthorityClass.vendor_quote)
