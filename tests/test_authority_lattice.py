from __future__ import annotations

from app.core.authority import choose_governing_atoms, score_authority
from app.core.ids import stable_id
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _atom(
    atom_id: str,
    *,
    authority: AuthorityClass,
    atom_type: AtomType = AtomType.scope_item,
    entity_keys: list[str] | None = None,
    review_status: ReviewStatus = ReviewStatus.auto_accepted,
    locator: dict | None = None,
    confidence: float = 0.9,
) -> EvidenceAtom:
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id="art_1",
        atom_type=atom_type,
        raw_text=f"{atom_type.value} {atom_id}",
        normalized_text=f"{atom_type.value} {atom_id}",
        value={"context": "vendor_mismatch" if authority == AuthorityClass.vendor_quote else "scope"},
        entity_keys=entity_keys or ["site:west_wing"],
        source_refs=[
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id="art_1",
                artifact_type=ArtifactType.txt,
                filename="fixture.txt",
                locator=locator or {},
                extraction_method="test",
                parser_version="test",
            )
        ],
        authority_class=authority,
        confidence=confidence,
        review_status=review_status,
        review_flags=[],
        parser_version="test",
    )


def test_current_customer_exclusion_beats_meeting_note_exclusion() -> None:
    customer = _atom("a_customer", authority=AuthorityClass.customer_current_authored, atom_type=AtomType.exclusion)
    meeting = _atom("a_meeting", authority=AuthorityClass.meeting_note, atom_type=AtomType.exclusion)
    winners = choose_governing_atoms([meeting, customer], context={"packet_family": "scope_exclusion"})
    assert [atom.id for atom in winners] == ["a_customer"]


def test_meeting_note_exclusion_can_govern_when_only_candidate() -> None:
    meeting = _atom(
        "a_meeting_only",
        authority=AuthorityClass.meeting_note,
        atom_type=AtomType.exclusion,
        review_status=ReviewStatus.needs_review,
    )
    winners = choose_governing_atoms([meeting], context={"packet_family": "scope_exclusion"})
    assert [atom.id for atom in winners] == ["a_meeting_only"]


def test_vendor_quote_cannot_govern_scope_inclusion() -> None:
    vendor_scope = _atom("a_vendor_scope", authority=AuthorityClass.vendor_quote, atom_type=AtomType.scope_item)
    roster_scope = _atom("a_roster_scope", authority=AuthorityClass.approved_site_roster, atom_type=AtomType.scope_item)
    winners = choose_governing_atoms([vendor_scope, roster_scope], context={"packet_family": "scope_inclusion"})
    assert [atom.id for atom in winners] == ["a_roster_scope"]


def test_vendor_quote_can_govern_vendor_mismatch_procurement_context() -> None:
    vendor_qty = _atom("a_vendor_qty", authority=AuthorityClass.vendor_quote, atom_type=AtomType.quantity)
    machine_qty = _atom("a_machine_qty", authority=AuthorityClass.machine_extractor, atom_type=AtomType.quantity)
    winners = choose_governing_atoms([vendor_qty, machine_qty], context={"packet_family": "vendor_mismatch"})
    assert [atom.id for atom in winners] == ["a_vendor_qty"]


def test_approved_meeting_note_can_govern_without_higher_authority() -> None:
    approved_meeting = _atom(
        "a_meeting_approved",
        authority=AuthorityClass.meeting_note,
        review_status=ReviewStatus.approved,
    )
    machine = _atom("a_machine", authority=AuthorityClass.machine_extractor)
    winners = choose_governing_atoms([approved_meeting, machine], context={"packet_family": "meeting_decision"})
    assert [atom.id for atom in winners] == ["a_meeting_approved"]


def test_rejected_atom_never_governs() -> None:
    rejected_customer = _atom(
        "a_rejected",
        authority=AuthorityClass.customer_current_authored,
        review_status=ReviewStatus.rejected,
    )
    roster = _atom("a_roster", authority=AuthorityClass.approved_site_roster)
    winners = choose_governing_atoms([rejected_customer, roster], context={"packet_family": "scope_inclusion"})
    assert [atom.id for atom in winners] == ["a_roster"]


def test_quoted_old_email_never_governs_when_current_customer_exists() -> None:
    quoted = _atom("a_quoted", authority=AuthorityClass.quoted_old_email, atom_type=AtomType.customer_instruction)
    current = _atom("a_current", authority=AuthorityClass.customer_current_authored, atom_type=AtomType.customer_instruction)
    winners = choose_governing_atoms([quoted, current], context={"packet_family": "customer_override"})
    assert [atom.id for atom in winners] == ["a_current"]


def test_same_inputs_produce_same_authority_scores() -> None:
    atom = _atom(
        "a_stable",
        authority=AuthorityClass.approved_site_roster,
        atom_type=AtomType.quantity,
        locator={"timestamp": "2026-04-01T10:00:00", "speaker_role": "customer"},
    )
    score_one = score_authority(atom, [atom], context={"packet_family": "scope_inclusion"})
    score_two = score_authority(atom, [atom], context={"packet_family": "scope_inclusion"})
    assert score_one.model_dump() == score_two.model_dump()
