"""Tests for open-question resolution + gap-question generation."""

from __future__ import annotations

from app.core.open_question_resolution import (
    filter_unhelpful_open_questions,
    generate_gap_questions,
    is_answered_question,
    is_unhelpful_pm_question,
    resolve_open_questions,
)
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(atom_type, text, *, entity_keys=None, value=None, aid="art_x", rid=None):
    return EvidenceAtom(
        id=rid or f"atm_{abs(hash((atom_type, text))) % (10**12):012x}",
        project_id="p",
        artifact_id=aid,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=value if value is not None else {},
        entity_keys=entity_keys or [],
        source_refs=[
            SourceRef(
                id="src_1",
                artifact_id=aid,
                artifact_type=ArtifactType.txt,
                filename="f.txt",
                locator={},
                extraction_method="test",
                parser_version="t",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.8,
        confidence_raw=0.8,
        calibrated_confidence=0.8,
        review_status=ReviewStatus.needs_review,
        review_flags=[],
        parser_version="t",
    )


def test_question_answered_by_device_atom_is_suppressed():
    atoms = [
        _atom(AtomType.open_question, "What size TVs?", entity_keys=["device:display"],
              value={"kind": "paragraph", "qa_split": True}, rid="atm_q1"),
        _atom(AtomType.entity, "LG 65UN570H0UD 65 inch display", entity_keys=["device:display"], rid="atm_d1"),
    ]
    n = resolve_open_questions(atoms)
    assert n == 1
    q = atoms[0]
    assert is_answered_question(q)
    assert q.review_status == ReviewStatus.auto_accepted
    assert q.value.get("answered") is True


def test_question_answered_by_requirement_atom_is_suppressed():
    key = "requirement:perform_an_onsite_inventory_count"
    atoms = [
        _atom(AtomType.open_question, "Will techs need to perform an inventory count?",
              entity_keys=[key], value={"kind": "paragraph", "qa_split": True}, rid="atm_q2"),
        _atom(AtomType.requirement, "Perform an onsite inventory count before work begins.",
              entity_keys=[key], rid="atm_r1"),
    ]
    assert resolve_open_questions(atoms) == 1
    assert is_answered_question(atoms[0])


def test_genuinely_open_question_not_suppressed():
    atoms = [
        _atom(AtomType.open_question, "Who is the site contact?",
              entity_keys=["stakeholder:unknown"], rid="atm_q3"),
        _atom(AtomType.entity, "some display", entity_keys=["device:display"], rid="atm_d2"),
    ]
    # stakeholder: is not answer-bearing, and no other atom shares it
    assert resolve_open_questions(atoms) == 0
    assert not is_answered_question(atoms[0])


def test_visual_page_marker_never_suppressed():
    atoms = [
        _atom(AtomType.open_question, "PDF page 3 has unextracted diagram",
              entity_keys=["device:display"], value={"kind": "visual_page_marker"}, rid="atm_q4"),
        _atom(AtomType.entity, "display", entity_keys=["device:display"], rid="atm_d3"),
    ]
    assert resolve_open_questions(atoms) == 0
    assert not is_answered_question(atoms[0])


def test_generic_site_key_does_not_answer():
    atoms = [
        _atom(AtomType.open_question, "What are the access hours at the site?",
              entity_keys=["site:santa_fe_87506"], rid="atm_q5"),
        _atom(AtomType.physical_site, "Santa Fe site", entity_keys=["site:santa_fe_87506"], rid="atm_s1"),
    ]
    # site: is intentionally NOT answer-bearing
    assert resolve_open_questions(atoms) == 0


def test_transcript_dialogue_questions_are_filtered_from_pm_gaps():
    noisy = [
        _atom(AtomType.open_question, "have the what G6?  Yeah, we have some.", rid="atm_n1"),
        _atom(
            AtomType.open_question,
            "Part of it. Jacob Vander-Plaats [21:50] Yep. Daniel Peterson [21:51] How quickly do you think you could be able to have the style over?",
            rid="atm_n2",
        ),
        _atom(
            AtomType.open_question,
            "Yeah, we'll start working on it now. Eddie, is there anything else you need in order to start getting the price?",
            rid="atm_n3",
        ),
        _atom(AtomType.open_question, "How much is it for? Daniel Peterson [02:55] It's just time and materials.", rid="atm_n4"),
    ]
    kept, dropped = filter_unhelpful_open_questions(noisy)
    assert kept == []
    assert len(dropped) == 4
    assert all(is_unhelpful_pm_question(q) for q in dropped)


def test_hybrid_transcript_pricing_question_kept_for_conversation_graph():
    """Diarized 'How much is it for?' must stay — not PM-gap noise."""
    q = _atom(AtomType.open_question, "How much is it for?", rid="atm_price_q")
    q.source_refs[0].locator = {
        "page": 2,
        "block_kind": "transcript_turn",
        "utterance_index": 20,
        "speaker": "Jacob Vander-Plaats",
        "timestamp_start": "02:53",
        "hybrid_plan": "hybrid",
    }
    q.value = {"kind": "transcript_turn", "speaker": "Jacob Vander-Plaats", "text": "How much is it for?"}
    kept, dropped = filter_unhelpful_open_questions([q])
    assert kept == [q]
    assert dropped == []
    assert not is_unhelpful_pm_question(q)


def test_real_pm_question_survives_quality_filter():
    atoms = [
        _atom(AtomType.open_question, "Who is the on-site contact for the Pittsburgh office?", rid="atm_real"),
    ]
    kept, dropped = filter_unhelpful_open_questions(atoms)
    assert kept == atoms
    assert dropped == []


def test_generate_gap_questions_from_missing():
    checklist = {
        "missing": [
            {"field_id": "site_contact"},
            {"field_id": "kickoff_date"},
            {"field_id": "payment_terms"},
            {"field_id": "some_unmapped_field"},
        ]
    }
    gaps = generate_gap_questions(checklist)
    ids = {g["field_id"] for g in gaps}
    assert "site_contact" in ids
    assert "kickoff_date" in ids
    assert "payment_terms" in ids
    assert "some_unmapped_field" not in ids  # only curated fields produce questions
    assert all(g["kind"] == "generated_gap" and g["summary"] for g in gaps)


def test_generate_gap_questions_empty_when_present():
    assert generate_gap_questions({"missing": []}) == []
    assert generate_gap_questions({}) == []
