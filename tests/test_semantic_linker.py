from __future__ import annotations

from app.core.ids import stable_id
from app.core.packetizer import build_packets
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EdgeType,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.domain.loader import load_domain_pack
from app.semantic.linker import propose_semantic_link_candidates
from app.semantic.vectorizer import sentence_transformer_enabled


def _atom(
    atom_id: str,
    *,
    atom_type: AtomType,
    text: str,
    entity_keys: list[str],
    authority: AuthorityClass = AuthorityClass.machine_extractor,
) -> EvidenceAtom:
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id="art_1",
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value={"text": text},
        entity_keys=entity_keys,
        source_refs=[
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id="art_1",
                artifact_type=ArtifactType.txt,
                filename="fixture.txt",
                locator={"line_start": 1, "line_end": 1},
                extraction_method="test",
                parser_version="test_v1",
            )
        ],
        authority_class=authority,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test_v1",
    )


def test_alias_terms_can_propose_same_as_with_domain_pack() -> None:
    pack = load_domain_pack("default_pack")
    a = _atom("a", atom_type=AtomType.entity, text="IP Camera install", entity_keys=["device:ip_camera"])
    b = _atom("b", atom_type=AtomType.entity, text="security camera install", entity_keys=["device:ip_camera"])
    candidates = propose_semantic_link_candidates([a, b], domain_pack=pack)
    assert any(
        row.proposed_edge_type == EdgeType.same_as and row.status == "accepted"
        for row in candidates
    )


def test_constraint_phrases_support_candidate() -> None:
    a = _atom(
        "a",
        atom_type=AtomType.constraint,
        text="Main Campus escort after 5pm",
        entity_keys=["site:main_campus"],
        authority=AuthorityClass.customer_current_authored,
    )
    b = _atom(
        "b",
        atom_type=AtomType.constraint,
        text="escort access after hours at Main Campus",
        entity_keys=["site:main_campus"],
        authority=AuthorityClass.meeting_note,
    )
    candidates = propose_semantic_link_candidates([a, b])
    assert any(row.proposed_edge_type == EdgeType.supports for row in candidates)


def test_unrelated_sites_are_not_merged() -> None:
    a = _atom("a", atom_type=AtomType.entity, text="Main Campus", entity_keys=["site:main_campus"])
    b = _atom("b", atom_type=AtomType.entity, text="West Wing", entity_keys=["site:west_wing"])
    candidates = propose_semantic_link_candidates([a, b])
    assert not any(row.proposed_edge_type == EdgeType.same_as and row.status == "accepted" for row in candidates)


def test_semantic_linker_never_creates_contradiction_edge() -> None:
    a = _atom("a", atom_type=AtomType.scope_item, text="Install cameras", entity_keys=["site:main_campus"])
    b = _atom("b", atom_type=AtomType.scope_item, text="Deploy security cameras", entity_keys=["site:main_campus"])
    candidates = propose_semantic_link_candidates([a, b])
    assert all(row.proposed_edge_type != EdgeType.contradicts for row in candidates)


def test_semantic_edges_cannot_govern_packets_alone() -> None:
    a = _atom("a", atom_type=AtomType.entity, text="IP Camera", entity_keys=["device:ip_camera"])
    b = _atom("b", atom_type=AtomType.entity, text="security camera", entity_keys=["device:ip_camera"])
    packets = build_packets("proj_1", atoms=[a, b], entities=[], edges=[], attach_metadata=False)
    assert packets == []


def test_tfidf_output_is_deterministic() -> None:
    a = _atom("a", atom_type=AtomType.constraint, text="Escort after 5pm", entity_keys=["site:main_campus"])
    b = _atom("b", atom_type=AtomType.constraint, text="escort access after hours", entity_keys=["site:main_campus"])
    first = propose_semantic_link_candidates([a, b])
    second = propose_semantic_link_candidates([a, b])
    assert [row.model_dump(mode="json") for row in first] == [row.model_dump(mode="json") for row in second]


def test_sentence_transformer_feature_flag_disabled_by_default() -> None:
    assert sentence_transformer_enabled() is False
