from __future__ import annotations

from app.core.entity_resolution import (
    extract_entity_records,
    normalize_entity_key,
    resolve_aliases,
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


def _atom(atom_id: str, entity_keys: list[str]) -> EvidenceAtom:
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id="art_1",
        atom_type=AtomType.scope_item,
        raw_text="scope text",
        normalized_text="scope text",
        value={"text": "scope text"},
        entity_keys=entity_keys,
        source_refs=[
            SourceRef(
                id=stable_id("src", atom_id),
                artifact_id="art_1",
                artifact_type=ArtifactType.xlsx,
                filename="site_list.xlsx",
                locator={},
                extraction_method="test",
                parser_version="test",
            )
        ],
        authority_class=AuthorityClass.approved_site_roster,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def test_west_wing_variants_resolve_to_same_canonical_key() -> None:
    atoms = [
        _atom("a1", ["site:West Wing"]),
        _atom("a2", ["site:west-wing"]),
    ]
    records = extract_entity_records("proj_1", atoms)
    resolved = resolve_aliases(records)
    site_records = [r for r in resolved if r.entity_type == "site"]
    assert len(site_records) == 1
    assert site_records[0].canonical_key == "site:west_wing"


def test_ip_cameras_normalizes_to_device_ip_camera() -> None:
    assert normalize_entity_key("device", "IP Cameras") == "device:ip_camera"
