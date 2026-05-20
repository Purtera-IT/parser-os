"""Regression tests for site-code promotion and integration-doc boilerplate filtering."""

from __future__ import annotations

from app.core.entity_extraction import _emit_sites, is_site_boilerplate_slug
from app.core.entity_resolution import extract_entity_records
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def test_emit_sites_promotes_explicit_site_codes():
    text = "Sites ATL-HQ, ATL-WEST, and ATL-AIR are in scope. 1180 Peachtree St NE."
    keys = _emit_sites(text)
    assert "site:atl_hq" in keys
    assert "site:atl_west" in keys
    assert "site:atl_air" in keys
    assert any(k.startswith("address:") for k in keys)


def test_integration_doc_phrase_not_emitted_as_site_from_proper_noun_path():
    text = (
        "PurPulse HubSpot dev deal mock confidential integration notes "
        "procurement packet pdf orbitbrief parser azure"
    )
    keys = _emit_sites(text)
    assert "site:hubspot_dev_deal" not in keys
    assert "site:mock_confidential_this" not in keys


def test_is_site_boilerplate_slug_keeps_codes_drops_noise():
    assert not is_site_boilerplate_slug("atl_hq")
    assert is_site_boilerplate_slug("hubspot_dev_deal")
    assert is_site_boilerplate_slug("mock_confidential_this")


def test_extract_entity_records_drops_boilerplate_site_entities():
    atom = EvidenceAtom(
        id="atm_test",
        project_id="proj",
        artifact_id="art",
        atom_type=AtomType.scope_item,
        authority_class=AuthorityClass.contractual_scope,
        raw_text="HubSpot dev deal mock confidential procurement packet",
        normalized_text="hubspot dev deal mock confidential procurement packet",
        source_refs=[
            SourceRef(
                id="src_test",
                artifact_id="art",
                artifact_type=ArtifactType.txt,
                filename="x.txt",
                locator={"line": 1},
                extraction_method="test",
                parser_version="test",
            )
        ],
        entity_keys=["site:hubspot_dev_deal", "site:atl_hq"],
        confidence=1.0,
        review_status=ReviewStatus.auto_accepted,
        parser_version="test",
    )
    records = extract_entity_records("proj", [atom])
    keys = {r.canonical_key for r in records}
    assert "site:hubspot_dev_deal" not in keys
    assert "site:atl_hq" in keys
