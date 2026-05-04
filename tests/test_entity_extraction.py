"""Regression tests for app.core.entity_extraction.

Locks in the universal entity extractor's behavior on real-world
samples drawn from the STRESS_* corpus.  Without these tests, future
parser changes could silently regress the entity-key population and
re-introduce the empty-keys bug from PRODUCTION_GAPS P0.2.
"""
from __future__ import annotations

import pytest

from app.core.entity_extraction import enrich_atoms, extract_keys
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.domain import load_domain_pack


SECURITY_PACK = load_domain_pack("security_camera")
WIRELESS_PACK = load_domain_pack("wireless")
AV_PACK = load_domain_pack("av")
ACCESS_PACK = load_domain_pack("access_control")
COPPER_PACK = load_domain_pack("copper_cabling")


def _make_atom(text: str, value: dict | None = None) -> EvidenceAtom:
    return EvidenceAtom(
        id="atm_test",
        project_id="test",
        artifact_id="art_test",
        atom_type=AtomType.scope_item,
        raw_text=text,
        normalized_text=text.lower(),
        value=value or {},
        entity_keys=[],
        source_refs=[
            SourceRef(
                id="src_test",
                artifact_id="art_test",
                artifact_type=ArtifactType.pdf,
                filename="test.pdf",
                locator={"page": 1},
                extraction_method="test",
                parser_version="test_v1",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test_v1",
    )


# ─── Vendor extraction ───
class TestVendorExtraction:
    def test_cisco_in_dna_license(self) -> None:
        keys = extract_keys(
            "Wireless Cisco DNA On-Prem Essential, 5Y Term Lic",
            pack=WIRELESS_PACK,
        )
        assert "vendor:cisco" in keys

    def test_genetec_in_acs_context(self) -> None:
        keys = extract_keys(
            "OCTA is currently using Lenel and Milestone, transitioning to Genetec.",
            pack=ACCESS_PACK,
        )
        assert "vendor:genetec" in keys
        assert "vendor:lenel" in keys
        assert "vendor:milestone" in keys

    def test_compound_vendor_naming(self) -> None:
        # "Mercury/Genetec SY-MP1502" should produce both vendor keys
        keys = extract_keys(
            "Provide and install one MP 1502 Mercury Intelligent controller - Model: Mercury/Genetec SY-MP1502",
            pack=ACCESS_PACK,
        )
        assert "vendor:mercury" in keys
        assert "vendor:genetec" in keys

    def test_bosch_dicentis(self) -> None:
        keys = extract_keys(
            "13 Bosch DCNM-DVT908 Bosch DICENTIS Discussion Device with Voting",
            pack=AV_PACK,
        )
        assert "vendor:bosch" in keys

    def test_aiphone_typo_handled(self) -> None:
        # The OCTA RFP uses the typo "Airphone" for Aiphone — accept both
        keys = extract_keys(
            "Zenitel, Grandstream, and Airphone IP based intercom systems",
            pack=ACCESS_PACK,
        )
        assert "vendor:aiphone" in keys
        assert "vendor:zenitel" in keys
        assert "vendor:grandstream" in keys


# ─── Part-number extraction ───
class TestPartNumberExtraction:
    def test_cisco_sku_with_suffix(self) -> None:
        keys = extract_keys(
            "Part Number: CW9166I-B | Description: Catalyst 9166I AP",
            pack=WIRELESS_PACK,
        )
        assert "part_number:cw9166i_b" in keys

    def test_cisco_dna_sku_multi_segment(self) -> None:
        keys = extract_keys(
            "Part Number: AIR-DNA-E-T-5Y | Description: Wireless Cisco DNA On-Prem Essential",
            pack=WIRELESS_PACK,
        )
        assert "part_number:air_dna_e_t_5y" in keys

    def test_bosch_dicentis_sku(self) -> None:
        keys = extract_keys(
            "13 Bosch DCNM-DVT908 Bosch DICENTIS Discussion Device with Voting",
            pack=AV_PACK,
        )
        assert "part_number:dcnm_dvt908" in keys


# ─── Quantity extraction ───
class TestQuantityExtraction:
    def test_qty_pattern(self) -> None:
        keys = extract_keys(
            "Part Number: CW9166I-B | Qty: 136",
            pack=WIRELESS_PACK,
        )
        assert "quantity:136" in keys

    def test_quantity_in_value(self) -> None:
        keys = extract_keys(
            "Some text",
            pack=WIRELESS_PACK,
            value={"quantity": 500},
        )
        assert "quantity:500" in keys

    def test_quantity_in_value_overrides_text(self) -> None:
        # Both should produce keys; the union covers each
        keys = extract_keys(
            "Qty: 136",
            pack=WIRELESS_PACK,
            value={"quantity": 500},
        )
        assert "quantity:136" in keys
        assert "quantity:500" in keys


# ─── Site / address extraction ───
class TestSiteExtraction:
    def test_school_with_suffix(self) -> None:
        keys = extract_keys(
            "Alameda Elementary School, 8613 Alameda St., Downey, CA 90242",
            pack=COPPER_PACK,
        )
        assert "site:alameda_elementary_school" in keys
        assert "address:8613_alameda_st" in keys

    def test_andrews_information_systems_bldg(self) -> None:
        keys = extract_keys(
            "A14. Storage will be centralized at Andrews Information Systems Bldg, 1700 Pratt Drive.",
            pack=SECURITY_PACK,
        )
        assert any(k.startswith("site:andrews_information_systems") for k in keys), keys
        assert "address:1700_pratt_drive" in keys

    def test_does_not_emit_site_for_form_field_template(self) -> None:
        keys = extract_keys(
            "FULL LEGAL NAME (PRINT) (Company name as it appears with your Federal "
            "Taxpayer Number): CONTACT NAME/TITLE (PRINT) | FEDERAL TAXPAYER "
            "NUMBER (ID#): SIGNATURE (IN INK) | col_4: DATE",
            pack=SECURITY_PACK,
        )
        # Form-field templates should produce NO site keys
        site_keys = [k for k in keys if k.startswith("site:")]
        assert site_keys == [], f"Form-field text leaked sites: {site_keys}"


# ─── Q&A markers ───
class TestQAMarkers:
    def test_qa_markers_extracted(self) -> None:
        keys = extract_keys(
            "Q1. I assume the Parking garage is the first project. A1. Yes, Perry Street is the first project.",
            pack=SECURITY_PACK,
        )
        assert "qa:q1" in keys
        assert "qa:a1" in keys


# ─── CSI MasterFormat ───
class TestCsiSection:
    def test_section_id_extracted(self) -> None:
        keys = extract_keys(
            "Section 28 13 00 — Electronic Access Control System",
            pack=ACCESS_PACK,
        )
        assert "spec_section:28_13_00" in keys

    def test_subsection_id_extracted(self) -> None:
        keys = extract_keys(
            "Per Section 25 50 02.01 — Vykon JACE 9000",
            pack=COPPER_PACK,
        )
        assert any(k.startswith("spec_section:25_50_02") for k in keys), keys


# ─── enrich_atoms (mutating helper) ───
class TestEnrichAtoms:
    def test_enriches_empty_keys_only(self) -> None:
        a1 = _make_atom("Provide IP cameras at Perry Street Parking Deck.")
        a2 = _make_atom("Vendor: Cisco Catalyst 9166I AP")
        # Pre-populate a2 with a key — should be left alone
        a2.entity_keys = ["vendor:custom"]
        enriched, total = enrich_atoms([a1, a2], SECURITY_PACK)
        assert enriched == 1  # only a1 was changed
        assert a2.entity_keys == ["vendor:custom"]  # untouched
        assert any(k.startswith("device:") for k in a1.entity_keys)
        assert total > 0

    def test_handles_empty_text(self) -> None:
        # extract_keys must handle empty / whitespace-only text without
        # crashing — the EvidenceAtom model rejects empty raw_text so we
        # test the underlying function directly here.
        assert extract_keys("", pack=SECURITY_PACK) == []
        assert extract_keys("   \n  ", pack=SECURITY_PACK) == []
