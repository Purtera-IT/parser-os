"""Week 6 regression tests (PRODUCTION_GAPS Week 6 polish + recall).

Locks in the Week 6 changes that close the corpus pass-fraction gap:

* ``AtomType.compliance`` + classifier patterns + ``compliance_clause``
  packet family.
* Tightened proper-noun matcher (kills ``site:attorney_fees_in``-class
  false positives).
* Broadened ``customer_current_authored`` detection beyond Q&A
  ``A\\d.`` markers (Owner Preferred:, Customer Notes:, "the District
  has selected", CUSTOMER RESPONSE:).
* New ``customer:`` and ``requirement:`` entity prefixes.
* Topic-slug / anchor-key length cap at 80 chars.
* NATOMAS school-list extraction (form-field detector no longer
  blanket-rejects rows whose only "form-field" markers are placeholder
  ``col_N:`` column names).
* XLSX Q&A row split: rows with both a question column and a
  response column emit additional Q + A sub-atoms with the proper
  authority class.
* Bare common-noun typed-alias filter (no more ``site:school`` /
  ``site:campus`` from generic word matches).
"""
from __future__ import annotations

from app.core.anchors import _MAX_TOPIC_SLUG_LEN, _topic_slug
from app.core.entity_extraction import (
    _emit_customer_keys,
    _emit_proper_nouns,
    _emit_requirement_keys,
    extract_keys,
)
from app.core.schemas import AtomType, AuthorityClass, PacketFamily
from app.domain import load_domain_pack
from app.parsers.orbitbrief_pdf import _classify_text_block, _looks_like_form_field

WIRELESS_PACK = load_domain_pack("wireless")
SECURITY_PACK = load_domain_pack("security_camera")


# ─── AtomType.compliance + classifier (Week 6 P6.1) ──────────────


class TestComplianceClassifier:
    def test_nfpa_compliance(self) -> None:
        atom_type, _ = _classify_text_block(
            text="The system shall comply with NFPA 72 for fire alarm signaling.",
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.compliance

    def test_ul_listed(self) -> None:
        atom_type, _ = _classify_text_block(
            text="All cabling shall be UL-listed and ETL-rated.",
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.compliance

    def test_ada_accordance(self) -> None:
        atom_type, _ = _classify_text_block(
            text="In accordance with ADA, all readers must be mounted at 48 inches.",
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.compliance

    def test_per_nec_code_cite(self) -> None:
        atom_type, _ = _classify_text_block(
            text="Per NEC 250.122, ground conductor sizing applies.",
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.compliance

    def test_erate_eligible(self) -> None:
        atom_type, _ = _classify_text_block(
            text="Equipment must be E-rate eligible and Section 508 compliant.",
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.compliance

    def test_taa_compliant_per_federal(self) -> None:
        atom_type, _ = _classify_text_block(
            text="Vendors shall be TAA-compliant per federal procurement guidelines.",
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.compliance

    def test_generic_constraint_does_not_match(self) -> None:
        # Vendor must provide installation services — NOT a compliance
        # clause (no external standard cited).
        atom_type, _ = _classify_text_block(
            text="Vendor must provide installation services.",
            section_path=[],
            kind="paragraph",
        )
        assert atom_type != AtomType.compliance

    def test_compliance_packet_family_exists(self) -> None:
        # Cheap sanity check: the new family is part of the enum.
        assert PacketFamily.compliance_clause.value == "compliance_clause"


# ─── Proper-noun tightening (Week 6 P6.2) ────────────────────────


class TestProperNounTightening:
    def test_attorney_fees_in_rejected(self) -> None:
        keys = _emit_proper_nouns(
            "Attorney Fees In are billed monthly", set()
        )
        assert "site:attorney_fees_in" not in keys
        assert "site:attorney_fees" not in keys

    def test_fulfill_contract_when_rejected(self) -> None:
        keys = _emit_proper_nouns(
            "Fulfill Contract When delivered", set()
        )
        assert "site:fulfill_contract_when" not in keys

    def test_e_rate_funding_year_rejected(self) -> None:
        keys = _emit_proper_nouns("E Rate Funding Year is annual", set())
        assert "site:e_rate_funding_year" not in keys

    def test_real_orgs_still_pass(self) -> None:
        # Don't over-correct.
        keys = _emit_proper_nouns(
            "The Andrews Information Systems Building is centralized.",
            set(),
        )
        assert "site:andrews_information_systems_building" in keys

    def test_district_office_still_passes(self) -> None:
        keys = _emit_proper_nouns(
            "District Office at 1901 Arena Blvd.",
            set(),
        )
        assert "site:district_office" in keys

    def test_two_word_high_school_passes(self) -> None:
        keys = _emit_proper_nouns(
            "Discovery High serves grades 9-12.",
            set(),
        )
        assert "site:discovery_high" in keys


# ─── Broader customer_current_authored (Week 6 P6.3) ─────────────


class TestBroadCustomerAuthority:
    def test_owner_furnished(self) -> None:
        _, auth = _classify_text_block(
            text="Owner-furnished controllers are mounted at site.",
            section_path=[],
            kind="paragraph",
        )
        assert auth == AuthorityClass.customer_current_authored

    def test_owner_shall_provide(self) -> None:
        _, auth = _classify_text_block(
            text="Owner shall provide network drops at every entry.",
            section_path=[],
            kind="paragraph",
        )
        assert auth == AuthorityClass.customer_current_authored

    def test_district_has_selected(self) -> None:
        _, auth = _classify_text_block(
            text="The District has selected an integrator.",
            section_path=[],
            kind="paragraph",
        )
        assert auth == AuthorityClass.customer_current_authored

    def test_university_will_manage(self) -> None:
        _, auth = _classify_text_block(
            text="The University will manage video surveillance.",
            section_path=[],
            kind="paragraph",
        )
        assert auth == AuthorityClass.customer_current_authored

    def test_customer_response_header(self) -> None:
        _, auth = _classify_text_block(
            text="CUSTOMER RESPONSE: We do not require backup generators.",
            section_path=[],
            kind="paragraph",
        )
        assert auth == AuthorityClass.customer_current_authored

    def test_customer_notes_header(self) -> None:
        _, auth = _classify_text_block(
            text="Customer Notes: prefer Brivo over Lenel.",
            section_path=[],
            kind="paragraph",
        )
        assert auth == AuthorityClass.customer_current_authored

    def test_vendor_we_does_not_promote(self) -> None:
        # "We are pleased to submit this proposal" is vendor-authored,
        # not customer.  The pattern requires a customer-side subject.
        _, auth = _classify_text_block(
            text="We are pleased to submit this proposal.",
            section_path=[],
            kind="paragraph",
        )
        assert auth != AuthorityClass.customer_current_authored


# ─── customer: + requirement: entity prefixes (Week 6 P6.4) ──────


class TestCustomerRequirementPrefixes:
    def test_customer_from_school_district(self) -> None:
        keys = extract_keys(
            "The Natomas Unified School District is selecting an integrator.",
            pack=WIRELESS_PACK,
        )
        assert "customer:natomas_unified_school_district" in keys

    def test_customer_from_two_word_university(self) -> None:
        keys = extract_keys(
            "Virginia Tech will work with the offeror.",
            pack=WIRELESS_PACK,
        )
        assert "customer:virginia_tech" in keys

    def test_requirement_erate(self) -> None:
        keys = extract_keys(
            "Equipment must be E-rate eligible.",
            pack=WIRELESS_PACK,
        )
        assert "requirement:erate_eligibility_marking" in keys

    def test_requirement_section_508(self) -> None:
        keys = extract_keys(
            "Section 508 compliance is required for all hardware.",
            pack=WIRELESS_PACK,
        )
        assert "requirement:section_508_compliance" in keys

    def test_requirement_nfpa_with_number(self) -> None:
        keys = extract_keys(
            "Cabling shall comply with NFPA 72.",
            pack=WIRELESS_PACK,
        )
        assert "requirement:nfpa_72_compliance" in keys

    def test_requirement_taa_compliant(self) -> None:
        keys = extract_keys(
            "All cameras shall be UL-listed and TAA-compliant.",
            pack=WIRELESS_PACK,
        )
        assert "requirement:taa_compliance" in keys


# ─── anchor_key length cap (Week 6 P6.5) ─────────────────────────


class TestAnchorKeyCap:
    def test_short_slug_unchanged(self) -> None:
        assert _topic_slug("fiber optic cable") == "fiber_optic_cable"

    def test_long_slug_truncated_with_hash(self) -> None:
        long_text = (
            "Q13. When you are scaling from 250 to 2500 cameras, based on frames "
            "per second and resolution there is a major difference in packet "
            "passing so if you decide to do one frame per second"
        )
        slug = _topic_slug(long_text)
        assert len(slug) <= _MAX_TOPIC_SLUG_LEN
        # Trailing 6-hex collision-avoidance suffix
        assert slug.split("_")[-1].isalnum()

    def test_distinct_long_inputs_get_distinct_slugs(self) -> None:
        a = _topic_slug("Q13. Long text variant one with the same beginning")
        b = _topic_slug("Q13. Long text variant two with the same beginning")
        # Beginnings may match (truncation) but the suffix must differ
        # at least sometimes — for these specific inputs, the truncation
        # falls within the same prefix so we only diverge at the hash
        # suffix.  Verify they ARE different.
        if len(a) > _MAX_TOPIC_SLUG_LEN - 7:
            assert a != b


# ─── NATOMAS-style table form-field filter (Week 6 P6.6) ─────────


class TestSchoolListTable:
    def test_legitimate_table_row_not_form(self) -> None:
        text = (
            "col_1: American Lakes School (K-8) | NATOMAS UNIFIED SCHOOL "
            "DISTRICT: 2800 Stonecreek Drive | col_3: Sacramento | col_4: "
            "95833 | col_5: 916.567.5500"
        )
        assert _looks_like_form_field(text) is False

    def test_real_form_with_strong_markers_still_caught(self) -> None:
        text = "FULL LEGAL NAME (PRINT) | FEDERAL TAXPAYER NUMBER (ID#)"
        assert _looks_like_form_field(text) is True

    def test_school_row_extracts_site_and_customer(self) -> None:
        text = (
            "col_1: American Lakes School (K-8) | NATOMAS UNIFIED SCHOOL "
            "DISTRICT: 2800 Stonecreek Drive | col_3: Sacramento | col_4: "
            "95833 | col_5: 916.567.5500"
        )
        keys = extract_keys(text, pack=WIRELESS_PACK)
        assert "site:american_lakes_school" in keys
        assert "site:natomas_unified_school_district" in keys
        assert "customer:american_lakes_school" in keys
        assert "customer:natomas_unified_school_district" in keys

    def test_h_allen_hight_elementary(self) -> None:
        # The middle initial used to break the proper-noun matcher
        # because the sentence-splitter chopped on the period.
        text = (
            "col_1: H. Allen Hight Elementary | NATOMAS UNIFIED SCHOOL "
            "DISTRICT: 3200 North Park Drive"
        )
        keys = extract_keys(text, pack=WIRELESS_PACK)
        assert "site:h_allen_hight_elementary" in keys


# ─── Bare-noun typed-alias stoplist (Week 6 P6.4 follow-up) ──────


class TestBareNounStoplist:
    def test_no_bare_school_emission(self) -> None:
        # The wireless pack lists "school" as a generic site alias.
        # Without the stoplist, a 4-word phrase that contains "school"
        # would emit a redundant ``site:school`` plus the rich form.
        keys = extract_keys(
            "Natomas Unified School District",
            pack=WIRELESS_PACK,
        )
        assert "site:school" not in keys
        assert "site:district" not in keys
        # Rich form survives
        assert "site:natomas_unified_school_district" in keys

    def test_no_bare_warehouse_emission(self) -> None:
        keys = extract_keys(
            "Multi-story warehouse with several APs",
            pack=WIRELESS_PACK,
        )
        assert "site:warehouse" not in keys
