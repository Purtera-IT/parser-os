"""Week 5 regression tests (PRODUCTION_GAPS Week 5 polish + recall).

Locks in the Week 5 changes that close the VT_CAM gold-standard gap:

* Q+A answer body drives atom_type classification (decision /
  action_item / exclusion / customer_instruction) and the ``A\\d.``
  authority marker promotes scope/open_question atoms to
  ``customer_instruction``.
* Cross-pack vendor catalog now covers T2 Systems, ThyssenKrupp,
  ESRI / ArcSDE.
* Two-word organization names ("Virginia Tech", "Boston College") and
  leading-article runs ("The Andrews Information Systems Building")
  emit ``site:`` keys.
* ``security_camera`` pack carries a ``device:ups`` alias group.
* Single-token vendor candidates surface in
  ``ontology_gaps.collect_gap_candidates``.
* ``app.core.graph_invariants`` accepts ``customer_instruction`` atoms
  whose text matches an exclusion / constraint pattern as valid
  endpoints for ``excludes`` / ``requires`` edges (matches what
  ``graph_builder`` actually produces).
* ``app.core.quality_metrics._stage_durations_ms`` reads ``stage_name``
  from CompileTrace stages (was silently empty in Week 4).
"""
from __future__ import annotations

from app.core.entity_extraction import enrich_atoms, extract_keys
from app.core.graph_invariants import check_graph_invariants
from app.core.ontology_gaps import detect_ontology_gaps
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    CompileStageTrace,
    CompileTrace,
    EdgeType,
    EvidenceAtom,
    EvidenceEdge,
    ReviewStatus,
    SourceRef,
)
from app.domain import load_domain_pack, set_active_domain_pack
from app.parsers.orbitbrief_pdf import _classify_text_block, _split_qa_blob

SECURITY_PACK = load_domain_pack("security_camera")
ACCESS_PACK = load_domain_pack("access_control")


def _make_atom(text: str, atom_type: AtomType = AtomType.scope_item) -> EvidenceAtom:
    return EvidenceAtom(
        id=f"atm_{abs(hash(text)) % 10**12}",
        project_id="test",
        artifact_id="art_test",
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value={},
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


# ─── Q+A classifier (Week 5 P5.1) ─────────────────────────────────


class TestQAClassifier:
    def test_pure_question_classifies_as_open_question(self) -> None:
        atom_type, auth = _classify_text_block(
            text="Q1. Is this in scope?",
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.open_question
        assert auth == AuthorityClass.contractual_scope

    def test_q_with_decision_answer_promotes_to_decision(self) -> None:
        atom_type, auth = _classify_text_block(
            text=(
                "Q66. Where will video management be located? A66. "
                "Centralized at the Andrews Information Systems Building."
            ),
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.decision
        assert auth == AuthorityClass.customer_current_authored

    def test_q_with_action_item_answer_promotes(self) -> None:
        atom_type, auth = _classify_text_block(
            text=(
                "Q5. Vendor must describe? A5. Vendor must describe the on-site "
                "support available."
            ),
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.action_item
        assert auth == AuthorityClass.customer_current_authored

    def test_q_with_exclusion_answer_promotes(self) -> None:
        atom_type, auth = _classify_text_block(
            text=(
                "Q12. Do you need centralized controllers? A12. We would "
                "not be needing centralized controllers."
            ),
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.exclusion
        assert auth == AuthorityClass.customer_current_authored

    def test_q_with_plain_answer_falls_back_to_customer_instruction(self) -> None:
        # A47 from VT_CAM — plain declarative answer, no strong override.
        atom_type, auth = _classify_text_block(
            text=(
                "Q47. Cabling? A47. Fiber has been pulled to the parking "
                "structure and is available in the communications closet."
            ),
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.customer_instruction
        assert auth == AuthorityClass.customer_current_authored

    def test_a_marker_alone_promotes_scope_item(self) -> None:
        # When a chunk starts directly at the answer ("A18. ..."),
        # classify atom_type from the answer body and authority from the
        # A-marker.
        atom_type, auth = _classify_text_block(
            text="A18. The RFP requests that the proposed solution protect existing investments.",
            section_path=[],
            kind="paragraph",
        )
        assert atom_type == AtomType.customer_instruction
        assert auth == AuthorityClass.customer_current_authored


class TestQABlobSplit:
    def test_paragraph_with_two_pairs_splits_into_two_chunks(self) -> None:
        text = (
            "Q1. Question one? A1. Answer one. Q2. Question two? A2. Answer two."
        )
        chunks = _split_qa_blob(text)
        assert len(chunks) == 2
        assert chunks[0].startswith("Q1.")
        assert chunks[1].startswith("Q2.")
        # Each Q+A pair stays merged
        assert "A1." in chunks[0]
        assert "A2." in chunks[1]

    def test_single_pair_does_not_split(self) -> None:
        chunks = _split_qa_blob("Q1. Question. A1. Answer.")
        # Single Q+A pair coalesced into one chunk → returns either the
        # original ([text]) or the singleton coalesced result; either is
        # acceptable as long as no spurious splitting happens.
        assert len(chunks) <= 2


# ─── Cross-pack vendor catalog (Week 5 P5.2) ──────────────────────


class TestCrossPackVendors:
    def test_t2_systems_detected(self) -> None:
        keys = extract_keys(
            "We currently use a T2 Systems parking management product.",
            pack=SECURITY_PACK,
        )
        assert "vendor:t2_systems" in keys

    def test_thyssenkrupp_detected(self) -> None:
        keys = extract_keys(
            "Elevators are managed by ThyssenKrupp service.",
            pack=SECURITY_PACK,
        )
        assert "vendor:thyssenkrupp" in keys

    def test_esri_arcsde_detected(self) -> None:
        keys = extract_keys(
            "Mapped through a middleware product called ArcSDE — ESRI integration.",
            pack=SECURITY_PACK,
        )
        # ESRI canonical, ArcSDE rolls into the same vendor
        assert "vendor:esri" in keys


# ─── Two-word org sites (Week 5 P5.4) ─────────────────────────────


class TestTwoWordOrgSites:
    def test_virginia_tech(self) -> None:
        keys = extract_keys(
            "Virginia Tech will work with the successful offeror.",
            pack=SECURITY_PACK,
        )
        assert "site:virginia_tech" in keys

    def test_boston_college(self) -> None:
        keys = extract_keys(
            "Boston College has selected an integrator.",
            pack=SECURITY_PACK,
        )
        assert "site:boston_college" in keys

    def test_houston_isd(self) -> None:
        keys = extract_keys(
            "The Houston ISD has approved the cabling plan.",
            pack=SECURITY_PACK,
        )
        assert "site:houston_isd" in keys

    def test_random_two_word_does_not_emit(self) -> None:
        # Two capitalized words whose tail isn't an org-suffix should NOT
        # produce a site key.  Keeps the matcher conservative.
        keys = extract_keys(
            "Some Random words appearing in text.",
            pack=SECURITY_PACK,
        )
        site_keys = [k for k in keys if k.startswith("site:")]
        # We tolerate other site detections (street addresses, etc.) but
        # the bare 2-word run should not be in the output.
        assert "site:some_random" not in site_keys

    def test_leading_article_stripped(self) -> None:
        # "The Andrews Information Systems Building" → drop The, slug
        # the rest.
        keys = extract_keys(
            "The Andrews Information Systems Building is the central facility.",
            pack=SECURITY_PACK,
        )
        assert any(k.startswith("site:andrews_information_systems") for k in keys)


# ─── UPS device alias (Week 5 P5.4 follow-up) ─────────────────────


class TestUpsDeviceAlias:
    def test_ups_in_security_camera_pack(self) -> None:
        keys = extract_keys(
            "Vendors should recommend UPS units as part of their turn-key response.",
            pack=SECURITY_PACK,
        )
        assert "device:ups" in keys

    def test_uninterruptible_power_supply_alias(self) -> None:
        keys = extract_keys(
            "Each rack will include a rackmount uninterruptible power supply.",
            pack=SECURITY_PACK,
        )
        assert "device:ups" in keys


# ─── Single-token vendor gap detection (Week 5 P5.3) ──────────────


class TestSingleTokenVendorGaps:
    def test_thyssenkrupp_surfaces_in_gap_report(self) -> None:
        # ThyssenKrupp does not match the multi-word "vendor catalog"
        # path — it must show up via the single-token candidate
        # detector for ontology_gaps.
        atoms = [
            _make_atom("Vendor identification: ThyssenKrupp service contract."),
            _make_atom("ThyssenKrupp manages elevator maintenance on this site."),
        ]
        report = detect_ontology_gaps(atoms=atoms, pack=SECURITY_PACK)
        vendor_gaps = [g for g in report.get("vocab_gaps", []) if g.get("kind") == "vendor"]
        phrases = [g.get("phrase", "").lower() for g in vendor_gaps]
        assert any("thyssen" in p for p in phrases), phrases


# ─── graph_invariants alignment (Week 5 P5.5) ─────────────────────


class TestGraphInvariantsAlignment:
    def test_excludes_edge_from_customer_instruction_with_pack_pattern(self) -> None:
        # graph_builder treats a customer_instruction whose text matches
        # one of the *active pack's* exclusion patterns as
        # exclusion-bearing.  The validator must accept that.  We use
        # ``not in scope`` because it's in the security_camera pack's
        # exclusion_patterns list.
        set_active_domain_pack(SECURITY_PACK)
        ci = _make_atom(
            "Backup generators are not in scope for this project.",
            atom_type=AtomType.customer_instruction,
        )
        target = _make_atom(
            "Provide UPS units throughout the facility.",
            atom_type=AtomType.scope_item,
        )
        edge = EvidenceEdge(
            id="edge_test",
            project_id="test",
            from_atom_id=ci.id,
            to_atom_id=target.id,
            edge_type=EdgeType.excludes,
            reason="Exclusion atom applies to target entity context",
            confidence=0.9,
            metadata={"edge_family": "exclusion_application"},
        )
        errors = check_graph_invariants([ci, target], [edge])
        # No "excludes edge must involve exclusion atom" error.
        assert not any("excludes edge must involve exclusion atom" in e for e in errors), errors

    def test_excludes_edge_from_unrelated_customer_instruction_rejected(self) -> None:
        # The relaxation only accepts customer_instruction atoms whose
        # text matches an exclusion pattern; a customer_instruction
        # without a match must still fail validation.
        set_active_domain_pack(SECURITY_PACK)
        ci = _make_atom(
            "Please provide IP cameras at every entrance.",
            atom_type=AtomType.customer_instruction,
        )
        target = _make_atom(
            "Install controllers at the loading dock.",
            atom_type=AtomType.scope_item,
        )
        edge = EvidenceEdge(
            id="edge_bad",
            project_id="test",
            from_atom_id=ci.id,
            to_atom_id=target.id,
            edge_type=EdgeType.excludes,
            reason="bogus excludes edge",
            confidence=0.9,
            metadata={},
        )
        errors = check_graph_invariants([ci, target], [edge])
        assert any("excludes edge must involve exclusion atom" in e for e in errors), errors

    def test_literal_exclusion_atom_still_accepted(self) -> None:
        # Smoke test: a real ``exclusion`` atom_type must continue to
        # validate without depending on the active pack at all.
        ex = _make_atom(
            "Cabling is not included in this scope.",
            atom_type=AtomType.exclusion,
        )
        target = _make_atom(
            "Provide IP cameras with PoE backed by UPS.",
            atom_type=AtomType.scope_item,
        )
        edge = EvidenceEdge(
            id="edge_real_excl",
            project_id="test",
            from_atom_id=ex.id,
            to_atom_id=target.id,
            edge_type=EdgeType.excludes,
            reason="Exclusion atom applies",
            confidence=0.9,
            metadata={},
        )
        errors = check_graph_invariants([ex, target], [edge])
        assert not any("excludes edge must involve exclusion atom" in e for e in errors), errors


# ─── stage_durations_ms (Week 5 P5.6) ─────────────────────────────


class TestStageDurationsMs:
    def test_quality_metrics_reads_stage_name(self) -> None:
        from app.core.quality_metrics import _stage_durations_ms

        trace = CompileTrace(
            project_id="test",
            compile_id="cmp_test",
            stages=[
                CompileStageTrace(
                    stage_name="parse_artifacts",
                    started_at="2026-05-03T00:00:00Z",
                    completed_at="2026-05-03T00:00:00Z",
                    duration_ms=42.5,
                    input_count=1,
                    output_count=10,
                ),
                CompileStageTrace(
                    stage_name="graph_build",
                    started_at="2026-05-03T00:00:00Z",
                    completed_at="2026-05-03T00:00:00Z",
                    duration_ms=17.0,
                    input_count=10,
                    output_count=4,
                ),
            ],
            total_duration_ms=59.5,
            artifact_count=1,
            atom_count=10,
            entity_count=4,
            edge_count=2,
            packet_count=3,
        )
        durations = _stage_durations_ms(trace)
        assert durations == {"parse_artifacts": 42.5, "graph_build": 17.0}

    def test_handles_none_trace(self) -> None:
        from app.core.quality_metrics import _stage_durations_ms

        assert _stage_durations_ms(None) == {}
