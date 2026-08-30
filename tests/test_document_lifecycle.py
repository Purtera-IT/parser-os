"""The routing table and the state-claim check.

Both exist because a model got these wrong on real data, so the cases below are
the ones that actually failed, not invented ones.
"""
import pytest

from app.core.document_lifecycle import TAXONOMY, TYPES, check_state_claim, normalise, route


class TestRouting:
    def test_our_own_output_is_never_evidence(self):
        # 60% of the corpus is output. Admitting any of it as evidence is the leak
        # this package exists to close.
        for t in ("DEAL_KIT", "QUOTE", "ROM", "PROPOSAL", "CASE_MANIFEST", "WIN_WIRE"):
            stage, adm = route(t)
            assert stage == "QUOTED_OUTPUT"
            assert adm == "label", f"{t} must be label-only, got {adm}"

    def test_contract_paper_is_label_not_evidence(self):
        for t in ("SOW_DRAFT", "SOW_SIGNED", "CONTRACT", "CONTRACT_EXHIBIT"):
            assert route(t)[1] == "label"

    def test_delivery_documents_go_to_atlas_only(self):
        for t in ("PURCHASE_ORDER", "RUNBOOK", "INSTALL_INSTRUCTIONS", "CHANGE_ORDER"):
            assert route(t)[1] == "atlas"

    def test_discovery_and_quoting_are_the_only_evidence(self):
        evidence = {t for t in TYPES if route(t)[1] == "evidence"}
        stages = {route(t)[0] for t in evidence}
        assert stages <= {"DISCOVERY", "QUOTING"}

    @pytest.mark.parametrize("written", ["deal kit", "Deal-Kit", "DEAL_KIT", " deal  kit "])
    def test_type_spelling_does_not_change_routing(self, written):
        assert route(written) == ("QUOTED_OUTPUT", "label")

    def test_unknown_type_quarantines_rather_than_guessing(self):
        # A type we have never seen must reach a human, not fall into evidence.
        assert route("SOMETHING_NEW") == ("UNKNOWN", "quarantine")
        assert route("") == ("UNKNOWN", "quarantine")
        assert route(None) == ("UNKNOWN", "quarantine")

    def test_every_type_routes_somewhere(self):
        for t in TYPES:
            stage, adm = route(t)
            assert stage and adm
            assert t in TAXONOMY

    def test_normalise(self):
        assert normalise("sow signed") == "SOW_SIGNED"
        assert normalise("  Purchase-Order ") == "PURCHASE_ORDER"


class TestStateClaims:
    def test_sow_boilerplate_does_not_prove_signature(self):
        # The exact failure: 20 documents called SOW_SIGNED on this receipt.
        receipt = 'This Project Services Statement of Work ("SOW") is made by and between'
        kept, demoted, reason = check_state_claim("SOW_SIGNED", receipt, "scope, deliverables, pricing")
        assert kept == "SOW_DRAFT"
        assert demoted is True
        assert "no evidence" in reason

    @pytest.mark.parametrize("phrase", [
        "fully executed", "Authorized Signature", "signed by", "DocuSign Envelope",
        "countersigned", "Date Signed",
    ])
    def test_real_signature_evidence_is_accepted(self, phrase):
        kept, demoted, _ = check_state_claim("SOW_SIGNED", f"...{phrase}...", "")
        assert (kept, demoted) == ("SOW_SIGNED", False)

    def test_evidence_in_the_body_is_kept_but_flagged(self):
        kept, demoted, reason = check_state_claim("SOW_SIGNED", "scope of work", "fully executed on 1 May")
        assert kept == "SOW_SIGNED"
        assert demoted is False
        assert reason  # the quote was weak even though the claim holds

    def test_types_without_a_state_claim_pass_through(self):
        for t in ("DEAL_KIT", "SITE_LIST", "RATE_CARD"):
            assert check_state_claim(t, "", "") == (t, False, None)

    def test_change_order_needs_change_language(self):
        kept, demoted, _ = check_state_claim("CHANGE_ORDER", "quarterly summary", "nothing relevant")
        assert (kept, demoted) == ("OTHER", True)


class TestDataset:
    """The precomputed table the envelope reads. No model call in a compile."""

    def test_dataset_is_present_and_substantial(self):
        from app.core.document_lifecycle.dataset import coverage
        assert coverage() > 1000

    def test_lookup_accepts_a_full_length_sha(self):
        # The envelope holds a full sha256; the table is keyed on a 32-char prefix.
        import json
        from pathlib import Path
        from app.core.document_lifecycle.dataset import lookup
        data = json.loads((Path("app/core/document_lifecycle/data/lifecycle_by_sha.json")).read_text())
        key = next(iter(data))
        assert lookup(key + "f" * 32) == data[key]
        assert lookup(key.upper()) == data[key]

    def test_unknown_content_is_not_guessed(self):
        from app.core.document_lifecycle.dataset import lookup
        assert lookup("0" * 64) is None
        assert lookup("") is None
        assert lookup(None) is None

    def test_every_stored_record_routes_consistently(self):
        # A stored record must agree with the table it was derived from, or the
        # dataset and the routing logic have drifted apart.
        import json
        from pathlib import Path
        from app.core.document_lifecycle import route
        data = json.loads((Path("app/core/document_lifecycle/data/lifecycle_by_sha.json")).read_text())
        for sha, rec in data.items():
            stage, adm = route(rec["type"])
            assert (stage, adm) == (rec["stage"], rec["admissible_for"]), f"{sha}: {rec['type']}"

    def test_our_own_output_is_never_stored_as_evidence(self):
        # The whole point. If this fails, the leak is back.
        import json
        from pathlib import Path
        data = json.loads((Path("app/core/document_lifecycle/data/lifecycle_by_sha.json")).read_text())
        leaked = [s for s, r in data.items()
                  if r["stage"] in ("QUOTED_OUTPUT", "CONTRACTED") and r["admissible_for"] == "evidence"]
        assert not leaked, f"{len(leaked)} output document(s) marked as evidence"


class TestVisionFallback:
    """PDFs that are drawings, not documents."""

    def test_detects_a_drawing_wearing_a_text_layer(self):
        from app.core.document_lifecycle.vision_fallback import has_empty_text_layer
        # The real cases: a 13-page policy that yielded 596 chars of page furniture,
        # and a one-page AP placement drawing with 37 characters on it.
        assert has_empty_text_layer("x" * 596, 13) is True
        assert has_empty_text_layer("x" * 37, 1) is True
        assert has_empty_text_layer("", 2) is True

    def test_leaves_real_documents_alone(self):
        from app.core.document_lifecycle.vision_fallback import has_empty_text_layer
        assert has_empty_text_layer("x" * 6000, 3) is False
        assert has_empty_text_layer("x" * 2000, 1) is False

    def test_zero_pages_is_not_a_drawing(self):
        from app.core.document_lifecycle.vision_fallback import has_empty_text_layer
        assert has_empty_text_layer("", 0) is False

    def test_vision_read_documents_are_marked_as_such(self):
        # A label derived from looking at a picture should say so, so a reader can
        # weigh it differently from one derived from extracted text.
        import json
        from pathlib import Path
        data = json.loads(Path("app/core/document_lifecycle/data/lifecycle_by_sha.json").read_text())
        seen = [v for v in data.values() if v.get("read_by")]
        assert seen, "no vision-read documents recorded"
        assert all("vision" in v["read_by"] for v in seen)

    def test_site_drawings_became_evidence(self):
        # The point of the exercise: floor plans and AP placements were invisible.
        import json
        from pathlib import Path
        data = json.loads(Path("app/core/document_lifecycle/data/lifecycle_by_sha.json").read_text())
        vision = [v for v in data.values() if v.get("read_by")]
        evidence = [v for v in vision if v["admissible_for"] == "evidence"]
        assert len(evidence) >= 10, f"only {len(evidence)} of {len(vision)} became evidence"


class TestBaseHealthGates:
    """The gates that stop the contamination coming back."""

    def _env(self, docs):
        return {"atoms": [{"id": "a", "text": "x"}], "documents": docs}

    def test_output_read_as_evidence_is_caught(self):
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{
            "artifact_id": "1", "filename": "010215 Sodexo Deal Kit.xlsx",
            "lifecycle": {"type": "DEAL_KIT", "stage": "QUOTED_OUTPUT", "admissible_for": "evidence"},
        }]), "d")
        assert m.counts["output_document_as_evidence"] == 1
        assert "Deal Kit" in m.examples["output_document_as_evidence"][0]

    def test_a_signed_sow_read_as_evidence_is_caught(self):
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{
            "artifact_id": "1", "filename": "SOW signed.pdf",
            "lifecycle": {"type": "SOW_SIGNED", "stage": "CONTRACTED", "admissible_for": "evidence"},
        }]), "d")
        assert m.counts["output_document_as_evidence"] == 1

    def test_correctly_routed_output_does_not_fire(self):
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{
            "artifact_id": "1", "filename": "Deal Kit.xlsx",
            "lifecycle": {"type": "DEAL_KIT", "stage": "QUOTED_OUTPUT", "admissible_for": "label"},
        }]), "d")
        assert m.counts["output_document_as_evidence"] == 0

    def test_real_evidence_does_not_fire(self):
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{
            "artifact_id": "1", "filename": "LA Office - Wireless AP Placement.pdf",
            "lifecycle": {"type": "FLOOR_PLAN", "stage": "DISCOVERY", "admissible_for": "evidence"},
        }]), "d")
        assert m.counts["output_document_as_evidence"] == 0

    def test_mock_data_reaching_a_training_label_is_caught(self):
        # This actually happened: 07_contracting_procurement_packet.pdf, a
        # document stamped "Fictional data", was routed to `label`.
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{
            "artifact_id": "1", "filename": "07_contracting_procurement_packet.pdf",
            "lifecycle": {"type": "TEST_FIXTURE", "stage": "UNKNOWN", "admissible_for": "label"},
        }]), "d")
        assert m.counts["test_fixture_admitted"] == 1

    def test_quarantined_fixture_does_not_fire(self):
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{
            "artifact_id": "1", "filename": "mock.pdf",
            "lifecycle": {"type": "TEST_FIXTURE", "stage": "UNKNOWN", "admissible_for": "quarantine"},
        }]), "d")
        assert m.counts["test_fixture_admitted"] == 0

    def test_documents_without_lifecycle_are_ignored_not_failed(self):
        # Absent lifecycle means never classified. That is a gap, not a defect,
        # and these gates must stay silent about it.
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{"artifact_id": "1", "filename": "x.pdf"}]), "d")
        assert m.counts["output_document_as_evidence"] == 0
        assert m.counts["test_fixture_admitted"] == 0
