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
