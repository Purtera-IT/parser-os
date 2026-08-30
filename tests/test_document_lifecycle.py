"""The routing table and the state-claim check.

Both exist because a model got these wrong on real data, so the cases below are
the ones that actually failed, not invented ones.
"""
import pathlib
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

    def test_evidence_arriving_after_the_cut_is_caught(self):
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{
            "artifact_id": "1", "filename": "RFP Network Switch upgrade.docx",
            "lifecycle": {
                "type": "RFP", "stage": "DISCOVERY", "admissible_for": "evidence",
                "delivered_at": "2026-07-09T20:28:13Z", "after_cut": True,
            },
        }]), "d")
        assert m.counts["post_cut_evidence"] == 1
        assert "after the cut" in m.examples["post_cut_evidence"][0]

    def test_post_cut_output_is_not_double_counted_as_evidence(self):
        # A Deal Kit produced after the quote is the NORMAL case -- it is the
        # answer. It must not land in this gate; `output_document_as_evidence`
        # is the one that owns output, and only when output claims to be
        # evidence.
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{
            "artifact_id": "1", "filename": "Deal Kit.xlsx",
            "lifecycle": {
                "type": "DEAL_KIT", "stage": "QUOTED_OUTPUT", "admissible_for": "label",
                "delivered_at": "2026-08-20T10:00:00Z", "after_cut": True,
            },
        }]), "d")
        assert m.counts["post_cut_evidence"] == 0

    def test_evidence_before_the_cut_does_not_fire(self):
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{
            "artifact_id": "1", "filename": "floor plan.pdf",
            "lifecycle": {
                "type": "FLOOR_PLAN", "stage": "DISCOVERY", "admissible_for": "evidence",
                "delivered_at": "2026-05-01T10:00:00Z", "after_cut": False,
            },
        }]), "d")
        assert m.counts["post_cut_evidence"] == 0

    def test_documents_without_lifecycle_are_ignored_not_failed(self):
        # Absent lifecycle means never classified. That is a gap, not a defect,
        # and these gates must stay silent about it.
        from tools.base_health import measure_envelope
        m = measure_envelope(self._env([{"artifact_id": "1", "filename": "x.pdf"}]), "d")
        assert m.counts["output_document_as_evidence"] == 0
        assert m.counts["test_fixture_admitted"] == 0


class TestDealTimeline:
    """When a deal committed to an answer, and what that makes admissible."""

    def test_coverage_is_substantial(self):
        from app.core.document_lifecycle.timeline import coverage
        deals, with_cut = coverage()
        assert deals > 150
        assert with_cut > 50

    def test_events_are_ordered_and_carry_their_receipt(self):
        import json
        from pathlib import Path
        data = json.loads(Path("app/core/document_lifecycle/data/deal_timeline.json").read_text())
        for deal, rec in data.items():
            dates = [e["date"] for e in rec["events"]]
            assert dates == sorted(dates), f"{deal}: events out of order"
            for e in rec["events"]:
                assert e.get("receipt"), f"{deal}: event with no receipt was stored"

    def test_the_cut_is_one_of_the_deals_own_events(self):
        # The cut must be a moment that actually happened on the deal, never
        # interpolated.
        import json
        from pathlib import Path
        from app.core.document_lifecycle.timeline import COMMITTING_EVENTS
        data = json.loads(Path("app/core/document_lifecycle/data/deal_timeline.json").read_text())
        for deal, rec in data.items():
            cut = rec.get("quote_asof")
            if not cut:
                continue
            committing = [e["date"] for e in rec["events"] if e["type"] in COMMITTING_EVENTS]
            assert cut == min(committing), f"{deal}: cut is not its earliest committing event"

    def test_a_deal_still_in_discovery_has_no_cut(self):
        # None is a real answer: nothing has been promised, so nothing is stale.
        from app.core.document_lifecycle.timeline import quote_asof, is_after_cut
        assert quote_asof("a-deal-that-does-not-exist") is None
        assert is_after_cut("a-deal-that-does-not-exist", "2099-01-01") is False

    def test_unknowable_never_reads_as_stale(self):
        # Missing timestamp, missing cut: admissible until something proves
        # otherwise. Silence must not exclude evidence.
        from app.core.document_lifecycle.timeline import is_after_cut
        assert is_after_cut(None, "2099-01-01") is False
        assert is_after_cut("whatever", None) is False

    def test_before_and_after_a_real_cut(self):
        import json
        from pathlib import Path
        from app.core.document_lifecycle.timeline import is_after_cut
        data = json.loads(Path("app/core/document_lifecycle/data/deal_timeline.json").read_text())
        deal = next(d for d, v in data.items() if v.get("quote_asof"))
        assert is_after_cut(deal, "2099-01-01") is True
        assert is_after_cut(deal, "1999-01-01") is False


class TestPackaging:
    """The dataset is data, and setuptools ships no file it is not told about.

    This was a real, silent failure: the package installed cleanly into the
    container without its JSON, every lookup returned None, and the envelope
    carried `lifecycle: null` on all 65 documents of a deal whose 21 documents
    were all in the table. Nothing raised. A missing dataset must be loud.
    """

    def test_pyproject_ships_the_lifecycle_data(self):
        import tomllib
        from pathlib import Path
        cfg = tomllib.loads(Path("pyproject.toml").read_text())
        patterns = cfg["tool"]["setuptools"]["package-data"]["app"]
        assert any("document_lifecycle/data" in p for p in patterns), (
            "app/core/document_lifecycle/data/*.json is not in package-data; "
            "the wheel will install without its labels and silently classify nothing"
        )

    def test_both_datasets_exist_where_the_loaders_look(self):
        from app.core.document_lifecycle import dataset, timeline
        assert dataset._DATA.exists(), dataset._DATA
        assert timeline._DATA.exists(), timeline._DATA

    def test_loaders_report_real_coverage(self):
        # If this drops to zero in a deployed environment, the data did not ship.
        from app.core.document_lifecycle.dataset import coverage as doc_coverage
        from app.core.document_lifecycle.timeline import coverage as tl_coverage
        assert doc_coverage() > 1000
        assert tl_coverage()[0] > 150


class TestTheCut:
    """Timestamps, and the one comparison the cut actually turns on."""

    def test_the_same_instant_spelled_two_ways_is_not_after_itself(self):
        # The bug this replaced: the timeline writes naive UTC and a delivery
        # stamp arrives with a Z, so `"...:07Z" > "...:07"` was True as a string
        # and a document delivered exactly at the cut was ruled inadmissible.
        from app.core.document_lifecycle.timeline import parse_ts
        assert "2026-08-12T19:28:07Z" > "2026-08-12T19:28:07"      # the old test
        assert parse_ts("2026-08-12T19:28:07Z") == parse_ts("2026-08-12T19:28:07")

    def test_an_offset_is_normalised_to_utc(self):
        from app.core.document_lifecycle.timeline import parse_ts
        assert parse_ts("2026-08-12T15:28:07-04:00") == parse_ts("2026-08-12T19:28:07Z")

    def test_microseconds_parse(self):
        from app.core.document_lifecycle.timeline import parse_ts
        assert parse_ts("2026-05-22T19:35:16.654000") is not None

    def test_junk_is_none_not_an_exception(self):
        from app.core.document_lifecycle.timeline import parse_ts
        assert parse_ts("last Tuesday") is None
        assert parse_ts("") is None
        assert parse_ts(None) is None

    def test_delivered_at_takes_the_earliest_send(self):
        # A document re-sent after the quote did not become post-quote material.
        from app.core.document_lifecycle.timeline import delivered_at
        lc = {"delivered": [
            {"ts": "2026-08-20T10:00:00Z"},
            {"ts": "2026-06-01T09:00:00Z"},
            {"ts": "2026-07-04T12:00:00Z"},
        ]}
        assert delivered_at(lc) == "2026-06-01T09:00:00Z"

    def test_delivered_at_is_none_without_a_delivering_message(self):
        from app.core.document_lifecycle.timeline import delivered_at
        assert delivered_at({"delivered": []}) is None
        assert delivered_at({}) is None
        assert delivered_at(None) is None

    def test_an_unparseable_stamp_does_not_win_the_min(self):
        from app.core.document_lifecycle.timeline import delivered_at
        lc = {"delivered": [{"ts": "whenever"}, {"ts": "2026-06-01T09:00:00Z"}]}
        assert delivered_at(lc) == "2026-06-01T09:00:00Z"

    def test_missing_information_keeps_a_document_admissible(self):
        # Every unanswerable case must be False. Ruling real evidence OUT is the
        # expensive direction of this error.
        from app.core.document_lifecycle.timeline import is_after_cut
        assert is_after_cut(None, "2026-08-20T10:00:00Z") is False
        assert is_after_cut("no-such-deal", "2026-08-20T10:00:00Z") is False
        assert is_after_cut("no-such-deal", None) is False

    def test_a_real_deal_orders_around_its_own_cut(self):
        from app.core.document_lifecycle.timeline import _table, is_after_cut, quote_asof
        deal = next(d for d, v in _table().items() if v.get("quote_asof"))
        cut = quote_asof(deal)
        assert is_after_cut(deal, cut) is False, "the cut is not after itself"
        assert is_after_cut(deal, "2099-01-01T00:00:00Z") is True
        assert is_after_cut(deal, "2000-01-01T00:00:00Z") is False


class TestLookupIsolation:
    """The dataset is cached and shared; a caller's annotation must not stick."""

    def test_lookup_returns_a_copy(self):
        from app.core.document_lifecycle.dataset import lookup, _table
        sha = next(iter(_table()))
        first = lookup(sha)
        first["after_cut"] = True
        assert "after_cut" not in lookup(sha), "annotation leaked into the shared table"


class TestTheCutReachesTheEnvelope:
    """The wiring itself: a real compile must carry the deal's timeline."""

    def _compile(self, tmp_path, project_id):
        import pytest
        from app.core.compiler import compile_project
        from app.core.orbitbrief_envelope import build_orbitbrief_envelope

        src = (
            pathlib.Path(__file__).resolve().parents[1]
            / "real_data_cases" / "COPPER_001_SPRING_LAKE_AUDITORIUM" / "CASE_DOSSIER.pdf"
        )
        if not src.is_file():
            pytest.skip(f"Fixture PDF not present: {src}")
        project = tmp_path / "proj"
        project.mkdir()
        (project / src.name).write_bytes(src.read_bytes())
        result = compile_project(project_dir=project, project_id=project_id)
        return build_orbitbrief_envelope(project_dir=project, compile_result=result)

    def test_a_deal_with_a_cut_carries_it(self, tmp_path):
        from app.core.document_lifecycle.timeline import _table, quote_asof
        deal = next(d for d, v in _table().items() if v.get("quote_asof"))
        env = self._compile(tmp_path, deal)

        tl = env["deal_timeline"]
        assert tl["known"] is True
        assert tl["quote_asof"] == quote_asof(deal)
        assert tl["events"], "a deal with a cut has the events the cut came from"
        assert all(e.get("receipt") for e in tl["events"]), "every event keeps its sentence"
        assert isinstance(tl["documents_after_cut"], int)

    def test_an_unknown_deal_says_so_rather_than_implying_discovery(self, tmp_path):
        # null quote_asof is ambiguous on its own -- "still in discovery" and
        # "we have no timeline for this deal" are different facts. `known`
        # separates them, and this is the case that would otherwise read as the
        # permissive one.
        env = self._compile(tmp_path, "00000000-0000-0000-0000-000000000000")
        tl = env["deal_timeline"]
        assert tl["known"] is False
        assert tl["quote_asof"] is None
        assert tl["events"] == []

    def test_every_document_is_annotated_or_honestly_absent(self, tmp_path):
        env = self._compile(tmp_path, "env_smoke_cut")
        for d in env["documents"]:
            life = d.get("lifecycle")
            if life is None:
                continue          # never classified -- a gap, not a guess
            assert "after_cut" in life and "delivered_at" in life
            assert life["after_cut"] is False, "an unknown deal has no cut to be after"
