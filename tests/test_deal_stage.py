"""The stage/direction composition, pinned on the cases that made it necessary.

Fixtures are deal 010215's real transitions, read from HubSpot on 2026-08-31 and
cross-checked against the HubSpot UI timeline entry ("moved from Open- Awaiting
Scope to Submitted for Quoting", 2:11 PM EDT == 18:11:39Z).
"""
from app.core.document_lifecycle import deal_stage


TIMELINE = {
    "created_at": "2026-08-12T18:03:46.264Z",
    "transitions": [
        {"ts": "2026-08-12T18:03:46.264Z", "label": "Open- Awaiting Scope", "order": 0},
        {"ts": "2026-08-12T18:11:39.978Z", "label": "Submitted for Quoting", "order": 1},
        {"ts": "2026-08-13T15:53:00.994Z", "label": "Decision Pending", "order": 2},
        {"ts": "2026-08-28T12:40:20.296Z", "label": "Closed Won", "order": 3},
    ],
}


class TestStageAtArrival:
    def test_places_by_the_stage_in_force(self):
        at = deal_stage.stage_at_arrival
        assert at("2026-08-12T18:05:00Z", TIMELINE) == "Open- Awaiting Scope"
        assert at("2026-08-12T18:31:00Z", TIMELINE) == "Submitted for Quoting"
        assert at("2026-08-21T10:00:00Z", TIMELINE) == "Decision Pending"
        assert at("2026-08-29T10:00:00Z", TIMELINE) == "Closed Won"

    def test_pre_deal_material_is_its_own_bucket(self):
        assert deal_stage.stage_at_arrival("2026-08-01T00:00:00Z", TIMELINE) == "Before deal created"

    def test_boundary_belongs_to_the_stage_it_opens(self):
        assert deal_stage.stage_at_arrival("2026-08-12T18:11:39.978Z", TIMELINE) == "Submitted for Quoting"

    def test_undated_is_unplaced_not_bucketed(self):
        # Filing an unknown date under the first or last stage asserts a date we
        # do not have. None is the honest answer.
        assert deal_stage.stage_at_arrival(None, TIMELINE) is None
        assert deal_stage.stage_at_arrival("", TIMELINE) is None

    def test_no_timeline_is_unplaced(self):
        assert deal_stage.stage_at_arrival("2026-08-21T10:00:00Z", None) is None
        assert deal_stage.stage_at_arrival("2026-08-21T10:00:00Z", {"transitions": []}) is None


class TestAdmissibility:
    def test_the_case_stage_alone_gets_wrong(self):
        # Marion County School District Locations.docx: the customer sent us a
        # site list on 2026-08-21, deep inside Decision Pending. Stage alone
        # calls that "negotiation over our own output". It is evidence.
        adm, why = deal_stage.admissibility(
            stage="Decision Pending", direction="inbound", classified_as=None, timeline=TIMELINE,
        )
        assert adm == "evidence"
        assert "inbound" in why

    def test_our_own_output_stays_output_whenever_it_arrived(self):
        # A Deal Kit is our answer. Readmitting it as evidence teaches a model to
        # copy itself, which is the whole reason the label class exists.
        adm, _ = deal_stage.admissibility(
            stage="Open- Awaiting Scope", direction="outbound", classified_as="label", timeline=TIMELINE,
        )
        assert adm == "label"

    def test_outbound_after_quoting_is_our_output(self):
        adm, _ = deal_stage.admissibility(
            stage="Decision Pending", direction="outbound", classified_as=None, timeline=TIMELINE,
        )
        assert adm == "label"

    def test_outbound_before_quoting_is_still_scope_setting(self):
        adm, _ = deal_stage.admissibility(
            stage="Open- Awaiting Scope", direction="outbound", classified_as=None, timeline=TIMELINE,
        )
        assert adm == "evidence"

    def test_post_close_is_atlas(self):
        adm, _ = deal_stage.admissibility(
            stage="Closed Won", direction="inbound", classified_as=None, timeline=TIMELINE,
        )
        assert adm == "atlas"

    def test_pre_deal_is_discovery(self):
        adm, _ = deal_stage.admissibility(
            stage="Before deal created", direction=None, classified_as=None, timeline=TIMELINE,
        )
        assert adm == "evidence"

    def test_unknown_stage_defers_to_the_classifier(self):
        adm, why = deal_stage.admissibility(
            stage=None, direction="inbound", classified_as="reference", timeline=TIMELINE,
        )
        assert adm == "reference"
        assert "no stage" in why

    def test_a_file_with_no_direction_does_not_get_overruled(self):
        # Files carry no direction of their own. Stage alone is too weak to move
        # a classified document, so it defers rather than asserts.
        adm, why = deal_stage.admissibility(
            stage="Decision Pending", direction=None, classified_as="reference", timeline=TIMELINE,
        )
        assert adm == "reference"
        assert "no direction" in why

    def test_every_verdict_carries_a_reason(self):
        for stage in (None, "Before deal created", "Open- Awaiting Scope",
                      "Submitted for Quoting", "Decision Pending", "Closed Won", "Archive"):
            for direction in (None, "inbound", "outbound", "internal"):
                _, why = deal_stage.admissibility(
                    stage=stage, direction=direction, classified_as=None, timeline=TIMELINE,
                )
                assert why and isinstance(why, str), (stage, direction)


class TestAnnotate:
    def test_reports_when_it_overrules_the_classifier(self):
        out = deal_stage.annotate(
            {"admissible_for": "neither"},
            authored_at="2026-08-21T10:00:00Z",
            direction="inbound",
            timeline=TIMELINE,
        )
        assert out["stage_at_arrival"] == "Decision Pending"
        assert out["admissible_for"] == "evidence"
        assert out["changed_from_classifier"] is True
        assert out["unplaced"] is False

    def test_marks_an_undated_document_unplaced(self):
        out = deal_stage.annotate(
            None, authored_at=None, direction=None, timeline=TIMELINE,
        )
        assert out["unplaced"] is True
        assert out["stage_at_arrival"] is None

    def test_survives_a_missing_lifecycle_and_timeline(self):
        out = deal_stage.annotate(None, authored_at=None, direction=None, timeline=None)
        assert out["admissible_for"] is None
        assert out["unplaced"] is True


class TestRealHubspotLabels:
    """Stage labels are typed by people and are not clean strings.

    The live pipeline on 2026-08-31 reads "Submitted for Quoting " with a
    TRAILING SPACE and "Closed Won: 100%". The first version of this module
    compared them literally, found nothing, and `_stage_index` returned -1 --
    which made every stage compare as "after quoting", so our own outbound mail
    during discovery came out labelled as produced output. The fixtures above
    used idealised labels and sailed straight past it.
    """

    LIVE = {
        "created_at": "2026-08-12T18:03:46.264Z",
        "transitions": [
            {"ts": "2026-08-12T18:03:46.264Z", "label": "Open- Awaiting Scope", "order": 0},
            {"ts": "2026-08-12T18:11:39.978Z", "label": "Submitted for Quoting ", "order": 1},
            {"ts": "2026-08-13T15:53:00.994Z", "label": "Decision Pending", "order": 2},
            {"ts": "2026-08-28T12:40:20.296Z", "label": "Closed Won: 100%", "order": 3},
        ],
    }

    def test_outbound_before_quoting_is_evidence_despite_the_trailing_space(self):
        adm, _ = deal_stage.admissibility(
            stage="Open- Awaiting Scope", direction="outbound",
            classified_as=None, timeline=self.LIVE,
        )
        assert adm == "evidence"

    def test_outbound_after_quoting_is_still_label(self):
        adm, _ = deal_stage.admissibility(
            stage="Decision Pending", direction="outbound",
            classified_as=None, timeline=self.LIVE,
        )
        assert adm == "label"

    def test_the_quoting_stage_itself_counts_as_quoted(self):
        adm, _ = deal_stage.admissibility(
            stage="Submitted for Quoting ", direction="outbound",
            classified_as=None, timeline=self.LIVE,
        )
        assert adm == "label"

    def test_closed_won_with_a_percentage_suffix_routes_to_atlas(self):
        adm, _ = deal_stage.admissibility(
            stage="Closed Won: 100%", direction="inbound",
            classified_as=None, timeline=self.LIVE,
        )
        assert adm == "atlas"

    def test_a_timeline_without_a_quoting_stage_does_not_assume_one(self):
        # If we cannot see a quote going out, we must not claim outbound material
        # postdates it. Previously -1 >= -1 made this "label".
        tl = {"created_at": "2026-08-12T18:03:46Z", "transitions": [
            {"ts": "2026-08-12T18:03:46Z", "label": "Open- Awaiting Scope", "order": 0},
        ]}
        adm, _ = deal_stage.admissibility(
            stage="Open- Awaiting Scope", direction="outbound", classified_as=None, timeline=tl,
        )
        assert adm == "evidence"

    def test_display_label_is_stripped(self):
        # The section heading must not render a trailing space.
        assert deal_stage.stage_at_arrival("2026-08-12T18:31:00Z", self.LIVE) == "Submitted for Quoting"


class TestManifestSidecarIsNotADocument:
    """Purpulse's manifest is bookkeeping ABOUT the artifacts, never an artifact.

    Left in the scan, parser-os parsed it as a document. On deal 010215 that
    produced 789 of the envelope's 1,933 atoms -- 40% -- reading like:

        artifacts[55].mime_type: message/rfc822
        artifacts[42].metadata.ingestedAt: 2026-08-17T16:27:10.831Z
        artifacts[9].attachment_id: d5690dc6-6907-403e-b9f2-1614dddd24e3

    It also leaked a DIFFERENT deal's filename into this deal's evidence, and it
    was the only "undated" item in the whole deal -- every real HubSpot
    attachment carries a timestamp.
    """

    def test_the_sidecar_is_excluded_from_parsing(self):
        from pathlib import Path
        from app.core.compiler import _is_excluded_artifact

        assert _is_excluded_artifact(Path("/p/.parser_manifest.json"), Path("/p")) is True

    def test_real_documents_are_still_parsed(self):
        from pathlib import Path
        from app.core.compiler import _is_excluded_artifact

        for name in ("SOW Marion.docx", "Sodexo Breakdown.xlsx", "010215-hs-email-1.eml"):
            assert _is_excluded_artifact(Path(f"/p/{name}"), Path("/p")) is False, name
