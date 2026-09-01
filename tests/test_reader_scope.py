"""What a model was allowed to know when it did the work.

Training a Deal Kit model on a finished deal is only honest if it sees what the
person saw. Feed it documents that arrived afterwards and it learns from its own
answer -- brilliant on history, useless live. On a manually-worked corpus that
is the default outcome, not an edge case.

The timeline below is deal 010215's, exactly as HubSpot recorded it.
"""
from __future__ import annotations

import pytest

from app.core.document_lifecycle.reader_scope import audit, consumers, visible_to

TIMELINE = {
    "created_at": "2026-08-12T18:03:46Z",
    "transitions": [
        {"ts": "2026-08-12T18:03:46Z", "label": "Open- Awaiting Scope", "order": 1},
        {"ts": "2026-08-12T18:11:39Z", "label": "Submitted for Quoting", "order": 2},
        {"ts": "2026-08-13T15:53:00Z", "label": "Decision Pending", "order": 3},
        {"ts": "2026-08-28T12:40:20Z", "label": "Closed Won: 100%", "order": 4},
    ],
}


def ok(consumer, stage, adm="evidence"):
    return visible_to(consumer, stage=stage, admissible_for=adm, timeline=TIMELINE)


def test_deal_kit_reads_through_the_quoting_window():
    # Submitted for Quoting is working time, not output time: the kit is built
    # DURING it. On 010215 the kit was authored 15:50:31 and the stage moved on
    # at 15:53:00 -- two and a half minutes later.
    assert ok("deal_kit", "Open- Awaiting Scope")[0] is True
    assert ok("deal_kit", "Submitted for Quoting")[0] is True


def test_deal_kit_cannot_read_what_came_after_the_quote():
    seen, why = ok("deal_kit", "Decision Pending")
    assert seen is False
    assert "postdates" in why


def test_sow_reads_the_negotiation_the_deal_kit_cannot():
    # A SOW is negotiated after the quote goes out, so it legitimately sees a
    # window the Deal Kit must not.
    assert ok("sow", "Decision Pending")[0] is True
    assert ok("deal_kit", "Decision Pending")[0] is False


def test_neither_reads_past_close():
    assert ok("sow", "Closed Won: 100%")[0] is False
    assert ok("deal_kit", "Closed Won: 100%")[0] is False


def test_atlas_reads_the_whole_deal_including_delivery_material():
    # The default readable set is evidence+reference, which silently gave atlas
    # the same 27 documents as the SOW model on deal 010215 and called that
    # "the whole deal". Delivery material is exactly what atlas is for.
    assert ok("atlas", "Closed Won: 100%")[0] is True
    assert ok("atlas", "Closed Won: 100%", adm="atlas")[0] is True
    # It still may not read our quote as input.
    assert ok("atlas", "Decision Pending", adm="label")[0] is False


def test_our_own_output_is_never_input_however_early_it_arrived():
    # The Deal Kit sits INSIDE the Deal Kit's own window. It is excluded by
    # type, not by a time buffer -- a buffer would be arbitrary and would drop
    # real customer mail that landed late in the window.
    seen, why = ok("deal_kit", "Submitted for Quoting", adm="label")
    assert seen is False
    assert "not readable by deal_kit" in why


def test_delivery_material_is_not_scope_input():
    assert ok("deal_kit", "Open- Awaiting Scope", adm="atlas")[0] is False
    assert ok("deal_kit", "Open- Awaiting Scope", adm="neither")[0] is False


def test_a_customer_document_in_the_window_is_readable():
    # The case the whole feature turns on: ten "SOW Smarthands" files, the
    # customer's, arriving at 18:05 -- inside Open/Awaiting Scope.
    assert ok("deal_kit", "Open- Awaiting Scope", adm="evidence")[0] is True


def test_pre_deal_material_is_discovery_context():
    seen, why = ok("deal_kit", "Before deal created")
    assert seen is True
    assert "predates" in why


def test_an_unplaced_document_is_refused_out_loud():
    # Excluding it loses real material; including it asserts a date we do not
    # have. Refused, and the reason says which, so an auditor sees the gap.
    seen, why = ok("deal_kit", None)
    assert seen is False
    assert "no stage" in why


def test_a_stage_absent_from_the_timeline_is_refused_not_guessed():
    seen, why = visible_to(
        "deal_kit", stage="Some Stage Nobody Configured", admissible_for="evidence", timeline=TIMELINE
    )
    assert seen is False
    assert "not in this deal's timeline" in why


def test_a_deal_that_never_reached_quoting_refuses_rather_than_admits_everything():
    # If the cut itself is not in the timeline, ordering is undefined. Admitting
    # everything would hand a model the whole deal and call it gated.
    short = {"transitions": [{"ts": "2026-08-12T18:03:46Z", "label": "Open- Awaiting Scope", "order": 1}]}
    seen, why = visible_to(
        "deal_kit", stage="Open- Awaiting Scope", admissible_for="evidence", timeline=short
    )
    assert seen is False
    assert "cannot place the cut" in why


def test_an_unknown_consumer_reads_nothing():
    seen, why = ok("sowsmith_v9", "Open- Awaiting Scope")
    assert seen is False
    assert "unknown consumer" in why


@pytest.mark.parametrize("consumer", ["deal_kit", "sow", "atlas"])
def test_every_consumer_gives_a_reason_for_every_answer(consumer):
    for stage in [None, "Before deal created", "Open- Awaiting Scope", "Decision Pending"]:
        _, why = ok(consumer, stage)
        assert why and isinstance(why, str)


def test_audit_splits_a_deal_and_explains_each_side():
    docs = [
        {
            "filename": "SOW Smarthands Marion County SD Marion High School.docx",
            "direction": "inbound",
            "authored_at": "2026-08-12T18:05:00Z",
            "deal_stage": {"stage_at_arrival": "Open- Awaiting Scope", "admissible_for": "evidence"},
        },
        {
            "filename": "010215  Sodexo Deal Kit.xlsx",
            "authored_at": "2026-08-13T15:50:31Z",
            "deal_stage": {"stage_at_arrival": "Submitted for Quoting", "admissible_for": "label"},
        },
        {
            "filename": "SOW_204630 203891.pdf",
            "authored_at": "2026-08-26T17:13:31Z",
            "deal_stage": {"stage_at_arrival": "Decision Pending", "admissible_for": "label"},
        },
    ]
    out = audit(docs, consumer="deal_kit", timeline=TIMELINE)
    assert out["visible_count"] == 1
    assert out["visible"][0]["filename"].startswith("SOW Smarthands")
    assert out["hidden_count"] == 2
    assert all(row["why"] for row in out["hidden"])


def test_consumers_are_discoverable():
    assert set(consumers()) >= {"deal_kit", "sow", "atlas"}


# ── cutting at the produced artifact's own timestamp ────────────────────────
#
# The stage boundary is administrative and the work is not. On deal 010215 the
# Deal Kit was authored 15:50:30 and the deal moved to Decision Pending at
# 15:53:00 -- so a note that arrived 15:52:24, 114 seconds AFTER the kit
# existed, sat inside the model's readable set. That is the model reading its
# own future.

from app.core.document_lifecycle.reader_scope import cut_at

KIT = {
    "filename": "010215  Sodexo Deal Kit.xlsx",
    "authored_at": "2026-08-13T15:50:30.720Z",
    "deal_stage": {"stage_at_arrival": "Submitted for Quoting", "admissible_for": "label"},
}
LATE_NOTE = {
    "filename": "010215-hs-note-114812740653-Note.txt",
    "authored_at": "2026-08-13T15:52:24.032Z",
    "deal_stage": {"stage_at_arrival": "Submitted for Quoting", "admissible_for": "evidence"},
}
CUSTOMER_SOW = {
    "filename": "SOW Smarthands Marion County SD Marion High School.docx",
    "authored_at": "2026-08-12T18:05:00Z",
    "deal_stage": {"stage_at_arrival": "Open- Awaiting Scope", "admissible_for": "evidence"},
}


def test_the_cut_is_the_moment_our_own_kit_was_authored():
    assert cut_at("deal_kit", [KIT, LATE_NOTE, CUSTOMER_SOW]) == "2026-08-13T15:50:30.720Z"


def test_a_customer_document_named_sow_never_sets_the_cut():
    # The customer's files are named "SOW Smarthands ..." and are exactly what
    # the model is meant to read. Letting one anchor the boundary would cut the
    # deal at the customer's first message.
    assert cut_at("sow", [CUSTOMER_SOW]) is None


def test_the_earliest_output_wins_when_there_are_several():
    later = dict(KIT, filename="010215 Deal Kit v2.xlsx", authored_at="2026-08-20T09:00:00Z")
    # A revised kit issued later does not license reading what came between.
    assert cut_at("deal_kit", [later, KIT]) == KIT["authored_at"]


def test_no_output_on_the_deal_means_no_cut_to_apply():
    # A cut we cannot locate is not a cut we should invent; the stage boundary
    # stands instead.
    assert cut_at("deal_kit", [CUSTOMER_SOW]) is None
    assert cut_at("deal_kit", []) is None


def test_the_114_second_leak_is_closed():
    out = audit([KIT, LATE_NOTE, CUSTOMER_SOW], consumer="deal_kit", timeline=TIMELINE)
    names = {r["filename"] for r in out["visible"]}
    assert CUSTOMER_SOW["filename"] in names
    assert LATE_NOTE["filename"] not in names
    why = next(r["why"] for r in out["hidden"] if r["filename"] == LATE_NOTE["filename"])
    assert "after deal_kit output was authored" in why


def test_the_stage_boundary_still_applies_without_a_produced_artifact():
    out = audit([CUSTOMER_SOW, LATE_NOTE], consumer="deal_kit", timeline=TIMELINE)
    assert out["produced_at"] is None
    # Both are inside Submitted for Quoting, so both remain readable.
    assert out["visible_count"] == 2


def test_our_own_output_is_still_excluded_by_type_not_by_the_clock():
    # The SOW attached alongside the kit is withheld because of WHAT it is.
    # The clock only closes the gap for genuine inbound material.
    out = audit([KIT, LATE_NOTE, CUSTOMER_SOW], consumer="deal_kit", timeline=TIMELINE)
    why = next(r["why"] for r in out["hidden"] if r["filename"] == KIT["filename"])
    assert "not readable by deal_kit" in why


def test_orbitbrief_reads_every_stage_but_not_our_own_answer():
    # The compile builds the picture of the deal, so no stage cut applies. It
    # still may not read our own output, or the brief quotes us back to
    # ourselves.
    assert ok("orbitbrief", "Decision Pending")[0] is True
    assert ok("orbitbrief", "Closed Won: 100%")[0] is True
    assert ok("orbitbrief", "Decision Pending", adm="label")[0] is False


def test_every_named_consumer_is_present():
    assert set(consumers()) == {"deal_kit", "sow", "orbitbrief", "atlas"}
