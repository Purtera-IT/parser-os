"""Each head is served the context its answer actually turns on.

The clearest case: "If you all would be able to do something like this, I will
get a conversation going with the club owner" is blocked on US when a reseller
writes it to us and on THEM when we write it to them. Identical words. A head
served only the words cannot learn that -- it can only memorise the sentence.
"""
from __future__ import annotations

import json

from app.learning.human_labels import CLOSED_READS, PRESENT, rows_for_deal
from app.learning.label_context import V2_PARTS, context_text, parts_for

GATE = "If you all would be able to do something like this, I will get a conversation going with the club owner."

THEIRS = {
    "text": GATE,
    "said_by": {"company": "cdw.com", "role": "reseller", "side": "theirs", "email": "alecandrich@cdw.com"},
    "said_to": [{"company": "purtera-it.com", "role": "internal", "side": "ours", "email": "aj@purtera-it.com"}],
    "doc_type": "email",
    "neighbors_below": ["Note the diagram is labeled by the vendor/Huzzard"],
}
OURS = {
    **THEIRS,
    "said_by": THEIRS["said_to"][0],
    "said_to": [THEIRS["said_by"]],
}


def test_the_type_head_still_sees_exactly_what_it_always_saw():
    # Its gold predates per-head context. If this string moves, every row in
    # _training_*.db is out of distribution against every new one.
    assert parts_for("atom_type") == V2_PARTS
    lb = {"text": "Relay", "lead_in": ["Provided by us:"], "section": ["Bill of materials"], "table_ref": "t1"}
    assert context_text("atom_type", lb) == "Relay [table: t1] [section: Bill of materials] [intro: Provided by us:]"
    # and none of the new parts leak into it
    assert "[from:" not in context_text("atom_type", THEIRS)


def test_blocked_on_can_see_who_spoke_and_the_same_words_read_differently():
    theirs = context_text("reads:blocked_on", THEIRS)
    ours = context_text("reads:blocked_on", OURS)
    assert theirs != ours, "the head must be able to tell these apart"
    assert "[from: cdw.com (reseller, theirs) -> purtera-it.com (internal, ours)]" in theirs
    assert "[from: purtera-it.com (internal, ours) -> cdw.com (reseller, theirs)]" in ours


def test_a_party_is_a_company_and_a_role_never_a_name():
    # "Alec" appears in one deal; "cdw.com (reseller, theirs)" is what carries
    # to the next reseller.
    got = context_text("reads:blocked_on", THEIRS)
    assert "alecandrich" not in got and "aj@" not in got


def test_an_internal_note_says_so_instead_of_naming_our_own_people():
    lb = {**THEIRS, "internal_only": True, "said_by": {"company": "purtera-it.com", "role": "internal", "side": "ours"}}
    assert "our own people" in context_text("about", lb)


def test_a_head_is_not_handed_context_it_cannot_use():
    assert "below" not in parts_for("reads:blocked_on")
    assert "[below:" not in context_text("reads:blocked_on", THEIRS)
    assert "below" in parts_for("reads:opens_block")


def _rows(**label):
    doc = {
        "deal_id": "c2a3bdce-53bb-4da6-a840-a5260841685a",
        "purpose": "train",
        "labels": [{"label_type": "deal_metadata", "labeler": "developer@purtera-it.com", **THEIRS, **label}],
    }
    return rows_for_deal(doc)


def test_about_wants_and_every_reading_become_rows():
    rows = _rows(about="account", wants="decide-internally",
                 reads_set={"blocked_on": "us", "commitment": "get a conversation going with the club owner"},
                 reads_shown=["commitment"])
    by = {r["relation"]: r for r in rows}
    assert "about" in by and by["about"]["label"] == "account"
    assert "wants" in by and by["wants"]["label"] == "decide-internally"
    # a closed set is a class
    assert by["reads:blocked_on"]["label"] == "us"
    # a phrase is not; what is learnable is that the atom carries one
    assert by["reads:commitment"]["label"] == PRESENT
    prov = json.loads(by["reads:commitment"]["provenance"])
    assert prov["value"] == "get a conversation going with the club owner"
    assert prov["parser_proposed"] is True
    assert json.loads(by["reads:blocked_on"]["provenance"])["parser_proposed"] is False
    # and each row carries the context it was built from
    assert json.loads(by["about"]["provenance"])["context_parts"][-1] == "from"


def test_a_reading_outside_its_values_is_refused_not_trained():
    rows = _rows(reads_set={"blocked_on": "the club owner"})
    assert not [r for r in rows if r["relation"] == "reads:blocked_on"]
    assert "us" in CLOSED_READS["blocked_on"]


def test_the_type_row_and_the_axis_rows_do_not_share_a_string():
    rows = _rows(about="account", reads_set={"blocked_on": "us"})
    by = {r["relation"]: r for r in rows}
    assert by["atom_type"]["raw_text"] != by["reads:blocked_on"]["raw_text"]
    assert "[from:" not in by["atom_type"]["raw_text"]


def test_a_governs_link_becomes_a_structure_edge():
    doc = {
        "deal_id": "d1", "purpose": "train", "labels": [],
        "links": [{"relation": "governs", "from_text": "Here are the details for the small job",
                   "to_text": "Provided by us: Relay", "labeler": "developer@purtera-it.com"}],
    }
    rows = rows_for_deal(doc)
    edges = [r for r in rows if r["relation"] == "edge_relation"]
    assert len(edges) == 1 and edges[0]["label"] == "governs"


def test_an_assistants_draft_is_never_gold():
    # A draft written for a labeler to accept or replace sits on the card on
    # purpose. Ingesting it as teacher="human" would train the heads on the
    # assistant's own answers and call them a person's.
    doc = {
        "deal_id": "d1", "purpose": "train",
        "labels": [
            {"label_type": "deal_metadata", "labeler": "claude-code (assistant)", **THEIRS, "about": "account"},
            {"label_type": "deal_metadata", "labeler": "developer@purtera-it.com", **THEIRS, "about": "account"},
        ],
        "links": [{"relation": "governs", "from_text": "Here are the details", "to_text": "Relay",
                   "labeler": "claude-code (assistant)"}],
    }
    rows = rows_for_deal(doc)
    labelers = {json.loads(r["provenance"]).get("labeler") for r in rows if r["relation"] == "about"}
    assert labelers == {"developer@purtera-it.com"}
    assert not [r for r in rows if r["relation"] == "edge_relation"]


def test_a_reading_the_parser_proposed_and_a_human_removed_is_the_negative():
    # A chip nobody ticked may simply not have been considered. A chip the
    # parser ticked and the labeler cleared is a decision -- and without it a
    # presence task has one class and cannot train at all.
    rows = _rows(reads_set={"job_scale": "small"}, reads_shown=["job_scale", "small_talk"])
    by = {r["relation"]: r for r in rows}
    assert by["reads:job_scale"]["label"] == "small"
    assert by["reads:small_talk"]["label"] == "absent"
    assert json.loads(by["reads:small_talk"]["provenance"])["removed_by_human"] is True
