# -*- coding: utf-8 -*-
"""A totals block states several figures under one category.

`_value_key` keyed a `commercial_total` on its CATEGORY alone, so every figure
in one row of a totals block became one atom. Live 010237's labour block is

    Total Labor Revenue: $121,519   metric=revenue
    Total Labor Cost:     $96,000   metric=cost
    Total Labor Margin:   $25,519   metric=margin
    Margin % on Labor:        21%   metric=margin_pct

and all four keyed as ("commercial_total", "total labor"). Revenue survived.
The cost, the margin and the margin percentage were deleted -- on a deal whose
margin is the reason anyone reads it.

`amount` was already being read in that branch and then dropped on the floor,
so the intent was right and only the return was wrong.

The second half of the damage is `_merge_values`. It fills the survivor's empty
fields from the atom it is deleting and, where both are strings, takes the
LONGER one. So the kept revenue atom came out carrying `metric="margin_pct"`
beside `value=121518.99`: not a figure that went missing, a figure that became
a false statement about a different quantity.
"""
from __future__ import annotations

from app.core.semantic_dedup import semantic_dedup_atoms


class _Atom:
    def __init__(self, atom_type, text, value, confidence=0.8):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.value = value
        self.confidence = confidence
        self.source_refs = [f"src::{text[:20]}"]
        self.receipts = []
        self.entity_keys = []
        self.review_flags = []


def _labour_block():
    return [
        _Atom("commercial_total", "Total Labor Revenue: $121,519",
              {"category": "Total Labor", "metric": "revenue", "value": 121518.99}),
        _Atom("commercial_total", "Total Labor Cost: $96,000",
              {"category": "Total Labor", "metric": "cost", "value": 96000.0}),
        _Atom("commercial_total", "Total Labor Margin: $25,519",
              {"category": "Total Labor", "metric": "margin", "value": 25518.99}),
        _Atom("commercial_total", "Margin % on Labor: 21%",
              {"category": "Total Labor", "metric": "margin_pct", "value": 21.0}),
    ]


def test_the_margin_is_not_deleted_by_the_revenue() -> None:
    out = semantic_dedup_atoms(_labour_block())
    kept = {a.value["metric"]: a.value["value"] for a in out}
    assert kept == {"revenue": 121518.99, "cost": 96000.0,
                    "margin": 25518.99, "margin_pct": 21.0}


def test_no_survivor_states_a_figure_it_did_not_carry() -> None:
    """The corruption is worse than the loss: a kept atom must not come out
    labelled with a metric belonging to an atom that was deleted."""
    for a in semantic_dedup_atoms(_labour_block()):
        assert a.value["metric"] in a.raw_text.lower().replace(" ", "_") \
            or a.value["metric"].split("_")[0] in a.raw_text.lower()


def test_one_figure_stated_twice_still_collapses() -> None:
    """The stage's actual job, unchanged: a restatement is one fact."""
    same = [
        _Atom("commercial_total", "Total Labor Revenue: $121,519",
              {"category": "Total Labor", "metric": "revenue", "value": 121518.99}),
        _Atom("commercial_total", "TOTAL LABOR REVENUE $121,519",
              {"category": "Total Labor", "metric": "revenue", "value": 121519.0}),
    ]
    assert len(semantic_dedup_atoms(same)) == 1


def test_a_longer_string_is_not_authority() -> None:
    """`_merge_values` may only replace a populated field when the longer value
    CONTAINS the shorter -- "Rhonda Sharp" -> "Rhonda Sharp <r@cdw.com>". A
    placeholder that merely has one more character must not win."""
    from app.core.semantic_dedup import _merge_values

    winner = _Atom("stakeholder", "WIFI Example", {"name": "WIFI Example"})
    loser = _Atom("stakeholder", "Chase Smith", {"name": "Chase Smith"})
    _merge_values(winner, loser)
    assert winner.value["name"] == "WIFI Example", "a fold rewrote the survivor's name"

    short = _Atom("stakeholder", "x", {"name": "Rhonda Sharp"})
    full = _Atom("stakeholder", "y", {"name": "Rhonda Sharp <rhonda.sharp@cdw.com>"})
    _merge_values(short, full)
    assert short.value["name"] == "Rhonda Sharp <rhonda.sharp@cdw.com>"


# ── the same defect, one branch up: a constant used as an identity ──────────
#
# `deal_metadata` fell through to `_first("field_name", "value")`, and for two
# kinds `field_name` is a CONSTANT. Every HubSpot note on the deal keyed as
# ("deal_metadata", "hubspot_note_meta") and every message header as
# ("deal_metadata", "email_metadata"), so each kind collapsed to one atom for
# the whole deal -- live 010237 lost four notes with four distinct ids to one,
# and seven message headers spanning 19 August to 29 September to one.


def _note(note_id, date, author="Trent Torrence"):
    return _Atom("deal_metadata", f"note_id={note_id} | author={author}",
                 {"field_name": "hubspot_note_meta", "kind": "hubspot_note_meta",
                  "hubspot_note_id": note_id, "date": date, "author": author})


def _header(sender, date, subject="RE: NMC Budgetary Quote"):
    return _Atom("deal_metadata", f"From: {sender} | Subject: {subject} | Date: {date}",
                 {"field_name": "email_metadata", "kind": "email_header",
                  "from": sender, "subject": subject, "date": date})


def test_four_notes_are_four_notes() -> None:
    notes = [_note("115206652190", "2026-08-19T19:26:22.009Z"),
             _note("115206706200", "2026-08-19T19:29:14.215Z"),
             _note("115297769875", "2026-08-20T15:29:55.567Z", "Chase Smith"),
             _note("115298050361", "2026-08-20T15:29:55.570Z", "Chase Smith")]
    out = semantic_dedup_atoms(notes)
    assert {a.value["hubspot_note_id"] for a in out} == {
        "115206652190", "115206706200", "115297769875", "115298050361"}


def test_one_note_synced_twice_is_one_note() -> None:
    same = [_note("115206652190", "2026-08-19T19:26:22.009Z"),
            _note("115206652190", "2026-08-19T19:26:22.009Z")]
    assert len(semantic_dedup_atoms(same)) == 1


def test_a_thread_keeps_its_messages() -> None:
    """Same sender, same subject, different day: different messages."""
    thread = [_header("snalla@nmcms.com", "Wed, 19 Aug 2026 19:45:49 +0000"),
              _header("snalla@nmcms.com", "Tue, 29 Sep 2026 18:19:15 +0000"),
              _header("t@purtera-it.com", "Wed, 19 Aug 2026 19:34:03 +0000")]
    assert len(semantic_dedup_atoms(thread)) == 3


def test_one_message_quoted_in_many_replies_still_collapses() -> None:
    quoted = [_header("snalla@nmcms.com", "Wed, 19 Aug 2026 19:45:49 +0000")
              for _ in range(33)]
    assert len(semantic_dedup_atoms(quoted)) == 1


def test_an_unidentifiable_header_stays_out_of_dedup() -> None:
    """No date recorded: fold on a guess, or keep both? Keep both -- the same
    policy the quoted-header branch above already follows."""
    vague = [_header("snalla@nmcms.com", ""), _header("snalla@nmcms.com", "")]
    assert len(semantic_dedup_atoms(vague)) == 2
