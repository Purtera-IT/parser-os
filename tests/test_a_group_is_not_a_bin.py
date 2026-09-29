# -*- coding: utf-8 -*-
"""A cross-type group keeps one atom PER TYPE, not one atom.

`cross_type_dedup_atoms` exists to collapse one sentence emitted under several
atom types: the technician line typed scope_item AND task AND service_line AND
raw_table_row is one fact read four ways. Its docstring has always added that
"groups that are all one type are left untouched (intra-type dedup is
semantic_dedup's job)".

That was true of a group of two and false of every larger one. The guard only
spared a group in which EVERY member shared a type, so a single atom of a
second type joining four quotes of one email thread turned "collapse a
retyping" into "keep one atom and delete the other four".

Two properties make that expensive rather than cosmetic:

* the group key STRIPS digits and cuts at 80 characters, so two quotes of one
  thread, or two rows of one totals table, arrive identical by construction --
  the very atoms that differ only in the number they carry
* `_merge_atom_metadata` carries source_refs, receipts, entity_keys and
  review_flags, and NOT `value` -- so the survivor keeps its own figures and
  the deleted atom's are simply gone

Measured over three live deals: 76 deal_metadata atoms deleted into other
deal_metadata atoms, 46 stakeholders into stakeholders, 18 commercial_totals
into commercial_totals. Eighteen of those left a commercial_total stating a
`value` the deleted atom did not hold, and seventeen left a stakeholder under
another person's name.
"""
from __future__ import annotations

from app.core.semantic_dedup import cross_type_dedup_atoms


class _Atom:
    def __init__(self, atom_type, text, confidence=0.8, value=None):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.confidence = confidence
        self.value = value or {}
        self.source_refs = [f"src::{atom_type}::{text[:12]}::{id(self)}"]
        self.receipts = []
        self.entity_keys = []
        self.review_flags = []


def test_a_stray_second_type_does_not_delete_the_others() -> None:
    """The case the guard missed: four same-type atoms and one outsider."""
    txt = "Estimated total fees for the engagement as quoted above"
    quotes = [_Atom("commercial_total", txt, value={"value": v, "metric": "total"})
              for v in (21560.0, 18200.0, 24310.0, 19995.0)]
    out = cross_type_dedup_atoms(quotes + [_Atom("raw_table_row", txt)])

    kept = [a for a in out if a.atom_type == "commercial_total"]
    assert len(kept) == 4, "a cross-type fold deleted same-typed totals"
    assert {a.value["value"] for a in kept} == {21560.0, 18200.0, 24310.0, 19995.0}
    # The outsider is still collapsed: that is what the stage is for.
    assert not [a for a in out if a.atom_type == "raw_table_row"]


def test_the_original_retyping_still_collapses() -> None:
    """The four-way technician line is why the stage exists; it must not move."""
    txt = "Technician #1- TV Install | $98.00 | Per Hour | 55 | $5,390.00"
    out = cross_type_dedup_atoms([
        _Atom("scope_item", txt), _Atom("task", txt),
        _Atom("service_line", txt), _Atom("raw_table_row", txt),
    ])
    assert len(out) == 1
    assert out[0].atom_type == "service_line"


def test_two_of_the_losing_type_both_go() -> None:
    """Keeping per-type must not let a raw row leak through beside its twin."""
    txt = "Large conference room fit-out as described in the attached schedule"
    out = cross_type_dedup_atoms([
        _Atom("raw_table_row", txt), _Atom("raw_table_row", txt),
        _Atom("service_line", txt),
    ])
    assert [a.atom_type for a in out] == ["service_line"]


def test_the_survivors_keep_document_order() -> None:
    txt = "Monthly rate and estimated hours for the six month term"
    a, b, c = (_Atom("deal_metadata", txt, value={"date": "1"}),
               _Atom("scope_item", txt),
               _Atom("deal_metadata", txt, value={"date": "2"}))
    out = cross_type_dedup_atoms([a, b, c])
    assert out == [a, c], "a dedup pass reordered the atoms it kept"


# ── and a retyping may not take a FIGURE with it ────────────────────────────


def test_the_values_do_not_go_with_the_label() -> None:
    """Live 010238. The group key strips digits by design, so the row that
    KEPT the numbers and the label that DROPPED them key identically, and the
    label outranks the raw row. What survived read "Effective Date: : Exp":
    the account number, the effective date and the expiry date all left the
    compile and nothing else in the deal stated them."""
    row = _Atom("raw_table_row",
                "Effective Date:: Account # | 2022-10-01 00:00:00: 2701149/5698885")
    label = _Atom("signatory", "Effective Date: : Exp")
    out = cross_type_dedup_atoms([row, label])
    assert any("2701149" in a.raw_text for a in out), \
        "a fold deleted the only atom stating the account number"


def test_a_money_column_still_folds_when_its_value_survives() -> None:
    """The case the words-guard was written for must NOT change: the typed
    atom carries the amount in `value`, so the raw row adds nothing."""
    row = _Atom("raw_table_row", "ESTIMATED TOTAL FEES | $21,560.00")
    typed = _Atom("commercial_total", "ESTIMATED TOTAL FEES",
                  value={"category": "estimated total fees", "amount": 21560.00})
    out = cross_type_dedup_atoms([row, typed])
    assert [a.atom_type for a in out] == ["commercial_total"]


def test_the_same_number_written_two_ways_is_one_number() -> None:
    row = _Atom("raw_table_row", "Subtotal | 1,200.50")
    typed = _Atom("service_line", "Subtotal", value={"amount": 1200.5})
    assert [a.atom_type for a in cross_type_dedup_atoms([row, typed])] == ["service_line"]


def test_a_richer_type_must_actually_say_it() -> None:
    """`_suppress_table_row_blob_doubles` drops a table row's generic blob when
    a richer-typed atom exists at the SAME CELL, on the reasoning that the
    richer atom is the classifier's reading of that row. `merge_signature_rows`
    breaks that: it retypes one row of a signature page to `signatory` and
    rewrites it as a merged party record, so the rich atom at live 010238's
    cell Lift:r11 read "Effective Date: : Exp" while the blob it displaced held
    the account number and both contract dates."""
    from app.core.semantic_dedup import _suppress_table_row_blob_doubles

    class _Ref:
        def __init__(self, table, row):
            self.locator = {"table_index": table, "row_index": row}

    def _at(atom_type, text, value=None):
        a = _Atom(atom_type, text, value=value)
        a.source_refs = [_Ref(3, 11)]
        a.artifact_id = "art_1"
        return a

    blob = _at("scope_item",
               "Effective Date:: Account # | 2022-10-01 00:00:00: 2701149/5698885",
               value={"kind": "table_row"})
    rich = _at("signatory", "Effective Date: : Exp", value={"kind": "signature_block"})
    out = _suppress_table_row_blob_doubles([blob, rich])
    assert any("2701149" in a.raw_text for a in out)

    # And the case it was written for is unchanged: the rich atom states the
    # row's figures, so the blob is a duplicate.
    blob2 = _at("scope_item", "Acme Corp | 12 units | $4,800",
                value={"kind": "table_row"})
    rich2 = _at("bom_line", "Acme Corp 12 units $4,800",
                value={"qty": 12, "amount": 4800})
    assert [a.atom_type for a in _suppress_table_row_blob_doubles([blob2, rich2])] \
        == ["bom_line"]
