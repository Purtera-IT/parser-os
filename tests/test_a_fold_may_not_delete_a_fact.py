# -*- coding: utf-8 -*-
"""The floor under every collapse in the compile.

A dozen stages fold atoms together and each one is right about its own case.
What none of them had was a shared floor, so the same defect was found and
fixed EIGHT separate times in eight separate shapes -- 81 per-plate cable
quantities, 39 number-differing folds including a bid deadline, a totals block
reduced to its revenue line, four HubSpot notes collapsed to one, a contract
row deleted as a signature, and then deleted again one stage later as a
duplicate of the record that replaced it.

They are all one sentence: a fold may not delete what only the loser held.

`app.core.fold_invariants` is that sentence, once. This file holds it to the
real collapse functions rather than to a mock, because the bug was never in
the rule -- it was always in a stage that had not heard of it.
"""
from __future__ import annotations

import pytest

from app.core import fold_invariants as fold


class _Atom:
    def __init__(self, atom_type="scope_item", text="", value=None,
                 confidence=0.8, artifact_id="art_1", entity_keys=None):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.normalized_text = text
        self.value = dict(value or {})
        self.confidence = confidence
        self.artifact_id = artifact_id
        self.source_refs = []
        self.receipts = []
        self.entity_keys = list(entity_keys or [])
        self.review_flags = []


# ── the rule itself ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("a,b", [
    ("1,200", "1200"),      # thousands separators are formatting
    ("2.50", "2.5"),        # trailing decimal zeros are formatting
    ("1200.0", "1200"),     # and so is a whole number written as a float
    ("$4,182.00", "4182"),  # currency is not part of the figure
])
def test_formatting_is_not_a_different_number(a: str, b: str) -> None:
    assert fold.figures(a) == fold.figures(b), \
        "a fold that should happen would be blocked"


@pytest.mark.parametrize("a,b", [
    ("500 access points", "250 access points"),
    ("Total Labor Revenue: $121,519", "Total Labor Cost: $96,000"),
    ("Exp Date: 2023-10-01", "Exp Date: 2024-10-01"),
])
def test_a_different_number_is_a_different_fact(a: str, b: str) -> None:
    assert fold.figures(a) != fold.figures(b)


def test_a_figure_in_the_value_counts_as_stated() -> None:
    """The half that gets forgotten: a typed atom carries the money column in
    `value`, so folding the raw row away loses nothing -- and the moment it
    does NOT carry it, the same fold is a deletion."""
    typed = _Atom("commercial_total", "ESTIMATED TOTAL FEES",
                  value={"amount": 21560.00})
    row = _Atom("raw_table_row", "ESTIMATED TOTAL FEES | $21,560.00")
    assert not fold.figures_only_the_loser_states(typed, row)

    label = _Atom("signatory", "Effective Date: : Exp")
    assert fold.figures_only_the_loser_states(label, row)


def test_identity_needs_both_sides_to_be_placed() -> None:
    """An atom that records no plate or site is unplaced, not elsewhere.
    Refusing every fold involving one would stop dedup entirely."""
    avl2 = _Atom(value={"plate_id": "AVL 2"})
    avl3 = _Atom(value={"plate_id": "AVL 3"})
    nowhere = _Atom(value={})
    assert fold.identities_differ(avl2, avl3)
    assert not fold.identities_differ(avl2, nowhere)
    assert not fold.identities_differ(nowhere, nowhere)


def test_refuse_fold_names_what_would_be_lost() -> None:
    reason = fold.refuse_fold(_Atom(text="Quantity RJ45"),
                              _Atom(text="Quantity RJ45 2"))
    assert reason and "2" in reason
    assert fold.refuse_fold(_Atom(text="same"), _Atom(text="same")) is None


# ── and held to every real fold site ────────────────────────────────────────
#
# One property, run against the functions that actually collapse atoms. Each
# builds its own atom shape, so the cases are per-site; the ASSERTION is the
# same one every time.


def test_collapse_duplicate_atoms_keeps_both_plates() -> None:
    from app.core.entity_resolution import collapse_duplicate_atoms

    out = collapse_duplicate_atoms([
        _Atom("quantity", "Quantity RJ45 2", value={"plate_id": "AVL 2"}),
        _Atom("quantity", "Quantity RJ45 2", value={"plate_id": "AVL 3"}),
    ])
    assert {a.value["plate_id"] for a in out} == {"AVL 2", "AVL 3"}


def test_cross_type_dedup_keeps_the_figure() -> None:
    from app.core.semantic_dedup import cross_type_dedup_atoms

    row = _Atom("raw_table_row",
                "Effective Date:: Account # | 2022-10-01 00:00:00: 2701149/5698885")
    label = _Atom("signatory", "Effective Date: : Exp")
    assert any("2701149" in a.raw_text for a in cross_type_dedup_atoms([row, label]))


def test_semantic_dedup_keeps_each_metric() -> None:
    from app.core.semantic_dedup import semantic_dedup_atoms

    block = [
        _Atom("commercial_total", "Total Labor Revenue: $121,519",
              value={"category": "Total Labor", "metric": "revenue", "value": 121518.99}),
        _Atom("commercial_total", "Total Labor Cost: $96,000",
              value={"category": "Total Labor", "metric": "cost", "value": 96000.0}),
    ]
    assert len(semantic_dedup_atoms(block)) == 2


def test_the_blob_suppressor_keeps_the_figure() -> None:
    from app.core.semantic_dedup import _suppress_table_row_blob_doubles

    class _Ref:
        def __init__(self):
            self.locator = {"table_index": 3, "row_index": 11}

    blob = _Atom("scope_item", "Account # | 2701149/5698885",
                 value={"kind": "table_row"})
    rich = _Atom("signatory", "Effective Date: : Exp",
                 value={"kind": "signature_block"})
    for a in (blob, rich):
        a.source_refs = [_Ref()]
    out = _suppress_table_row_blob_doubles([blob, rich])
    assert any("2701149" in a.raw_text for a in out)


def test_merge_signature_rows_keeps_the_contract_row() -> None:
    from app.core.atom_type_sanity import merge_signature_rows

    atoms = [
        _Atom("signatory", "PurTera LLC: By: Trent Torrence", value={"page": 1}),
        _Atom("signatory", "PurTera LLC: Title: Executive Vice President", value={"page": 1}),
        _Atom("scope_item", "Effective Date:: Account # | 2701149/5698885", value={"page": 1}),
    ]
    merge_signature_rows(atoms)
    assert any("2701149" in a.raw_text for a in atoms)


def test_there_is_one_definition_of_a_figure() -> None:
    """Four private copies existed and three disagreed -- one folded decimals,
    one stripped trailing zeros, one could not see a decimal point at all. A
    shared rule that is re-implemented per site is not a shared rule."""
    from app.core.atom_type_sanity import _sig_figures
    from app.core.entity_resolution import _numbers
    from app.core.semantic_dedup import _figures

    sample = "1,200 / 2.50 / 1200.0 / $4,182.00"
    assert set(_numbers(sample)) == fold.figures(sample)
    assert set(_figures(sample)) == fold.figures(sample)
    assert set(_sig_figures(sample)) == fold.figures(sample)
