# -*- coding: utf-8 -*-
"""A verdict on a dropped atom has to outlive the compile that dropped it.

The **Dropped** stage of the labelling workspace is where the suppression head
gets its training set, and where "should this rule have fired?" is actually
answerable. Until now each dropped atom reached that stage carrying its text,
its stage and its survivor -- and nothing that said who PRODUCED it, or any key
that survives a reparse.

Both gaps cost the same thing: a label that cannot be joined to anything
later.

* `emitted_by` is what makes the verdict attributable. Measured over one live
  deal, the share of an emitter's output that a later stage deletes runs from
  0% to 100% -- `xlsx_block_raw_table_row` had 26 atoms kept and 241 deleted,
  `prose_list_split_v50` 172 and 515. That is the ranking of which rules are
  worth labelling, and it was invisible on the card.
* `label_key` is sha256(deal | filename | page | text) -- three quarters
  location, one quarter the atom's own words -- so a verdict re-attaches on any
  later compile that still produces the atom. Without it a verdict is pinned to
  one compile's atom id.
"""
from __future__ import annotations

import pytest

from app.core.label_key import label_key
from app.core.orbitbrief_envelope import _emitted_by, _suppressed_for_review, _where


class _Ref:
    def __init__(self, filename="quote.xlsx", page=3, method="xlsx_block_raw_table_row"):
        self.filename = filename
        self.locator = {"page": page}
        self.extraction_method = method


class _Receipt:
    def __init__(self, name="xlsx_parser", method="deterministic_rule"):
        self.extractor_name = name
        self.extraction_method = method


class _Atom:
    def __init__(self, text, stage="pre_classify_dedup", refs=None, receipts=None):
        self.id = "atom_1"
        self.artifact_id = "art_1"
        self.atom_type = "scope_item"
        self.raw_text = text
        self.text = text
        self.value = {}
        self.entity_keys = []
        self.source_refs = refs if refs is not None else [_Ref()]
        self.receipts = receipts or []
        self.review_flags = [f"suppressed:{stage}"]


class _Result:
    def __init__(self, dropped, kept=()):
        self.project_id = "01491cca"
        self.suppressed_atoms = list(dropped)
        self.atoms = list(kept)


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")


def test_the_card_says_who_made_it() -> None:
    row = _suppressed_for_review(_Result([_Atom("Rack 14 | 42U | $4,182.00")]), [])[0]
    assert row["emitted_by"] == "xlsx_block_raw_table_row"
    assert row["stage"] == "pre_classify_dedup"


def test_a_receipt_beats_a_source_ref_for_attribution() -> None:
    atom = _Atom("x", receipts=[_Receipt()])
    assert _emitted_by(atom) == "xlsx_parser:deterministic_rule"


def test_the_verdict_survives_a_reparse() -> None:
    text = "Rack 14 | 42U | $4,182.00"
    row = _suppressed_for_review(_Result([_Atom(text)]), [])[0]
    assert row["label_key"] == label_key("01491cca", "quote.xlsx", "3", text)
    assert row["filename"] == "quote.xlsx" and row["page"] == "3"


def test_an_atom_with_no_source_is_still_shown() -> None:
    """No filename means no key -- but the atom must still reach the labeller,
    because "this was dropped and nobody can say where it came from" is itself
    worth seeing."""
    row = _suppressed_for_review(_Result([_Atom("orphan", refs=[])]), [])[0]
    assert row["label_key"] == ""
    assert row["text"] == "orphan"


def test_the_gate_still_holds(monkeypatch) -> None:
    monkeypatch.delenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", raising=False)
    assert _suppressed_for_review(_Result([_Atom("x")]), []) == []


def test_where_prefers_a_page_then_a_sheet() -> None:
    ref = _Ref()
    ref.locator = {"sheet": "Pricing", "row_index": 7}
    assert _where(_Atom("x", refs=[ref])) == ("quote.xlsx", "Pricing")
