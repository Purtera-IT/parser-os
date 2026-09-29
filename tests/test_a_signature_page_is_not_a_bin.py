# -*- coding: utf-8 -*-
"""A contract row on the signature page is not a signature row.

`merge_signature_rows` groups every row on a page that LOOKS like a signature
row -- anything carrying a by/name/title/date/signature label -- and rewrites
the first one as a merged record holding party, name, title and signed_at. It
then deleted every other row in the group outright, whether or not the merged
record said anything of it.

Live 010238 kept a contract table on the same page:

    Effective Date:: Account # | 2022-10-01 00:00:00: 2701149/5698885
    Effective Date:: Exp Date: | 2022-10-01 00:00:00: 2023-10-01 00:00:00

Both matched on their date label, both were deleted, and what survived read
"Effective Date: : Exp" -- the labels with the values torn off. The deal's
account number and its contract effective and expiry dates left the compile
and nothing else in it stated them.

The loss was invisible for a second reason, fixed alongside: `atom_type_sanity`
filed nothing to the suppression ledger, so the audits could not see the stage
drop anything at all. See `test_every_departure_leaves_a_receipt`.
"""
from __future__ import annotations

from app.core.atom_type_sanity import merge_signature_rows


class _Atom:
    def __init__(self, atom_type, text, page=1, artifact_id="art_1"):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.artifact_id = artifact_id
        self.value = {"page": page}
        self.confidence = 0.8
        self.source_refs = []
        self.receipts = []
        self.entity_keys = []
        self.review_flags = []


def _signature_page():
    return [
        _Atom("signatory", "PurTera LLC: By: Trent Torrence"),
        _Atom("signatory", "PurTera LLC: Title: Executive Vice President"),
        _Atom("signatory", "PurTera LLC: Date: Mar 26, 2026"),
        _Atom("scope_item",
              "Effective Date:: Account # | 2022-10-01 00:00:00: 2701149/5698885"),
        _Atom("scope_item",
              "Effective Date:: Exp Date: | 2022-10-01 00:00:00: 2023-10-01 00:00:00"),
    ]


def test_the_account_number_is_not_a_signature() -> None:
    atoms = _signature_page()
    merge_signature_rows(atoms)
    blob = " ".join(a.raw_text for a in atoms)
    assert "2701149" in blob, "the only atom stating the account number was deleted"
    assert "5698885" in blob
    assert "2023-10-01" in blob, "the contract expiry date was deleted"


def test_the_signature_rows_still_merge() -> None:
    """What the stage is for must not change: one record per party, not one
    atom per row."""
    atoms = _signature_page()
    folded = merge_signature_rows(atoms)
    assert folded >= 2, "the signature rows stopped merging"
    signers = [a for a in atoms
               if isinstance(a.value, dict) and a.value.get("kind") == "signature_block"]
    assert len(signers) == 1
    assert signers[0].value["name"] == "Trent Torrence"


def test_a_row_the_record_fully_states_is_still_folded() -> None:
    atoms = [
        _Atom("signatory", "PurTera LLC: By: Trent Torrence"),
        _Atom("signatory", "PurTera LLC: Title: Executive Vice President"),
        _Atom("signatory", "PurTera LLC: By: Trent Torrence"),
    ]
    before = len(atoms)
    merge_signature_rows(atoms)
    assert len(atoms) < before


def test_the_rewritten_row_does_not_lose_its_own_content() -> None:
    """The row chosen to BECOME the merged record is overwritten in place, so
    it must be one the merged record already says everything of. When the
    contract row sorts first, picking it destroys the account number just as
    surely as deleting it."""
    atoms = [
        _Atom("scope_item",
              "Effective Date:: Account # | 2022-10-01 00:00:00: 2701149/5698885"),
        _Atom("signatory", "PurTera LLC: By: Trent Torrence"),
        _Atom("signatory", "PurTera LLC: Title: Executive Vice President"),
        _Atom("signatory", "PurTera LLC: Date: Mar 26, 2026"),
    ]
    merge_signature_rows(atoms)
    blob = " ".join(a.raw_text for a in atoms)
    assert "2701149" in blob, "the row rewritten as the merged record lost its own content"
