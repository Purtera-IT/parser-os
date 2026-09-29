# -*- coding: utf-8 -*-
"""Two plates with the same count are two facts, not two copies of one.

``collapse_duplicate_atoms`` keeps the higher-confidence copy and drops the
other outright -- it does not merge -- so anything the twin held and the
survivor lacks leaves the compile silently, and the only trace is an atom
count nobody reads.

Measured on live COPPER_001 before this fix: 4,615 folds, and **81 of them
dropped a twin whose identity differed**. Every one was an
``atom_type=quantity`` row::

    "Quantity RJ45 2"     plate AVL 3  -->  folded into plate AVL 2's copy
    "Quantity Cat6 UTP 2" plate AVL 3  -->  folded into plate AVL 2's copy
    "Quantity Cat6 STP 0" plate AVL 4  -->  folded into plate AVL 2's copy

On a cabling job a plate is a location, so that is one location's cable and
jack quantities deleted. After the fix: 81 folds fewer, and identity loss 0,
while the other 4,534 folds still happen.
"""
from __future__ import annotations

from app.core.entity_resolution import _identity, collapse_duplicate_atoms


class _Atom:
    """Only what the stage reads."""

    def __init__(self, text, *, atom_type="quantity", artifact_id="art_1",
                 confidence=0.8, value=None, entity_keys=None):
        self.raw_text = text
        self.normalized_text = text.lower()
        self.atom_type = atom_type
        self.artifact_id = artifact_id
        self.confidence = confidence
        self.value = value or {}
        self.entity_keys = entity_keys or []


def _plate(text, plate, conf=0.8):
    return _Atom(text, value={"plate_id": plate, "quantity": 2},
                 entity_keys=[f"plate:{plate.lower().replace(' ', '_')}"],
                 confidence=conf)


def test_the_same_count_on_two_plates_survives_as_two_atoms() -> None:
    kept = collapse_duplicate_atoms([
        _plate("Quantity RJ45 2", "AVL 2", conf=0.9),
        _plate("Quantity RJ45 2", "AVL 3", conf=0.8),
    ])
    plates = sorted(a.value["plate_id"] for a in kept)
    assert plates == ["AVL 2", "AVL 3"], (
        "one plate's quantity was deleted because another plate had the same count"
    )


def test_a_true_duplicate_still_folds() -> None:
    """The fix must not turn dedup off."""
    kept = collapse_duplicate_atoms([
        _plate("Quantity RJ45 2", "AVL 2", conf=0.9),
        _plate("Quantity RJ45 2", "AVL 2", conf=0.5),
    ])
    assert len(kept) == 1
    assert kept[0].confidence == 0.9, "the higher-confidence copy is the survivor"


def test_a_differing_entity_key_is_enough_to_keep_both() -> None:
    a = _Atom("Cat6 drop", entity_keys=["site:atl_hq"])
    b = _Atom("Cat6 drop", entity_keys=["site:bos_dc"])
    assert len(collapse_duplicate_atoms([a, b])) == 2


def test_identity_ignores_where_the_row_SAT() -> None:
    """Positional keys differ for every row; keying on them would dedup nothing.

    Identity is about WHICH THING the atom is about, not where it was found.
    """
    a = _Atom("Quantity RJ45 2", value={"sheet": "Sheet1", "row": 4})
    b = _Atom("Quantity RJ45 2", value={"sheet": "Sheet1", "row": 9})
    assert _identity(a) == _identity(b) == ()
    assert len(collapse_duplicate_atoms([a, b])) == 1, (
        "two copies of one fact should still fold"
    )


def test_identity_is_stable_whatever_order_the_keys_arrive_in() -> None:
    a = _Atom("x", entity_keys=["plate:avl_3", "site:atl"])
    b = _Atom("x", entity_keys=["site:atl", "plate:avl_3"])
    assert _identity(a) == _identity(b)


def test_an_atom_with_nothing_identifying_has_an_empty_identity() -> None:
    assert _identity(_Atom("just words")) == ()
    assert _identity(_Atom("just words", value={"quantity": 2})) == (), (
        "a quantity is not an identity -- two plates can hold the same count"
    )
