"""An address where a scope item should be is not merely wasteful — it misleads.

Measured on deal 010215's Deal Kit context: 41% of the prompt handed to a model
was bare identifiers typed as ``scope_item`` — "t", "Q", "M: 404-918-0783",
"quinton.james@cdw.com", "Quinton James <", "3 | 4 | 5".

A model reading `t` as a scope item is being actively misled, and so is anything
downstream that counts atoms rather than weighting them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.atom_substance_gate import _is_contact_chrome, drop_contact_chrome


@dataclass
class A:
    raw_text: str
    atom_type: Any = "scope_item"
    value: dict = field(default_factory=dict)
    entity_keys: list = field(default_factory=list)


def types(atoms):
    return [a.raw_text for a in atoms]


def test_a_bare_email_is_not_a_scope_item():
    for t in ["quinton.james@cdw.com", "Joe.Buda@sodexo.com", "t@purtera-it.com"]:
        assert _is_contact_chrome(t), t


def test_a_bare_phone_is_not_a_scope_item():
    for t in ["404.771.3490", "M: 404-918-0783", "(704) 351-7537", "+1 847-371-3000"]:
        assert _is_contact_chrome(t), t


def test_a_recipient_list_cut_mid_address_is_chrome():
    for t in ["Quinton James <", "; Finn, Melody <", "Donnelly, Bernie <"]:
        assert _is_contact_chrome(t), t


def test_a_table_row_of_separators_is_chrome():
    for t in ["3 | 4 | 5", "1 . 2 . 3", "  |  "]:
        assert _is_contact_chrome(t), t


def test_one_or_two_characters_cannot_state_anything():
    for t in ["t", "Q", "AB", " x "]:
        assert _is_contact_chrome(t), t


def test_a_sentence_mentioning_an_address_is_content():
    # Anchored to the WHOLE string. A claim that happens to contain an address
    # is exactly what the model should read.
    keep = [
        "email Bernie at bernie.donnelly@sodexo.com about the SOWs",
        "Install 10 time clocks at Marion County School District",
        "Verify the power is on",
        "ADA Compliance 8",
        "Call Quinton on 404.771.3490 before the site visit",
    ]
    for t in keep:
        assert not _is_contact_chrome(t), t


def test_only_generic_prose_types_are_judged():
    """An email address IS the content of a stakeholder atom.

    Dropping it there would destroy the very fact the atom exists to carry.
    This rule is about an address appearing where a SCOPE ITEM should be.
    """
    atoms = [
        A("quinton.james@cdw.com", atom_type="stakeholder"),
        A("quinton.james@cdw.com", atom_type="scope_item"),
        A("404.771.3490", atom_type="contact"),
    ]
    kept, dropped = drop_contact_chrome(atoms)
    assert len(dropped) == 1
    assert dropped[0].atom_type == "scope_item"
    assert {a.atom_type for a in kept} == {"stakeholder", "contact"}


def test_dropped_atoms_are_returned_not_discarded():
    # The compiler routes these into the suppression ledger; losing them would
    # make the drop unauditable.
    atoms = [A("t"), A("Install 10 clocks at ten schools")]
    kept, dropped = drop_contact_chrome(atoms)
    assert types(kept) == ["Install 10 clocks at ten schools"]
    assert types(dropped) == ["t"]


def test_the_gate_runs_this_pass():
    from app.core.atom_substance_gate import apply_substance_gate

    kept, dropped = apply_substance_gate([A("t"), A("quinton.james@cdw.com"), A("Install 10 clocks")])
    assert types(kept) == ["Install 10 clocks"]
    assert len(dropped) == 2
