"""Different JSON form fields with identical text are different facts.

Live 010353: the OxBlue intake JSON listed three equipment lines, each with
"quantity: 1". The dedup text key strips digits and punctuation, so
equipment.items[0..2].quantity all keyed as one, and the solar unit's count
was folded into the camera's and the mount's.
"""

from __future__ import annotations

import json

from app.core.schemas import AtomType
from app.core.semantic_dedup import cross_type_dedup_atoms, semantic_dedup_atoms
from app.parsers.json_parser import JsonParser


def _intake_atoms(tmp_path):
    p = tmp_path / "intake.json"
    p.write_text(json.dumps({"equipment": {"items": [
        {"name": "Sapphire PTZ camera", "quantity": 1},
        {"name": "Solar unit", "quantity": 1},
        {"name": "Roof mount", "quantity": 1},
    ]}}))
    atoms = JsonParser().parse_artifact("p", "art_intake", p)
    return [a for a in atoms if a.raw_text.endswith("quantity: 1")]


def _paths(atoms):
    return sorted(a.source_refs[0].locator["json_pointer"] for a in atoms)


def test_cross_type_pass_keeps_every_quantity_field(tmp_path):
    qty = _intake_atoms(tmp_path)
    assert len(qty) == 3
    # A later stage retypes one leaf; the three now span two types.
    qty[1].atom_type = AtomType.quantity
    kept = cross_type_dedup_atoms(qty)
    assert _paths(kept) == [
        "/equipment/items/0/quantity",
        "/equipment/items/1/quantity",
        "/equipment/items/2/quantity",
    ]


def test_cross_type_pass_keeps_fields_whose_text_is_identical(tmp_path):
    qty = _intake_atoms(tmp_path)
    for a in qty:  # the form shows each line as just "quantity: 1"
        a.raw_text = "quantity: 1"
    qty[1].atom_type = AtomType.quantity
    assert len(cross_type_dedup_atoms(qty)) == 3


def test_semantic_pass_keeps_every_quantity_field(tmp_path):
    qty = _intake_atoms(tmp_path)
    for a in qty:
        a.atom_type = AtomType.deal_metadata
        a.value = {**a.value, "field_name": "quantity", "value": "1"}
    assert len(semantic_dedup_atoms(qty)) == 3


def test_the_same_field_typed_twice_still_collapses(tmp_path):
    qty = _intake_atoms(tmp_path)
    a = qty[0]
    twin = a.model_copy(deep=True)
    twin.id = a.id + "_twin"
    twin.atom_type = AtomType.quantity
    assert len(cross_type_dedup_atoms([a, twin])) == 1
