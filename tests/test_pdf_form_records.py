"""A PDF form's address and contact fields arrive as one record each.

Live 000037 (Maricopa County freight form): "SITE ADDRESS (Pickup Location)",
"CITY: Phoenix", "STATE: AZ" and "ZIP: 85008" were four atoms, so the labeling
page showed three lines that meant nothing alone, and the form-answer pass
emitted "SITE CONTACT (for Pickup): ..." a second time.
"""

from __future__ import annotations

from app.core.schemas import AtomType, AuthorityClass
from app.parsers.pdf._shared import _make_atom
from app.parsers.pdf.form_records import (
    drop_duplicate_form_answers, join_address_pieces, join_contact_phone, join_form_records,
)


def _atom(text, n, page=0, form_answer=False):
    return _make_atom(
        text=text, project_id="p", artifact_id="a", filename="form.pdf", parser_version="t",
        atom_type=AtomType.scope_item, authority_class=AuthorityClass.customer_current_authored,
        confidence=0.9, locator={"page": page, "block_id": "b", "row_index": n},
        value={"kind": "form_answer"} if form_answer else {},
    )


def _texts(atoms):
    return [a.raw_text for a in atoms]


def _screenshot_page():
    rows = [
        "SITE NAME: Iron Mountain Data Center",
        "SITE ADDRESS (Pickup Location): 615 N 48th St",
        "CITY: Phoenix",
        "SITE CONTACT (for Pickup): Adam Smith",
        "STATE: AZ",
        "ALTERNATE SITE CONTACT | PHONE: ☐",
        "ZIP: 85008",
        "Does the site require a pre-call before arrival?: Yes",
    ]
    atoms = [_atom(t, i) for i, t in enumerate(rows)]
    atoms += [_atom("SITE CONTACT (for Pickup): Adam Smith", 20, form_answer=True),
              _atom("Does the site require a pre-call before arrival?: Yes", 21, form_answer=True),
              _atom("PHONE: 6027232087; Work", 22, form_answer=True)]
    return atoms


def test_address_pieces_join_the_address():
    out = join_form_records(_screenshot_page())
    texts = _texts(out)
    assert "SITE ADDRESS (Pickup Location): 615 N 48th St, Phoenix, AZ 85008" in texts
    for piece in ("CITY: Phoenix", "STATE: AZ", "ZIP: 85008"):
        assert piece not in texts
    head = next(a for a in out if a.raw_text.startswith("SITE ADDRESS"))
    assert head.value["form_record"] == {
        "address": "615 N 48th St", "city": "Phoenix", "state": "AZ", "zip": "85008",
        "full_address": "615 N 48th St, Phoenix, AZ 85008"}
    roles = {(r["text"], r["role"]) for r in head.value["joined_from"]}
    assert ("SITE ADDRESS (Pickup Location): 615 N 48th St", "head") in roles
    assert ("ZIP: 85008", "piece") in roles


def test_form_answer_copy_of_page_text_is_dropped():
    out = drop_duplicate_form_answers(_screenshot_page())
    texts = _texts(out)
    assert texts.count("SITE CONTACT (for Pickup): Adam Smith") == 1
    assert texts.count("Does the site require a pre-call before arrival?: Yes") == 1
    survivor = next(a for a in out if a.raw_text.startswith("SITE CONTACT"))
    assert survivor.value["joined_from"] == [
        {"text": "SITE CONTACT (for Pickup): Adam Smith", "role": "duplicate"}]
    # An answer the page text does not have stays.
    assert "PHONE: 6027232087; Work" in texts


def test_phone_is_not_guessed_when_there_is_an_alternate_contact():
    out = join_form_records(_screenshot_page())
    assert "PHONE: 6027232087; Work" in _texts(out)
    assert "SITE CONTACT (for Pickup): Adam Smith" in _texts(out)


def test_lone_contact_takes_the_lone_phone():
    atoms = [_atom("SITE CONTACT: Adam Smith", 0), _atom("PHONE: 602-723-2087", 1),
             _atom("EMAIL: adam@example.com", 2)]
    out = join_contact_phone(atoms)
    assert _texts(out) == ["SITE CONTACT: Adam Smith, PHONE 602-723-2087, EMAIL adam@example.com"]
    assert out[0].value["form_record"]["phone"] == "602-723-2087"


def test_two_addresses_keep_their_own_pieces():
    atoms = [_atom("PICK UP ADDRESS: 615 N 48th St", 0), _atom("CITY: Phoenix", 1), _atom("ZIP: 85008", 2),
             _atom("DELIVERY ADDRESS: 1 Main St", 3), _atom("CITY: Scottsdale", 4), _atom("ZIP: 85251", 5)]
    out = join_address_pieces(atoms)
    assert _texts(out) == ["PICK UP ADDRESS: 615 N 48th St, Phoenix, 85008",
                           "DELIVERY ADDRESS: 1 Main St, Scottsdale, 85251"]


def test_pieces_on_another_page_or_without_an_address_stay():
    atoms = [_atom("SITE ADDRESS: 615 N 48th St", 0, page=0), _atom("CITY: Phoenix", 1, page=1)]
    assert _texts(join_address_pieces(atoms)) == ["SITE ADDRESS: 615 N 48th St", "CITY: Phoenix"]
    lone = [_atom("CITY: Phoenix", 0), _atom("STATE: AZ", 1)]
    assert _texts(join_address_pieces(lone)) == ["CITY: Phoenix", "STATE: AZ"]


def test_value_must_have_the_field_shape():
    atoms = [_atom("SITE ADDRESS: 615 N 48th St", 0), _atom("ZIP: see attached", 1),
             _atom("STATE: N/A", 2)]
    assert _texts(join_address_pieces(atoms)) == [
        "SITE ADDRESS: 615 N 48th St", "ZIP: see attached", "STATE: N/A"]


def test_atom_id_is_kept():
    atoms = _screenshot_page()
    head_id = atoms[1].id
    out = join_form_records(atoms)
    assert next(a for a in out if a.raw_text.startswith("SITE ADDRESS")).id == head_id
