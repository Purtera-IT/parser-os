"""Read a person out of a row the way site_property_block reads a place.

    On Site Contact Name | Rosalyn Hemingway | OSC email | Rosalyn.Hemingway@Sodexo.com
    OSC phone             | 843-423-8335 x3069 | Backup Contact | Bernard Donnelly

Shape decides what a value IS; a role label only decides WHICH person it
belongs to. Get the role wrong and the value still lands on a real person,
never in the void.
"""

from __future__ import annotations

from app.parsers.contact_property_block import (
    contacts_from_property_rows,
    fields_from_contact_row,
)


def _row(*pairs):
    return {str(i): v for i, v in enumerate(pairs)}


def test_the_real_easterling_block_reads_clean():
    rows = [
        _row("On Site Contact Name", "Rosalyn Hemingway", "OSC email", "Rosalyn.Hemingway@Sodexo.com"),
        _row("OSC phone", "843-423-8335 x3069", "Backup Contact", "Bernard Donnelly"),
        _row("Backup email", "bernie.donnelly@sodexo.com", "Backup Phone", "404-918-0783"),
    ]
    people = contacts_from_property_rows(rows)
    assert len(people) == 2
    primary, backup = people
    assert primary == {
        "role": "On Site Contact", "kind": "person",
        "name": "Rosalyn Hemingway", "email": "Rosalyn.Hemingway@Sodexo.com",
        "phone": "843-423-8335 x3069",
    }
    assert backup["name"] == "Bernard Donnelly"
    assert backup["email"] == "bernie.donnelly@sodexo.com"
    assert backup["phone"] == "404-918-0783"


def test_a_label_is_never_taken_as_a_name():
    """The bug: name="OSC phone", email="843-423-2571 x3645" -- position 0 and
    1 taken regardless of what they say. Shape refuses this."""
    fields = fields_from_contact_row(_row("OSC phone", "843-423-2571 x3645", "Backup Contact", "Bernard Donnelly"))
    assert fields.get("primary_name") is None
    assert fields.get("primary_phone") == "843-423-2571 x3645"
    assert fields.get("backup_name") == "Bernard Donnelly"


def test_a_phone_is_never_taken_as_an_email():
    fields = fields_from_contact_row(_row("Backup Contact", "Bernard Donnelly", "OSC phone", "843-464-3710 x5002"))
    assert fields.get("primary_phone") == "843-464-3710 x5002"
    assert "email" not in " ".join(fields)  # no field accepted the phone as an email


def test_a_malformed_email_is_left_unset_not_guessed():
    """Real customer typo on 010215: 'Wanda.Brayfield@sodexo,com' -- a comma,
    not a period. Shape correctly refuses it rather than passing it through."""
    fields = fields_from_contact_row(_row("On Site Contact Name", "Wanda Brayfield", "OSC email", "Wanda.Brayfield@sodexo,com"))
    assert fields.get("primary_name") == "Wanda Brayfield"
    assert "primary_email" not in fields


def test_em_dash_and_capital_x_are_still_a_phone():
    """Real customer formatting on 010215: an em dash for a hyphen, capital X
    for the extension marker. Neither is vendor vocabulary."""
    fields = fields_from_contact_row(_row("OSC phone", "843—464-3725 X 5110", "Backup Contact", "Bernard Donnelly"))
    assert fields.get("primary_phone") == "843—464-3725 X 5110"


def test_an_account_number_is_never_a_phone():
    fields = fields_from_contact_row(_row("OSC phone", "94575001", "Backup Contact", "Bernard Donnelly"))
    assert "primary_phone" not in fields


def test_no_recognised_label_yields_nothing():
    assert fields_from_contact_row(_row("Escalation Contact", "", "Escalation Contact", "")) == {}
    assert contacts_from_property_rows([_row("Project Name:", "MCSD Time Clock Installation")]) == []


def test_a_role_with_no_shape_valid_value_names_no_one():
    """A label match alone must not manufacture a person from nothing."""
    fields = fields_from_contact_row(_row("On Site Contact Name", "", "OSC email", "not-an-email"))
    assert fields == {}
