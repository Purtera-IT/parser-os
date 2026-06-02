"""Execution-block boilerplate hygiene.

Grounded in the Yonah DOCX, whose signature page produced scope_item /
raw_table_row atoms like "Name: \\nDate: | Name: \\nDate:",
"Services By:\\nPurTera | Agreed By:" and "Signature: | Signature:".
Those are not scope; this validator drops them while leaving real scope
lines (even ones containing a colon) untouched.
"""

from __future__ import annotations

from app.core.entity_hygiene import (
    drop_execution_boilerplate,
    is_execution_boilerplate,
)


class _Atom:
    def __init__(self, text, atom_type="scope_item"):
        self.raw_text = text
        self.text = text
        self.atom_type = atom_type


# ── the real signature-page strings ─────────────────────────────────


def test_name_date_block_is_boilerplate() -> None:
    assert is_execution_boilerplate(_Atom("Name: \nDate: | Name: \nDate:"))


def test_services_by_block_is_boilerplate() -> None:
    assert is_execution_boilerplate(_Atom("Services By:\nPurTera | Agreed By:"))


def test_signature_block_is_boilerplate() -> None:
    assert is_execution_boilerplate(_Atom("Signature: | Signature:"))


# ── real scope lines must survive ───────────────────────────────────


def test_real_scope_with_colon_not_boilerplate() -> None:
    # Contains a colon but the label isn't an exec field.
    assert not is_execution_boilerplate(
        _Atom("Note: Install each display and connect to Wi-Fi when credentials provided.")
    )


def test_access_constraint_not_boilerplate() -> None:
    assert not is_execution_boilerplate(
        _Atom("Provide access to all 23 dwellings and all installation locations.")
    )


def test_label_with_long_value_not_boilerplate() -> None:
    # "Title: ..." with a real sentence value is content, not a form field.
    assert not is_execution_boilerplate(
        _Atom("Title: Remove existing TVs and mounts from each dwelling location")
    )


# ── list filter ─────────────────────────────────────────────────────


def test_drop_filters_only_boilerplate() -> None:
    atoms = [
        _Atom("Signature: | Signature:"),
        _Atom("Provide access to all 23 dwellings."),
        _Atom("Name: \nDate:"),
        _Atom("Technician #1- TV Install | $98.00 | Per Hour | 55 | $5,390.00"),
    ]
    out = drop_execution_boilerplate(atoms)
    texts = [a.raw_text for a in out]
    assert "Provide access to all 23 dwellings." in texts
    assert any("Technician #1" in t for t in texts)
    assert len(out) == 2
