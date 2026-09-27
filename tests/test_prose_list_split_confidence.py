"""A stage that fails silently is worse than a stage that is switched off.

Live 010180, compile cmp_135c10268e527d45:

    WARNING: prose_list_split failed: TypeError: unsupported operand
             type(s) for -: 'NoneType' and 'float'

One line in the whole compile, and the consequence was that no prose list in
the deal was split. A paragraph naming six stakeholders, or four payment
tiers, stayed a single atom — which is the same "an atom must be a statement"
failure the substance gate exists to catch, arriving from the other direction:
not a fragment torn off its label, but six facts glued into one.

The cause is a `getattr` default that reads as a guard and is not:

    confidence_raw=max(0.5, getattr(parent, "confidence_raw", 0.8) - 0.05)

`confidence_raw` and `calibrated_confidence` EXIST on every EvidenceAtom and
are routinely `None` — nothing has calibrated them yet at this point in the
pipeline, which is stage 1105 of a compile whose calibration runs at 2100.
`getattr` returns the default only when the attribute is absent, so it handed
back `None`, and `None - 0.05` raised. The stage's own `except` caught it,
appended a warning, and the compile reported success.

So the failure needed an atom with `calibrated_confidence=None` AND a
splittable paragraph in the same deal, and it announced itself only as one
line among a hundred INFO lines.
"""
from __future__ import annotations

from app.core.compiler import _a_shade_less_than


class _Atom:
    """An atom at stage 1105: confidence set by the parser, the calibrated
    fields still empty because calibration runs a thousand lines later."""
    confidence = 0.9
    confidence_raw = None
    calibrated_confidence = None


def test_a_child_is_a_shade_less_certain_than_its_parent():
    assert _a_shade_less_than(_Atom(), "confidence") == 0.85


def test_none_is_treated_as_unset_rather_than_raising():
    """The bug. Both of these were `None - 0.05`."""
    assert _a_shade_less_than(_Atom(), "confidence_raw") == 0.75
    assert _a_shade_less_than(_Atom(), "calibrated_confidence") == 0.75


def test_an_absent_attribute_still_falls_back():
    assert _a_shade_less_than(_Atom(), "not_a_field") == 0.75


def test_the_floor_holds():
    """A child of a barely-trusted parent does not go below the floor."""
    assert _a_shade_less_than(type("A", (), {"c": 0.2})(), "c") == 0.5


def test_it_returns_a_number_for_every_shape_an_atom_arrives_in():
    """The point of the test: whatever these fields hold, the stage runs."""
    for value in (None, 0.0, 0.5, 1.0, 0.63):
        got = _a_shade_less_than(type("A", (), {"c": value})(), "c")
        assert isinstance(got, float)
        assert 0.5 <= got <= 1.0
