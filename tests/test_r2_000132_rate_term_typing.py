"""An after-hours rate is a pricing term, not a change-order rule.

Deal 000132's SOW: "After-hours work is billed at 150% of the standard rate."
was typed ``change_order_rule`` (whose registry description listed
"after-hours rate"), so it sat in the change-order packet instead of beside
the deal's other rates.
"""
from __future__ import annotations

from app.core.atom_type_sanity import apply_type_sanity, retype_rate_terms_off_change_orders
from app.core.schemas import AtomType


class _A:
    def __init__(self, text, atom_type=AtomType.change_order_rule):
        self.raw_text = text
        self.atom_type = atom_type
        self.value = {"text": text}
        self.review_flags = []
        self.entity_keys = []
        self.source_refs = []


def test_after_hours_rate_is_a_pricing_term():
    a = _A("After-hours work is billed at 150% of the standard rate.")
    assert retype_rate_terms_off_change_orders([a]) == 1
    assert a.atom_type == AtomType.pricing_assumption
    assert "change_order_rule" in a.value["alt_atom_types"]
    assert a.value["term_kind"] == "rate_term"


def test_multiplier_shapes():
    for t in ("Weekend work: 1.5x the hourly rate", "Holiday work is charged at double-time.",
              "Services outside business hours are calculated at 200% of the Unit Rates."):
        a = _A(t)
        retype_rate_terms_off_change_orders([a])
        assert a.atom_type == AtomType.pricing_assumption, t


def test_real_change_order_rules_stay():
    keep = [
        "Change orders are billed at 150% of the standard rate.",
        "Any out-of-scope work is billed at the T&M rate of $125/hr.",
        "A change order is required before any additional work begins.",
        "Materials on change requests carry a 15% markup.",
    ]
    atoms = [_A(t) for t in keep]
    retype_rate_terms_off_change_orders(atoms)
    assert all(a.atom_type == AtomType.change_order_rule for a in atoms)


def test_other_types_untouched_and_pass_wired():
    a = _A("After-hours work is billed at 150% of the standard rate.", AtomType.constraint)
    b = _A("After-hours work is billed at 150% of the standard rate.")
    apply_type_sanity([a, b], project_id="p")
    assert a.atom_type == AtomType.constraint
    assert b.atom_type == AtomType.pricing_assumption
