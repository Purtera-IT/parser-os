"""schematic_pm_review: existing schematic atoms -> PM verification triage.
present:False for non-schematic deals (no noise)."""
from __future__ import annotations
from dataclasses import dataclass, field
from app.core.schematic_pm_review import schematic_pm_review


@dataclass
class _T:  # minimal atom stand-in (mimics enum .value)
    value: str
@dataclass
class _Atom:
    atom_type: _T
    value: dict = field(default_factory=dict)


def _legend(entry_id, label, count=None):
    return _Atom(_T("schematic_legend"), {"entry_id": entry_id, "label_text": label, "count_column": count})
def _det(eid, conf=1.0):
    return _Atom(_T("schematic_symbol_detection"), {"legend_entry_id": eid, "confidence": conf})


def test_non_schematic_deal_is_absent():
    atoms = [_Atom(_T("requirement"), {"text": "x"}), _Atom(_T("stakeholder"), {})]
    assert schematic_pm_review(atoms) == {"present": False}


def test_triage_confident_flagged_missing():
    atoms = [
        _legend("cam", "camera", 2), _legend("smk", "smoke detector", 2), _legend("door", "card reader", 1),
        _det("cam"), _det("cam"),        # 2==2 confident
        _det("smk"),                      # 1!=2 flagged
        # door: 0 -> missing
    ]
    r = schematic_pm_review(atoms)
    assert r["present"] is True
    assert r["confident_count"] == 1
    assert r["needs_review_count"] == 2
    types = {q["type"] for q in r["review_queue"]}
    assert "smoke detector" in types and "card reader" in types


def test_pm_dashboard_includes_schematic_review():
    from app.core.orbitbrief_core import build_pm_dashboard
    out = build_pm_dashboard(atoms=[], packets=[], edges=[], entities=[])
    assert "schematic_review" in out
    assert out["schematic_review"] == {"present": False}  # empty deal
