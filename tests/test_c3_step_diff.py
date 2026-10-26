"""scripts/c3_step_diff.py: a C3 step off vs on, matched by place and text.
Made-up atoms only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "c3_step_diff", Path(__file__).resolve().parents[1] / "scripts" / "c3_step_diff.py")
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)


def _atom(i, t, text, page=1):
    return {"id": f"id{i}", "atom_type": t, "normalized_text": text,
            "source_refs": [{"filename": "sow.pdf", "locator": {"page": page}}]}


def _compile(*atoms):
    return {"atoms": list(atoms)}


def test_atoms_match_by_place_and_text_not_id():
    off = sd.atoms_of(_compile(_atom(1, "task", "Install the racks")))
    on = sd.atoms_of(_compile(_atom(99, "task", "install  the racks")))
    d = sd.diff(off, on)
    assert d["atoms"]["matched"] == 1 and d["lost"] == {} and d["retyped"] == {}


def test_a_lost_protected_atom_fails_and_a_retype_is_counted():
    off = sd.atoms_of(_compile(_atom(1, "task", "Install the racks"), _atom(2, "requirement", "Use cat6"),
                               _atom(3, "scope_item", "Kickoff call")))
    on = sd.atoms_of(_compile(_atom(1, "task", "Install the racks"), _atom(3, "task", "Kickoff call")))
    d = sd.diff(off, on)
    assert d["protected_lost"] == 1 and d["protected_lost_ids"] == ["id2"]
    assert d["retyped"] == {"scope_item->task": 1}
    ok, _ = sd.verdict(d, None)
    assert not ok


def test_loss_within_the_llms_own_churn_passes():
    off = sd.atoms_of(_compile(_atom(1, "task", "A"), _atom(2, "requirement", "B")))
    on = sd.atoms_of(_compile(_atom(1, "task", "A")))
    again = sd.atoms_of(_compile(_atom(2, "requirement", "B")))
    ok, why = sd.verdict(sd.diff(off, on), sd.diff(off, again))
    assert ok and "1" in why
