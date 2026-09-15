"""Learned hours per unit of work."""
from types import SimpleNamespace

import pytest

from app.core import decide as decide_mod
from app.core.decide import Decision
from app.core.schemas import AtomType
from app.core.task_hours import (
    RELATION, encode_hours_verdict, estimate_task_hours, parse_hours_verdict, quantity_for_unit,
)


class _Corr(SimpleNamespace):
    pass


class _Store:
    def __init__(self, taught):
        self.taught = taught  # text -> verdict

    def all_corrections(self, active_only=True):
        return [_Corr(relation=RELATION, verdict=v) for v in set(self.taught.values())]

    def resolve(self, *, relation, text, candidates, **_):
        v = self.taught.get(text) if relation == RELATION else None
        return Decision(verdict=v, confidence=0.9, source="store", correction_id="c1") if v in candidates else None


@pytest.fixture
def store():
    prev = decide_mod.get_store()
    yield lambda taught: decide_mod.set_store(_Store(taught))
    decide_mod.set_store(prev)


def _task(text):
    return SimpleNamespace(atom_type=AtomType.task, raw_text=text, value={})


def test_verdict_round_trip():
    v = encode_hours_verdict(3, per="cable drop", role="L2")
    assert parse_hours_verdict(v) == {"hours": 3.0, "per": "cable drop", "role": "L2"}
    assert parse_hours_verdict("per=drop") is None


@pytest.mark.parametrize("text,unit,want", [
    ("Install EMT conduit drops at 39 AP locations", "AP location", 39.0),
    ("Install 39 new CAT6 drops", "cable drop", 39.0),
    ("Production: 21 AP (lower existing AP, may require new cabling)", "AP", 21.0),
    ("Install pendant mount assemblies and remount APs", "AP location", None),
    ("Relocate APs from 35 ft to 15-20 ft ceiling", "AP", None),
])
def test_quantity_follows_the_taught_unit(text, unit, want):
    assert quantity_for_unit(text, unit) == want


def test_fixed_hours_are_stamped(store):
    store({"Remove the 1518 SonicWall from service, label it, and store as a backup": "hours=0.75"})
    atoms = [_task("Remove the 1518 SonicWall from service, label it, and store as a backup")]
    assert estimate_task_hours(atoms) == 1
    assert atoms[0].value["estimated_hours"] == 0.75
    assert atoms[0].value["hours_basis"] == "0.75 h"


def test_per_unit_hours_scale_by_the_stated_quantity(store):
    store({"Install 39 new CAT6 drops": "hours=3;per=cable drop"})
    atoms = [_task("Install 39 new CAT6 drops")]
    estimate_task_hours(atoms)
    assert atoms[0].value["estimated_hours"] == 117.0
    assert atoms[0].value["hours_basis"] == "3 h per cable drop x 39"


def test_per_unit_without_a_quantity_keeps_the_rate_not_a_guess(store):
    store({"Install pendant mount assemblies and remount APs": "hours=1;per=AP location"})
    atoms = [_task("Install pendant mount assemblies and remount APs")]
    estimate_task_hours(atoms)
    assert atoms[0].value["estimated_hours"] is None
    assert atoms[0].value["hours_per_unit"] == 1.0


def test_only_tasks_and_only_taught_text(store):
    store({"Reset Ubiquiti gateway": "hours=1"})
    atoms = [_task("Something untaught"), SimpleNamespace(atom_type=AtomType.scope_item, raw_text="Reset Ubiquiti gateway", value={})]
    assert estimate_task_hours(atoms) == 0
    assert atoms[1].value == {}


def test_no_store_no_op():
    prev = decide_mod.get_store()
    decide_mod.set_store(None)
    try:
        assert estimate_task_hours([_task("Install 39 new CAT6 drops")]) == 0
    finally:
        decide_mod.set_store(prev)


def test_a_per_unit_lesson_carries_the_kits_numbers_not_a_rounded_rate(store):
    """010043: 16 h for 3 cameras. Taught as 5.33/camera the compile said 15.99."""
    assert encode_hours_verdict(16, per="camera", qty=3) == "hours=16;per=camera;qty=3"
    assert parse_hours_verdict("hours=16;per=camera;qty=3") == {"hours": 16.0, "per": "camera", "qty": 3.0}
    store({"3 Verkada cameras install": "hours=16;per=camera;qty=3"})
    atoms = [_task("3 Verkada cameras install")]
    assert estimate_task_hours(atoms) == 1
    assert atoms[0].value["estimated_hours"] == 16.0
    assert atoms[0].value["hours_per_unit"] == 5.3333
    assert atoms[0].value["hours_basis"] == "16 h for 3 cameras = 5.333 h per camera x 3"


def test_a_quantity_without_a_unit_or_below_one_is_ignored():
    assert parse_hours_verdict("hours=16;qty=3") == {"hours": 16.0}
    assert parse_hours_verdict("hours=16;per=camera;qty=0") == {"hours": 16.0, "per": "camera"}
    assert parse_hours_verdict("hours=16;per=camera;qty=x") == {"hours": 16.0, "per": "camera"}
    assert encode_hours_verdict(16, per="camera", qty=0) == "hours=16;per=camera"
