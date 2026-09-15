"""The commercial shape of a kit, learned from finished kits on the request line."""
from types import SimpleNamespace

import pytest

from app.core import decide as decide_mod
from app.core.decide import Decision
from app.core.schemas import AtomType
from app.core.commercial_terms import (
    RELATION, encode_commercial_verdict, parse_commercial_verdict, stamp_commercial_terms,
)


class _Corr(SimpleNamespace):
    pass


class _Store:
    def __init__(self, taught, created_by="deal-kit:010198"):
        self.taught = taught; self.created_by = created_by; self.scopes = []

    def all_corrections(self, active_only=True):
        return [_Corr(relation=RELATION, verdict=v) for v in set(self.taught.values())]

    def resolve(self, *, relation, text, candidates, scope=None, **_):
        self.scopes.append(getattr(scope, "deal_id", None))
        v = self.taught.get(text) if relation == RELATION else None
        return Decision(verdict=v, confidence=0.91, source="store", correction_id="c1") if v in candidates else None

    def get(self, cid):
        return _Corr(exemplars=list(self.taught), created_by=self.created_by)


@pytest.fixture
def store():
    prev = decide_mod.get_store()
    yield lambda taught: decide_mod.set_store(_Store(taught))
    decide_mod.set_store(prev)


def _task(text, project_id="deal-1", **value):
    return SimpleNamespace(atom_type=AtomType.task, raw_text=text, value=dict(value), project_id=project_id)


def test_verdict_round_trip():
    v = encode_commercial_verdict(billing="Fixed", pm_hours=2, pc_hours=1, travel_days=0)
    assert v == "billing=fixed;pm_hours=2;pc_hours=1;travel_days=0"
    assert parse_commercial_verdict(v) == {"billing": "fixed", "pm_hours": 2.0, "pc_hours": 1.0, "travel_days": 0.0}
    assert encode_commercial_verdict(billing="Hourly") == ""          # not a billing type the kit knows
    assert encode_commercial_verdict(billing="t_and_m", pm_hours=-1) == "billing=t_and_m"
    assert parse_commercial_verdict("billing=weekly;pm_hours=x") is None
    assert parse_commercial_verdict("") is None


def test_the_request_line_carries_the_kits_shape(store):
    """010198: one register and a printer, after hours -> Fixed, one PC line."""
    store({"We will be setting just 1 Square register and 1 kitchen printer.": "billing=fixed;pc_hours=1"})
    a = _task("We will be setting just 1 Square register and 1 kitchen printer.")
    b = _task("Ship the router to the site")
    assert stamp_commercial_terms([a, b]) == 1
    assert a.value["commercial_terms"] == {"billing": "fixed", "pc_hours": 1.0}
    assert a.value["commercial_terms_confidence"] == 0.91 and a.value["commercial_terms_correction_id"] == "c1"
    assert a.value["commercial_terms_evidence"] == 1 and a.value["commercial_terms_taught_by"] == "deal-kit:010198"
    assert "commercial_terms" not in b.value


def test_looks_up_within_the_deal_and_skips_folded_mentions(store):
    s = _Store({"Install two device servers": "billing=t_and_m;travel_days=1"}); decide_mod.set_store(s)
    folded = _task("Install two device servers", folded_into="atm_1")
    real = _task("Install two device servers", project_id="deal-9")
    assert stamp_commercial_terms([folded, real]) == 1
    assert "commercial_terms" not in folded.value and real.value["commercial_terms"]["travel_days"] == 1.0
    assert s.scopes == ["deal-9"]


def test_only_tasks_and_no_store_is_a_no_op(store):
    store({"x": "billing=fixed"})
    site = SimpleNamespace(atom_type=AtomType.physical_site, raw_text="x", value={}, project_id="d")
    assert stamp_commercial_terms([site]) == 0
    decide_mod.set_store(None)
    assert stamp_commercial_terms([_task("x")]) == 0
