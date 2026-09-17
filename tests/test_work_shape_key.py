"""PUR-14: a lesson keyed on the work, not the wording, behind a flag.

A is corrected. B does the same work in different words (different customer,
different template) and must change. C uses A's exact words, customer and
template for different work and must not.
"""

from __future__ import annotations

from app.core import estimator_head as eh
from app.core import work_shape as ws
from app.core.pm_feedback import HEAD_REGISTRY, pm_correction_to_correction
from app.eval.offline_embedder import offline_store

W = "Install 12 cameras at the warehouse"


def _deal(did, wording, obj, unit, count, *, customer="Northwind", delivery="onsite", **wo):
    return {
        "deal_id": did, "customer": customer, "document_title": f"{customer} SOW",
        "wording": wording, "delivery_model": delivery,
        "work_order": {"work_lines": [{"work": wording, "object": obj, "unit": unit, "count": count}],
                       "site_count": 1, "after_hours": False, "no_onsite_hands": False,
                       "customer_supplies_equipment": False, **wo},
    }


A = _deal("A", W, "camera", "camera", 12)
B = _deal("B", "Mount and cable forty security cameras across the distribution center",
          "security camera", "camera", 40, customer="Contoso")
C = _deal("C", W, "camera license", "license", 12, delivery="remote")


def _teach(store, mode, field="hours_per_visit", accepted=18.0):
    p = eh.propose(A, store, key_mode=mode)
    lesson = eh.lesson_from_override(A, p, line_index=0, field_name=field, accepted=accepted,
                                     reason_code="other", reason_text="lift per camera", key_mode=mode)
    store.add(lesson)
    return lesson


def _hpv(deal, store, mode):
    return eh.propose(deal, store, key_mode=mode).field_value(0, "hours_per_visit")


def test_flag_defaults_to_wording(monkeypatch):
    monkeypatch.delenv(ws.KEY_MODE_ENV, raising=False)
    assert ws.lesson_key_mode() == ws.MODE_WORDING
    monkeypatch.setenv(ws.KEY_MODE_ENV, "work_shape")
    assert ws.lesson_key_mode() == ws.MODE_WORK_SHAPE
    monkeypatch.setenv(ws.KEY_MODE_ENV, "nonsense")
    assert ws.lesson_key_mode() == ws.MODE_WORDING


def test_work_shape_key_transfers_a_to_b_and_not_to_c():
    store = offline_store()
    b0, c0 = _hpv(B, store, "work_shape"), _hpv(C, store, "work_shape")
    _teach(store, "work_shape")
    assert _hpv(A, store, "work_shape") == 18.0
    assert _hpv(B, store, "work_shape") != b0  # A -> B
    assert _hpv(B, store, "work_shape") == 1.5 * 40  # the rate, not A's number
    assert _hpv(C, store, "work_shape") == c0  # A -/-> C


def test_wording_key_reproduces_the_failure():
    """The current key: C (same words) changes, B (same work) does not."""
    store = offline_store()
    b0, c0 = _hpv(B, store, "wording"), _hpv(C, store, "wording")
    _teach(store, "wording")
    assert _hpv(B, store, "wording") == b0
    assert _hpv(C, store, "wording") != c0


def test_key_text_ignores_customer_title_wording_and_count():
    s1 = eh.line_shape(A, A["work_order"]["work_lines"][0])
    other = _deal("Z", "totally different sentence", "cameras", "cameras", 17, customer="Zeta")
    s2 = eh.line_shape(other, other["work_order"]["work_lines"][0])
    k1, k2 = ws.key_text(s1, "units"), ws.key_text(s2, "units")
    assert "northwind" not in k1.lower() and "warehouse" not in k1.lower() and "12" not in k1
    assert k1.replace("action:install", "") == k2.replace("action:totally", "")


def test_gate_blocks_on_disagreeing_facet_but_not_on_unstated():
    taught = ws.WorkShape(object="camera", unit="camera", delivery_model="onsite").as_dict()
    gate = ws.field_key("hours_per_visit").gate
    assert ws.gate_holds(taught, dict(taught), gate)
    assert not ws.gate_holds(taught, {**taught, "delivery_model": "remote"}, gate)
    assert ws.gate_holds(taught, {**taught, "delivery_model": ""}, gate)
    assert not ws.gate_holds(taught, None, gate)


def test_ungated_corrections_are_untouched_by_the_store_gate():
    assert ws.correction_gate_holds({}, None)
    assert ws.correction_gate_holds({"when": {"field": "x"}}, {"work_shape": {}})


def test_every_field_has_a_key_spec_and_head():
    for f in eh.FIELDS:
        assert f in ws.FIELD_KEYS
        assert eh.RELATIONS[f] in {s.relation for s in HEAD_REGISTRY.values()}


def test_pm_payload_with_work_shape_is_keyed_on_shape_only_when_flag_on(monkeypatch):
    shape = eh.line_shape(A, A["work_order"]["work_lines"][0]).as_dict()
    payload = {"head": "estimate_hours_per_visit", "dealId": "A", "targetId": "line0",
               "text": W, "oldValue": "12", "newValue": "rate=1.5", "scope": "global",
               "field": "hours_per_visit", "workShape": shape}
    monkeypatch.delenv(ws.KEY_MODE_ENV, raising=False)
    assert pm_correction_to_correction(payload).exemplars == [W]
    monkeypatch.setenv(ws.KEY_MODE_ENV, "work_shape")
    corr = pm_correction_to_correction(payload)
    assert corr.exemplars == [ws.key_text(ws.WorkShape.from_dict(shape), "hours_per_visit")]
    assert corr.relations[ws.REL_GATE] == list(ws.field_key("hours_per_visit").gate)
    assert corr.relations["wording"] == W
