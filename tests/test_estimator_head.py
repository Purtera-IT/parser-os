"""PUR-13: the estimator is a correctable head; a correction lands on a named field."""

from __future__ import annotations

import pytest

from app.core import estimator_head as eh
from app.core.pm_feedback import HEAD_REGISTRY
from app.eval.offline_embedder import offline_store

DEAL = {
    "deal_id": "SYN-X", "wording": "Refresh POS terminals at 6 stores after close", "delivery_model": "onsite",
    "work_order": {"work_lines": [{"work": "Refresh POS terminals", "object": "POS terminal", "unit": "terminal", "count": 24}],
                   "site_count": 6, "after_hours": True, "no_onsite_hands": False, "customer_supplies_equipment": False},
}


def test_estimator_fields_are_registered_heads():
    rels = {h: s.relation for h, s in HEAD_REGISTRY.items()}
    for f in eh.FIELDS:
        head = f"estimate_{f}"
        assert rels[head] == eh.RELATIONS[f]
        assert HEAD_REGISTRY[head].mode == "extract"
    assert "total_hours" not in eh.FIELDS


def test_baseline_proposal_is_labelled_and_total_is_derived():
    p = eh.propose(DEAL, offline_store(), key_mode="work_shape")
    line = p.lines[0]
    assert {f.source for f in line.fields.values()} == {"baseline"}
    assert line.fields["units"].value == 24 and line.fields["visits"].value == 6
    assert line.total_hours == line.fields["hours_per_visit"].value * line.fields["visits"].value
    assert p.proposal_id.startswith("prop_") and p.version.startswith(eh.ESTIMATOR_VERSION)


@pytest.mark.parametrize("field,accepted", [("units", 30), ("visits", 12), ("hours_per_visit", 2)])
def test_a_correction_lands_on_exactly_one_named_field(field, accepted):
    store = offline_store()
    before = eh.propose(DEAL, store, key_mode="work_shape")
    lesson = eh.lesson_from_override(DEAL, before, line_index=0, field_name=field, accepted=accepted,
                                     reason_code="other", reason_text="x", key_mode="work_shape")
    assert lesson.relation == eh.RELATIONS[field]
    assert lesson.relations["source_proposal_id"] == before.proposal_id
    store.add(lesson)
    after = eh.propose(DEAL, store, key_mode="work_shape")
    assert after.field_value(0, field) == pytest.approx(accepted)
    assert after.lines[0].fields[field].source == "lesson"
    assert after.lines[0].fields[field].correction_id == lesson.id
    # Upstream fields are untouched; only fields that derive from the corrected
    # one move (hours_per_visit reads units and visits).
    order = list(eh.FIELDS)
    for other in order[: order.index(field)]:
        assert after.lines[0].fields[other].source == "baseline"
    assert after.version != before.version  # the lesson set is part of the version


def test_totals_cannot_be_corrected():
    p = eh.propose(DEAL, offline_store())
    with pytest.raises(ValueError):
        eh.rate_from_accepted(p, 0, "total_hours", 10)
    with pytest.raises(ValueError):
        eh.rate_from_accepted(p, 0, "units", 0)


def test_preview_never_writes_the_live_store():
    store = offline_store()
    p = eh.propose(DEAL, store, key_mode="work_shape")
    lesson = eh.lesson_from_override(DEAL, p, line_index=0, field_name="visits", accepted=12,
                                     reason_code="wrong_site_count", key_mode="work_shape")
    eh.preview_transfer(store, lesson, [DEAL], key_mode="work_shape")
    assert store.all_corrections() == []
