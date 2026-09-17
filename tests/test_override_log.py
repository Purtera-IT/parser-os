"""PUR-54: every override recorded against the proposal id/version that produced it."""

from __future__ import annotations

import pytest

from app.core.override_log import REASON_CODES, OverrideLog, OverrideRecord


def _rec(**kw):
    base = dict(deal_id="SYN-A01", proposal_id="prop_1", proposal_version="estimator-v1+lessons:00000000",
                field="hours_per_visit", proposed_value=12.0, accepted_value=18.0,
                reason_code="after_hours", actor="pm@example.test", inputs={"count": 12})
    base.update(kw)
    return OverrideRecord(**base)


def test_record_round_trips_per_deal(tmp_path):
    log = OverrideLog(str(tmp_path / "o.sqlite"))
    rid = log.record(_rec())
    log.record(_rec(field="visits", proposed_value=1, accepted_value=2, reason_code="wrong_visit_count"))
    log.record(_rec(deal_id="SYN-B01"))
    rows = OverrideLog(str(tmp_path / "o.sqlite")).for_deal("SYN-A01")
    assert [r.field for r in rows] == ["hours_per_visit", "visits"]
    assert rows[0].id == rid and rows[0].inputs == {"count": 12}
    assert rows[0].proposal_id == "prop_1" and rows[0].proposal_version.startswith("estimator-v1")


def test_queryable_across_deals():
    log = OverrideLog()
    log.record(_rec())
    log.record(_rec(deal_id="SYN-B01"))
    log.record(_rec(deal_id="SYN-C01", field="units", proposed_value=8, accepted_value=16, reason_code="wrong_unit_count"))
    assert {r.deal_id for r in log.query(field="hours_per_visit")} == {"SYN-A01", "SYN-B01"}
    assert [r.deal_id for r in log.query(reason_code="wrong_unit_count")] == ["SYN-C01"]


@pytest.mark.parametrize("kw,msg", [
    ({"proposal_id": ""}, "proposal_id"),
    ({"proposal_version": ""}, "proposal_version"),
    ({"field": "total_hours"}, "field"),
    ({"reason_code": "because"}, "reason_code"),
    ({"reason_code": "other", "reason_text": ""}, "reason_text"),
    ({"accepted_value": 12.0}, "not an override"),
    ({"actor": ""}, "actor"),
])
def test_refuses_incomplete_overrides(kw, msg):
    with pytest.raises(ValueError, match=msg):
        OverrideLog().record(_rec(**kw))


def test_reason_codes_cover_the_issue_examples():
    for code in ("wrong_site_count", "after_hours", "customer_supplies_equipment", "other"):
        assert code in REASON_CODES
