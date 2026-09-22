"""Kit-taught lessons fire on the next deal's work lines.

A Deal Kit teaches on a work line and the PM's own words name a circumstance:
"drop ceilings, ladder work". That becomes a `mentions` condition the judged
text must satisfy — no deal facts needed — so the lesson fires on a sentence
that says "ladder" and stays silent on one that doesn't.
"""

from __future__ import annotations

from app.core.feedback_store import condition_holds
from app.core.pm_feedback import _threshold_for
from app.core.pm_note_router import extract_mentions


def test_mentions_come_from_the_pms_words_not_the_kits_nouns():
    assert extract_mentions("Drop ceilings over 14 ft, ladder work") == ["drop", "ceil", "ladder"]
    assert extract_mentions("Bigger job than it looks") == []
    assert extract_mentions("After hours at the mall; union site") == ["after", "mall", "union"]
    assert extract_mentions("") == []


def test_mentions_condition_holds_on_the_judged_text_alone():
    when = {"when": {"mentions": ["ladder", "ceil"]}}
    assert condition_holds(when, {"text": "Install 24 Cat6 drops in the drop ceiling from a ladder"})
    assert condition_holds(when, {"text": "Ceiling-mounted APs at 30 stores"})
    assert not condition_holds(when, {"text": "Rack and stack the core switch"})
    assert not condition_holds(when, None)
    assert condition_holds({}, None)


def test_kit_heads_take_a_looser_bar_than_the_default():
    assert _threshold_for("hours", "global") < _threshold_for("gap", "global")
    assert _threshold_for("hours", "global") == 0.76
    assert _threshold_for("task_tier", "deal") == 0.72
    assert _threshold_for("commercial", "global") == 0.76
    assert _threshold_for("bom_owner", "global") == 0.78


def test_correction_route_turns_the_pms_words_into_a_mentions_condition(monkeypatch):
    from tests.test_routes_feedback import _client, _store
    from app.core.decide import set_store

    store = _store()
    set_store(store)
    try:
        r = _client().post(
            "/projects/deal-1/feedback/correction",
            json={
                "head": "hours",
                "deal_id": "deal-1",
                "target_id": "task:cable drops cat6",
                "text": "Install 24 Cat6 cable drops to the new APs in the drop ceiling",
                "old_value": "hours=0.66;per=drop;qty=24",
                "new_value": "hours=1;per=drop;qty=24;role=L2 Tech",
                "scope": "global",
                "rationale": "Drop ceilings over 14 ft, ladder work",
                "relations": {"outcome": "correct", "words": "Drop ceilings over 14 ft, ladder work"},
                "pm": "pm@purtera-it.com",
            },
        )
        assert r.status_code == 200, r.text
        corr = store.get(r.json()["correction_id"])
        assert corr.relations["when"] == {"mentions": ["drop", "ceil", "ladder"]}
        assert "words" not in corr.relations
        # A chip-only reason (no typed words) leaves the lesson unconditional.
        r2 = _client().post(
            "/projects/deal-1/feedback/correction",
            json={"head": "hours", "deal_id": "deal-1", "target_id": "task:ap swap", "text": "Swap 8 ceiling APs",
                  "old_value": "", "new_value": "hours=0.75;per=AP;qty=8", "scope": "global",
                  "rationale": "Bigger job than it looks", "relations": {"outcome": "new"}, "pm": "pm@x"},
        )
        assert r2.status_code == 200, r2.text
        assert "when" not in store.get(r2.json()["correction_id"]).relations
    finally:
        set_store(None)
