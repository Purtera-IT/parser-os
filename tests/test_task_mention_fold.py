"""One unit of work stated more than once is one quote line, priced once.

Live 010043 (compile fc2db7e, 2026-09-15): "3 Verkada cameras" in the email,
"3 cameras-Verkada" in a note and "3 Verkada cameras intsall." in the note's
title were three parent tasks; two carried the learned 5.33 h per camera, so
the Deal Kit was offered 32 hours for 16 hours of work.
"""
from types import SimpleNamespace

from app.core.schemas import AtomType
from app.core.task_hours import estimate_task_hours
from app.core.task_tier_classifier import fold_task_mentions


def _task(aid, text, tier="parent"):
    return SimpleNamespace(id=aid, atom_type=AtomType.task, raw_text=text,
                           value={"task_tier": tier, "is_quote_line": tier == "parent"}, review_flags=["task_tier_" + tier])


def test_three_mentions_of_the_same_work_fold_into_the_fullest():
    a, b, c = _task("t1", "3 Verkada cameras"), _task("t2", "3 cameras-Verkada"), _task("t3", "3 Verkada cameras intsall.")
    assert fold_task_mentions([a, b, c]) == 2
    assert c.value["task_tier"] == "parent" and c.value["is_quote_line"] is True
    for m in (a, b):
        assert m.value["task_tier"] == "child" and m.value["is_quote_line"] is False
        assert m.value["folded_into"] == "t3" and m.value["parent_task_id"] == "t3"
        assert "task_mention_folded" in m.review_flags and "task_tier_parent" not in m.review_flags


def test_different_work_on_the_same_subnet_stays_two_tasks():
    a = _task("t1", "Update QS1 Host PC static IP for the 1517 subnet")
    b = _task("t2", "Update any hardcoded printer IPs from 1518 to the 1517 subnet")
    c = _task("t3", "Remove 1518 Wi-Fi or combine it with 1517 Wi-Fi as needed")
    assert fold_task_mentions([a, b, c]) == 0
    assert all(t.value["task_tier"] == "parent" for t in (a, b, c))


def test_a_survey_said_five_ways_is_not_folded_by_this_rule():
    lines = ["Conduct site survey during regular business hours to assess AP mounting",
             "Chase will conduct the survey to confirm wiring, mounting points, and lift needs.",
             "Site survey to be scheduled promptly to verify all assumptions",
             "I'll also include a site survey as well too.",
             "Expected four-hour survey will confirm AP count"]
    tasks = [_task(f"t{i}", l) for i, l in enumerate(lines)]
    assert fold_task_mentions(tasks) == 0


def test_one_word_mentions_never_fold_and_children_are_left_alone():
    a, b = _task("t1", "Install"), _task("t2", "Install cameras")
    kid = _task("t3", "3 Verkada cameras", tier="child")
    assert fold_task_mentions([a, b, kid]) == 0


def test_a_folded_mention_is_not_priced_again():
    class _Store:
        def all_corrections(self, active_only=True):
            return [SimpleNamespace(relation="task_hours", verdict="hours=5.33;per=camera;role=r1")]

    parent = _task("t3", "3 Verkada cameras intsall.")
    folded = _task("t1", "3 Verkada cameras")
    folded.value.update({"task_tier": "child", "is_quote_line": False, "folded_into": "t3"})
    import app.core.task_hours as th
    calls = []

    def _decide(relation, text, candidates, **kw):
        calls.append(text)
        return SimpleNamespace(verdict="hours=5.33;per=camera;role=r1", source="store", correction_id="c1", confidence=0.95)

    orig = th.decide if hasattr(th, "decide") else None
    import app.core.decide as dm
    real = dm.decide
    dm.decide = _decide
    try:
        n = estimate_task_hours([parent, folded], store=_Store())
    finally:
        dm.decide = real
    assert n == 1 and calls == ["3 Verkada cameras intsall."]
    assert parent.value["estimated_hours"] == 15.99 and "estimated_hours" not in folded.value
