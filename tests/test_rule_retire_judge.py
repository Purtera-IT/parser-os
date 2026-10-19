"""rule_retire_judge: the judge scored against each rule's own decisions."""
from ml.c3.rule_retire_judge import judged_answer, score


def test_answers_map_per_head():
    v = {"section": "customer_responsibilities", "owner": "customer"}
    assert judged_answer("sow_section", v) == "customer_responsibilities"
    assert judged_answer("sow_party", v) == "customer"
    assert judged_answer("sow_supplier", v) == "customer"
    assert judged_answer("sow_supplier", {"owner": "technician"}) == "us"
    assert judged_answer("sow_keep", v) is None


def test_ready_needs_30_and_95_percent():
    d = {"rule": "bom_lists", "head": "sow_section", "value": "provider_responsibilities", "was": "scope_of_work"}
    ok = {"section": "provider_responsibilities", "owner": "technician"}
    bad = {"section": "scope_of_work", "owner": "technician"}
    r = score([("x", d, ok)] * 29)
    assert r[0]["status"] == "learning" and r[0]["blocker"] == "n=29 of 30 needed"
    r = score([("x", d, ok)] * 30)
    assert r[0]["status"] == "ready" and r[0]["judge_agrees"] == 1.0
    r = score([("x", d, ok)] * 30 + [("x", d, bad)] * 2)
    assert r[0]["status"] == "learning" and r[0]["judge_says_was"] == 2
