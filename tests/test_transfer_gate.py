"""PUR-15 / PUR-59: the cross-deal transfer gate over frozen A/B/C triples."""

from __future__ import annotations

import copy
import json

from app.eval import transfer_gate as tg

# Both are object synonyms the structural head-noun key cannot see
# (POS terminal / register, laptop / notebook). Tracked, not hidden: when a
# learned object identity lands (docs/LESSON_KEYS.md, option C) this set must
# shrink, and this test will say so.
KNOWN_SHAPE_KEY_FAILURES = {"t04_pos_rollout_visits", "t05_laptop_deploy_visits"}


def test_fixture_set_is_valid_and_large_enough():
    doc = tg.load_triples()
    assert len(doc["triples"]) >= 10
    assert {t["correction"]["field"] for t in doc["triples"]} == {"units", "visits", "hours_per_visit"}
    for t in doc["triples"]:
        a, b, c = t["a"], t["b"], t["c"]
        assert c["wording"] == a["wording"]  # C is adversarial: A's own words
        assert b["wording"] != a["wording"] and b["customer"] != a["customer"]
        assert b["document_title"] != a["document_title"]
        assert all(d["deal_id"].startswith("SYN-") for d in (a, b, c))  # synthetic only


def test_work_shape_key_beats_wording_key_with_no_leaks():
    reports = tg.run_gate()
    wording, shape = reports["wording"], reports["work_shape"]
    # The wording key is the failure the milestone names: C changes, B does not.
    assert wording.leak_rate == 1.0
    assert wording.transfer_rate == 0.0
    # The work-shape key: nothing leaks to C, nothing collateral across the set.
    assert shape.leak_rate == 0.0
    assert shape.collateral == 0
    assert shape.pass_rate > wording.pass_rate
    failed = {r.id for r in shape.results if not r.passed}
    assert failed == KNOWN_SHAPE_KEY_FAILURES


def test_gate_fails_a_key_that_changes_everything():
    """A key loose enough to change every deal passes the B half and must fail
    on C."""
    doc = tg.load_triples()
    loose = copy.deepcopy(doc)
    for t in loose["triples"]:
        t["c"] = copy.deepcopy(t["b"]) | {"deal_id": t["c"]["deal_id"], "wording": t["a"]["wording"],
                                            "customer": t["a"].get("customer")}
    r = tg.run_gate(loose, key_modes=("work_shape",))["work_shape"]
    assert r.leak_rate > 0 and not r.passed


def test_validate_rejects_bad_pairs():
    doc = tg.load_triples()
    bad = copy.deepcopy(doc)
    t = bad["triples"][0]
    t["b"]["wording"] = t["a"]["wording"]
    t["b"]["customer"] = t["a"]["customer"]
    t["correction"]["field"] = "total_hours"
    problems = tg.validate_triples(bad)
    assert any("wording" in p for p in problems)
    assert any("customer" in p for p in problems)
    assert any("correction.field" in p for p in problems)


def test_cli_exit_code_and_json(capsys, tmp_path):
    assert tg.main(["--json"]) == 1  # not every triple passes yet
    out = json.loads(capsys.readouterr().out)
    assert out["work_shape"]["collateral_changes"] == 0
    assert tg.main(["--min-pass-rate", "0.8"]) == 0
    assert tg.main(["--gate-mode", "wording", "--min-pass-rate", "0.1"]) == 1
