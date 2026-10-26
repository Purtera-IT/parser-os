"""Queued main-head corrections become the c3 loader's records; drafts never train without approval."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from ml.c3 import labels_from_corrections as lfc  # noqa: E402
from ml.c3.data import IGNORE, featurize  # noqa: E402
from ml.c3.loop import load_deal  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

DEAL = "11111111-2222-3333-4444-555555555555"
A = ["atm_a1", "atm_a2", "atm_a3", "atm_a4"]


def _env():
    return {"crm": {"deal_name": "010999 - Widgets"},
            "documents": [{"artifact_id": "f", "artifact_type": "eml"}],
            "atoms": [{"id": A[0], "artifact_id": "f", "text": "Mount four widgets in room A."},
                      {"id": A[1], "artifact_id": "f", "text": "Customer supplies the widgets."},
                      {"id": A[2], "artifact_id": "f", "text": "7 Penn Plaza, New York"},
                      {"id": A[3], "artifact_id": "f", "text": "7 Penn Plaza floor 12"}],
            "suppressed": [{"id": "atm_s1", "artifact_id": "f", "text": "Hope all is well!"}]}


def _row(head, key, verdict, atoms, person=True, cid=None, at=1.0, size="line", **kw):
    return {"id": f"{cid or head}#0", "correction_id": cid or f"pm_{head}_{verdict}", "head": head, "key": key,
            "size": size, "verdict": verdict, "atoms": atoms, "deal_uuid": DEAL, "deal": "010999",
            "by_person": person, "author": "pm@purtera-it.com" if person else "claude-agent (review)",
            "at": at, "skip": None, **kw}


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def _build(tmp_path, rows, approvals=None):
    envs = tmp_path / "envs"
    envs.mkdir(exist_ok=True)
    (envs / f"{DEAL}.json").write_text(json.dumps(_env()))
    s = lfc.build(rows, str(envs), str(tmp_path / "out"), approvals)
    part = lfc.split_of("010999")
    return s, tmp_path / "out" / part / "010999.json", part


def _targets(path, schema):
    d = load_deal(path)
    b = featurize(d, schema)
    keys = [a.atom_id for a in d.atoms]
    return {k: {keys[i]: v for i, v in enumerate(b.targets.get(k, [])) if v != IGNORE}
            for k in ("col:label_type", "col:admission", "col:supplier", "jdg:site_role", "jdg:facility")}, b


def test_person_type_trains_and_draft_does_not(tmp_path, schema):
    rows = [_row("type", "col:label_type", "scope_item", [A[0]]),
            _row("type", "col:label_type", "task", [A[1]], person=False, cid="pm_type_draft")]
    s, path, part = _build(tmp_path, rows)
    t, _ = _targets(path, schema)
    opp = schema.by_key()["col:label_type"]
    assert t["col:label_type"] == {A[0]: opp.index("scope_item")}
    assert s["heads"]["col:label_type"][part] == {"files": [str(path)], "person": 1, "approved": 0, "draft": 1}
    rec = json.loads(path.read_text())
    draft = [lb for lb in rec["labels"] if lb["label_key"] == A[1]][0]
    assert draft["labeler"] == lfc.DRAFT_LABELER and draft["why_author"] == "machine_draft"


def test_owner_approval_makes_a_draft_train(tmp_path, schema):
    rows = [_row("type", "col:label_type", "task", [A[1]], person=False, cid="pm_type_draft")]
    approvals = lfc.read_approvals(None) or {"pm_type_draft": "griffin@purtera-it.com"}
    _, path, _ = _build(tmp_path, rows, approvals)
    t, _ = _targets(path, schema)
    assert A[1] in t["col:label_type"]
    rec = json.loads(path.read_text())
    assert rec["labels"][0]["labeler"] == f"griffin@purtera-it.com {lfc.APPROVED_DRAFT}"


def test_approval_file_needs_who(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"approved": [{"id": "x"}]}))
    with pytest.raises(ValueError):
        lfc.read_approvals(str(p))


def test_person_beats_draft_and_latest_person_wins(tmp_path, schema):
    rows = [_row("type", "col:label_type", "task", [A[0]], at=1.0, cid="c1"),
            _row("type", "col:label_type", "scope_item", [A[0]], at=2.0, cid="c2"),
            _row("type", "col:label_type", "exclusion", [A[0]], person=False, at=3.0, cid="c3")]
    s, path, _ = _build(tmp_path, rows)
    t, _ = _targets(path, schema)
    assert t["col:label_type"][A[0]] == schema.by_key()["col:label_type"].index("scope_item")
    assert s["counts"]["clash_person"] == 1 and s["counts"]["draft_shadowed_by_person"] == 1


def test_judgment_heads_reach_their_super_heads(tmp_path, schema):
    rows = [_row("admission", "col:admission", "drop", ["atm_s1"]),
            _row("bom_owner", "col:supplier", "customer_furnished", [A[1]]),
            _row("site_role", "jdg:site_role", "job_site", [A[2]]),
            _row("facility", "jdg:facility", "keep_facility", [A[3]]),
            _row("site", "jdg:site", "same_site", [A[2], A[3]], size="pair"),
            _row("conflict", "jdg:conflict", "supports", [A[0], A[1]], size="pair", also=["rel:supports"])]
    s, path, part = _build(tmp_path, rows)
    t, b = _targets(path, schema)
    by = schema.by_key()
    assert t["col:admission"]["atm_s1"] == by["col:admission"].index("drop")      # a suppressed line is a line
    assert t["col:supplier"][A[1]] == by["col:supplier"].index("customer")
    assert t["jdg:site_role"][A[2]] == by["jdg:site_role"].index("job_site")
    assert t["jdg:facility"][A[3]] == by["jdg:facility"].index("keep_facility")
    judged = {j.key for j in b.judged}
    assert {"jdg:site", "jdg:conflict"} <= judged
    assert s["heads"]["rel:supports"][part]["person"] == 1


def test_bad_answer_is_counted_not_trained(tmp_path):
    rows = [_row("type", "col:label_type", "raw_utterance", [A[0]]), _row("type", "col:label_type", "task", [A[1]])]
    s, path, _ = _build(tmp_path, rows)
    assert s["counts"]["bad_answer:col:label_type=raw_utterance"] == 1
    assert [lb["label_key"] for lb in json.loads(path.read_text())["labels"]] == [A[1]]


def test_draft_only_deal_is_written_but_trains_nothing(tmp_path, schema):
    rows = [_row("type", "col:label_type", "task", [A[0]], person=False)]
    s, path, part = _build(tmp_path, rows)
    t, _ = _targets(path, schema)
    assert t["col:label_type"] == {}
    assert s["heads"]["col:label_type"][part]["files"] == []      # a refit never reads it for this head


def test_split_is_stable_by_deal():
    assert {lfc.split_of(f"0100{i:02d}") for i in range(60)} == {"train", "val", "holdout"}
    assert lfc.split_of("010999") == lfc.split_of("010999")


def test_main_fails_loudly_with_nothing_to_write(tmp_path):
    qf = tmp_path / "queue.jsonl"
    qf.write_text(json.dumps(dict(_row("type", "col:label_type", "task", [A[0]]), skip="unresolved")) + "\n")
    assert lfc.main(["--queue", str(qf), "--envelopes", str(tmp_path), "--out", str(tmp_path / "o")]) == 3
