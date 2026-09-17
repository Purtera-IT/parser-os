"""PUR-16 / PUR-57: predict -> correct + why -> preview -> commit -> re-predict, on fixtures."""

from __future__ import annotations

import json

import pytest

from app.core.override_log import OverrideLog
from app.learning import one_deal_loop as loop


def _run(capsys, argv):
    assert loop.main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_predict_shows_every_field_with_evidence(capsys):
    out = _run(capsys, ["predict", "--deal", "SYN-A01", "--key-mode", "work_shape"])
    line = out["lines"][0]
    assert set(line["fields"]) == {"units", "visits", "hours_per_visit"}
    assert line["shape"]["object"] == "camera" and line["wording"]
    assert all(f["source"] == "baseline" for f in line["fields"].values())


def test_preview_without_commit_writes_nothing(capsys, tmp_path):
    store, ovr = str(tmp_path / "l.sqlite"), str(tmp_path / "o.sqlite")
    out = _run(capsys, ["correct", "--deal", "SYN-A01", "--field", "hours_per_visit", "--value", "18",
                        "--reason-code", "other", "--reason-text", "lift per camera",
                        "--key-mode", "work_shape", "--store", store, "--overrides", ovr])
    assert out["committed"] is False
    assert [c["deal_id"] for c in out["would_also_change"]] == ["SYN-B01"]
    assert loop.open_store(store).all_corrections() == []
    assert OverrideLog(ovr).for_deal("SYN-A01") == []


def test_commit_stores_lesson_records_override_and_repredicts(capsys, tmp_path):
    store, ovr = str(tmp_path / "l.sqlite"), str(tmp_path / "o.sqlite")
    before = _run(capsys, ["predict", "--deal", "SYN-B01", "--key-mode", "work_shape", "--store", store])
    out = _run(capsys, ["correct", "--deal", "SYN-A01", "--field", "hours_per_visit", "--value", "18",
                        "--reason-code", "other", "--reason-text", "lift per camera",
                        "--key-mode", "work_shape", "--store", store, "--overrides", ovr, "--commit"])
    assert out["committed"] is True
    assert out["repredicted"]["lines"][0]["fields"]["hours_per_visit"]["value"] == 18.0
    rows = OverrideLog(ovr).for_deal("SYN-A01")
    assert len(rows) == 1 and rows[0].proposal_id == out["proposal_id"]
    assert rows[0].proposal_version == out["proposal_version"] and rows[0].lesson_id == out["lesson"]["id"]
    after = _run(capsys, ["predict", "--deal", "SYN-B01", "--key-mode", "work_shape", "--store", store])
    b0 = before["lines"][0]["fields"]["hours_per_visit"]
    b1 = after["lines"][0]["fields"]["hours_per_visit"]
    assert b1["value"] != b0["value"] and b1["source"] == "lesson"
    c = _run(capsys, ["predict", "--deal", "SYN-C01", "--key-mode", "work_shape", "--store", store])
    assert c["lines"][0]["fields"]["hours_per_visit"]["source"] == "baseline"


def test_other_reason_without_text_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="reason_text"):
        loop.main(["correct", "--deal", "SYN-A01", "--field", "units", "--value", "20",
                   "--reason-code", "other", "--key-mode", "work_shape"])
