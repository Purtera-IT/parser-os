"""PUR-58: operator stage selection on re-parse. Synthetic fixtures only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.core.compiler import compile_project
from app.core.stage_plan import (
    PARTIAL_WARNING_PREFIX,
    SELECTABLE_STAGES,
    StagePlan,
    StagePlanError,
    is_partial_result,
    plan_for_reparse,
)


# ---- plan validation -------------------------------------------------------

def test_none_is_full_parse():
    plan = StagePlan.from_request(None)
    assert not plan.partial
    assert all(plan.runs(s) for s in SELECTABLE_STAGES)
    assert plan.record()["stages_skipped"] == []
    assert plan.warning() is None


def test_unknown_stage_lists_valid_names():
    with pytest.raises(StagePlanError) as ei:
        StagePlan.from_request(["work_order", "bogus"])
    msg = str(ei.value)
    assert "bogus" in msg
    for name in SELECTABLE_STAGES:
        assert name in msg


def test_refuses_stage_without_its_input_with_reason():
    with pytest.raises(StagePlanError) as ei:
        plan_for_reparse("quoted_history_dedup")
    assert "email_threading" in str(ei.value)
    assert "refused" in str(ei.value)


def test_dependency_satisfied_when_selected():
    plan = plan_for_reparse("quoted_history_dedup, email_threading")
    assert plan.requested == ["email_threading", "quoted_history_dedup"]
    assert plan.partial


def test_core_stages_always_run_and_record_labels_partial():
    plan = plan_for_reparse(["work_order"])
    assert plan.runs("parse_artifacts") and plan.runs("graph_build")
    assert plan.runs("work_order") and not plan.runs("task_hours")
    rec = plan.record()
    assert rec["partial"] is True
    assert rec["stages_ran"] == ["work_order"]
    assert "task_hours" in rec["stages_skipped"]
    assert "parse_artifacts" in rec["stages_reused"]
    assert plan.warning().startswith(PARTIAL_WARNING_PREFIX)
    assert plan_for_reparse(["work_order"], use_cache=False).record()["stages_reused"] == {}


def test_selecting_every_stage_is_not_partial():
    plan = plan_for_reparse(list(SELECTABLE_STAGES))
    assert not plan.partial and plan.warning() is None


# ---- compile integration ---------------------------------------------------

def test_default_compile_has_no_stage_plan(tmp_path: Path):
    (tmp_path / "p").mkdir()
    result = compile_project(tmp_path / "p", allow_errors=True, allow_unverified_receipts=True)
    assert result.stage_plan is None
    assert not any(w.startswith(PARTIAL_WARNING_PREFIX) for w in result.warnings)
    assert not is_partial_result(result)


def test_partial_compile_is_labelled_everywhere(demo_project: Path):
    result = compile_project(
        demo_project, allow_errors=True, allow_unverified_receipts=True, stages=["task_hours"]
    )
    assert result.stage_plan["partial"] is True
    assert any(w.startswith(PARTIAL_WARNING_PREFIX) for w in result.warnings)
    stage_names = [s.stage_name for s in result.trace.stages]
    assert "stage_plan" in stage_names
    assert "email_threading" not in stage_names and "task_hours" in stage_names
    dumped = json.loads(result.model_dump_json())
    assert is_partial_result(dumped)


def test_compile_refuses_bad_plan_before_work(tmp_path: Path):
    with pytest.raises(StagePlanError):
        compile_project(tmp_path / "does-not-matter", stages=["nope"])


# ---- CLI -------------------------------------------------------------------

def test_cli_rejects_bad_stages(tmp_path: Path):
    (tmp_path / "p").mkdir()
    res = CliRunner().invoke(
        cli_app,
        ["compile", str(tmp_path / "p"), "--out", str(tmp_path / "o.json"), "--stages", "quoted_history_dedup"],
    )
    assert res.exit_code == 2
    assert not (tmp_path / "o.json").exists()


def test_cli_partial_output_labelled(tmp_path: Path):
    (tmp_path / "p").mkdir()
    out = tmp_path / "o.json"
    res = CliRunner().invoke(
        cli_app,
        ["compile", str(tmp_path / "p"), "--out", str(out), "--stages", "work_order", "--skip-orbitbrief"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(out.read_text())
    assert payload["stage_plan"]["partial"] is True
    assert any(w.startswith(PARTIAL_WARNING_PREFIX) for w in payload["warnings"])


# ---- API -------------------------------------------------------------------

def _client(monkeypatch, captured: dict) -> TestClient:
    from app.api import routes_compile

    monkeypatch.setattr(routes_compile, "list_artifacts", lambda pid: ["x"])

    def fake_compile(**kwargs):
        captured.update(kwargs)
        return {"project_id": kwargs["project_id"]}

    monkeypatch.setattr(routes_compile, "compile_project", fake_compile)
    app = FastAPI()
    app.include_router(routes_compile.router)
    return TestClient(app)


def test_route_rejects_bad_plan_with_reason(monkeypatch):
    captured: dict = {}
    r = _client(monkeypatch, captured).post("/projects/p1/compile?stages=quoted_history_dedup")
    assert r.status_code == 422
    assert "email_threading" in r.json()["detail"]
    assert captured == {}


def test_route_passes_plan_and_default_omits_it(monkeypatch):
    captured: dict = {}
    client = _client(monkeypatch, captured)
    assert client.post("/projects/p1/compile?stages=work_order").status_code == 200
    assert captured["stages"].requested == ["work_order"]
    captured.clear()
    assert client.post("/projects/p1/compile").status_code == 200
    assert "stages" not in captured
