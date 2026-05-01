from __future__ import annotations

import json
from pathlib import Path

from app.core.compiler import compile_project
from scripts.run_perf_benchmark import run_benchmark


def test_perf_script_writes_json(tmp_path: Path) -> None:
    report = run_benchmark(sites=5, devices_per_site=1)
    out = tmp_path / "perf.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["artifacts"] >= 1
    assert parsed["atoms"] >= 1
    assert parsed["total_duration_ms"] >= 0.0


def test_demo_compile_under_generous_threshold(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert result.trace is not None
    assert result.trace.total_duration_ms < 20000.0
