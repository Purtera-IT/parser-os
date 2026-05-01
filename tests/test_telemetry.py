from __future__ import annotations

from collections import Counter
from pathlib import Path

from app.core.compiler import compile_project


EXPECTED_STAGES = {
    "discover_artifacts",
    "parse_artifacts",
    "source_replay",
    "entity_resolution",
    "graph_build",
    "packetize",
    "packet_certificates",
    "quality_gates",
}


def test_trace_exists_in_compile_result(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert result.trace is not None
    assert result.trace.compile_id == result.compile_id


def test_trace_has_all_expected_stages(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert result.trace is not None
    stage_names = {stage.stage_name for stage in result.trace.stages}
    assert EXPECTED_STAGES.issubset(stage_names)


def test_stage_durations_are_non_negative(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert result.trace is not None
    assert result.trace.total_duration_ms >= 0.0
    assert all(stage.duration_ms >= 0.0 for stage in result.trace.stages)


def test_packet_family_counts_are_correct(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project", allow_unverified_receipts=True)
    assert result.trace is not None
    expected = dict(sorted(Counter(packet.family.value for packet in result.packets).items()))
    assert result.trace.packet_family_counts == expected
