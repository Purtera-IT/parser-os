from __future__ import annotations

from pathlib import Path

from app.core.compiler import compile_project
from app.core.risk import packet_pm_sort_key
from app.core.schemas import (
    AUTHORITY_POLICY_VERSION,
    COMPILER_VERSION,
    PACKETIZER_VERSION,
    SCHEMA_VERSION,
    CompileResult,
)


def test_compile_result_includes_manifest_and_versions(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project")
    assert result.manifest is not None
    assert result.schema_version == SCHEMA_VERSION
    assert result.compiler_version == COMPILER_VERSION
    assert result.compile_id
    assert result.manifest.schema_version == SCHEMA_VERSION
    assert result.manifest.compiler_version == COMPILER_VERSION
    assert result.manifest.packetizer_version == PACKETIZER_VERSION
    assert result.manifest.authority_policy_version == AUTHORITY_POLICY_VERSION
    assert result.manifest.input_signature
    assert result.manifest.output_signature
    assert result.manifest.compile_id == result.compile_id


def test_output_json_roundtrip_preserves_manifest(demo_project: Path, tmp_path: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project")
    out_path = tmp_path / "compiled.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    rebuilt = CompileResult.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert rebuilt.manifest is not None
    assert rebuilt.manifest.input_signature == result.manifest.input_signature
    assert rebuilt.manifest.output_signature == result.manifest.output_signature
def test_deterministic_ordering_in_compile_result(demo_project: Path) -> None:
    result = compile_project(demo_project, project_id="demo_project")
    atom_ids = [a.id for a in result.atoms]
    entity_ids = [e.id for e in result.entities]
    edge_ids = [e.id for e in result.edges]
    packet_sort_keys = [
        packet_pm_sort_key(p) if p.risk is not None else (50, 50, 0.0, p.anchor_key, p.id)
        for p in result.packets
    ]
    assert atom_ids == sorted(atom_ids)
    assert entity_ids == sorted(entity_ids)
    assert edge_ids == sorted(edge_ids)
    assert packet_sort_keys == sorted(packet_sort_keys)
    assert result.warnings == sorted(result.warnings)
