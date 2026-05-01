from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.core.compiler import compile_project
from app.eval.real_data import compile_case, read_case_manifest_domain_pack


def test_read_case_manifest_domain_pack_precedence_keys(tmp_path: Path) -> None:
    case = tmp_path / "CASE_KEYS"
    case.mkdir()
    (case / "case_manifest.json").write_text(
        json.dumps({"domain": "ignored", "domain_pack": "also_ignored", "compiler_domain_pack": "copper_cabling"}),
        encoding="utf-8",
    )
    assert read_case_manifest_domain_pack(case) == "copper_cabling"


def test_read_case_manifest_falls_back_domain_pack(tmp_path: Path) -> None:
    case = tmp_path / "CASE_DP"
    case.mkdir()
    (case / "case_manifest.json").write_text(json.dumps({"domain_pack": "copper_cabling"}), encoding="utf-8")
    assert read_case_manifest_domain_pack(case) == "copper_cabling"


def test_compile_case_manifest_selects_copper_cabling(tmp_path: Path) -> None:
    case_id = "MANIFEST_COPPER"
    cdir = tmp_path / case_id
    (cdir / "artifacts").mkdir(parents=True)
    (cdir / "outputs").mkdir(parents=True)
    (cdir / "case_manifest.json").write_text(
        json.dumps({"compiler_domain_pack": "copper_cabling"}),
        encoding="utf-8",
    )
    (cdir / "artifacts" / "note.txt").write_text("noop", encoding="utf-8")
    summary = compile_case(tmp_path, case_id)
    assert summary["domain_pack_id"] == "copper_cabling"
    assert summary["domain_pack_version"] == "0.4.0-generated"


def test_compile_case_cli_domain_pack_overrides_manifest(tmp_path: Path) -> None:
    case_id = "CLI_OVERRIDE"
    cdir = tmp_path / case_id
    (cdir / "artifacts").mkdir(parents=True)
    (cdir / "outputs").mkdir(parents=True)
    (cdir / "case_manifest.json").write_text(
        json.dumps({"compiler_domain_pack": "copper_cabling"}),
        encoding="utf-8",
    )
    (cdir / "artifacts" / "note.txt").write_text("noop", encoding="utf-8")
    summary = compile_case(tmp_path, case_id, domain_pack="default_pack")
    assert summary["domain_pack_id"] == "default_pack"
    assert summary["domain_pack_version"] == "1.0.0"


def test_compile_case_no_manifest_uses_default_pack(tmp_path: Path) -> None:
    case_id = "NO_MANIFEST"
    cdir = tmp_path / case_id
    (cdir / "artifacts").mkdir(parents=True)
    (cdir / "outputs").mkdir(parents=True)
    (cdir / "artifacts" / "note.txt").write_text("noop", encoding="utf-8")
    summary = compile_case(tmp_path, case_id)
    assert summary["domain_pack_id"] == "default_pack"


@pytest.mark.skipif(
    not (
        Path(
            os.environ.get(
                "COPPER_VALIDATION_ROOT",
                r"c:\Users\lilli\Downloads\purtera_copper_low_voltage_public_validation_packs"
                r"\purtera_copper_low_voltage_validation_packs\real_data_cases",
            )
        )
        / "COPPER_001_SPRING_LAKE_AUDITORIUM"
        / "artifacts"
        / "extracted"
    ).is_dir(),
    reason="COPPER_001 artifacts not present",
)
def test_copper_001_compile_keeps_material_aggregate_packets_with_copper_domain() -> None:
    root = Path(
        os.environ.get(
            "COPPER_VALIDATION_ROOT",
            r"c:\Users\lilli\Downloads\purtera_copper_low_voltage_public_validation_packs"
            r"\purtera_copper_low_voltage_validation_packs\real_data_cases",
        )
    )
    result = compile_project(
        project_dir=root / "COPPER_001_SPRING_LAKE_AUDITORIUM" / "artifacts",
        project_id="COPPER_001_SPRING_LAKE_AUDITORIUM",
        allow_errors=True,
        allow_unverified_receipts=True,
        domain_pack="copper_cabling",
    )
    assert result.manifest
    assert result.manifest.domain_pack_id == "copper_cabling"
    mat_edges = [
        e
        for e in result.edges
        if e.edge_type.value == "contradicts"
        and (e.metadata or {}).get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote"
    ]
    assert len(mat_edges) >= 3
    identities = {(e.metadata or {}).get("identity") for e in mat_edges}
    assert {"rj45", "cat6_utp", "cat6_stp"}.issubset(identities)
    for p in result.packets:
        if p.certificate and p.anchor_key.startswith("material:"):
            assert p.certificate.domain_pack_id == "copper_cabling"
