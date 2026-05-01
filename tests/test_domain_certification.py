from __future__ import annotations

import json
from pathlib import Path

import yaml

from app.eval.domain_certification import CertificationReport, certify_domain_pack


def _write_gold(fixtures_dir: Path, *, scenario_id: str, project_dir: Path) -> None:
    scenario = fixtures_dir / scenario_id
    scenario.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario_id": scenario_id,
        "project_dir": str(project_dir),
        "expected_packets": [
            {
                "family": "quantity_conflict",
                "anchor_key_contains": "",
                "must_contain_quantities": [91, 72],
                "expected_status": "needs_review",
                "forbidden_governing_authority": ["vendor_quote"],
            }
        ],
        "expected_governing": [],
        "forbidden": [
            {"condition": "deleted_text_governs"},
            {"condition": "quoted_old_email_governs_current_conflict"},
        ],
    }
    (scenario / "gold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clone_pack(src: Path, dst: Path) -> dict:
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    dst.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def test_valid_security_camera_pack_passes(demo_project: Path, tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    _write_gold(fixtures, scenario_id="security_ok", project_dir=demo_project)
    report = certify_domain_pack(
        domain_pack_path=Path("app/domain/security_camera_pack.yaml"),
        fixtures_dir=fixtures,
    )
    assert report.passed is True
    assert all(row.passed or row.severity == "warning" for row in report.checks)


def test_pack_with_duplicate_alias_collision_fails(demo_project: Path, tmp_path: Path) -> None:
    pack_path = tmp_path / "bad_pack.yaml"
    payload = _clone_pack(Path("app/domain/security_camera_pack.yaml"), pack_path)
    payload["entity_types"] = [
        {"name": "site", "aliases": ["camera"], "examples": ["Main Campus"]},
        {"name": "device", "aliases": ["camera"], "examples": ["IP Camera"]},
    ]
    pack_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    fixtures = tmp_path / "fixtures"
    _write_gold(fixtures, scenario_id="security_bad_alias", project_dir=demo_project)

    report = certify_domain_pack(domain_pack_path=pack_path, fixtures_dir=fixtures)
    check = next(row for row in report.checks if row.check_id == "check_03_site_device_alias_collision")
    assert report.passed is False
    assert check.passed is False


def test_pack_missing_risk_defaults_warns_or_fails_by_severity(demo_project: Path, tmp_path: Path) -> None:
    pack_path = tmp_path / "risk_pack.yaml"
    payload = _clone_pack(Path("app/domain/security_camera_pack.yaml"), pack_path)
    payload["risk_defaults"].pop("ip_camera_unit_exposure", None)
    pack_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    fixtures = tmp_path / "fixtures"
    _write_gold(fixtures, scenario_id="security_missing_risk", project_dir=demo_project)

    report = certify_domain_pack(domain_pack_path=pack_path, fixtures_dir=fixtures)
    check = next(row for row in report.checks if row.check_id == "check_04_risk_defaults_for_key_devices")
    assert check.passed is False
    assert check.severity in {"warning", "error"}


def test_report_serializes() -> None:
    report = CertificationReport(
        pack_id="security_camera",
        pack_version="1.0.0",
        passed=True,
        checks=[],
        parser_coverage={"match_rate": 1.0},
        alias_collision_count=0,
        required_fixture_count=1,
        benchmark_metrics={"compile_success_rate": 1.0},
        recommendations=[],
    )
    payload = report.model_dump_json(indent=2)
    assert '"pack_id": "security_camera"' in payload
