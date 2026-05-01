from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from app.core.compiler import compile_project
from app.learning.promotion import apply_approved_suggestion, promote_review_to_fixture


def _write_review_labels(path: Path, packet_id: str, family: str, anchor_key: str) -> None:
    payload = {
        "reviews": [
            {
                "packet_id": packet_id,
                "family": family,
                "anchor_key": anchor_key,
                "correct_packet": False,
                "correct_governing_atom": None,
                "correct_severity": None,
                "should_be_status": "needs_review",
                "missing_evidence": "Need explicit confirmation",
                "false_positive_reason": "Regression seed",
                "reviewer_notes": "Seed this for promotion fixture.",
                "reviewed_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        "metadata": {"source_compile_result": "local"},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compile_demo_json(path: Path, demo_project: Path) -> dict:
    result = compile_project(
        project_dir=demo_project,
        project_id="promotion_case",
        allow_errors=True,
        allow_unverified_receipts=True,
    )
    payload = json.loads(result.model_dump_json())
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def test_promotion_creates_fixture_metadata(tmp_path: Path, demo_project: Path) -> None:
    compile_path = tmp_path / "compile.json"
    payload = _compile_demo_json(compile_path, demo_project)
    first_packet = payload["packets"][0]
    review_path = tmp_path / "packet_reviews.json"
    _write_review_labels(review_path, first_packet["id"], first_packet["family"], first_packet["anchor_key"])
    out_dir = tmp_path / "regression" / "CASE_X"

    artifact = promote_review_to_fixture(
        review_labels_path=review_path,
        compile_result_path=compile_path,
        out_dir=out_dir,
    )
    assert artifact.status == "proposed"
    assert (out_dir / "promotion_metadata.json").exists()
    assert (out_dir / "promotion_artifact.json").exists()
    assert (out_dir / "project" / "review_regression_email.txt").exists()
    assert (out_dir / "gold.json").exists()


def test_apply_requires_approve(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(Path("app/domain/default_pack.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    suggestion = {
        "suggestion_id": "sugg_x",
        "suggestion_type": "domain_alias",
        "proposed_change": {"device_aliases": {"ip_camera": ["CCTV cam"]}},
        "evidence_count": 2,
        "positive_examples": ["a", "b"],
        "negative_examples": [],
        "confidence": 0.9,
        "requires_human_approval": True,
        "target_file": str(pack_path),
        "test_file": str(tmp_path / "regression.txt"),
    }
    suggestion_path = tmp_path / "suggestion.json"
    suggestion_path.write_text(json.dumps(suggestion, indent=2), encoding="utf-8")
    before = pack_path.read_text(encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/apply_approved_suggestion.py", "--suggestion", str(suggestion_path)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    after = pack_path.read_text(encoding="utf-8")
    assert before == after


def test_domain_pack_change_includes_test_file(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(Path("app/domain/default_pack.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    suggestion = {
        "suggestion_id": "sugg_domain_pack",
        "suggestion_type": "domain_alias",
        "proposed_change": {"device_aliases": {"ip_camera": ["CCTV cam"]}},
        "evidence_count": 3,
        "positive_examples": ["x", "y", "z"],
        "negative_examples": [],
        "confidence": 1.0,
        "requires_human_approval": True,
        "target_file": str(pack_path),
        "test_file": str(tmp_path / "promotion_test.txt"),
    }
    artifact = apply_approved_suggestion(suggestion_payload=suggestion, approve=True)
    assert artifact.status == "applied"
    assert any("test" in Path(path).name.lower() for path in artifact.proposed_files)
    parsed = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    assert "CCTV cam" in parsed["device_aliases"]["ip_camera"]
    test_file = next(Path(path) for path in artifact.proposed_files if "test" in Path(path).name.lower())
    assert test_file.exists()


def test_rejected_suggestion_does_not_modify_files(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(Path("app/domain/default_pack.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    suggestion = {
        "suggestion_id": "sugg_rejected",
        "suggestion_type": "domain_alias",
        "proposed_change": {"device_aliases": {"ip_camera": ["NeverApply"]}},
        "evidence_count": 2,
        "positive_examples": ["x", "y"],
        "negative_examples": [],
        "confidence": 0.9,
        "requires_human_approval": True,
        "target_file": str(pack_path),
        "test_file": str(tmp_path / "rejected_test.txt"),
        "status": "rejected",
    }
    before = pack_path.read_text(encoding="utf-8")
    artifact = apply_approved_suggestion(suggestion_payload=suggestion, approve=True)
    after = pack_path.read_text(encoding="utf-8")
    assert artifact.status == "rejected"
    assert before == after


def test_generated_regression_fixture_can_be_compiled(tmp_path: Path, demo_project: Path) -> None:
    compile_path = tmp_path / "compile.json"
    payload = _compile_demo_json(compile_path, demo_project)
    packet = payload["packets"][0]
    review_path = tmp_path / "packet_reviews.json"
    _write_review_labels(review_path, packet["id"], packet["family"], packet["anchor_key"])
    out_dir = tmp_path / "regression" / "CASE_COMPILE"
    promote_review_to_fixture(
        review_labels_path=review_path,
        compile_result_path=compile_path,
        out_dir=out_dir,
    )

    result = compile_project(
        project_dir=out_dir / "project",
        project_id="promotion_regression_compile",
        allow_errors=True,
        allow_unverified_receipts=True,
    )
    assert result.manifest is not None
    assert result.project_id == "promotion_regression_compile"
