from __future__ import annotations

import json
from pathlib import Path

from app.core.compiler import compile_project
from app.eval.real_data import compile_case, init_case, summarize_cases, write_packet_label_skeleton


def test_init_case_creates_structure(tmp_path: Path) -> None:
    root = tmp_path / "real_data_cases"
    cdir = init_case(root, "CASE_001", notes="test", redaction_status="synthetic", allowed_for_tests=True)
    assert cdir.exists()
    assert (cdir / "artifacts").exists()
    assert (cdir / "labels").exists()
    assert (cdir / "outputs").exists()
    manifest = json.loads((cdir / "case_manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_id"] == "CASE_001"
    assert manifest["allowed_for_tests"] is True


def test_gitignore_protects_artifacts() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "real_data_cases/*/artifacts/*" in text
    assert "real_data_cases/*/outputs/*" in text
    assert "!real_data_cases/.gitkeep" in text
    assert "!real_data_cases/*/labels/*.json" in text
    assert "!real_data_cases/*/case_manifest.json" in text


def test_label_skeleton_writes(tmp_path: Path, demo_project: Path) -> None:
    root = tmp_path / "real_data_cases"
    cdir = init_case(root, "CASE_002", redaction_status="synthetic", allowed_for_tests=True)
    result = compile_project(demo_project, project_id="CASE_002", allow_errors=True, allow_unverified_receipts=True)
    outputs = cdir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "compile_result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    labels_path = write_packet_label_skeleton(root, "CASE_002")
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    assert payload["packet_labels"]
    first = payload["packet_labels"][0]
    assert {"packet_id", "family", "anchor_key", "predicted_status", "human_label"} <= set(first.keys())


def test_summarize_works_on_fake_labels(tmp_path: Path) -> None:
    root = tmp_path / "real_data_cases"
    case_a = init_case(root, "CASE_A", redaction_status="synthetic", allowed_for_tests=True)
    case_b = init_case(root, "CASE_B", redaction_status="redacted", allowed_for_tests=False)

    (case_a / "labels" / "packet_labels.json").write_text(
        json.dumps(
            {
                "case_id": "CASE_A",
                "packet_labels": [
                    {
                        "packet_id": "p1",
                        "family": "scope_exclusion",
                        "anchor_key": "site:west_wing",
                        "predicted_status": "needs_review",
                        "human_label": {
                            "correct_packet": True,
                            "correct_governing_atom": True,
                            "severity_correct": False,
                            "notes": "severity too high",
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (case_b / "labels" / "packet_labels.json").write_text(
        json.dumps(
            {
                "case_id": "CASE_B",
                "packet_labels": [
                    {
                        "packet_id": "p2",
                        "family": "vendor_mismatch",
                        "anchor_key": "device:ip_camera",
                        "predicted_status": "needs_review",
                        "human_label": {
                            "correct_packet": False,
                            "correct_governing_atom": False,
                            "severity_correct": True,
                            "notes": "wrong anchor",
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = summarize_cases(root)
    assert summary["labeled_case_count"] == 2
    assert summary["packet_precision_estimate"] == 0.5
    assert summary["governing_accuracy_estimate"] == 0.5
    assert summary["severity_accuracy_estimate"] == 0.5
    assert summary["common_failure_modes"]


def test_compile_case_writes_outputs(tmp_path: Path, demo_project: Path) -> None:
    root = tmp_path / "real_data_cases"
    cdir = init_case(root, "CASE_003", redaction_status="synthetic", allowed_for_tests=True)
    artifacts = cdir / "artifacts"
    for src in demo_project.iterdir():
        if src.is_file():
            (artifacts / src.name).write_bytes(src.read_bytes())
    summary = compile_case(root, "CASE_003")
    assert summary["atom_count"] >= 1
    assert (cdir / "outputs" / "compile_result.json").exists()
    assert (cdir / "outputs" / "benchmark_summary.json").exists()
