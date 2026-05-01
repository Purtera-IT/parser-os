from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.review.schemas import PacketReviewFile


def _run_review(compile_json: Path, out_json: Path, *extra: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/review_packets.py",
            str(compile_json),
            "--out",
            str(out_json),
            "--non-interactive",
            *extra,
        ],
        check=True,
    )


def test_can_create_labels_non_interactive(tmp_path: Path, demo_project: Path) -> None:
    compile_out = tmp_path / "compile.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "compile",
            str(demo_project),
            "--allow-unverified-receipts",
            "--out",
            str(compile_out),
        ],
        check=True,
    )
    labels_out = tmp_path / "packet_reviews.json"
    _run_review(compile_out, labels_out, "--limit", "3")
    payload = json.loads(labels_out.read_text(encoding="utf-8"))
    assert payload["reviews"]
    assert len(payload["reviews"]) <= 3


def test_label_file_schema_valid(tmp_path: Path, demo_project: Path) -> None:
    compile_out = tmp_path / "compile.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "compile",
            str(demo_project),
            "--allow-unverified-receipts",
            "--out",
            str(compile_out),
        ],
        check=True,
    )
    labels_out = tmp_path / "packet_reviews.json"
    _run_review(compile_out, labels_out, "--limit", "2")
    parsed = PacketReviewFile.model_validate_json(labels_out.read_text(encoding="utf-8"))
    assert parsed.reviews


def test_filtering_by_family_works(tmp_path: Path, demo_project: Path) -> None:
    compile_out = tmp_path / "compile.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "compile",
            str(demo_project),
            "--allow-unverified-receipts",
            "--out",
            str(compile_out),
        ],
        check=True,
    )
    labels_out = tmp_path / "packet_reviews.json"
    _run_review(compile_out, labels_out, "--family", "vendor_mismatch")
    payload = PacketReviewFile.model_validate_json(labels_out.read_text(encoding="utf-8"))
    assert payload.reviews
    assert all(review.family == "vendor_mismatch" for review in payload.reviews)


def test_existing_labels_preserved_and_updated(tmp_path: Path, demo_project: Path) -> None:
    compile_out = tmp_path / "compile.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "compile",
            str(demo_project),
            "--allow-unverified-receipts",
            "--out",
            str(compile_out),
        ],
        check=True,
    )
    labels_out = tmp_path / "packet_reviews.json"
    _run_review(compile_out, labels_out, "--limit", "1")
    first = PacketReviewFile.model_validate_json(labels_out.read_text(encoding="utf-8"))
    _run_review(compile_out, labels_out, "--family", "vendor_mismatch")
    second = PacketReviewFile.model_validate_json(labels_out.read_text(encoding="utf-8"))
    assert len(second.reviews) >= len(first.reviews)
    first_ids = {row.packet_id for row in first.reviews}
    second_ids = {row.packet_id for row in second.reviews}
    assert first_ids.issubset(second_ids)
