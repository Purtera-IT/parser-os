"""PR9 — schematic-specific gold metrics + stress-case behavior."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

fitz = pytest.importorskip("fitz")

from app.core.gold_compare import compare_to_gold
from app.core.ids import stable_id
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser


def _compile_schematic_pdf(pdf: Path, pack_id: str = "security_camera") -> dict[str, Any]:
    """Run OrbitBriefPdfParser and return a ``compiled`` dict in gold_compare shape."""

    parser = OrbitBriefPdfParser()
    pack = load_domain_pack(pack_id)
    art_id = stable_id("art", str(pdf))
    out = parser.parse_artifact("proj_test", art_id, pdf, domain_pack=pack)
    return {
        "project_id": "proj_test",
        "atoms": [a.model_dump(mode="json") for a in out.atoms],
        "edges": [],
        "packets": [],
    }


def _build_camera_pdf(path: Path, ptz_marks: int = 5, include_legend: bool = True) -> None:
    doc = fitz.open()
    if include_legend:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
        page.insert_text((72, 90), "SYMBOL", fontsize=10)
        page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
        page.insert_text((300, 90), "COUNT", fontsize=10)
        page.insert_text((72, 110), "PTZ", fontsize=10)
        page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
        page.insert_text((300, 110), str(ptz_marks), fontsize=10)
        page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    if include_legend:
        page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    for i in range(ptz_marks):
        col = i % 3
        row = i // 3
        page.insert_text((100 + col * 150, 200 + row * 80), "PTZ", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(path))
    doc.close()


# ─── core metric behavior ───


def test_skips_schematic_metrics_when_gold_keys_absent(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_camera_pdf(pdf, ptz_marks=3)
    compiled = _compile_schematic_pdf(pdf)
    gold: dict[str, Any] = {"case_id": "no_schematic_keys"}
    result = compare_to_gold(gold=gold, compiled=compiled)
    sch_metrics = {
        k: v
        for k, v in result["metrics"].items()
        if k in {
            "legend_entries_min",
            "detection_targets_include",
            "symbol_counts",
            "missing_legend_pages",
            "unknown_symbol_max",
            "all_schematic_atoms_have_bbox",
        }
    }
    for name, m in sch_metrics.items():
        assert m["verdict"] == "skipped", f"{name} unexpectedly graded: {m}"


def test_legend_entries_min_pass(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_camera_pdf(pdf, ptz_marks=2)
    compiled = _compile_schematic_pdf(pdf)
    result = compare_to_gold(gold={"expected_legend_entries_min": 1}, compiled=compiled)
    assert result["metrics"]["legend_entries_min"]["verdict"] == "pass"


def test_legend_entries_min_fail_when_below_threshold(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_camera_pdf(pdf, ptz_marks=2)
    compiled = _compile_schematic_pdf(pdf)
    result = compare_to_gold(gold={"expected_legend_entries_min": 50}, compiled=compiled)
    assert result["metrics"]["legend_entries_min"]["verdict"] == "fail"


def test_detection_targets_include_pass(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_camera_pdf(pdf, ptz_marks=2)
    compiled = _compile_schematic_pdf(pdf)
    result = compare_to_gold(
        gold={"expected_detection_targets_include": ["ptz_camera"]},
        compiled=compiled,
    )
    m = result["metrics"]["detection_targets_include"]
    assert m["verdict"] == "pass"
    assert "ptz_camera" in m["actual_present"]


def test_detection_targets_include_fail_lists_missing(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_camera_pdf(pdf, ptz_marks=2)
    compiled = _compile_schematic_pdf(pdf)
    result = compare_to_gold(
        gold={"expected_detection_targets_include": ["bullet_camera"]},
        compiled=compiled,
    )
    m = result["metrics"]["detection_targets_include"]
    assert m["verdict"] == "fail"
    assert "bullet_camera" in m["missing"]


def test_symbol_counts_pass(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_camera_pdf(pdf, ptz_marks=5)
    compiled = _compile_schematic_pdf(pdf)
    result = compare_to_gold(
        gold={"expected_symbol_counts": {"ptz_camera": 5}},
        compiled=compiled,
    )
    m = result["metrics"]["symbol_counts"]
    assert m["verdict"] == "pass", m


def test_symbol_counts_fail_lists_misses(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_camera_pdf(pdf, ptz_marks=2)
    compiled = _compile_schematic_pdf(pdf)
    result = compare_to_gold(
        gold={"expected_symbol_counts": {"ptz_camera": 99}},
        compiled=compiled,
    )
    m = result["metrics"]["symbol_counts"]
    assert m["verdict"] == "fail"
    assert "ptz_camera" in m["misses"]


def test_missing_legend_pages_pass(tmp_path: Path) -> None:
    # Build a drawing-only PDF with no legend; the parser should emit
    # missing_legend warnings that the gold can assert on.
    pdf = tmp_path / "orphan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN ORPHAN", fontsize=14)
    page.insert_text((500, 740), "E5.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    compiled = _compile_schematic_pdf(pdf)
    result = compare_to_gold(gold={"expected_missing_legend_pages": [0]}, compiled=compiled)
    assert result["metrics"]["missing_legend_pages"]["verdict"] == "pass"


def test_unknown_symbol_max_pass(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_camera_pdf(pdf, ptz_marks=5)
    compiled = _compile_schematic_pdf(pdf)
    result = compare_to_gold(
        gold={"expected_unknown_symbol_count_max": 3},
        compiled=compiled,
    )
    assert result["metrics"]["unknown_symbol_max"]["verdict"] == "pass"


def test_all_schematic_atoms_have_bbox_pass(tmp_path: Path) -> None:
    pdf = tmp_path / "drawings.pdf"
    _build_camera_pdf(pdf, ptz_marks=3)
    compiled = _compile_schematic_pdf(pdf)
    result = compare_to_gold(
        gold={"expected_all_schematic_atoms_have_bbox": True},
        compiled=compiled,
    )
    m = result["metrics"]["all_schematic_atoms_have_bbox"]
    assert m["verdict"] == "pass", m


def test_existing_metrics_unchanged_when_schematic_added(tmp_path: Path) -> None:
    # Run gold_compare on an arbitrary compiled dict with only legacy
    # metrics in gold — the comparator should still grade those metrics
    # the same way it did before PR9.
    legacy_gold = {
        "case_id": "legacy",
        "expected_min_atom_count": 0,
        "expected_min_packet_count": 0,
    }
    compiled = {"project_id": "p", "atoms": [], "edges": [], "packets": []}
    result = compare_to_gold(gold=legacy_gold, compiled=compiled)
    assert result["metrics"]["atom_count"]["verdict"] == "pass"
    assert result["metrics"]["packet_count"]["verdict"] == "pass"
    # Schematic metrics must all be skipped on a non-schematic gold.
    for k in (
        "legend_entries_min",
        "detection_targets_include",
        "symbol_counts",
        "missing_legend_pages",
        "unknown_symbol_max",
        "all_schematic_atoms_have_bbox",
    ):
        assert result["metrics"][k]["verdict"] == "skipped"


def test_schematic_stress_case_full_battery(tmp_path: Path) -> None:
    """Single PDF exercises every schematic gold field at once.

    This mirrors what a real ``SCHEMATIC_*/labels/gold_standard.json``
    case looks like.
    """
    pdf = tmp_path / "schematic_demo.pdf"
    _build_camera_pdf(pdf, ptz_marks=5)
    compiled = _compile_schematic_pdf(pdf)
    gold = {
        "case_id": "SCHEMATIC_SECURITY_CAMERA_DEMO",
        "service_line": "security_camera",
        "expected_legend_entries_min": 1,
        "expected_detection_targets_include": ["ptz_camera"],
        "expected_symbol_counts": {"ptz_camera": 5},
        "expected_unknown_symbol_count_max": 5,
        "expected_all_schematic_atoms_have_bbox": True,
    }
    result = compare_to_gold(gold=gold, compiled=compiled)
    for name in (
        "legend_entries_min",
        "detection_targets_include",
        "symbol_counts",
        "unknown_symbol_max",
        "all_schematic_atoms_have_bbox",
    ):
        assert result["metrics"][name]["verdict"] == "pass", (
            name,
            result["metrics"][name],
        )
