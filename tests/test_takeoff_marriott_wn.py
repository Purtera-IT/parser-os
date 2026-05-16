"""Real-PDF regression for the Marriott Atlanta T-set takeoff.

This test runs the full pipeline against the source PDF and asserts
every WN count called out in the gold table. The test is skipped
gracefully when the PDF is missing so a CI without the asset doesn't
turn red.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "real_data_cases"
    / "LOWVOLT_002_MARRIOTT_ATLANTA_T"
    / "artifacts"
    / "2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf"
)


@pytest.fixture(scope="module")
def marriott_takeoff():
    if not PDF_PATH.exists():
        pytest.skip(f"Marriott source PDF not available: {PDF_PATH}")
    pytest.importorskip("fitz")
    from app.takeoff.pipeline import build_low_voltage_takeoff

    return build_low_voltage_takeoff(PDF_PATH)


def _wn_count_by_sheet(takeoff) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in takeoff.devices:
        if d.normalized_class != "wireless_node_outlet":
            continue
        key = d.sheet_number or "?"
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_sheet_count(marriott_takeoff) -> None:
    assert len(marriott_takeoff.sheets) == 25


def test_wn_base_counts_per_sheet(marriott_takeoff) -> None:
    counts = _wn_count_by_sheet(marriott_takeoff)
    expected = {
        "T1.01": 13,
        "T1.02": 12,
        "T1.03": 23,
        "T1.04": 16,
        "T1.05": 14,
        "T1.06": 12,
        "T1.07": 12,
        "T1.08": 12,
        "T1.09": 13,
        "T1.10": 13,
        "T1.11": 12,
        "T1.12": 22,
    }
    for sheet, expected_count in expected.items():
        assert counts.get(sheet, 0) == expected_count, (
            f"sheet {sheet}: expected {expected_count} WN, got {counts.get(sheet, 0)}"
        )


def test_wn_extended_total_is_335(marriott_takeoff) -> None:
    summary = marriott_takeoff.summary.get("wireless_node_outlet") or {}
    assert summary.get("extended_count") == 335


def test_wn_rejected_on_legend_and_detail_pages(marriott_takeoff) -> None:
    counts = _wn_count_by_sheet(marriott_takeoff)
    assert counts.get("T0.01", 0) == 0  # legend
    assert counts.get("T9.02", 0) == 0  # detail
    assert counts.get("T1.00", 0) == 0  # service level NOT IN SCOPE
    assert counts.get("T1.13", 0) == 0  # roof — no WN in PDF


def test_sheet_multipliers(marriott_takeoff) -> None:
    by_num = {s.sheet_number: s for s in marriott_takeoff.sheets}
    assert by_num["T1.06"].multiplier == 9
    assert by_num["T1.09"].multiplier == 2
    assert by_num["T1.10"].multiplier == 5
    assert by_num["T1.01"].multiplier == 1


def test_at_least_one_zone_warning_or_open_question(marriott_takeoff) -> None:
    blob = "\n".join(marriott_takeoff.warnings + marriott_takeoff.open_questions)
    assert "T1.06" in blob or "T1.10" in blob
    assert "ambiguous_homerun_zone" in blob or "T1.01" in blob or "missing_homerun_zone" in blob


def test_summary_rejection_buckets(marriott_takeoff) -> None:
    wn = marriott_takeoff.summary.get("wireless_node_outlet") or {}
    assert wn.get("excluded_not_in_scope_count", 0) >= 1
    assert wn.get("rejected_non_plan_count", 0) >= 1


def test_t100_is_out_of_scope(marriott_takeoff) -> None:
    by_num = {s.sheet_number: s for s in marriott_takeoff.sheets}
    assert by_num["T1.00"].in_scope is False
