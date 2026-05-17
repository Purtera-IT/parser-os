"""Tests for the typical-plan expander.

The expander pulls panel titles + room codes off a typical-plan sheet,
counts the symbols inside each panel, then multiplies by per-floor room
counts to get a building-wide rollup.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.takeoff.schemas import BBox, SheetRecord, SymbolCandidate
from app.takeoff.typical_plan_expander import (
    TypicalPanel,
    TypicalPlanReport,
    build_expansion_summary,
    count_room_types_on_floor,
    parse_panel_titles_from_text,
)


# ─── Panel title parsing ───


def test_parse_panel_titles_numbered_form_marriott() -> None:
    text = (
        "1\n"
        "K1 - GUESTROOM PLAN\n"
        "2\n"
        "K2 - GUESTROOM PLAN\n"
        "3\n"
        "QQ1 - GUESTROOM PLAN\n"
        "4\n"
        "QQ2 - GUESTROOM PLAN\n"
    )
    panels = parse_panel_titles_from_text(text)
    assert panels == [(1, "K1"), (2, "K2"), (3, "QQ1"), (4, "QQ2")]


def test_parse_panel_titles_inline_fallback() -> None:
    text = "K1 - GUESTROOM PLAN\nK2 - GUESTROOM PLAN\n"
    panels = parse_panel_titles_from_text(text)
    assert panels == [(1, "K1"), (2, "K2")]


def test_parse_panel_titles_empty_text_returns_empty() -> None:
    assert parse_panel_titles_from_text("") == []
    assert parse_panel_titles_from_text("totally unrelated text") == []


# ─── Aggregation ───


def _stub_report(
    sheet_number: str,
    panels: dict[str, dict[str, int]],
) -> TypicalPlanReport:
    """Build a synthetic TypicalPlanReport without touching PyMuPDF."""
    typical_panels = [
        TypicalPanel(index=i + 1, room_type=room, device_counts=dict(counts))
        for i, (room, counts) in enumerate(panels.items())
    ]
    return TypicalPlanReport(sheet_number=sheet_number, page_index=99, panels=typical_panels)


def test_expansion_summary_multiplies_per_room_then_per_floor() -> None:
    report = _stub_report(
        "T4.00",
        {
            "K1": {"matv_outlet": 1, "house_phone_outlet": 1},
            "K2": {"matv_outlet": 1},
            "QQ1": {"matv_outlet": 1, "house_phone_outlet": 1},
            "QQ2": {"matv_outlet": 1},
        },
    )
    floor_counts = {
        "T1.06": {"K1": 4, "K2": 2, "QQ1": 7, "QQ2": 2},  # 15 rooms x9 floors
        "T1.09": {"K1": 3, "K2": 2, "QQ1": 6, "QQ2": 2},  # 13 rooms x2 floors
    }
    sheets = [
        SheetRecord(page_index=9, sheet_number="T1.06", page_type="floor_plan", multiplier=9),
        SheetRecord(page_index=12, sheet_number="T1.09", page_type="floor_plan", multiplier=2),
    ]
    summary = build_expansion_summary(
        typical_reports=[report],
        floor_room_counts=floor_counts,
        sheet_records=sheets,
    )

    # 4 K1 + 2 K2 + 7 QQ1 + 2 QQ2 = 15 TVs/floor on T1.06; x9 multiplier = 135
    # 3 K1 + 2 K2 + 6 QQ1 + 2 QQ2 = 13 TVs/floor on T1.09; x2 multiplier = 26
    # 4 K1 + 7 QQ1 = 11 H/floor on T1.06 x9 = 99
    # 3 K1 + 6 QQ1 = 9 H/floor on T1.09 x2 = 18
    expanded = summary["expanded_device_totals"]
    assert expanded["matv_outlet"] == 135 + 26
    assert expanded["house_phone_outlet"] == 99 + 18

    per_floor = summary["per_floor_expansion"]
    assert per_floor["T1.06"]["matv_outlet"] == 135
    assert per_floor["T1.09"]["matv_outlet"] == 26
    assert summary["unresolved_floors"] == []


def test_expansion_summary_emits_unresolved_floors_when_no_rooms() -> None:
    report = _stub_report("T4.00", {"K1": {"matv_outlet": 1}})
    floor_counts = {
        "T1.10": {"K1": 0, "K2": 0, "QQ1": 0, "QQ2": 0},
    }
    sheets = [SheetRecord(page_index=13, sheet_number="T1.10", page_type="floor_plan", multiplier=5)]
    summary = build_expansion_summary(
        typical_reports=[report],
        floor_room_counts=floor_counts,
        sheet_records=sheets,
    )
    assert summary["unresolved_floors"] == ["T1.10"]
    assert summary["expanded_device_totals"] == {}


def test_assign_candidates_to_panels_partitions_by_bbox() -> None:
    panels = [
        TypicalPanel(
            index=1,
            room_type="K1",
            bbox=BBox(x0=0, y0=0, x1=500, y1=800, coord_space="pdf_pt"),
        ),
        TypicalPanel(
            index=2,
            room_type="K2",
            bbox=BBox(x0=500, y0=0, x1=1000, y1=800, coord_space="pdf_pt"),
        ),
    ]
    candidates = [
        SymbolCandidate(
            id="a", page_index=17, raw_symbol="TV", normalized_class="matv_outlet",
            bbox=BBox(x0=100, y0=100, x1=110, y1=110),
        ),
        SymbolCandidate(
            id="b", page_index=17, raw_symbol="TV", normalized_class="matv_outlet",
            bbox=BBox(x0=700, y0=200, x1=710, y1=210),
        ),
        SymbolCandidate(
            id="c", page_index=17, raw_symbol="WN", normalized_class="wireless_node_outlet",
            bbox=BBox(x0=100, y0=400, x1=110, y1=410),
        ),
    ]
    from app.takeoff.typical_plan_expander import assign_candidates_to_panels
    assign_candidates_to_panels(panels, candidates)
    assert panels[0].device_counts == {"matv_outlet": 1, "wireless_node_outlet": 1}
    assert panels[1].device_counts == {"matv_outlet": 1}


# ─── Real PDF integration (slow) ───


PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "real_data_cases"
    / "LOWVOLT_002_MARRIOTT_ATLANTA_T"
    / "artifacts"
    / "2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf"
)


@pytest.mark.skipif(
    not PDF_PATH.exists() or not os.environ.get("RUN_SLOW_TESTS"),
    reason="Marriott source PDF + RUN_SLOW_TESTS=1 required",
)
def test_marriott_typical_plan_expansion_populates_summary() -> None:
    """T4.00 should expand into per-floor totals once room counts are
    parsed.

    Acceptance: the summary's ``typical_plan_pages`` mentions T4.00,
    each typical_room_device_counts entry is a non-empty dict, and
    either ``expanded_device_totals`` is non-empty *or*
    ``unresolved_floors`` is populated. The point of the test is to
    catch a regression where the expander silently stops working.
    """
    from app.takeoff.pipeline import build_low_voltage_takeoff

    takeoff = build_low_voltage_takeoff(PDF_PATH)
    expansion = takeoff.summary.get("typical_plan_expansion") or {}
    assert expansion, "typical_plan_expansion missing from summary"
    sheets_seen = {p["sheet"] for p in expansion.get("typical_plan_pages", [])}
    assert "T4.00" in sheets_seen
    room_counts = expansion.get("typical_room_device_counts", {})
    assert room_counts, "typical_room_device_counts is empty"
    # The expansion either produced totals OR surfaced unresolved floors —
    # both are acceptable v0 outcomes per the spec.
    expanded = expansion.get("expanded_device_totals") or {}
    unresolved = expansion.get("unresolved_floors") or []
    assert expanded or unresolved, "neither expansion nor unresolved-floors emitted"


def test_markdown_renders_typical_plan_section() -> None:
    """The takeoff markdown gains a Typical-Plan Expansion section when
    the summary has the expansion block populated."""
    from app.takeoff.exports import takeoff_doc_to_markdown
    from app.takeoff.schemas import TakeoffDocument

    doc = TakeoffDocument(
        source_pdf="x.pdf",
        summary={
            "typical_plan_expansion": {
                "typical_plan_pages": [],
                "typical_room_device_counts": {"K1": {"matv_outlet": 1}},
                "floor_room_counts": {"T1.06": {"K1": 4}},
                "expanded_device_totals": {"matv_outlet": 36},
                "per_floor_expansion": {"T1.06": {"matv_outlet": 36}},
                "unresolved_floors": ["T1.10"],
            }
        },
    )
    md = takeoff_doc_to_markdown(doc)
    assert "## Typical-Plan Expansion" in md
    assert "matv_outlet" in md
    assert "T1.06" in md
    assert "T1.10" in md  # unresolved-floors callout


def test_atoms_include_typical_plan_expansion_quantity() -> None:
    """takeoff_to_atoms yields one rollup-quantity + assumption atom
    when a typical-plan expansion is present in the summary."""
    from app.takeoff.exports import takeoff_to_atoms
    from app.takeoff.schemas import TakeoffDocument

    doc = TakeoffDocument(
        source_pdf="x.pdf",
        summary={
            "typical_plan_expansion": {
                "typical_plan_pages": [],
                "typical_room_device_counts": {"K1": {"matv_outlet": 1}},
                "floor_room_counts": {"T1.06": {"K1": 4}},
                "expanded_device_totals": {"matv_outlet": 36},
                "per_floor_expansion": {"T1.06": {"matv_outlet": 36}},
                "unresolved_floors": [],
            }
        },
    )
    atoms = list(
        takeoff_to_atoms(
            takeoff=doc,
            project_id="P",
            artifact_id="A",
            filename="x.pdf",
            parser_version="v0",
        )
    )
    raws = [a.raw_text for a in atoms]
    assert any("matv_outlet: 36 drops on T1.06" in r for r in raws)
    assert any("typical-plan expansion" in r and "all guest-room" in r for r in raws)
    assert any("Typical-plan expansion is a heuristic" in r for r in raws)
