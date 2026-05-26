"""Universal schematic parser tests — Marriott Atlanta DD shape.

Pins the new modules added in the universal-parser sweep:

* ``discipline_router`` — sheet-prefix → discipline mapping
* ``region_proposals`` — vector-stroke clustering
* ``keyed_notes_graph`` — keyed-note box + callout linking
* ``typed_schedule_extractors`` — door/panel/equipment/fixture/cable/room
* ``room_detector`` — vector polygon room boundaries
* ``cable_run_tracer`` — multi-segment cable path tracing
* ``revision_diff`` — DD revision comparison

Each test uses lightweight stubs so the suite stays fast and doesn't
require fitz or a vision endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


# ── Stubs ─────────────────────────────────────────────────────────


@dataclass
class _Block:
    text: str
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 20.0)


@dataclass
class _Det:
    detection_id: str
    page_index: int
    bbox_pdf: tuple[float, float, float, float]
    matched_label_text: str = ""
    matched_symbol_text: str = ""


class _Rect:
    def __init__(self, w: float, h: float) -> None:
        self.width = w
        self.height = h
        self.x0 = 0.0
        self.y0 = 0.0
        self.x1 = w
        self.y1 = h


class _Page:
    def __init__(self, *, drawings: Sequence[dict] | None = None) -> None:
        self.rect = _Rect(2160.0, 3240.0)
        self._drawings = list(drawings or [])

    def get_drawings(self) -> list[dict]:
        return self._drawings


# ── discipline_router ─────────────────────────────────────────────


def test_discipline_router_telecom_prefix() -> None:
    from orbitbrief_page_os.segmentation.schematic.discipline_router import (
        parse_discipline_from_sheet_number,
    )
    code, label = parse_discipline_from_sheet_number("T0.01")
    assert code == "T"
    assert label == "telecom"


def test_discipline_router_fire_alarm_two_char_prefix() -> None:
    from orbitbrief_page_os.segmentation.schematic.discipline_router import (
        parse_discipline_from_sheet_number,
    )
    code, label = parse_discipline_from_sheet_number("FA-101")
    assert code == "FA"
    assert label == "fire_alarm"


def test_discipline_router_assigns_all_pages() -> None:
    from orbitbrief_page_os.segmentation.schematic.discipline_router import (
        assign_disciplines,
    )
    inputs = {0: "T0.01", 1: "E0.01", 2: "M-101", 3: None}
    out = assign_disciplines(inputs)
    assert out[0].discipline_code == "T"
    assert out[1].discipline_code == "E"
    assert out[2].discipline_code == "M"
    assert out[3].discipline_code is None
    assert out[3].confidence == 0.0


def test_discipline_router_legend_scope_cross_discipline() -> None:
    from orbitbrief_page_os.segmentation.schematic.discipline_router import (
        legend_scope_for_discipline,
    )
    # Same discipline: applies
    assert legend_scope_for_discipline(
        legend_page_discipline="telecom", target_page_discipline="telecom"
    ) == "same_discipline:telecom"
    # Cross-discipline: does NOT apply
    assert legend_scope_for_discipline(
        legend_page_discipline="telecom", target_page_discipline="electrical"
    ) is None


# ── region_proposals ──────────────────────────────────────────────


def test_region_proposals_empty_page() -> None:
    from orbitbrief_page_os.segmentation.schematic.region_proposals import (
        propose_regions,
    )
    page = _Page(drawings=[])
    assert propose_regions(page=page, page_index=0) == []


def test_region_proposals_filters_oversized() -> None:
    from orbitbrief_page_os.segmentation.schematic.region_proposals import (
        propose_regions,
    )
    # One giant title-block-sized rect (should be rejected as too big)
    # + two small symbol-sized rects close together (should merge into
    # one cluster).
    class FakeRect:
        def __init__(self, x0, y0, x1, y1) -> None:
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1

    drawings = [
        {"rect": FakeRect(0, 0, 2000, 1500), "items": []},          # giant: rejected
        {"rect": FakeRect(100, 100, 120, 120), "items": []},        # small (20x20 pt)
        {"rect": FakeRect(115, 105, 130, 120), "items": []},        # close to above (15x15 pt)
    ]
    page = _Page(drawings=drawings)
    proposals = propose_regions(page=page, page_index=0)
    # Only the two small ones should survive, clustered into one region
    assert len(proposals) == 1
    assert proposals[0].primitive_count == 2


# ── keyed_notes_graph ────────────────────────────────────────────


def test_keyed_notes_box_detection_and_parsing() -> None:
    from orbitbrief_page_os.segmentation.schematic.keyed_notes_graph import (
        locate_keyed_notes_boxes,
    )
    blocks = [
        _Block(text="KEYED NOTES", bbox=(100.0, 50.0, 250.0, 70.0)),
        _Block(text="1. PROVIDE 1-1/4\" CONDUIT FROM MDF TO IDF.",
               bbox=(100.0, 75.0, 600.0, 95.0)),
        _Block(text="2. COORDINATE MOUNTING HEIGHT WITH ARCH.",
               bbox=(100.0, 100.0, 600.0, 120.0)),
        _Block(text="3. CARD READER COMPATIBLE W/ EXISTING.",
               bbox=(100.0, 125.0, 600.0, 145.0)),
    ]
    boxes = locate_keyed_notes_boxes(page_index=0, blocks=blocks)
    assert len(boxes) == 1
    box = boxes[0]
    assert box.header_text == "KEYED NOTES"
    assert len(box.notes) == 3
    nums = sorted(n.note_number for n in box.notes)
    assert nums == [1, 2, 3]


def test_keyed_notes_callout_linking() -> None:
    from orbitbrief_page_os.segmentation.schematic.keyed_notes_graph import (
        KeyedNote,
        KeyedNotesBox,
        link_callouts_to_notes,
    )
    notes = (
        KeyedNote(page_index=0, note_number=1, note_text="conduit"),
        KeyedNote(page_index=0, note_number=2, note_text="height"),
    )
    box = KeyedNotesBox(
        page_index=0,
        header_text="KEYED NOTES",
        bbox_pdf=(100.0, 50.0, 700.0, 150.0),
        notes=notes,
    )
    # Two callout markers OUTSIDE the notes box
    callout_blocks = [
        _Block(text="1", bbox=(1000.0, 500.0, 1015.0, 515.0)),
        _Block(text="(2)", bbox=(1200.0, 800.0, 1220.0, 815.0)),
        _Block(text="99", bbox=(1300.0, 900.0, 1320.0, 915.0)),  # no matching note
    ]
    matches = link_callouts_to_notes(
        page_index=0, blocks=callout_blocks, notes_box=box
    )
    matched_nums = sorted(m.note_number for m in matches)
    assert matched_nums == [1, 2]


# ── typed_schedule_extractors ────────────────────────────────────


def test_typed_schedule_door_kind_detection() -> None:
    from orbitbrief_page_os.segmentation.schematic.typed_schedule_extractors import (
        detect_schedule_kind,
    )
    assert detect_schedule_kind("DOOR SCHEDULE") == "door_schedule"
    assert detect_schedule_kind("Panel Schedule") == "panel_schedule"
    assert detect_schedule_kind("EQUIPMENT LEGEND") == "equipment_schedule"
    assert detect_schedule_kind("Lighting Schedule") == "fixture_schedule"
    assert detect_schedule_kind("RANDOM TEXT") is None


def test_typed_schedule_column_header_mapping() -> None:
    from orbitbrief_page_os.segmentation.schematic.typed_schedule_extractors import (
        map_header_to_columns,
    )
    headers = [
        _Block(text="DOOR NO.", bbox=(0.0, 0.0, 50.0, 20.0)),
        _Block(text="TYPE",     bbox=(60.0, 0.0, 100.0, 20.0)),
        _Block(text="SIZE",     bbox=(110.0, 0.0, 160.0, 20.0)),
        _Block(text="HARDWARE", bbox=(170.0, 0.0, 230.0, 20.0)),
    ]
    column_map = map_header_to_columns(
        schedule_kind="door_schedule", header_blocks=headers
    )
    assert column_map[0] == "door_number"
    assert column_map[1] == "door_type"
    assert column_map[2] == "size"
    assert column_map[3] == "hardware"


# ── room_detector ─────────────────────────────────────────────────


def test_room_detector_empty_page() -> None:
    from orbitbrief_page_os.segmentation.schematic.room_detector import (
        detect_room_polygons,
    )
    page = _Page(drawings=[])
    assert detect_room_polygons(page=page, page_index=0) == []


def test_room_detector_point_in_polygon_helper() -> None:
    from orbitbrief_page_os.segmentation.schematic.room_detector import (
        _point_in_polygon,
    )
    # Square 0,0 -> 100,100
    poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    assert _point_in_polygon((50.0, 50.0), poly) is True
    assert _point_in_polygon((150.0, 50.0), poly) is False
    assert _point_in_polygon((-10.0, 50.0), poly) is False


def test_room_detector_polygon_area() -> None:
    from orbitbrief_page_os.segmentation.schematic.room_detector import (
        _polygon_area,
    )
    # 100×100 square = 10_000 sqpt
    poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    assert _polygon_area(poly) == 10_000.0


# ── cable_run_tracer ─────────────────────────────────────────────


def test_cable_run_tracer_empty_page() -> None:
    from orbitbrief_page_os.segmentation.schematic.cable_run_tracer import (
        trace_cable_runs,
    )
    page = _Page(drawings=[])
    assert trace_cable_runs(page=page, page_index=0, detections=[]) == []


def test_cable_run_tracer_shortest_path_basic() -> None:
    from orbitbrief_page_os.segmentation.schematic.cable_run_tracer import (
        _shortest_path,
        _build_adjacency_indexed,
    )
    # 0 — 1 — 2 with weights 10, 20 (path 0→2 = 30)
    edges = [(0, 1, 10.0), (1, 2, 20.0)]
    adj = _build_adjacency_indexed(3, edges)
    result = _shortest_path(adj, 0, 2)
    assert result is not None
    length, path = result
    assert length == 30.0
    assert path == [0, 1, 2]


def test_cable_run_tracer_disconnected_returns_none() -> None:
    from orbitbrief_page_os.segmentation.schematic.cable_run_tracer import (
        _shortest_path,
        _build_adjacency_indexed,
    )
    edges = [(0, 1, 10.0)]
    adj = _build_adjacency_indexed(3, edges)
    result = _shortest_path(adj, 0, 2)
    assert result is None


# ── revision_diff ─────────────────────────────────────────────────


def test_revision_diff_added_and_removed() -> None:
    from orbitbrief_page_os.segmentation.schematic.revision_diff import (
        diff_detections,
    )
    a = [
        _Det("d1", 0, (100.0, 100.0, 110.0, 110.0), "CARD READER"),
        _Det("d2", 0, (200.0, 200.0, 210.0, 210.0), "CAMERA"),
    ]
    b = [
        _Det("d1b", 0, (100.0, 100.0, 110.0, 110.0), "CARD READER"),
        _Det("d3", 0, (300.0, 300.0, 310.0, 310.0), "PANIC BUTTON"),
    ]
    added, removed, moved, renamed = diff_detections(a, b)
    assert len(added) == 1
    assert added[0].matched_label_text == "PANIC BUTTON"
    assert len(removed) == 1
    assert removed[0].matched_label_text == "CAMERA"


def test_revision_diff_moved_detection() -> None:
    from orbitbrief_page_os.segmentation.schematic.revision_diff import (
        diff_detections,
    )
    a = [_Det("d1", 0, (100.0, 100.0, 110.0, 110.0), "CARD READER")]
    # Same label, bbox moved > tolerance (24pt)
    b = [_Det("d1b", 0, (200.0, 200.0, 210.0, 210.0), "CARD READER")]
    added, removed, moved, renamed = diff_detections(a, b)
    assert len(moved) == 1
    assert len(added) == 0
    assert len(removed) == 0


def test_revision_diff_full_summary() -> None:
    from orbitbrief_page_os.segmentation.schematic.revision_diff import (
        diff_revisions,
    )
    diff = diff_revisions(
        revision_a_id="DD-100",
        revision_b_id="DD-50",
        detections_a=[_Det("d1", 0, (100.0, 100.0, 110.0, 110.0), "CARD READER")],
        detections_b=[
            _Det("d1b", 0, (100.0, 100.0, 110.0, 110.0), "CARD READER"),
            _Det("d2", 0, (200.0, 200.0, 210.0, 210.0), "PANIC BUTTON"),
        ],
    )
    assert diff.summary["added"] == 1
    assert diff.summary["removed"] == 0
    assert diff.revision_a_id == "DD-100"
    assert diff.revision_b_id == "DD-50"
