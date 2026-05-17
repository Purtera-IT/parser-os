"""Tests for spatial zone polygon assignment."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.takeoff.schemas import BBox
from app.takeoff.spatial_zones import (
    ZoneRegion,
    assign_home_run_spatial,
    build_zone_regions,
)
from app.takeoff.zones import HomeRunZone


class _FakeWord(tuple):
    """Mimic a PyMuPDF (x0, y0, x1, y1, text, ...) tuple."""

    def __new__(cls, x0, y0, x1, y1, text):
        return super().__new__(cls, (x0, y0, x1, y1, text, 0, 0, 0))


class _FakePage:
    """Stand-in for a fitz.Page exposing get_text('words') and .rect."""

    def __init__(self, words, width=1000.0, height=600.0):
        self._words = words
        self.rect = SimpleNamespace(width=width, height=height)

    def get_text(self, mode):
        if mode == "words":
            return self._words
        return ""


def _words_for_sentence(start_x: float, y: float, sentence: str) -> list:
    """Build a list of fake (x, y, ...) tuples for each whitespace-
    separated token in ``sentence``, starting at ``start_x``."""
    out = []
    x = start_x
    for tok in sentence.split():
        w = len(tok) * 6.0  # 6pt per character (rough)
        out.append(_FakeWord(x, y, x + w, y + 10.0, tok))
        x += w + 4.0
    return out


# ─── Region building ───


def test_build_zone_regions_single_zone_returns_full_page() -> None:
    page = _FakePage(words=[], width=1000.0, height=600.0)
    zones = [
        HomeRunZone(raw_text="z", target="IDF-1", applies_to_all_levels=True),
    ]
    regions = build_zone_regions(page=page, zones=zones)
    assert len(regions) == 1
    assert regions[0].region.x0 == 0.0
    assert regions[0].region.x1 == 1000.0


def test_build_zone_regions_two_zones_split_horizontally() -> None:
    """Two HOMERUN sentences on the same sheet split the page into a
    LEFT region (assigned to the leftmost sentence's zone) and a RIGHT
    region (the rightmost sentence's zone)."""
    left_zone = HomeRunZone(
        raw_text="HOMERUN ALL CABLES ON THIS LEVEL TO IDF-1, THIS LEVEL.",
        target="IDF-1", applies_to_all_levels=True,
    )
    right_zone = HomeRunZone(
        raw_text="HOMERUN ALL CABLES ON THIS LEVEL TO MDF ROOM, THIS LEVEL.",
        target="MDF ROOM", applies_to_all_levels=True,
    )
    words = []
    # LEFT zone sentence at x=200
    words += _words_for_sentence(200.0, 430.0, left_zone.raw_text)
    # RIGHT zone sentence at x=600
    words += _words_for_sentence(600.0, 430.0, right_zone.raw_text)
    page = _FakePage(words=words, width=1000.0, height=600.0)
    regions = build_zone_regions(page=page, zones=[left_zone, right_zone])
    assert len(regions) == 2
    # The two regions should partition the page horizontally.
    sorted_by_x = sorted(regions, key=lambda r: r.region.x0)
    left_region, right_region = sorted_by_x
    assert left_region.zone.target == "IDF-1"
    assert right_region.zone.target == "MDF ROOM"
    # No gap between regions.
    assert abs(left_region.region.x1 - right_region.region.x0) < 1.0
    # Cover the full page top-to-bottom.
    assert left_region.region.y0 == 0.0
    assert right_region.region.y1 == 600.0


def test_build_zone_regions_falls_back_to_equal_strips_when_sentences_not_found() -> None:
    """When the page has zones but their sentences can't be located,
    the builder splits the page into equal vertical strips in
    document order."""
    z1 = HomeRunZone(raw_text="nonsense 1", target="A", applies_to_all_levels=True)
    z2 = HomeRunZone(raw_text="nonsense 2", target="B", applies_to_all_levels=True)
    page = _FakePage(words=[], width=900.0, height=300.0)
    regions = build_zone_regions(page=page, zones=[z1, z2])
    assert len(regions) == 2
    # Equal strips.
    assert regions[0].region.x1 == pytest.approx(450.0)
    assert regions[1].region.x0 == pytest.approx(450.0)


# ─── Spatial assignment ───


def test_assign_home_run_spatial_picks_containing_region() -> None:
    z1 = HomeRunZone(raw_text="x", target="A", target_level="L1")
    z2 = HomeRunZone(raw_text="y", target="B", target_level="L2")
    regions = [
        ZoneRegion(zone=z1, region=BBox(x0=0, y0=0, x1=500, y1=600)),
        ZoneRegion(zone=z2, region=BBox(x0=500, y0=0, x1=1000, y1=600)),
    ]
    bbox_left = BBox(x0=100, y0=100, x1=110, y1=110)
    bbox_right = BBox(x0=700, y0=100, x1=710, y1=110)
    hr, lvl, _, flags = assign_home_run_spatial(regions=regions, device_bbox=bbox_left)
    assert hr == "A" and lvl == "L1" and flags == []
    hr, lvl, _, flags = assign_home_run_spatial(regions=regions, device_bbox=bbox_right)
    assert hr == "B" and lvl == "L2" and flags == []


def test_assign_home_run_spatial_returns_ambiguous_when_outside_all() -> None:
    z1 = HomeRunZone(raw_text="x", target="A")
    regions = [ZoneRegion(zone=z1, region=BBox(x0=0, y0=0, x1=500, y1=600))]
    bbox_outside = BBox(x0=900, y0=900, x1=910, y1=910)
    hr, _, _, flags = assign_home_run_spatial(regions=regions, device_bbox=bbox_outside)
    assert hr is None
    assert "ambiguous_homerun_zone" in flags


# ─── End-to-end: T1.01-style resolution via fusion ───


def test_fusion_uses_spatial_regions_when_provided() -> None:
    """Devices in different regions of a multi-zone page get DIFFERENT
    home-run targets — the spatial split unblocks them from
    ``ambiguous_homerun_zone``."""
    from app.takeoff.candidate_fusion import fuse_candidates_to_devices
    from app.takeoff.legend_extractor import load_default_legend_rules
    from app.takeoff.schemas import SheetRecord, SymbolCandidate

    rules = load_default_legend_rules()
    sheet = SheetRecord(
        page_index=4, page_type="floor_plan", in_scope=True,
        sheet_number="T1.01", floor_label="Lobby", multiplier=1,
        levels_represented=["Lobby"],
    )
    z_left = HomeRunZone(raw_text="x", target="IDF-1", applies_to_all_levels=True)
    z_right = HomeRunZone(raw_text="y", target="MDF ROOM", applies_to_all_levels=True)
    regions = [
        ZoneRegion(zone=z_left, region=BBox(x0=0, y0=0, x1=500, y1=600)),
        ZoneRegion(zone=z_right, region=BBox(x0=500, y0=0, x1=1000, y1=600)),
    ]
    candidates = [
        SymbolCandidate(
            id="c1", page_index=4, raw_symbol="WN",
            normalized_class="wireless_node_outlet",
            bbox=BBox(x0=100, y0=100, x1=110, y1=110),
            source_methods=["pdf_native_text"],
        ),
        SymbolCandidate(
            id="c2", page_index=4, raw_symbol="WN",
            normalized_class="wireless_node_outlet",
            bbox=BBox(x0=800, y0=100, x1=810, y1=110),
            source_methods=["pdf_native_text"],
        ),
    ]
    devices = fuse_candidates_to_devices(
        candidates=candidates, sheet=sheet, zones=[z_left, z_right],
        legend_rules=rules, zone_regions=regions,
    )
    assert len(devices) == 2
    by_pos = sorted(devices, key=lambda d: d.bbox.x0)
    assert by_pos[0].home_run_to == "IDF-1"
    assert by_pos[1].home_run_to == "MDF ROOM"
    # No ambiguity flags should fire.
    for d in devices:
        assert "ambiguous_homerun_zone" not in d.review_flags


# ─── Real PDF integration ───


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
def test_marriott_t101_ambiguity_resolved_by_spatial_split() -> None:
    """T1.01 has two zones (MDF + IDF-1) — spatial assignment should
    pick a specific target for every WN on the sheet."""
    from app.takeoff.pipeline import build_low_voltage_takeoff

    takeoff = build_low_voltage_takeoff(PDF_PATH)
    t101_devices = [
        d for d in takeoff.devices
        if d.sheet_number == "T1.01" and d.normalized_class == "wireless_node_outlet"
    ]
    assert t101_devices, "no T1.01 WN devices found"
    for d in t101_devices:
        assert "ambiguous_homerun_zone" not in d.review_flags, (
            f"T1.01 still has ambiguous WN at {d.bbox.center()} after spatial assignment"
        )
    targets = {d.home_run_to for d in t101_devices}
    assert targets == {"IDF-1", "MDF ROOM"}, (
        f"T1.01 should split across both targets; got {targets}"
    )


@pytest.mark.skipif(
    not PDF_PATH.exists() or not os.environ.get("RUN_SLOW_TESTS"),
    reason="Marriott source PDF + RUN_SLOW_TESTS=1 required",
)
def test_marriott_t106_kept_ambiguous_because_multi_level() -> None:
    """T1.06 has multi-LEVEL zones — each WN drawn on the sheet
    actually represents 9 WNs across 9 floors that may route to
    different IDFs. Spatial split would be WRONG, so the sheet should
    stay ambiguous."""
    from app.takeoff.pipeline import build_low_voltage_takeoff

    takeoff = build_low_voltage_takeoff(PDF_PATH)
    t106_devices = [
        d for d in takeoff.devices
        if d.sheet_number == "T1.06" and d.normalized_class == "wireless_node_outlet"
    ]
    assert t106_devices, "no T1.06 WN devices found"
    has_ambig = any("ambiguous_homerun_zone" in d.review_flags for d in t106_devices)
    assert has_ambig, "T1.06 should remain ambiguous (multi-level zones)"
