"""Tests for the connectivity extractor: line segments -> junction graph ->
symbol-to-symbol cable runs + lengths. The gap that produced 0 cable runs on
the real DD."""
from __future__ import annotations

import pytest

from app.core.schematic_connectivity import (
    Connection,
    Symbol,
    build_connections,
    extract_segments_vector,
)


def _syms():
    return [
        Symbol("A", (-8, -8, 8, 8), "CAMERA"),
        Symbol("B", (92, -8, 108, 8), "PANEL"),
        Symbol("C", (92, 72, 108, 88), "SENSOR"),
        Symbol("D", (300, 300, 316, 316), "SPEAKER"),  # isolated
    ]


def test_connections_and_lengths():
    segs = [(0, 0, 100, 0), (100, 0, 100, 80)]  # A-B len100, B-C len80
    got = {tuple(sorted((c.a, c.b))): c.length_units for c in build_connections(_syms(), segs)}
    assert abs(got[("A", "B")] - 100) < 1
    assert abs(got[("B", "C")] - 80) < 1
    assert abs(got[("A", "C")] - 180) < 1  # transitive shortest path


def test_isolated_symbol_has_no_connections():
    segs = [(0, 0, 100, 0), (100, 0, 100, 80)]
    conns = build_connections(_syms(), segs)
    assert not any("D" in (c.a, c.b) for c in conns)


def test_no_segments_no_connections():
    assert build_connections(_syms(), []) == []


def test_polyline_corner_is_one_run():
    """A bends through a corner (junction, not a symbol) to B -> single run whose
    length is the sum of the two legs."""
    syms = [Symbol("A", (-8, -8, 8, 8)), Symbol("B", (92, 72, 108, 88))]
    segs = [(0, 0, 100, 0), (100, 0, 100, 80)]  # corner at (100,0) is a junction
    conns = build_connections(syms, segs)
    assert len(conns) == 1
    assert abs(conns[0].length_units - 180) < 1


def test_vector_segment_extraction(tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.draw_line((20, 20), (200, 20))
    page.draw_line((200, 20), (200, 150))
    segs = extract_segments_vector(page)
    assert len(segs) >= 2
    # two symbols at the line ends should connect
    syms = [Symbol("X", (10, 10, 30, 30)), Symbol("Y", (190, 140, 210, 160))]
    conns = build_connections(syms, segs, snap_tol=4.0, attach_margin=10.0)
    assert len(conns) == 1
    assert conns[0].length_units > 250  # ~180+130 legs
