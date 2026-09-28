"""A room tag decides nothing. A measured distance can kill a design.

These pin the one thing reading CAD gives that reading a picture cannot: every
tag carries a coordinate, the header carries the units, and together they
answer how far the furthest drop runs -- which on a Cat6A job has a hard limit
and a five-figure consequence.
"""
from app.parsers.dwg_geometry import (
    CAT6A_PERMANENT_LINK_FT, Tag, cable_reach, tag_census,
)

IN = 1 / 12.0


def _plan(*pairs):
    return [Tag(t, x, y, "ROOM-TAG") for t, x, y in pairs]


def test_a_reachable_floor_reads_clear():
    """SP-6: the board room is the furthest room at ~100' routed from the IT
    closet, which derates to a 139' link against a 295' limit."""
    r = cable_reach(_plan(("IT", 0, 0), ("BOARD ROOM", 600, 600)), IN)
    assert r.verdict == "clear"
    assert 95 < r.routed_ft < 105
    assert r.headroom_ft > 100
    assert "BOARD ROOM" in r.note and "295" in r.note


def test_a_run_past_the_limit_is_called_over():
    """Past 90 m the drop does not underperform, it fails certification -- and
    the fix is a second closet nobody has priced."""
    r = cable_reach(_plan(("IT", 0, 0), ("FAR CORNER", 1800, 1800)), IN)
    assert r.verdict == "over"
    assert r.permanent_link_ft > CAT6A_PERMANENT_LINK_FT
    assert "second closet" in r.note


def test_a_near_miss_refuses_to_call_it():
    """A tag sits where its TEXT is placed, not at the furthest outlet, so a
    measured run is a FLOOR on the real one. When the headroom cannot absorb
    that gap the honest answer is 'measure it', not a pass."""
    r = cable_reach(_plan(("IT", 0, 0), ("CORNER", 1250, 1250)), IN)
    assert r.verdict == "tight"
    assert "site walk" in r.note


def test_cable_runs_orthogonally_not_diagonally():
    """Cable lies in tray. A plan distance is Manhattan, not the straight line,
    and using the straight line understates every run by up to 40%."""
    r = cable_reach(_plan(("IT", 0, 0), ("ROOM", 1200, 1200)), IN)
    assert r.routed_ft == 200            # (1200+1200) inches
    assert round(r.straight_ft) == 141   # the diagonal, which is NOT the cable


def test_a_drawing_that_cannot_answer_says_nothing():
    """No closet, no units, nothing to measure to -- each returns None. A
    finding that cannot be stood behind is worse than none, because it will be
    believed."""
    assert cable_reach(_plan(("PANTRY", 0, 0), ("BOARD ROOM", 10, 10)), IN) is None
    assert cable_reach(_plan(("IT", 0, 0), ("BOARD ROOM", 10, 10)), None) is None
    assert cable_reach(_plan(("IT", 0, 0)), IN) is None
    assert cable_reach([], IN) is None


def test_the_closet_is_found_by_the_names_a_drawing_uses():
    for name in ("IT", "MDF", "IDF 1", "TELECOM", "IT CLOSET", "SERVER ROOM"):
        r = cable_reach(_plan((name, 0, 0), ("ROOM", 600, 0)), IN)
        assert r is not None, name
        assert r.closet == name


def test_a_census_counts_what_the_plan_tags():
    """A schedule row saying 2 is checkable only against a count of the tags."""
    c = tag_census(_plan(("HUDDLE", 0, 0), ("HUDDLE", 50, 0), ("PANTRY", 90, 0)))
    assert c["HUDDLE"] == 2
    assert c["PANTRY"] == 1
