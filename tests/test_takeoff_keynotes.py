"""Tests for :mod:`app.takeoff.keynotes`."""
from __future__ import annotations

from app.takeoff.keynotes import (
    find_keynote_refs_near,
    parse_keynote_table,
    resolve_keynote,
)
from app.takeoff.pdf_native import PdfWord
from app.takeoff.schemas import BBox


def _word(text: str, x0: float, y0: float, w: float = 12.0, h: float = 10.0) -> PdfWord:
    return PdfWord(text=text, x0=x0, y0=y0, x1=x0 + w, y1=y0 + h, block_no=0, line_no=0, word_no=0)


# ─── parse_keynote_table ────────────────────────────────────────────

def test_parse_inline_keynotes() -> None:
    text = """
    KEYED NOTES
    1. INSTALL CAT6 CABLE FROM JACK TO PATCH PANEL.
    2. WIRELESS NODE - CEILING MOUNT, COORDINATE WITH WIFI VENDOR.
    3. PROVIDE 4" SQUARE 1 GANG BOX.
    SHEET NUMBER: T1.01
    """
    table = parse_keynote_table(page_index=4, page_text=text)
    assert table.notes["1"].startswith("INSTALL CAT6 CABLE")
    assert "WIRELESS NODE" in table.notes["2"]
    assert "1 GANG" in table.notes["3"]


def test_parse_column_split_keynotes() -> None:
    """Numbers in one column, descriptions in another (T1.01 Marriott style)."""
    text = """
    KEYED NOTES
    1
    2
    3
    INSTALL CAT6 CABLE FROM JACK TO PATCH PANEL.
    WIRELESS NODE - CEILING MOUNT, COORDINATE WITH WIFI VENDOR.
    PROVIDE 4" SQUARE 1 GANG BOX WITH MUD RING AND DEDICATED CONDUIT.
    SHEET NUMBER
    """
    table = parse_keynote_table(page_index=4, page_text=text)
    assert table.notes.get("1", "").startswith("INSTALL CAT6"), table.notes
    assert "WIRELESS NODE" in table.notes.get("2", ""), table.notes
    assert "GANG BOX" in table.notes.get("3", ""), table.notes


def test_block_end_header_stops_parsing() -> None:
    """Block-end headers must terminate the keynote section so later
    numbered lines (e.g. RFP clauses) don't pollute the table."""
    text = """
    KEYED NOTES
    1. SHOULD APPEAR
    SHEET NUMBER
    2. SHOULD NOT APPEAR
    """
    table = parse_keynote_table(page_index=0, page_text=text)
    assert table.notes.get("1") == "SHOULD APPEAR"
    assert "2" not in table.notes


def test_no_block_means_no_notes() -> None:
    text = "Just some random page text with 1. that looks like a note."
    table = parse_keynote_table(page_index=0, page_text=text)
    assert not table.notes
    assert not table.blocks


# ─── find_keynote_refs_near ─────────────────────────────────────────

def test_find_refs_picks_up_isolated_digits_near_bbox() -> None:
    bbox = BBox(x0=100, y0=100, x1=110, y1=110)
    words = [
        _word("4", 80, 100),    # keynote ref to the left — near
        _word("99", 500, 500),  # far away
        _word("WN", 115, 100),  # not a digit
    ]
    refs = find_keynote_refs_near(bbox=bbox, page_words=words, radius_pt=80)
    assert "4" in refs
    assert "99" not in refs
    assert "WN" not in refs


def test_find_refs_ignores_digits_inside_candidate_bbox() -> None:
    bbox = BBox(x0=100, y0=100, x1=200, y1=200)
    words = [_word("5", 150, 150)]  # inside the bbox
    refs = find_keynote_refs_near(bbox=bbox, page_words=words, radius_pt=200)
    assert "5" not in refs


# ─── resolve_keynote ───────────────────────────────────────────────

def test_resolve_picks_first_match_in_table() -> None:
    table = parse_keynote_table(page_index=0, page_text="""
    KEYED NOTES
    1. CAT6 CABLE
    2. WIRELESS NODE
    """)
    num, text = resolve_keynote(refs=["7", "2"], table=table)
    # 7 isn't in table; 2 is — should return the 2 entry.
    assert num == "2"
    assert "WIRELESS NODE" in text


def test_resolve_returns_ref_only_when_text_missing() -> None:
    table = parse_keynote_table(page_index=0, page_text="")
    num, text = resolve_keynote(refs=["7"], table=table)
    assert num == "7"
    assert text is None


def test_resolve_no_refs_returns_none_pair() -> None:
    table = parse_keynote_table(page_index=0, page_text="")
    assert resolve_keynote(refs=[], table=table) == (None, None)
