"""Tests for :mod:`app.takeoff.symbol_footprint`.

Universal hardening: the legend page can be parsed for per-symbol
inflation ratios that expand a text bbox into the full symbol shape.
These tests verify the math + the SYMBOL-column resolution rule, with
no dependency on PyMuPDF (the heavy ``build_symbol_footprints`` path
is exercised by the page-5 demo, not by unit tests).
"""
from __future__ import annotations

from app.takeoff.symbol_footprint import (
    NO_OP_FOOTPRINT,
    SymbolFootprint,
    _resolve_symbol_col,
    inflate_bbox,
)


# ────────────────────────── _resolve_symbol_col ─────────────────────


def test_resolve_symbol_col_finds_explicit_header() -> None:
    headers = [
        {"text": "DESCRIPTION"},
        {"text": "SYMBOL"},
        {"text": "CABLE COUNT"},
    ]
    assert _resolve_symbol_col(headers) == 1


def test_resolve_symbol_col_handles_lowercase() -> None:
    headers = [{"text": "symbol"}]
    assert _resolve_symbol_col(headers) == 0


def test_resolve_symbol_col_defaults_to_zero() -> None:
    """When no header says 'SYMBOL', column 0 is the universal fallback
    (99% of legend tables put the symbol in column 0)."""
    headers = [{"text": "WHATEVER"}, {"text": "DESCRIPTION"}]
    assert _resolve_symbol_col(headers) == 0


def test_resolve_symbol_col_returns_none_for_empty() -> None:
    assert _resolve_symbol_col(None) is None
    assert _resolve_symbol_col([]) is None


# ────────────────────────── inflate_bbox ────────────────────────────


def test_no_op_footprint_returns_original_bbox() -> None:
    """A no-op footprint reproduces the text bbox exactly (round-trip
    through the inflation math)."""
    x0, y0, x1, y1 = inflate_bbox(
        text_x0=10.0, text_y0=20.0, text_x1=30.0, text_y1=40.0,
        footprint=NO_OP_FOOTPRINT,
    )
    assert (x0, y0, x1, y1) == (10.0, 20.0, 30.0, 40.0)


def test_inflate_extends_below_for_wn_like_footprint() -> None:
    """A footprint with bot=+1.93 (WN antenna pattern) extends the bbox
    ~2x text-height downward while keeping the top + sides aligned."""
    fp = SymbolFootprint(
        symbol_code="WN", left=-0.5, right=0.5, top=-0.5, bot=1.93,
    )
    text_w, text_h = 16.0, 10.0
    x0, y0, x1, y1 = inflate_bbox(
        text_x0=100.0, text_y0=200.0,
        text_x1=100.0 + text_w, text_y1=200.0 + text_h,
        footprint=fp,
    )
    # Top edge + sides unchanged.
    assert x0 == 100.0
    assert y0 == 200.0
    assert x1 == 100.0 + text_w
    # Bottom extended by ~1.93 * text_h below the center.
    # center_y = 205, bot_y = 205 + 1.93 * 10 = 224.3
    assert abs(y1 - 224.3) < 1e-6


def test_inflate_handles_zero_size_text_bbox_gracefully() -> None:
    """Defensive: if upstream hands us a degenerate text bbox, inflate
    must still produce finite numbers (rather than dividing by zero)."""
    fp = SymbolFootprint(symbol_code="X", left=-1.0, right=1.0, top=-1.0, bot=1.0)
    x0, y0, x1, y1 = inflate_bbox(
        text_x0=50.0, text_y0=50.0, text_x1=50.0, text_y1=50.0,
        footprint=fp,
    )
    # All values finite — no inf / nan.
    for v in (x0, y0, x1, y1):
        assert v == v  # NaN check
        assert v != float("inf") and v != float("-inf")


def test_no_op_footprint_constant_is_text_bbox() -> None:
    """Sanity: the NO_OP_FOOTPRINT constant is the identity transform."""
    assert NO_OP_FOOTPRINT.left == -0.5
    assert NO_OP_FOOTPRINT.right == 0.5
    assert NO_OP_FOOTPRINT.top == -0.5
    assert NO_OP_FOOTPRINT.bot == 0.5
