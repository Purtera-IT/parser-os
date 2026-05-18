"""Tests for :mod:`app.takeoff.legend_self_extractor`."""
from __future__ import annotations

from app.takeoff.legend_self_extractor import (
    _classify_row,
    _normalized_class_for,
    merge_with_defaults,
)
from app.takeoff.pdf_native import PdfWord
from app.takeoff.schemas import LegendRule


def _word(text: str, x0: float, y0: float = 100.0, w: float = 30.0) -> PdfWord:
    return PdfWord(text=text, x0=x0, y0=y0, x1=x0 + w, y1=y0 + 12, block_no=0, line_no=0, word_no=0)


def _row(*texts_and_xs) -> list[PdfWord]:
    """``_row(("WN", 100), ("1 PORT WIRELESS NODE OUTLET", 200))`` → list[PdfWord].

    For multi-word strings, each word is emitted at the same x but
    incrementally — sufficient for the row tests.
    """
    out: list[PdfWord] = []
    for text, x0 in texts_and_xs:
        for i, w in enumerate(text.split()):
            out.append(_word(w, x0=x0 + i * 60))
    return out


# ─── _normalized_class_for ──────────────────────────────────────────

def test_normalized_class_for_known_descriptions() -> None:
    assert _normalized_class_for("1 PORT WIRELESS NODE OUTLET")["normalized_class"] == "wireless_node_outlet"
    assert _normalized_class_for("4 PORT POINT OF SALE TERMINAL OUTLET")["normalized_class"] == "pos_terminal_outlet"
    assert _normalized_class_for("4 PORT POINT OF SALE PRINTER OUTLET")["normalized_class"] == "pos_printer_outlet"
    assert _normalized_class_for("CARD READER")["normalized_class"] == "access_control_card_reader"
    assert _normalized_class_for("DURESS ALARM PUSH BUTTON")["normalized_class"] == "duress_alarm_push_button"
    assert _normalized_class_for("HOUSE PHONE OUTLET")["normalized_class"] == "house_phone_outlet"
    assert _normalized_class_for("MATV OUTLET")["normalized_class"] == "matv_outlet"


def test_normalized_class_for_unknown_returns_none() -> None:
    assert _normalized_class_for("RANDOM PROSE TEXT") is None


# ─── _classify_row ──────────────────────────────────────────────────

def test_classify_row_requires_anchor_phrase() -> None:
    """Rows whose description doesn't contain an OUTLET / READER / etc.
    anchor must be rejected even if a code-shaped token appears."""
    row = _row(("WN", 100), ("RANDOM TEXT WITHOUT ANCHOR PHRASE HERE PLEASE", 200))
    assert _classify_row(row) is None


def test_classify_row_picks_up_wireless_node() -> None:
    row = _row(("WN", 100), ("1 PORT WIRELESS NODE OUTLET", 200))
    result = _classify_row(row)
    assert result is not None
    assert result.symbol_code == "WN"
    assert "WIRELESS NODE" in result.description.upper()


def test_classify_row_rejects_cable_spec_codes() -> None:
    """CMP / CAT6 / AWG are cable specs, not symbol codes — even if the
    row's description happens to mention an anchor word like 'ALARM'."""
    row = _row(("CMP", 100), ("DURESS ALARM PUSH BUTTON LOCATION", 200))
    result = _classify_row(row)
    assert result is None or result.symbol_code != "CMP"


# ─── merge_with_defaults ────────────────────────────────────────────

def test_merge_keeps_yaml_only_rules() -> None:
    defaults = [
        LegendRule(raw_symbol="WN", normalized_class="wireless_node_outlet", system="x"),
        LegendRule(raw_symbol="POS-T", normalized_class="pos_terminal_outlet", system="x"),
    ]
    merged, info = merge_with_defaults(extracted=[], defaults=defaults)
    codes = {r.raw_symbol for r in merged}
    assert codes == {"WN", "POS-T"}


def test_merge_bumps_confidence_for_verified_symbols() -> None:
    defaults = [LegendRule(raw_symbol="WN", normalized_class="wireless_node_outlet", system="x", confidence=0.90)]
    extracted = [LegendRule(raw_symbol="WN", normalized_class="wireless_node_outlet", system="x", confidence=0.85)]
    merged, info = merge_with_defaults(extracted=extracted, defaults=defaults)
    wn = next(r for r in merged if r.raw_symbol == "WN")
    assert wn.confidence >= 0.97
    assert any("legend_verified_against_pdf: WN" in m for m in info)


def test_merge_drops_unknown_extracted_symbols_by_default() -> None:
    """Symbols extracted from the PDF that aren't in defaults are NOT added
    to the merged set unless accept_new_symbols=True."""
    defaults = [LegendRule(raw_symbol="WN", normalized_class="wireless_node_outlet", system="x")]
    extracted = [LegendRule(raw_symbol="ZIBGEE", normalized_class="access_control_card_reader", system="x")]
    merged, info = merge_with_defaults(extracted=extracted, defaults=defaults)
    codes = {r.raw_symbol for r in merged}
    assert "ZIBGEE" not in codes  # filtered out by default
    assert "WN" in codes
    assert any("ZIBGEE" in m and "not_in_defaults" in m for m in info)


def test_merge_accepts_new_symbols_when_opted_in() -> None:
    defaults = [LegendRule(raw_symbol="WN", normalized_class="wireless_node_outlet", system="x")]
    extracted = [LegendRule(raw_symbol="LV1", normalized_class="cctv_camera", system="x")]
    merged, info = merge_with_defaults(extracted=extracted, defaults=defaults, accept_new_symbols=True)
    codes = {r.raw_symbol for r in merged}
    assert "LV1" in codes
