"""Tests for :mod:`app.takeoff.parser_intelligence`.

These cover the universal hardening: symbol-code normalization across
firm-style variations, closet-pattern regex that rejects English words
that happen to share a prefix (TRAY / TRONICS / TROUGHS), cross-reference
keyword matching across device classes, and confidence scoring shape.

Every test is hermetic — no PDF rendering, just dict-driven inputs that
exercise the universal logic without depending on Marriott specifics.
"""
from __future__ import annotations

from app.takeoff.parser_intelligence import (
    _confidence_breakdown,
    _iter_legend_rows,
    _iter_schedule_rows,
    _iter_spec_paragraphs,
    _normalize_symbol_code,
    _project_zones,
    _schedule_cross_refs,
    _spec_paragraph_refs,
    build_detection_plan,
)


# ─────────────────────── _normalize_symbol_code ─────────────────────


def test_normalize_accepts_simple_codes() -> None:
    assert _normalize_symbol_code("WN") == "WN"
    assert _normalize_symbol_code("CR") == "CR"
    assert _normalize_symbol_code("TV") == "TV"
    assert _normalize_symbol_code("H") == "H"


def test_normalize_accepts_hyphenated_codes() -> None:
    assert _normalize_symbol_code("POS-T") == "POS-T"
    assert _normalize_symbol_code("POS-P") == "POS-P"
    assert _normalize_symbol_code("FACP-2") == "FACP-2"


def test_normalize_strips_trailing_port_placeholder() -> None:
    assert _normalize_symbol_code("POS-T #") == "POS-T"
    assert _normalize_symbol_code("A #") == "A"
    assert _normalize_symbol_code("TV #") == "TV"
    assert _normalize_symbol_code("F 2") == "F"


def test_normalize_rejects_paragraph_text() -> None:
    """Multi-word descriptions captured in the SYMBOL cell aren't codes."""
    assert _normalize_symbol_code("1 PORT WIRELESS NODE OUTLET") is None
    assert _normalize_symbol_code("SINGLE BUTTON EMERGENCY PHONE SYSTEM STATION") is None
    assert _normalize_symbol_code("ELEVATOR REMOTE ACCESS READER") is None


def test_normalize_rejects_pure_digits_and_glyphs() -> None:
    assert _normalize_symbol_code("#") is None
    assert _normalize_symbol_code("1") is None
    assert _normalize_symbol_code("180°") is None  # no leading ASCII letter
    assert _normalize_symbol_code("") is None
    assert _normalize_symbol_code(None) is None  # type: ignore[arg-type]


def test_normalize_lowercase_input_is_uppercased() -> None:
    assert _normalize_symbol_code("wn") == "WN"
    assert _normalize_symbol_code("pos-t") == "POS-T"


def test_normalize_rejects_too_long_codes() -> None:
    """A code-shaped token longer than 6 chars is rejected.
    Borderline 6-char codes (FACP-2, ZIGBEE) pass the shape gate —
    the operator-curated whitelist downstream can narrow further."""
    assert _normalize_symbol_code("ABCDEFG") is None  # 7 chars
    assert _normalize_symbol_code("TOOLONGCODE") is None
    # 6-char codes are deliberately allowed (FACP-2 is one of them).
    assert _normalize_symbol_code("ZIGBEE") == "ZIGBEE"
    assert _normalize_symbol_code("FACP-2") == "FACP-2"


# ────────────────────────── _iter_legend_rows ───────────────────────


def _legend_doc_with_rows(symbols_and_descs: list[tuple[str, str]]) -> dict:
    """Build a minimal legend doc matching the readable shape."""
    return {
        "legend": {
            "tables": [
                {
                    "sections": [
                        {
                            "title": "TEST LEGEND",
                            "column_headers": [
                                {"text": "SYMBOL"},
                                {"text": "DESCRIPTION"},
                            ],
                            "rows": [
                                {
                                    "cells_by_header": {
                                        "SYMBOL": sym,
                                        "DESCRIPTION": desc,
                                    }
                                }
                                for sym, desc in symbols_and_descs
                            ],
                        }
                    ]
                }
            ]
        }
    }


def test_iter_legend_rows_filters_noise() -> None:
    ref = _legend_doc_with_rows([
        ("WN", "1 PORT WIRELESS NODE OUTLET"),
        ("POS-T #", "POINT OF SALE TERMINAL OUTLET"),
        ("SINGLE BUTTON EMERGENCY PHONE", "ER PHONE STATION"),
        ("#", "FOO BAR PLACEHOLDER"),
        ("CR", "CARD READER"),
    ])
    rows = _iter_legend_rows(ref)
    codes = {r["symbol"] for r in rows}
    assert "WN" in codes
    assert "POS-T" in codes
    assert "CR" in codes
    assert "SINGLE" not in codes  # multi-word → rejected
    assert "#" not in codes


def test_iter_legend_rows_empty_reference() -> None:
    assert _iter_legend_rows({}) == []
    assert _iter_legend_rows({"legend": None}) == []


# ────────────────────────── _project_zones ──────────────────────────


def test_project_zones_accepts_real_closet_ids() -> None:
    ref = _legend_doc_with_rows([("WN", "patch to IDF-2")])
    extract = {"zone_notes": [{"target": "MDF ROOM"}, {"target": "IDF-5"}]}
    zones = _project_zones(ref, extract)
    assert "MDF ROOM" in zones
    assert "IDF-5" in zones
    assert "IDF-2" in zones


def test_project_zones_rejects_english_word_prefixes() -> None:
    """Words like TRAY / TRONICS / TROUGHS / TROOP / TRS happen to start
    with 'TR' but aren't TR-N closet refs. Same for ER (ERROR), BDF, IDF
    inside other words."""
    ref = _legend_doc_with_rows([
        ("X", "support on cable TRAY routed through TRONICS room"),
        ("Y", "see ERROR codes in TROUGHS"),
        ("Z", "INTRADEPARTMENTAL coordination required"),
    ])
    extract = {"zone_notes": []}
    zones = _project_zones(ref, extract)
    assert zones == [], zones


def test_project_zones_accepts_TR_with_suffix() -> None:
    """TR-3 / ER-A / BDF-5 should pass — they have explicit ID suffixes."""
    ref = _legend_doc_with_rows([
        ("X", "homerun to TR-3 on level 5"),
        ("Y", "see ER-A spec"),
    ])
    extract = {"zone_notes": []}
    zones = _project_zones(ref, extract)
    assert "TR-3" in zones
    assert "ER-A" in zones


# ────────────────────────── _schedule_cross_refs ────────────────────


def _schedule_doc(rows: list[tuple[str, dict[str, str]]]) -> dict:
    """Build a schedule doc with named sections."""
    return {
        "schedule": {
            "tables": [
                {
                    "sections": [
                        {
                            "title": section_title,
                            "column_headers": [],
                            "rows": [{"cells_by_header": cells}],
                        }
                        for section_title, cells in rows
                    ]
                }
            ]
        }
    }


def test_schedule_cross_refs_matches_wireless_for_wn_class() -> None:
    sched = _schedule_doc([
        ("COPPER COMPONENTS", {"DESCRIPTION": "CAT6 PLENUM CABLE", "PART": "X-1"}),
        ("FIBER", {"DESCRIPTION": "SINGLE-MODE FIBER", "PART": "F-1"}),
    ])
    schedule_rows = _iter_schedule_rows(sched)
    hits = _schedule_cross_refs(
        device_class="wireless_node_outlet",
        schedule_rows=schedule_rows,
    )
    descriptions = [h["row"]["DESCRIPTION"] for h in hits]
    assert "CAT6 PLENUM CABLE" in descriptions


def test_schedule_cross_refs_no_match_for_unknown_class() -> None:
    sched = _schedule_doc([
        ("ANY", {"DESCRIPTION": "RANDOM SPEC", "PART": "X"}),
    ])
    schedule_rows = _iter_schedule_rows(sched)
    hits = _schedule_cross_refs(
        device_class="unknown_class_with_no_keywords",
        schedule_rows=schedule_rows,
    )
    assert hits == []


def test_schedule_cross_refs_handles_empty_reference() -> None:
    hits = _schedule_cross_refs(
        device_class="wireless_node_outlet",
        schedule_rows=[],
    )
    assert hits == []


# ────────────────────────── _spec_paragraph_refs ────────────────────


def _spec_doc(sections: list[tuple[str, list[str]]]) -> dict:
    return {
        "spec": {
            "sections": [
                {"heading": heading, "paragraphs": paras, "bullets": []}
                for heading, paras in sections
            ]
        }
    }


def test_spec_paragraph_refs_matches_by_keyword() -> None:
    spec = _spec_doc([
        ("WIRELESS COVERAGE", ["DEPLOYMENT OF WIRELESS ACCESS POINTS REQUIRES …"]),
        ("FIRE ALARM",        ["ALL FIRE ALARM SYSTEMS PER NFPA 72 …"]),
    ])
    paras = _iter_spec_paragraphs(spec)
    hits = _spec_paragraph_refs(
        device_class="wireless_node_outlet",
        spec_paragraphs=paras,
    )
    assert len(hits) >= 1
    assert any("WIRELESS" in h["text"].upper() for h in hits)


def test_spec_paragraph_refs_caps_at_limit() -> None:
    spec = _spec_doc([
        ("X", ["WIRELESS A", "WIRELESS B", "WIRELESS C", "WIRELESS D", "WIRELESS E"]),
    ])
    paras = _iter_spec_paragraphs(spec)
    hits = _spec_paragraph_refs(
        device_class="wireless_node_outlet",
        spec_paragraphs=paras,
        limit=2,
    )
    assert len(hits) == 2


# ─────────────────────── _confidence_breakdown ──────────────────────


def test_confidence_full_signals() -> None:
    device = {
        "keynote_text": "INSTALL CAT6 ABOVE …",
        "home_run_to": "MDF ROOM",
        "room_guess": "LOBBY",
        "review_flags": [],
    }
    cb = _confidence_breakdown(device=device, legend_hit=True)
    assert cb["composite"] == 1.0
    assert cb["legend_lookup"] == 0.4


def test_confidence_no_legend_no_keynote() -> None:
    device = {
        "keynote_text": None, "home_run_to": None, "room_guess": None,
        "review_flags": [],
    }
    cb = _confidence_breakdown(device=device, legend_hit=False)
    # Only the "no review flags" 0.1 fires.
    assert cb["composite"] == 0.1


def test_confidence_partial_keynote_resolution() -> None:
    """keynote without text resolves to 0.1 not 0.2."""
    device = {
        "keynote": "4", "keynote_text": None,
        "home_run_to": None, "room_guess": None, "review_flags": ["ambiguous"],
    }
    cb = _confidence_breakdown(device=device, legend_hit=False)
    assert cb["keynote_resolution"] == 0.1
    assert cb["no_review_flags"] == 0.0


# ─────────────────────── build_detection_plan ───────────────────────


def test_build_detection_plan_lists_expected_symbols() -> None:
    ref = _legend_doc_with_rows([
        ("WN", "WIRELESS"), ("CR", "READER"), ("TV", "TELEVISION"),
    ])
    plan = build_detection_plan(ref)
    codes = plan["expected_symbol_codes"]
    assert "WN" in codes
    assert "CR" in codes
    assert "TV" in codes


def test_build_detection_plan_works_on_empty_reference() -> None:
    """Universal: no legend → empty plan, no crash."""
    plan = build_detection_plan({})
    assert plan["expected_symbol_codes"] == []
    assert plan["schedule_rows_available"] == 0
    assert plan["spec_paragraphs_available"] == 0
