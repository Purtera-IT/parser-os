"""PR4 — cross-sheet legend resolver tests."""
from __future__ import annotations

from app.parsers.schematic_models import ParsedLegend, ParsedLegendEntry
from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock
from orbitbrief_page_os.segmentation.schematic.legend_resolver import (
    LegendResolver,
    detect_inline_references,
    extract_sheet_number,
    parse_drawing_index,
)


def _b(text: str, x0: float, y0: float, w: float = 80, h: float = 12) -> TextBlock:
    return TextBlock(text=text, bbox=(x0, y0, x0 + w, y0 + h))


def _make_legend(page_index: int, sheet: str, scope: str, syms: list[str]) -> ParsedLegend:
    entries = []
    for s in syms:
        entries.append(
            ParsedLegendEntry.make(
                page_index=page_index,
                label_text=f"{s.upper()} LABEL",
                normalized_label=f"{s.lower()} label",
                raw_symbol_text=s,
                normalized_symbol_text=s.lower(),
                confidence=0.9,
            )
        )
    return ParsedLegend.make(
        page_index=page_index,
        sheet_number=sheet,
        title="GENERATED",
        scope=scope,  # type: ignore[arg-type]
        entries=tuple(entries),
        confidence=0.9,
    )


def test_extract_sheet_number_from_title_block() -> None:
    blocks = [
        _b("Big project floor plan body text up here", 50, 50, w=400),
        _b("T0.01", 500, 740),
    ]
    assert extract_sheet_number(blocks) == "T0.01"


def test_extract_sheet_number_returns_none_when_absent() -> None:
    blocks = [_b("Just some prose", 50, 50, w=200)]
    assert extract_sheet_number(blocks) is None


def test_parse_drawing_index_finds_sheets() -> None:
    blocks = [
        _b("DRAWING INDEX", 50, 50, w=200),
        _b("T0.01 SYMBOLS & LEGENDS", 50, 70, w=300),
        _b("T1.01 FIRST FLOOR PLAN", 50, 90, w=300),
        _b("E2.01 ELECTRICAL", 50, 110, w=300),
    ]
    idx = parse_drawing_index(blocks)
    assert idx == {
        "T0.01": "SYMBOLS & LEGENDS",
        "T1.01": "FIRST FLOOR PLAN",
        "E2.01": "ELECTRICAL",
    }


def test_parse_drawing_index_empty_when_no_header() -> None:
    blocks = [_b("Some non-index page", 50, 50, w=200)]
    assert parse_drawing_index(blocks) == {}


def test_detect_inline_see_sheet() -> None:
    blocks = [_b("See sheet E-001 for legend.", 50, 50, w=300)]
    refs = detect_inline_references(blocks)
    assert "E001" in refs["see_sheet"]


def test_detect_inline_continuation() -> None:
    blocks = [_b("Symbols continued from sheet T0.01", 50, 50, w=300)]
    refs = detect_inline_references(blocks)
    assert "T0.01" in refs["continuation"]


def test_resolver_priority_in_page_wins() -> None:
    res = LegendResolver()
    in_page = _make_legend(2, "T0.01", "page", ["WN"])
    res.ingest_page(page_index=2, blocks=[_b("T0.01", 500, 740)], legend=in_page)
    chosen = res.resolve_for_page(2)
    assert chosen.priority == 1
    assert chosen.legend is in_page


def test_resolver_priority_explicit_reference() -> None:
    res = LegendResolver()
    global_legend = _make_legend(1, "T0.01", "global", ["WN", "CR"])
    res.ingest_page(
        page_index=1,
        blocks=[_b("T0.01 SYMBOLS & LEGENDS", 50, 50, w=300)],
        legend=global_legend,
    )
    res.ingest_page(
        page_index=4,
        blocks=[
            _b("E1.01", 500, 740),
            _b("See sheet T0.01 for legend", 50, 100, w=300),
        ],
    )
    chosen = res.resolve_for_page(4)
    assert chosen.priority == 2
    assert chosen.legend is global_legend
    assert chosen.rationale.startswith("explicit_see_sheet")


def test_resolver_priority_same_discipline() -> None:
    res = LegendResolver()
    t_legend = _make_legend(1, "T0.01", "global", ["WN"])
    e_legend = _make_legend(2, "E0.01", "global", ["BREAKER"])
    res.ingest_page(page_index=1, blocks=[_b("T0.01", 500, 740)], legend=t_legend)
    res.ingest_page(page_index=2, blocks=[_b("E0.01", 500, 740)], legend=e_legend)
    res.ingest_page(page_index=5, blocks=[_b("E5.01", 500, 740)])
    chosen = res.resolve_for_page(5)
    assert chosen.priority == 3
    assert chosen.legend is e_legend


def test_resolver_priority_project_global() -> None:
    res = LegendResolver()
    legend = _make_legend(1, "G0.01", "global", ["X"])
    res.ingest_page(page_index=1, blocks=[_b("G0.01", 500, 740)], legend=legend)
    res.ingest_page(page_index=4, blocks=[_b("M1.01", 500, 740)])
    chosen = res.resolve_for_page(4)
    assert chosen.priority == 4
    assert chosen.legend is legend


def test_resolver_emits_missing_legend_warning() -> None:
    res = LegendResolver()
    res.ingest_page(page_index=4, blocks=[_b("T1.01", 500, 740)])
    chosen = res.resolve_for_page(4)
    assert chosen.legend is None
    assert chosen.priority == 99
    types = [w.warning_type for w in chosen.warnings]
    assert "missing_legend" in types


def test_resolver_emits_unresolved_reference_warning() -> None:
    res = LegendResolver()
    res.ingest_page(
        page_index=4,
        blocks=[
            _b("E1.01", 500, 740),
            _b("See sheet Z9.99 for legend", 50, 100, w=300),
        ],
    )
    chosen = res.resolve_for_page(4)
    types = {w.warning_type for w in chosen.warnings}
    assert "missing_legend" in types
    assert "unresolved_legend_reference" in types


def test_resolver_emits_ambiguous_warning_on_tie() -> None:
    res = LegendResolver()
    a = _make_legend(1, "T0.01", "global", ["WN"])
    b = _make_legend(2, "T0.02", "global", ["CR"])
    res.ingest_page(page_index=1, blocks=[_b("T0.01", 500, 740)], legend=a)
    res.ingest_page(page_index=2, blocks=[_b("T0.02", 500, 740)], legend=b)
    res.ingest_page(page_index=5, blocks=[_b("T5.01", 500, 740)])
    chosen = res.resolve_for_page(5)
    assert chosen.priority == 3
    # Lower page_index wins deterministically.
    assert chosen.legend is a
    types = [w.warning_type for w in chosen.warnings]
    assert "ambiguous_legend_reference" in types


def test_resolver_drawing_index_is_indexed() -> None:
    res = LegendResolver()
    blocks = [
        _b("DRAWING INDEX", 50, 50, w=200),
        _b("T0.01 SYMBOLS & LEGENDS", 50, 70, w=300),
        _b("T1.01 FIRST FLOOR", 50, 90, w=300),
    ]
    res.ingest_page(page_index=0, blocks=blocks)
    idx = res.drawing_index
    assert "T0.01" in idx
    assert "T1.01" in idx


def test_resolver_resolution_is_deterministic_across_runs() -> None:
    def build() -> "LegendResolver":
        r = LegendResolver()
        legend = _make_legend(1, "T0.01", "global", ["WN", "CR"])
        r.ingest_page(page_index=1, blocks=[_b("T0.01", 500, 740)], legend=legend)
        r.ingest_page(
            page_index=4,
            blocks=[_b("T1.01", 500, 740), _b("See sheet T0.01 for legend", 50, 100, w=300)],
        )
        return r

    a = build().resolve_for_page(4)
    b = build().resolve_for_page(4)
    assert a.legend is not None and b.legend is not None
    assert a.legend.legend_id == b.legend.legend_id
    assert a.priority == b.priority
    assert a.rationale == b.rationale
