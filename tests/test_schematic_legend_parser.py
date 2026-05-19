"""PR3 — legend locator + parser MVP tests."""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from orbitbrief_page_os.segmentation.schematic.legend_locator import (
    TextBlock,
    locate_legend_candidates,
    page_text_blocks,
)
from orbitbrief_page_os.segmentation.schematic.legend_parser import parse_legend


# ─── unit tests on synthetic TextBlock streams ───


def _b(text: str, x0: float, y0: float, w: float = 40, h: float = 12, bi: int = 0, li: int = 0) -> TextBlock:
    return TextBlock(text=text, bbox=(x0, y0, x0 + w, y0 + h), block_index=bi, line_index=li)


def test_locator_finds_strong_header() -> None:
    blocks = [
        _b("SYMBOL LEGEND", 50, 50),
        _b("WN", 50, 70),
        _b("WIRELESS NODE", 100, 70),
        _b("Some unrelated paragraph here", 50, 400),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    assert any(c.layer == "text_rule_strong" for c in cands), cands
    strong = next(c for c in cands if c.layer == "text_rule_strong")
    assert strong.score >= 0.55
    assert strong.header_text == "symbol legend"


def test_locator_finds_header_pair_with_count() -> None:
    blocks = [
        _b("SYMBOL", 50, 100, w=40),
        _b("DESCRIPTION", 100, 100, w=80),
        _b("COUNT", 200, 100, w=40),
        _b("WN", 50, 115, w=20),
        _b("Wireless Node", 100, 115, w=80),
        _b("4", 200, 115, w=10),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    pair = next(c for c in cands if c.layer == "header_pair")
    assert pair.score > 0.5  # base + count boost
    assert "count" in (pair.headers_seen[-1] if pair.headers_seen else "")


def test_locator_returns_continuation_hint() -> None:
    blocks = [
        _b("Symbols continued from sheet T0.01", 50, 60, w=300, h=12),
    ]
    cands = locate_legend_candidates(page_index=2, blocks=blocks)
    cont = next(c for c in cands if c.layer == "continuation")
    assert cont.continuation_ref == "T0.01"


def test_locator_dedupes_overlapping_candidates() -> None:
    blocks = [
        _b("SYMBOL LEGEND", 50, 50),
        _b("SYMBOL", 50, 80),
        _b("DESCRIPTION", 100, 80),
        _b("WN", 50, 95),
        _b("Wireless Node", 100, 95),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    pages = [c.page_index for c in cands]
    assert pages == sorted(pages)
    # Top-scoring candidate must come first.
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_parser_extracts_tabular_rows() -> None:
    blocks = [
        _b("SYMBOL", 50, 100, w=40),
        _b("DESCRIPTION", 100, 100, w=80),
        _b("WN", 50, 115, w=20),
        _b("WIRELESS NODE", 100, 115, w=120),
        _b("CR", 50, 130, w=20),
        _b("CARD READER", 100, 130, w=120),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    cand = max(cands, key=lambda c: c.score)
    legend = parse_legend(candidate=cand, page_blocks=blocks)
    assert legend is not None
    symbols = sorted((e.raw_symbol_text or "") for e in legend.entries)
    assert symbols == ["CR", "WN"]


def test_parser_handles_inline_dash_form() -> None:
    blocks = [
        _b("LEGEND", 50, 50, w=60),
        _b("WN - WIRELESS NODE", 50, 70, w=200),
        _b("CR = CARD READER", 50, 86, w=200),
        _b("TV: TELEVISION", 50, 102, w=200),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    cand = max(cands, key=lambda c: c.score)
    legend = parse_legend(candidate=cand, page_blocks=blocks)
    assert legend is not None
    syms = sorted((e.raw_symbol_text or "") for e in legend.entries)
    assert syms == ["CR", "TV", "WN"]


def test_parser_returns_none_for_no_rows() -> None:
    blocks = [_b("LEGEND", 50, 50)]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    cand = max(cands, key=lambda c: c.score)
    legend = parse_legend(candidate=cand, page_blocks=blocks)
    assert legend is None


def test_parser_ignores_prose_dashes() -> None:
    # Inline regex must reject "Note - the contractor shall..."
    blocks = [
        _b("LEGEND", 50, 50),
        _b("Note - the contractor shall verify all dimensions", 50, 70, w=300),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    cand = max(cands, key=lambda c: c.score)
    legend = parse_legend(candidate=cand, page_blocks=blocks)
    assert legend is None


def test_parser_assigns_count_column() -> None:
    blocks = [
        _b("SYMBOL", 50, 100, w=40),
        _b("DESCRIPTION", 100, 100, w=80),
        _b("COUNT", 200, 100, w=40),
        _b("WN", 50, 115, w=20),
        _b("Wireless Node", 100, 115, w=120),
        _b("4", 200, 115, w=10),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    cand = max(cands, key=lambda c: c.score)
    legend = parse_legend(candidate=cand, page_blocks=blocks)
    assert legend is not None
    wn = next(e for e in legend.entries if e.raw_symbol_text == "WN")
    assert wn.count_column == 4.0


def test_parser_is_deterministic_across_runs() -> None:
    blocks = [
        _b("SYMBOL", 50, 100),
        _b("DESCRIPTION", 100, 100, w=80),
        _b("WN", 50, 115),
        _b("Wireless Node", 100, 115, w=80),
        _b("CR", 50, 130),
        _b("Card Reader", 100, 130, w=80),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    cand = max(cands, key=lambda c: c.score)
    a = parse_legend(candidate=cand, page_blocks=blocks)
    b = parse_legend(candidate=cand, page_blocks=blocks)
    assert a is not None and b is not None
    assert a.legend_id == b.legend_id
    assert tuple(e.entry_id for e in a.entries) == tuple(e.entry_id for e in b.entries)


# ─── end-to-end test using a generated PDF ───


def _make_legend_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOL LEGEND", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    page.insert_text((72, 110), "WN", fontsize=10)
    page.insert_text((180, 110), "WIRELESS NODE", fontsize=10)
    page.insert_text((300, 110), "4", fontsize=10)
    page.insert_text((72, 128), "CR", fontsize=10)
    page.insert_text((180, 128), "CARD READER", fontsize=10)
    page.insert_text((300, 128), "12", fontsize=10)
    page.insert_text((72, 146), "TV", fontsize=10)
    page.insert_text((180, 146), "TELEVISION JACK", fontsize=10)
    page.insert_text((300, 146), "8", fontsize=10)
    doc.save(str(path))
    doc.close()


def test_pdf_e2e_locator_parser(tmp_path: Path) -> None:
    pdf_path = tmp_path / "legend.pdf"
    _make_legend_pdf(pdf_path)
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    blocks = page_text_blocks(page)
    assert blocks, "no text blocks extracted from generated PDF"
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    assert cands
    strong = [c for c in cands if c.is_strong or c.score >= 0.45]
    assert strong, [(c.layer, c.score) for c in cands]
    legend = parse_legend(candidate=strong[0], page_blocks=blocks)
    doc.close()
    assert legend is not None
    syms = sorted((e.raw_symbol_text or "") for e in legend.entries)
    assert syms == ["CR", "TV", "WN"]
    by_sym = {e.raw_symbol_text: e for e in legend.entries}
    assert by_sym["WN"].count_column == 4.0
    assert by_sym["CR"].count_column == 12.0
    assert by_sym["TV"].count_column == 8.0


def test_non_legend_page_returns_no_strong_candidate(tmp_path: Path) -> None:
    pdf_path = tmp_path / "no_legend.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "1. This is just a regular paragraph", fontsize=11)
    page.insert_text((72, 80), "of body text without any legend.", fontsize=11)
    page.insert_text((72, 100), "We then continue with more prose.", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    doc = fitz.open(str(pdf_path))
    blocks = page_text_blocks(doc.load_page(0))
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    doc.close()
    strong = [c for c in cands if c.is_strong]
    assert not strong, [(c.layer, c.score, c.header_text) for c in cands]


def test_classifier_callback_is_bounded() -> None:
    blocks = [
        _b("LEGEND", 50, 50),
        _b("WN - WIRELESS NODE", 50, 70, w=200),
    ]
    base = locate_legend_candidates(page_index=0, blocks=blocks)
    boosted = locate_legend_candidates(
        page_index=0, blocks=blocks, classifier=lambda c: 5.0  # absurdly large
    )
    # boost is capped at 0.20 so no candidate can exceed base + 0.20
    for b_cand in boosted:
        matching = [b for b in base if b.layer.split("+")[0] == b_cand.layer.split("+")[0]]
        if matching:
            assert b_cand.score <= matching[0].score + 0.21
