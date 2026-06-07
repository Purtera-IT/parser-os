"""End-to-end test of the neural grounding path (region proposer -> embedder ->
LegendIndex -> grounded detections). Synthesizes a drawing page with real symbol
instances + word/line distractors and asserts: correct per-legend meanings,
exact counts (dedup), and ZERO junk on the distractors (the heuristic's failure
mode: matching words like "TYPICAL" as devices)."""
from __future__ import annotations

import io
from collections import Counter

import pytest

pytest.importorskip("PIL.Image")

from app.core.schematic_heads import HeadRegistry
from app.core.schematic_symbol_head import LegendIndex
from app.core.schematic_neural_grounding import ground_page_image
from app.core.schematic_region_proposer import (
    LABEL_BACKGROUND,
    LABEL_SYMBOL,
    OBJECTNESS_HEAD,
    register_objectness_head,
)


def _sym(dr, x, y, k, s=18):
    if k == "circle":
        dr.ellipse((x - s, y - s, x + s, y + s), outline="black", width=3)
    elif k == "tri":
        dr.polygon([(x, y - s), (x - s, y + s), (x + s, y + s)], outline="black", width=3)
    elif k == "sq":
        dr.rectangle((x - s, y - s, x + s, y + s), outline="black", width=3)


def _cp(k):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (56, 56), "white")
    _sym(ImageDraw.Draw(im), 28, 28, k)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def _trained_registry():
    from PIL import Image, ImageDraw

    reg = HeadRegistry()
    register_objectness_head(reg)
    for deal in ["D1", "D2", "D3", "D4", "D5"]:
        for k in ["circle", "tri", "sq"]:
            for _ in range(6):
                reg.capture(OBJECTNESS_HEAD, deal_id=deal, raw=_cp(k), label=LABEL_SYMBOL, teacher="vlm")
        for kind in ["line", "text", "blank"]:
            for _ in range(6):
                im = Image.new("RGB", (56, 56), "white")
                d = ImageDraw.Draw(im)
                if kind == "line":
                    d.line((4, 28, 52, 28), fill="black", width=2)
                elif kind == "text":
                    d.text((6, 20), "TYP", fill="black")
                b = io.BytesIO()
                im.save(b, format="PNG")
                reg.capture(OBJECTNESS_HEAD, deal_id=deal, raw=b.getvalue(), label=LABEL_BACKGROUND, teacher="vlm")
    reg.train(OBJECTNESS_HEAD)
    return reg


def _page_and_legend():
    from PIL import Image, ImageDraw

    leg = LegendIndex()
    leg.add_symbol("CAMERA", _cp("circle"))
    leg.add_symbol("SPEAKER", _cp("sq"))
    leg.add_symbol("SENSOR", _cp("tri"))
    page = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(page)
    for x, y, k in [(80, 80, "circle"), (300, 90, "sq"), (120, 300, "tri"), (320, 320, "circle")]:
        _sym(d, x, y, k)
    # distractors the heuristic text-tag detector would falsely grab as devices
    d.text((180, 200), "TYPICAL", fill="black")
    d.text((40, 360), "VERIFY", fill="black")
    d.line((200, 40, 380, 40), fill="black", width=2)
    return page, leg


def test_grounds_real_symbols_to_correct_legend_meanings():
    reg = _trained_registry()
    page, leg = _page_and_legend()
    res = ground_page_image(page, registry=reg, legend_index=leg,
                            objectness_thresh=0.55, match_thresh=0.4, scales=(40, 56, 72))
    grounded = [g for g in res if g.source == "neural"]
    assert {g.meaning for g in grounded} == {"CAMERA", "SPEAKER", "SENSOR"}


def test_no_word_junk_grounded():
    reg = _trained_registry()
    page, leg = _page_and_legend()
    res = ground_page_image(page, registry=reg, legend_index=leg,
                            objectness_thresh=0.55, match_thresh=0.4, scales=(40, 56, 72))
    for g in res:
        if g.source == "neural":
            assert "TYP" not in g.meaning.upper() and "VERIF" not in g.meaning.upper()


def test_dedup_yields_exact_counts():
    reg = _trained_registry()
    page, leg = _page_and_legend()
    res = ground_page_image(page, registry=reg, legend_index=leg,
                            objectness_thresh=0.55, match_thresh=0.4, scales=(40, 56, 72))
    counts = Counter(g.meaning for g in res if g.source == "neural")
    assert dict(counts) == {"CAMERA": 2, "SPEAKER": 1, "SENSOR": 1}


def test_empty_legend_returns_nothing():
    reg = _trained_registry()
    page, _ = _page_and_legend()
    assert ground_page_image(page, registry=reg, legend_index=LegendIndex()) == []
