"""Tests for the neural region proposer (learned replacement for the heuristic
propose_regions). Synthetic symbol vs background crops; deterministic, offline."""
from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL.Image")

from app.core.schematic_heads import HeadRegistry
from app.core.schematic_region_proposer import (
    LABEL_BACKGROUND,
    LABEL_SYMBOL,
    OBJECTNESS_HEAD,
    propose_regions_neural,
    register_objectness_head,
    sliding_windows,
)


def _sym(k: str, o: int = 0) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (64, 64), "white")
    d = ImageDraw.Draw(im)
    if k == "circle":
        d.ellipse((12 + o, 12 + o, 52 + o, 52 + o), outline="black", width=3)
    elif k == "tri":
        d.polygon([(32 + o, 10), (10, 54), (54, 54)], outline="black", width=3)
    else:
        d.rectangle((14 + o, 14 + o, 50 + o, 50 + o), outline="black", width=3)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def _bg(kind: str, o: int = 0) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (64, 64), "white")
    d = ImageDraw.Draw(im)
    if kind == "line":
        d.line((2, 30 + o, 62, 30 + o), fill="black", width=2)
    elif kind == "text":
        d.text((6, 24), "A-12", fill="black")
    elif kind == "corner":
        d.line((2, 2, 2, 62), fill="black", width=2)
        d.line((2, 2, 62, 2), fill="black", width=2)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def _trained_registry() -> HeadRegistry:
    reg = HeadRegistry()
    register_objectness_head(reg)
    for deal in ["D1", "D2", "D3", "D4", "D5"]:
        for k in ["circle", "tri", "sq"]:
            for o in range(-4, 5, 2):
                reg.capture(OBJECTNESS_HEAD, deal_id=deal, raw=_sym(k, o), label=LABEL_SYMBOL, teacher="vlm")
        for k in ["line", "text", "blank", "corner"]:
            for o in range(-4, 5, 2):
                reg.capture(OBJECTNESS_HEAD, deal_id=deal, raw=_bg(k, o), label=LABEL_BACKGROUND, teacher="vlm")
    reg.train(OBJECTNESS_HEAD)
    return reg


def test_objectness_head_separates_symbol_from_background():
    reg = _trained_registry()
    sym_hits = sum(
        1 for k in ["circle", "tri", "sq"]
        if (p := reg.predict(OBJECTNESS_HEAD, _sym(k, 5))) and p[0] == LABEL_SYMBOL
    )
    bg_hits = sum(
        1 for k in ["line", "text", "corner"]
        if (p := reg.predict(OBJECTNESS_HEAD, _bg(k, 5))) and p[0] == LABEL_BACKGROUND
    )
    assert sym_hits >= 2 and bg_hits >= 2


def test_sliding_windows_yields_candidates():
    from PIL import Image

    img = Image.new("RGB", (200, 200), "white")
    wins = list(sliding_windows(img, scales=(48, 80)))
    assert len(wins) > 0
    box, png = wins[0]
    assert len(box) == 4 and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_untrained_proposer_returns_empty():
    from PIL import Image

    reg = HeadRegistry()
    register_objectness_head(reg)  # registered but NOT trained
    img = Image.new("RGB", (200, 200), "white")
    assert propose_regions_neural(reg, img) == []


def test_trained_proposer_finds_symbol_regions():
    from PIL import Image, ImageDraw

    reg = _trained_registry()
    img = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((40, 40, 90, 90), outline="black", width=3)
    d.rectangle((160, 160, 210, 210), outline="black", width=3)
    regions = propose_regions_neural(reg, img, score_thresh=0.5, scales=(48, 80))
    assert isinstance(regions, list)
    # should find at least one symbol region on a page that has two
    assert len(regions) >= 1
