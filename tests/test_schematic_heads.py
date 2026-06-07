"""Tests for the unified schematic neural-head framework: register -> teacher-
capture -> eval-gated train/promote -> predict -> rollback. Synthetic page
layouts stand in for real sheets so the test is deterministic and offline."""
from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL.Image")

from app.core.schematic_heads import default_registry


def _page(kind: str, seed: int = 0) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (300, 220), "white")
    d = ImageDraw.Draw(im)
    s = seed
    if kind == "SCHEDULE":
        for r in range(6):
            for c in range(5):
                d.rectangle((10 + c * 55, 10 + r * 30 + s, 60 + c * 55, 38 + r * 30 + s), outline="black")
    elif kind == "DRAWING":
        for i in range(5):
            d.line((10 + s, 10 + i * 30, 280, 200 - i * 20), fill="black", width=2)
        d.rectangle((120, 90, 160, 130), outline="black")
    elif kind == "COVER":
        d.rectangle((180, 150, 290, 210), outline="black", width=3)
        d.text((190, 170), "COVER", fill="black")
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def _seed(reg, head="page_kind"):
    for deal in ["D1", "D2", "D3", "D4", "D5"]:
        for k in ["SCHEDULE", "DRAWING", "COVER"]:
            for j in range(5):
                reg.capture(head, deal_id=deal, raw=_page(k, seed=j * 2), label=k, teacher="vlm")


def test_registry_has_global_heads_but_not_symbol():
    reg = default_registry()
    assert "page_kind" in reg.names()
    assert "discipline" in reg.names()
    # symbol grounding is per-document (LegendIndex), NOT a global head here
    assert "symbol" not in reg.names()


def test_head_trains_promotes_and_predicts():
    reg = default_registry()
    _seed(reg)
    res = reg.train("page_kind")
    assert res["promoted"] is True
    assert res["val_macro_f1"] >= 0.7
    correct = 0
    for k in ["SCHEDULE", "DRAWING", "COVER"]:
        r = reg.predict("page_kind", _page(k, seed=7))
        if r and r[0] == k:
            correct += 1
    assert correct >= 2


def test_unseeded_head_does_not_promote():
    reg = default_registry()
    res = reg.train("discipline")  # no data captured
    assert res["promoted"] is False
    assert res["reason"] == "insufficient_data"
    # and predict() on an un-promoted head abstains
    assert reg.predict("discipline", _page("SCHEDULE")) is None


def test_below_gate_rolls_back():
    reg = default_registry()
    reg.min_macro_f1 = 1.01  # impossible gate -> nothing can promote
    _seed(reg)
    res = reg.train("page_kind")
    assert res["promoted"] is False
    assert res["reason"] == "below_gate"
    assert reg.predict("page_kind", _page("SCHEDULE")) is None  # old (none) kept
