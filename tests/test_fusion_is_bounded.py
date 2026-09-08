"""Turning site fusion on must not turn on an unbounded sweep, or an LLM
deciding to delete locations.

Merging sites is destructive: it removes a place and moves the project tier,
which is scored on site count. Two properties make the flag safe to enable.
"""

from __future__ import annotations

import os

import pytest

import app.core.decide as decide_mod
from app.core.entity_resolution import (
    _EXHAUSTIVE_PAIR_CEILING,
    semantic_site_fusion_groups,
)


class _Undecided:
    verdict = None
    source = "fallback"


@pytest.fixture()
def asked(monkeypatch):
    """Capture what decide() is handed, and decide nothing."""
    seen: list[dict] = []

    def _spy(**kwargs):
        seen.append(kwargs)
        return _Undecided()

    monkeypatch.setattr(decide_mod, "decide", _spy)
    monkeypatch.setenv("SOWSMITH_NEURAL_SITE_FUSION", "1")
    return seen


def _rows(n: int) -> dict[str, dict]:
    """n located sites — the shape that produces no shortlist pairs."""
    return {
        f"site:s{i}": {
            "site": f"site:s{i}", "facility_name": f"Site {i}",
            "street_address": f"{i} Main St", "city": "Springfield", "state": "IL",
            "anchored": True,
        }
        for i in range(n)
    }


def test_a_small_deal_still_gets_the_exhaustive_sweep(asked) -> None:
    rows = _rows(5)
    semantic_site_fusion_groups(set(rows), rows)
    assert len(asked) == 10, "5 sites is 10 pairs"


def test_a_large_deal_does_not_enumerate_every_pair(asked) -> None:
    """135 sites is 9,045 pairs and a 437-site rollout is 95,266."""
    n = _EXHAUSTIVE_PAIR_CEILING + 20
    rows = _rows(n)
    semantic_site_fusion_groups(set(rows), rows)
    assert len(asked) < n * (n - 1) // 2
    # These rows are all located, so the shortlist proposes nothing at all.
    assert len(asked) == 0


def test_a_large_deal_still_asks_about_a_real_duplicate(asked) -> None:
    """The shortlist is a reduction, not a silencing."""
    rows = _rows(_EXHAUSTIVE_PAIR_CEILING + 5)
    rows["site:the_hillview_office"] = {
        "site": "site:the_hillview_office", "facility_name": "Hillview Office",
        "anchored": False,
    }
    rows["site:anchor"] = {
        "site": "site:anchor", "facility_name": "Palo Alto Office",
        "street_address": "3300 Hillview Ave", "city": "Palo Alto", "state": "CA",
        "anchored": True,
    }
    semantic_site_fusion_groups(set(rows), rows)
    texts = [k.get("text", "") for k in asked]
    assert any("Hillview" in t for t in texts), texts[:3]


def test_the_llm_can_be_locked_out_entirely(asked, monkeypatch) -> None:
    """With the budget at zero, only a PM-taught merge can fire — nothing is
    decided by a model on a destructive operation."""
    monkeypatch.setenv("SOWSMITH_SITE_FUSION_LLM_BUDGET", "0")
    rows = _rows(4)
    semantic_site_fusion_groups(set(rows), rows)
    assert asked, "the pass asked nothing at all"
    assert all(k.get("llm") is False for k in asked), [k.get("llm") for k in asked]


def test_the_flag_off_is_still_a_no_op(monkeypatch) -> None:
    monkeypatch.delenv("SOWSMITH_NEURAL_SITE_FUSION", raising=False)
    rows = _rows(5)
    assert semantic_site_fusion_groups(set(rows), rows) == []
