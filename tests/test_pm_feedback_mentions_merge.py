"""Two lessons with the same verdict and different PM words gate on either."""

from __future__ import annotations

import numpy as np

from app.core.feedback_store import FeedbackStore
from app.core.pm_feedback import apply_pm_correction


def _embed(texts):
    return np.ones((len(texts), 3), dtype=np.float32) / np.sqrt(3)


def _teach(store, text, when):
    return apply_pm_correction(store, {
        "head": "hours", "dealId": "deal-1", "compileId": "", "targetId": "task:cable drops",
        "text": text, "oldValue": "", "newValue": "hours=1;per=drop;qty=24;role=L2",
        "scope": "global", "context": "", "rationale": "", "relations": {"when": when}, "pm": "pm@x", "candidates": [],
    })


def test_mentions_union_on_merge_any_other_disagreement_ungates():
    store = FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)
    cid = _teach(store, "Install 24 Cat6 cable drops", {"mentions": ["ladder"]})
    assert _teach(store, "Install 24 Cat6 cable drops", {"mentions": ["ceil"]}) == cid
    assert store.get(cid).relations["when"] == {"mentions": ["ladder", "ceil"]}
    _teach(store, "Install 24 Cat6 cable drops", {"field": "owner", "equals": "chase"})
    assert "when" not in store.get(cid).relations
