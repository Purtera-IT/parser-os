"""Deals that never train, whichever run reads them.

``LABEL_HOLDOUT`` is the untouched labeling holdout. It was four deals
(``training_setup_2026-10-11/proposed_split.json``: 010172, 010038, 010387,
010194), scored once on run A. With no new deals to label, the owner moved
010172 and 010194 into training on 2026-10-14 (010194 holds nearly every
site-pair card) and kept the other two as the test. ``KIT_HOLDOUT`` is the
five published Deal Kits. Both runs refuse both lists, so a holdout cannot
slip into one run because it was only listed for the other. They are scored
once, at the end, and never used to choose steps or thresholds (that is the
validation deals' job).
"""
from __future__ import annotations

import re
from typing import Iterable

LABEL_HOLDOUT = ("010038", "010387")
KIT_HOLDOUT = ("000051", "010095", "010164", "010253", "010287")
#: Holdout B (runE/holdout2.json, 2026-10-22): four of the ten new Run E deals
#: that never train. The file is the source of truth (the owner may swap in
#: 010358 / 010447); these are refused even when the file is not on the box.
HOLDOUT_B = ("010215", "010269", "000026", "010155")


def _holdout_b_file() -> frozenset[str]:
    import json
    import os

    p = os.environ.get("C3_HOLDOUT2") or os.path.expanduser("~/c3-local/runE/holdout2.json")
    try:
        with open(p, encoding="utf-8") as fh:
            return frozenset(str(d) for d in json.load(fh).get("deals") or [])
    except (OSError, ValueError, AttributeError):
        return frozenset()


FIXED_HOLDOUT = frozenset(LABEL_HOLDOUT) | frozenset(KIT_HOLDOUT) | frozenset(HOLDOUT_B) | _holdout_b_file()


def deal_number(deal_id: str) -> str:
    """The source deal of a part (``<deal>#c3part<n>``), without spaces or a file suffix."""
    head = str(deal_id or "").partition("#c3part")[0].strip()
    return re.sub(r"\.json$", "", head)


def refuse_fixed_holdout(train_ids: Iterable[str]) -> None:
    # A deal view ("010038@pre_quote") is that deal.
    hit = sorted({deal_number(i).partition("@")[0] for i in train_ids} & FIXED_HOLDOUT)
    if hit:
        raise ValueError("A fixed holdout deal is in the training set: " + ", ".join(hit))
