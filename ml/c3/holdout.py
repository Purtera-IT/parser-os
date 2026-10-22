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
FIXED_HOLDOUT = frozenset(LABEL_HOLDOUT) | frozenset(KIT_HOLDOUT)


def deal_number(deal_id: str) -> str:
    """The source deal of a part (``<deal>#c3part<n>``), without spaces or a file suffix."""
    head = str(deal_id or "").partition("#c3part")[0].strip()
    return re.sub(r"\.json$", "", head)


def refuse_fixed_holdout(train_ids: Iterable[str]) -> None:
    # A deal view ("010038@pre_quote") is that deal.
    hit = sorted({deal_number(i).partition("@")[0] for i in train_ids} & FIXED_HOLDOUT)
    if hit:
        raise ValueError("A fixed holdout deal is in the training set: " + ", ".join(hit))
