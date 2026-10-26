"""On the Run E training data, person image verdicts on picture readings an
approved draft set aside now train (owner, 2026-10-26: "yes otpion a for me"),
and no other head gains or loses a single target. Local data only: skipped
where ~/c3-local/runE/runE_data_v2 is absent."""
from __future__ import annotations

import collections
import glob
import os

import pytest

pytest.importorskip("torch")

from ml.c3 import data as data_mod  # noqa: E402
from ml.c3.data import IGNORE  # noqa: E402
from ml.c3.loop import load_deal, read_draft_approvals, training_batches  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

DATA = os.path.expanduser("~/c3-local/runE/runE_data_v2")

pytestmark = pytest.mark.skipif(not os.path.exists(f"{DATA}/approval_all.json"),
                                reason="Run E training data not on this machine")


def _counts() -> collections.Counter:
    manifest, schema = read_draft_approvals(f"{DATA}/approval_all.json"), load_schema()
    out: collections.Counter = collections.Counter()
    for path in sorted(glob.glob(f"{DATA}/train_*/*@*.json")):
        if os.path.basename(path).startswith("010043"):
            continue
        parts, _ = training_batches(load_deal(path, manifest), schema, max_atoms=8192)
        for b in parts:
            for key, col in b.targets.items():
                out[key] += sum(1 for v in col if v != IGNORE)
            for j in b.judged:
                out["J:" + j.key] += 1
    return out


def test_image_verdicts_survive_and_nothing_else_moves(monkeypatch):
    after = _counts()
    monkeypatch.setattr(data_mod, "VERDICT_OVER_DRAFT_ASIDE", frozenset())
    before = _counts()
    assert before["jdg:image"] == 54
    assert 190 <= after["jdg:image"] <= 194
    changed = {k for k in set(before) | set(after) if before[k] != after[k]}
    assert changed == {"jdg:image"}
