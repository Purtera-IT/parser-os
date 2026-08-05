"""Bound the fit so the nightly stops being OOM-killed — without losing gold.

`fit_candidate` embeds every train row for a relation into one array. The
training log is append-only and grows ~1,500 rows per compile, so at 33,727 rows
the nightly was SIGKILLed at 3h48m inside a 4Gi container. Raising memory only
moves the wall; capping is the durable fix.

The one thing the cap must never do is drop gold. There are single digits of it,
and per the head ledger it is the only thing that can lift a head past the
teacher it distills.
"""
from __future__ import annotations

from app.core.training_log import TEACHER_LLM, TEACHER_PM, TrainingRow
from app.learning.retrain import _cap_train_rows


def _row(teacher: str, created_at: float, label: str = "scope_item") -> TrainingRow:
    return TrainingRow(
        relation="atom_type",
        label=label,
        raw_text="x",
        teacher=teacher,
        created_at=created_at,
    )


def test_under_the_cap_nothing_is_dropped():
    rows = [_row(TEACHER_LLM, i) for i in range(50)]
    assert len(_cap_train_rows(rows, 100)) == 50


def test_caps_to_the_limit():
    rows = [_row(TEACHER_LLM, i) for i in range(500)]
    assert len(_cap_train_rows(rows, 100)) == 100


def test_every_gold_row_survives_even_when_over_the_cap():
    gold = [_row(TEACHER_PM, i, label=f"gold-{i}") for i in range(20)]
    silver = [_row(TEACHER_LLM, 1000 + i) for i in range(5000)]
    capped = _cap_train_rows(gold + silver, 100)
    kept_gold = [r for r in capped if r.teacher == TEACHER_PM]
    assert len(kept_gold) == 20, "gold is scarce and irreplaceable — never drop it"
    assert len(capped) == 100


def test_gold_beyond_the_cap_is_still_kept_whole():
    # Pathological but must not silently truncate the only real labels.
    gold = [_row(TEACHER_PM, i) for i in range(150)]
    capped = _cap_train_rows(gold, 100)
    assert len(capped) == 150
    assert all(r.teacher == TEACHER_PM for r in capped)


def test_silver_is_dropped_oldest_first():
    silver = [_row(TEACHER_LLM, float(i)) for i in range(100)]
    capped = _cap_train_rows(silver, 10)
    kept = sorted(r.created_at for r in capped)
    assert kept == [float(i) for i in range(90, 100)], "keep the most recent silver"


def test_zero_or_negative_limit_disables_the_cap():
    rows = [_row(TEACHER_LLM, i) for i in range(30)]
    assert len(_cap_train_rows(rows, 0)) == 30
    assert len(_cap_train_rows(rows, -1)) == 30
