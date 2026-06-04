"""Tests for app.core.training_log — the teacher/PM dataset foundation.

Covers: masked_text auto-fill (generalization), deterministic deal-based
split (leave-one-deal-out), teacher weighting, census summary, and the
default-off / never-raise logging entrypoint.
"""

from __future__ import annotations

from app.core.training_log import (
    TEACHER_LLM,
    TEACHER_PM,
    TrainingLog,
    TrainingRow,
    assign_split,
    get_training_log,
    log_rows,
    set_training_log,
)


def test_add_autofills_masked_text_from_role_map():
    log = TrainingLog(":memory:")
    row = TrainingRow(
        relation="physical_site",
        label="not_site",
        raw_text="PurTera HQ at 11720 Amber Park Drive",
        teacher=TEACHER_PM,
        deal_id="d1",
        provenance={"role_map": {"PurTera": "<SELF_ORG>"}},
    )
    log.add(row)
    got = log.rows(relation="physical_site")[0]
    assert "PurTera" not in got.masked_text
    assert "<SELF_ORG>" in got.masked_text
    assert "<ADDR>" in got.masked_text


def test_split_is_deterministic_by_deal():
    # Same deal id → same split, every time.
    s1 = assign_split("deal-xyz")
    s2 = assign_split("deal-xyz")
    assert s1 == s2
    assert s1 in ("train", "holdout")
    # Empty deal id is never held out.
    assert assign_split("") == "train"


def test_all_rows_from_one_deal_share_a_split():
    log = TrainingLog(":memory:")
    rows = [
        TrainingRow(relation="atom_type", label="requirement",
                    raw_text=f"clause {i}", teacher=TEACHER_LLM, deal_id="dealA")
        for i in range(20)
    ]
    log.add_many(rows)
    splits = {r.split for r in log.rows(relation="atom_type")}
    assert len(splits) == 1  # leave-one-deal-out precondition


def test_teacher_weighting_pm_beats_llm():
    log = TrainingLog(":memory:")
    log.add(TrainingRow(relation="r", label="x", raw_text="t",
                        teacher=TEACHER_PM, deal_id="d"))
    log.add(TrainingRow(relation="r", label="y", raw_text="t2",
                        teacher=TEACHER_LLM, deal_id="d"))
    rows = {r.teacher: r.weight for r in log.rows(relation="r")}
    assert rows[TEACHER_PM] > rows[TEACHER_LLM]


def test_summary_census():
    log = TrainingLog(":memory:")
    log.add_many([
        TrainingRow(relation="atom_type", label="task", raw_text="a",
                    teacher=TEACHER_LLM, deal_id="d1"),
        TrainingRow(relation="payment_terms", label="net_30", raw_text="b",
                    teacher=TEACHER_PM, deal_id="d2"),
    ])
    s = log.summary()
    assert s["total"] == 2
    assert "atom_type" in s["by_relation"]
    assert "payment_terms" in s["by_relation"]


def test_count_filters():
    log = TrainingLog(":memory:")
    log.add_many([
        TrainingRow(relation="atom_type", label="task", raw_text="a",
                    teacher=TEACHER_LLM, deal_id="d1"),
        TrainingRow(relation="atom_type", label="task", raw_text="b",
                    teacher=TEACHER_PM, deal_id="d2"),
    ])
    assert log.count(relation="atom_type") == 2
    assert log.count(teacher=TEACHER_PM) == 1


def test_log_rows_is_noop_when_off(monkeypatch):
    # No injected log + no env var → default-off, returns 0, never raises.
    set_training_log(None)
    monkeypatch.delenv("SOWSMITH_TRAINING_LOG_DB", raising=False)
    assert get_training_log() is None
    assert log_rows([TrainingRow(relation="r", label="x", raw_text="t", deal_id="d")]) == 0


def test_log_rows_writes_when_injected():
    log = TrainingLog(":memory:")
    set_training_log(log)
    try:
        n = log_rows([TrainingRow(relation="r", label="x", raw_text="t", deal_id="d")])
        assert n == 1
        assert log.count(relation="r") == 1
    finally:
        set_training_log(None)


# ── teacher-logging seam: the label is the sub-type, never the relation ──
#
# Regression for the cross-deal label-schema poison: when the LLM returned a
# requirement with no ``category``, the old fallback logged ``label == relation``
# ("requirements"), so a deal whose model emitted real categories
# (deliverable/security/...) and a deal whose model omitted them trained the
# head on incompatible label schemas → 0% holdout accuracy. Guess-free: an
# item with no teacher sub-type is UNDECIDED and must not be logged at all.


def _log_results(results, atoms=()):
    from app.core.multi_entity_llm import _log_extraction_training_rows

    log = TrainingLog(":memory:")
    set_training_log(log)
    try:
        _log_extraction_training_rows(results, list(atoms))
    finally:
        set_training_log(None)
    return log


def test_list_item_with_category_logs_that_category_as_label():
    log = _log_results({"requirements": [
        {"text": "Guest Wi-Fi must stay off the corporate VLAN.", "category": "security"},
    ]})
    rows = log.rows(relation="requirements")
    assert len(rows) == 1
    assert rows[0].label == "security"


def test_list_item_without_subtype_is_skipped_not_relabelled_to_relation():
    # The exact poison shape: text but no category/role/type. Must NOT log a
    # row labelled with the bare relation name.
    log = _log_results({"requirements": [
        {"text": "Either party may terminate this SOW upon material breach."},
        {"text": "invoice for the Total Fees", "category": ""},
    ]})
    assert log.count(relation="requirements") == 0
    # And no row anywhere was labelled with its own relation name.
    assert all(r.label != r.relation for r in log.rows())


def test_mixed_batch_keeps_only_categorised_items():
    log = _log_results({"requirements": [
        {"text": "Rugged tablets must enrol into Intune before handoff.", "category": "deliverable"},
        {"text": "taxes will be invoiced but are not included"},  # no category → skip
        {"text": "All rooms support one-touch calendar join.", "category": "acceptance"},
    ]})
    labels = sorted(r.label for r in log.rows(relation="requirements"))
    assert labels == ["acceptance", "deliverable"]
