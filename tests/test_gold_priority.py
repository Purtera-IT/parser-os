"""Ranking a PM's next click by what it TEACHES, not just what looks wrong.

These pin the judgements that separate active learning from triage. Getting them
backwards is what makes a review queue feel like busywork: the PM answers ten
questions and the model learns nothing.
"""
from __future__ import annotations

from app.learning.gold_priority import (
    SATURATED_AT,
    boundary_uncertainty,
    class_counts_from_log,
    class_starvation,
    gold_priority,
    reach,
)

TAU = 0.72


def test_uncertainty_peaks_at_the_boundary_not_at_zero():
    """The whole point. `1 - confidence` would rank the 0.02 item highest, but
    that is usually OCR noise — labeling it teaches nothing."""
    at_boundary = boundary_uncertainty(TAU, TAU)
    noise = boundary_uncertainty(0.02, TAU)
    confident = boundary_uncertainty(0.99, TAU)
    assert at_boundary == 1.0
    assert noise < at_boundary
    assert confident < at_boundary


def test_a_confident_noise_atom_never_outranks_a_torn_one():
    counts = {("atom_type", "scope_item"): 5, ("atom_type", "risk"): 5}
    torn = gold_priority(confidence=TAU, label="scope_item", class_counts=counts, threshold=TAU)
    noise = gold_priority(confidence=0.01, label="risk", class_counts=counts, threshold=TAU)
    assert torn.score > noise.score
    assert "at_decision_boundary" in torn.reasons


def test_an_unseen_class_beats_a_saturated_one_at_equal_uncertainty():
    """With 12 fine classes sitting empty, this is where the leverage is: the
    500th scope_item teaches nothing, the first submission_req fills a hole."""
    counts = {("atom_type", "scope_item"): 500}
    empty = gold_priority(confidence=TAU, label="submission_req", class_counts=counts, threshold=TAU)
    saturated = gold_priority(confidence=TAU, label="scope_item", class_counts=counts, threshold=TAU)
    assert empty.score > saturated.score
    assert "unseen_class" in empty.reasons


def test_starvation_decays_and_bottoms_out_when_covered():
    assert class_starvation(0) == 1.0
    assert class_starvation(3) < class_starvation(0)
    assert class_starvation(SATURATED_AT) == 0.0
    assert class_starvation(SATURATED_AT + 100) == 0.0


def test_governing_a_packet_outranks_a_stray_atom():
    """A wrong label on a governing atom costs a deliverable, not just a metric."""
    counts = {("atom_type", "x"): 5}
    gov = gold_priority(confidence=TAU, label="x", class_counts=counts, threshold=TAU, governs_packet=True)
    stray = gold_priority(confidence=TAU, label="x", class_counts=counts, threshold=TAU, governs_packet=False)
    assert gov.score > stray.score
    assert "governs_a_packet" in gov.reasons


def test_a_recurring_shape_beats_a_one_off():
    counts = {("atom_type", "x"): 5}
    common = gold_priority(confidence=TAU, label="x", class_counts=counts, threshold=TAU, cohort_size=25)
    oneoff = gold_priority(confidence=TAU, label="x", class_counts=counts, threshold=TAU, cohort_size=1)
    assert common.score > oneoff.score
    assert reach(1) == 0.0


def test_already_corrected_items_are_never_asked_again():
    """Re-asking spends a click to learn nothing and reads to the PM as the
    system not listening to them."""
    p = gold_priority(confidence=TAU, label="x", threshold=TAU, already_gold=True, governs_packet=True)
    assert p.score == 0.0
    assert p.reasons == ["already_gold"]


def test_score_is_bounded_and_interpretable():
    best = gold_priority(
        confidence=TAU, label="never_seen", class_counts={}, threshold=TAU,
        governs_packet=True, cohort_size=50,
    )
    worst = gold_priority(
        confidence=0.999, label="x", class_counts={("atom_type", "x"): 10_000},
        threshold=TAU, governs_packet=False, cohort_size=1,
    )
    assert 0.0 <= worst.score <= best.score <= 1.0
    assert best.score > 0.9


def test_reasons_always_explain_the_ask():
    p = gold_priority(confidence=0.99, label="x", class_counts={("atom_type", "x"): 999}, threshold=TAU)
    assert p.reasons, "an ask with no stated reason is not auditable"
    assert "low_teaching_value" in p.reasons


class _Row:
    def __init__(self, relation, label):
        self.relation = relation
        self.label = label


class _Log:
    def __init__(self, rows):
        self._rows = rows

    def rows(self, relation=None):
        if relation:
            return [r for r in self._rows if r.relation == relation]
        return list(self._rows)


def test_class_counts_read_the_training_log():
    log = _Log([
        _Row("atom_type", "scope_item"),
        _Row("atom_type", "scope_item"),
        _Row("atom_type", "risk"),
        _Row("service_routing", "wireless"),
    ])
    counts = class_counts_from_log(log)
    assert counts[("atom_type", "scope_item")] == 2
    assert counts[("atom_type", "risk")] == 1
    assert counts[("service_routing", "wireless")] == 1


def test_an_unreadable_log_degrades_to_uncertainty_instead_of_breaking():
    class Broken:
        def rows(self, relation=None):
            raise RuntimeError("log unavailable")

    assert class_counts_from_log(Broken()) == {}
    # Still ranks, just without the starvation signal.
    p = gold_priority(confidence=TAU, label="x", class_counts={}, threshold=TAU)
    assert p.score > 0.0
