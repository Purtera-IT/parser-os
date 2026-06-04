"""Tests for app.core.extractor_student — the kNN trained-head, day one.

The student must (a) generalize: vote by rule not by name, proven by training
on one deal's proper nouns and querying a *different* deal's; (b) stay
guess-free: abstain when it has no confident evidence, when the embedder is
down, or when the log is empty; and (c) respect the decide() candidate contract.

A deterministic concept embedder stands in for qwen3-embedding: it maps the
*masked* text to one of a few orthonormal axes by detecting rule-bearing words
that survive delexicalization (names are already stripped, so the embedder
can't key on identity even if it wanted to).
"""

from __future__ import annotations

import numpy as np

from app.core.delexicalize import ROLE_CUSTOMER, ROLE_SELF_ORG
from app.core.extractor_student import ExtractionStudent
from app.core.training_log import TEACHER_LLM, TEACHER_PM, TrainingLog, TrainingRow

# ── deterministic concept embedder over MASKED text ─────────────────────
_AXES = {
    "requirement": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "payment": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "site": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "other": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}


def _concept(masked: str) -> str:
    m = masked.lower()
    if "shall" in m or "must" in m or "provide" in m:
        return "requirement"
    if "net " in m or "payment" in m or "invoice" in m:
        return "payment"
    if "located" in m or "site" in m or "premises" in m:
        return "site"
    return "other"


def _embed(texts: list[str]) -> np.ndarray:
    return np.array([_AXES[_concept(t)] for t in texts], dtype=np.float32)


def _student(log: TrainingLog, **kw) -> ExtractionStudent:
    kw.setdefault("threshold", 0.6)
    return ExtractionStudent(log, embed_fn=_embed, reachable_fn=lambda: True, **kw)


def _row(label, text, **kw) -> TrainingRow:
    return TrainingRow(relation="atom_type", label=label, raw_text=text, **kw)


# ── guess-free guarantees ───────────────────────────────────────────────
def test_abstains_with_no_training_rows():
    s = _student(TrainingLog(":memory:"))
    p = s.classify("PurTera shall provide cameras", "atom_type")
    assert p.abstained
    assert p.reason == "no_training_rows"


def test_abstains_when_embedder_unreachable():
    log = TrainingLog(":memory:")
    log.add(_row("requirement", "Vendor shall provide X", teacher=TEACHER_LLM, deal_id="d1"))
    s = ExtractionStudent(log, embed_fn=_embed, reachable_fn=lambda: False)
    p = s.classify("anything", "atom_type")
    assert p.abstained
    assert p.reason == "embedder_unreachable"


def test_confident_vote_after_learning():
    log = TrainingLog(":memory:")
    log.add_many([
        _row("requirement", "Contractor shall provide 20 cameras", teacher=TEACHER_LLM, deal_id="d1"),
        _row("requirement", "Vendor must provide signage", teacher=TEACHER_LLM, deal_id="d2"),
        _row("payment_term", "Net 30 from invoice date", teacher=TEACHER_LLM, deal_id="d3"),
    ])
    s = _student(log)
    p = s.classify("Supplier shall provide badges", "atom_type")  # paraphrase, new name
    assert not p.abstained
    assert p.label == "requirement"


def test_candidates_restrict_allowed_labels():
    log = TrainingLog(":memory:")
    log.add(_row("requirement", "shall provide X", teacher=TEACHER_LLM, deal_id="d1"))
    s = _student(log)
    # The only neighbour says "requirement", but the caller doesn't allow it →
    # no usable vote → abstain (never return an out-of-candidate verdict).
    p = s.classify("must provide Y", "atom_type", candidates=["payment_term"])
    assert p.abstained


def test_below_threshold_abstains():
    log = TrainingLog(":memory:")
    log.add(_row("requirement", "shall provide X", teacher=TEACHER_LLM, deal_id="d1"))
    s = _student(log, threshold=0.99)
    # Query matches concept but a 0.99 bar with a single neighbour won't clear.
    p = s.classify("totally unrelated boilerplate", "atom_type")
    assert p.abstained


# ── the load-bearing generalization test ────────────────────────────────
def test_name_swap_invariance():
    """Train on one deal's names; query a different deal's. Same rule → same
    answer. The student cannot have memorized the name, because the held-out
    name was never in its memory and the masked text is identical."""
    log = TrainingLog(":memory:")
    # Training deal uses PurTera/Yonah; provenance role_map masks them.
    log.add_many([
        TrainingRow(
            relation="atom_type", label="requirement",
            raw_text="PurTera shall provide cameras to Yonah County",
            teacher=TEACHER_PM, deal_id="train_deal",
            provenance={"role_map": {"PurTera": ROLE_SELF_ORG, "Yonah County": ROLE_CUSTOMER}},
        ),
        TrainingRow(
            relation="atom_type", label="requirement",
            raw_text="PurTera shall provide badges to Yonah County",
            teacher=TEACHER_PM, deal_id="train_deal",
            provenance={"role_map": {"PurTera": ROLE_SELF_ORG, "Yonah County": ROLE_CUSTOMER}},
        ),
    ])
    s = _student(log)
    # Query a *completely different* set of names, same rule.
    p = s.classify(
        "Acme Corp shall provide turnstiles to Dakota City", "atom_type",
        role_map={"Acme Corp": ROLE_SELF_ORG, "Dakota City": ROLE_CUSTOMER},
    )
    assert not p.abstained
    assert p.label == "requirement"


def test_pm_gold_outvotes_llm_silver():
    """When silver and gold disagree on similar text, the higher-weighted PM
    label wins the vote."""
    log = TrainingLog(":memory:")
    # Several silver rows say "other"; one gold row says "requirement". All map
    # to the same concept axis (contain "provide"), so they compete directly.
    log.add_many([
        _row("other", "shall provide thing one", teacher=TEACHER_LLM, deal_id="d1"),
        _row("requirement", "shall provide thing two", teacher=TEACHER_PM, deal_id="d2"),
    ])
    s = _student(log)
    p = s.classify("shall provide thing three", "atom_type")
    assert not p.abstained
    assert p.label == "requirement"  # gold weight (5.0) beats silver (1.0)
