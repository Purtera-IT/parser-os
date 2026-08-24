"""The two no-GPU bets: conformal abstention and the delta head.

Both are guess-free machinery over the PM hold-out, so the tests pin the
guarantees and the refusals with equal weight -- a gate that cannot refuse is
how this system got a 91% that meant nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.conformal import ConformalGate, coverage_on, fit
from app.core.delta_head import DeltaExample, DeltaHead, build_examples

# ── a deterministic toy scorer with a known noise level ──────────────────

_LABELS = ["wireless", "audio_visual", "low_voltage_cabling", "staff_augmentation"]


def _scores(i: int, true_label: str, noise: float) -> dict[str, float]:
    """Peaked on the truth, deterministic pseudo-noise from the index."""
    # Raw (un-normalised) scores on purpose: conformal needs no calibration of
    # the score function -- that is the whole point -- and raw scores let the
    # noise parameter actually control how loose the fitted quantile is.
    base = {lab: 0.05 + 0.01 * ((i * 7 + j) % 5) for j, lab in enumerate(_LABELS)}
    base[true_label] = 0.9 - noise * ((i * 13) % 10) / 10.0
    return base


def _rows(n: int, noise: float = 0.3) -> list[tuple[dict[str, float], str]]:
    return [
        (_scores(i, _LABELS[i % 4], noise), _LABELS[i % 4])
        for i in range(n)
    ]


# ── conformal: the guarantee ────────────────────────────────────────────


def test_the_guarantee_holds_empirically() -> None:
    """The point of the method: coverage >= 1 - alpha on fresh data,
    whatever the score function's raw calibration looks like."""
    gate = fit(_rows(80, noise=0.5), alpha=0.1)
    assert gate.fitted, gate.explain()
    report = coverage_on(gate, _rows(200, noise=0.5))
    assert report["empirical_coverage"] >= 0.9 - 0.05  # finite-sample slack
    assert 0.0 < report["answer_rate"] <= 1.0


def test_guaranteed_coverage_is_computed_not_promised() -> None:
    """With n calibration points you can only guarantee rank/(n+1) coverage.
    The gate stores what n actually buys, never the request."""
    gate = fit(_rows(20), alpha=0.1)
    assert gate.fitted
    assert gate.guaranteed_coverage == pytest.approx(19 / 21)
    assert gate.guaranteed_coverage < 0.95, "n=20 cannot promise 95%"


def test_singleton_answers_multi_label_abstains() -> None:
    """First written with a tight gate and a 0.45/0.45 tie -- which produced an
    EMPTY set, i.e. "this input conforms to nothing seen in calibration".
    That is legitimate conformal output (and still an abstention), but the
    claim under test is the two-label set, so the gate is calibrated loose
    enough (noise=0.6 -> threshold below 0.45) for the tie to land inside it.
    """
    gate = fit(_rows(60, noise=0.6), alpha=0.1)
    confident = {"wireless": 0.95, "audio_visual": 0.02,
                 "low_voltage_cabling": 0.02, "staff_augmentation": 0.01}
    answer, pset = gate.decide(confident)
    assert answer == "wireless" and pset == {"wireless"}

    torn = {"wireless": 0.45, "low_voltage_cabling": 0.45,
            "audio_visual": 0.05, "staff_augmentation": 0.05}
    answer, pset = gate.decide(torn)
    assert answer is None
    assert {"wireless", "low_voltage_cabling"} <= pset, (
        "a two-label set is information for the queue, not just an abstention"
    )


@pytest.mark.parametrize(
    "rows, alpha, refusal_word",
    [
        (_rows(3), 0.1, "calibration points"),          # too few
        (_rows(5), 0.05, "unreachable"),                # 95% from n=5: impossible
        (_rows(20), 1.5, "miscoverage"),                # nonsense alpha
    ],
)
def test_the_gate_refuses_what_it_cannot_guarantee(rows, alpha, refusal_word) -> None:
    """The house rule, applied to a theorem: no number that means nothing."""
    gate = fit(rows, alpha=alpha)
    assert not gate.fitted
    assert any(refusal_word in r for r in gate.refusals), gate.refusals


def test_a_refused_gate_fails_closed() -> None:
    gate = ConformalGate(refusals=["not fitted"])
    answer, pset = gate.decide({"wireless": 0.99, "audio_visual": 0.01})
    assert answer is None
    assert pset == {"wireless", "audio_visual"}, "refusal means the full set, never a guess"


def test_label_space_mismatch_is_refused_not_absorbed() -> None:
    rows = _rows(20)
    rows[3] = ({"wireless": 0.9, "audio_visual": 0.1}, "datacenter")
    gate = fit(rows, alpha=0.1)
    assert not gate.fitted
    assert any("label spaces disagree" in r for r in gate.refusals)


# ── delta head: censoring handled, refusals enforced ────────────────────


class _Row:
    """Duck-typed store row (Correction / TrainingRow shapes)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_negatives_require_review_evidence() -> None:
    """The censoring problem: 'nobody corrected it' usually means 'nobody
    looked'. A negative must come from a deal a human demonstrably worked."""
    corrections = [_Row(scope_key="deal_A", relation="service_routing",
                        exemplars=["install access points everywhere"])]
    shipped = [
        _Row(deal_id="deal_A", relation="service_routing",
             raw_text="terminate drops in IDF 3"),          # reviewed deal -> negative
        _Row(deal_id="deal_UNTOUCHED", relation="service_routing",
             raw_text="replace the UPS batteries"),          # nobody looked -> nothing
        _Row(deal_id="deal_A", relation="service_routing",
             raw_text="install access points everywhere"),   # the corrected text itself
    ]
    examples = build_examples(corrections, shipped)
    positives = [e for e in examples if e.corrected]
    negatives = [e for e in examples if not e.corrected]
    assert len(positives) == 1
    assert [e.text for e in negatives] == ["terminate drops in IDF 3"]
    assert all(e.deal_id != "deal_UNTOUCHED" for e in examples), (
        "silence from an unopened deal is not agreement"
    )


def _toy_embed(texts):
    """Two separable clusters: 'wrong-ish' texts mention cameras."""
    out = []
    for t in texts:
        v = np.zeros(8)
        v[0] = 1.0 if "camera" in t else -1.0
        v[1 + (hash_stable(t) % 7)] = 0.3
        out.append(v)
    return np.array(out)


def hash_stable(t: str) -> int:
    import zlib

    return zlib.crc32(t.encode())


def test_the_probe_learns_a_separable_error_pattern() -> None:
    examples = (
        [DeltaExample(f"camera count wrong at site {i}", True, f"d{i}") for i in range(10)]
        + [DeltaExample(f"terminate drops in IDF {i}", False, f"d{i}") for i in range(10)]
    )
    head = DeltaHead(embed_fn=_toy_embed).fit(examples)
    assert head.fitted, head.explain()
    hi, lo = head.score(["camera quantity at the new site", "certify drops with a Fluke"])
    assert hi is not None and lo is not None and hi > 0.5 > lo


def test_the_queue_ranks_by_error_probability() -> None:
    examples = (
        [DeltaExample(f"camera row {i}", True, f"d{i}") for i in range(9)]
        + [DeltaExample(f"cabling row {i}", False, f"d{i}") for i in range(9)]
    )
    head = DeltaHead(embed_fn=_toy_embed).fit(examples)
    ranked = head.rank_for_review([
        ("deal_safe", "terminate and certify every drop"),
        ("deal_risky", "camera counts per building"),
    ])
    assert [d for d, _ in ranked] == ["deal_risky", "deal_safe"]


def test_too_few_corrections_refuses() -> None:
    examples = (
        [DeltaExample("camera one", True, "d1"), DeltaExample("camera two", True, "d2")]
        + [DeltaExample(f"cabling {i}", False, f"d{i}") for i in range(10)]
    )
    head = DeltaHead(embed_fn=_toy_embed).fit(examples)
    assert not head.fitted
    assert any("corrected examples" in r for r in head.refusals)
    assert head.score(["anything"]) == [None]
    assert head.rank_for_review([("d", "t")]) == [], (
        "an unfitted error model must not order anyone's work"
    )


def test_no_negatives_refuses_rather_than_learning_a_slogan() -> None:
    examples = [DeltaExample(f"camera {i}", True, f"d{i}") for i in range(10)]
    head = DeltaHead(embed_fn=_toy_embed).fit(examples)
    assert not head.fitted
    assert any("slogan" in r for r in head.refusals)


# ── the multitask table: the CPU half of the one-backbone bet ───────────


def test_multitask_assembly_dedups_splits_and_names_what_it_skips(tmp_path) -> None:
    import sqlite3

    from app.learning.multitask_table import assemble

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE training_rows (relation TEXT, label TEXT, raw_text TEXT,"
        " deal_id TEXT, split TEXT, teacher TEXT)"
    )
    conn.executemany(
        "INSERT INTO training_rows VALUES (?,?,?,?,?,?)",
        [
            ("atom_type", "exclusion", "Mid-turn jumpers are excluded.", "d1", "train", "llm"),
            # same (task, text) labelled by a PM -> the PM row must win the dedup
            ("atom_type", "exclusion", "Mid-turn jumpers are excluded.", "d1", "train", "pm"),
            ("atom_type", "constraint", "Escort access required at dock.", "d2", "holdout", "llm"),
            ("edge_relation", "contradicts", "qty 40 vs 56", "d1", "train", "llm"),  # not a backbone task
            ("atom_type", "scope_item", "x", "d3", "train", "llm"),  # too short
        ],
    )
    conn.commit()
    conn.close()

    table = assemble([db])
    assert len(table.rows) == 2
    winner = next(r for r in table.rows if "jumpers" in r.text)
    assert winner.teacher == "pm", "dedup must keep the most trusted teacher"
    assert table.skipped["duplicate (kept most trusted teacher)"] == 1
    assert table.skipped["relation edge_relation not a backbone task"] == 1
    assert table.skipped["text too short"] == 1
    stats = table.per_task()["atom_type"]
    assert stats["train"] == 1 and stats["holdout"] == 1 and stats["pm"] == 1

    out = tmp_path / "table.db"
    assert table.write(out) == 2
    n = sqlite3.connect(out).execute("select count(*) from multitask_rows").fetchone()[0]
    assert n == 2
