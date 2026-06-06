"""Retrainer eval-gate — does accumulating data make a *safely better* served
head? (#72)

These tests drive :func:`app.learning.retrain.retrain_relation` end-to-end over
a synthetic training log with a deterministic, network-free ``embed_fn``. They
prove the contract that keeps the self-improving loop honest:

  * **promote_first** — the first head that clears the readiness bar becomes the
    champion;
  * **skip_no_new_data** — re-running with an unchanged data signature is a
    no-op (no churn, no needless re-promotion);
  * **promote_better** — more clean data that does not regress the champion is
    promoted;
  * **hold_not_ready** — a candidate that fails the absolute bar is NEVER served
    (a bad candidate can't reach serve, with or without an incumbent);
  * **_is_regression** — the gate logic refuses any candidate worse than the
    champion on accuracy, coverage, or gold-accuracy beyond tolerance.

Splits are by deal-id hash (leave-one-deal-out), so high holdout accuracy means
the head learned the rule, not the names — exactly what the gate credits.
"""

from __future__ import annotations

import hashlib

import numpy as np

from app.core.shadow_eval import RelationReport
from app.core.training_log import TEACHER_LLM, TrainingLog, TrainingRow, assign_split
from app.learning.head_registry import HeadRegistry
from app.learning.retrain import (
    _is_regression,
    data_signature,
    fit_candidate,
    retrain_relation,
)

_D = 24


def _embed(texts):
    """Deterministic embeddings: token 'AAA' → axis 0 cluster, 'BBB' → axis 1,
    'NNN' → random direction (the un-learnable, gate-failing relation). Plus a
    tiny per-text hash jitter so points are distinct but cluster cleanly."""
    out = np.zeros((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        v = np.zeros(_D, dtype=np.float32)
        if "AAA" in t:
            v[0] = 3.0
        elif "BBB" in t:
            v[1] = 3.0
        h = hashlib.sha256(t.encode("utf-8")).digest()
        for j in range(2, _D):
            v[j] = (h[j % len(h)] / 255.0 - 0.5) * 0.08
        if "NNN" in t:
            # fully hash-driven direction → not separable by label → gate fails
            for j in range(_D):
                v[j] = (h[j % len(h)] / 255.0 - 0.5)
        out[i] = v
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 0, n, 1.0)


def _separable_rows(relation, deal_ids, *, swap=False, token_for=None):
    """Build clean two-class rows (type_a↔AAA, type_b↔BBB) across deals.

    ``swap`` flips the text↔label mapping (to inject a wrong-labeled candidate).
    """
    rows = []
    for di in deal_ids:
        for k in range(3):  # 3 exemplars/class/deal → clears min_per_class
            for label, tok in (("type_a", "AAA"), ("type_b", "BBB")):
                lab = label
                if swap:
                    lab = "type_b" if label == "type_a" else "type_a"
                rows.append(TrainingRow(
                    relation=relation, label=lab,
                    raw_text=f"{tok} row {di} {k}",
                    masked_text=f"feature {tok} item {di} {k}",
                    teacher=TEACHER_LLM, deal_id=di,
                ))
    return rows


def _deal_ids(n, prefix=""):
    return [f"{prefix}deal_{i}" for i in range(n)]


def _split_counts(deal_ids):
    h = sum(1 for d in deal_ids if assign_split(d) == "holdout")
    return h, len(deal_ids) - h


def _fresh_log():
    return TrainingLog(":memory:")


def _seed_clean(log, relation, n_deals=60):
    """Seed enough deals that the holdout split has >= _MIN_HOLDOUT rows."""
    ids = _deal_ids(n_deals)
    log.add_many(_separable_rows(relation, ids))
    return ids


# ── the gate's pure logic ────────────────────────────────────────────────────
def test_is_regression_detects_each_axis():
    def rep(n_answered, n_correct, n_gold_correct, n_gold=10, n_ho=100):
        # explicit counts → exact ratios (no rounding drift)
        r = RelationReport(relation="x", n_holdout=n_ho)
        r.n_answered = n_answered
        r.n_correct = n_correct
        r.n_gold = n_gold
        r.n_gold_correct = n_gold_correct
        return r

    champ = rep(80, 78, 10)              # acc .975, cov .80, gold 1.0
    # equal → not a regression
    assert _is_regression(rep(80, 78, 10), champ, 0.01) is False
    # worse accuracy (.875 vs .975)
    assert _is_regression(rep(80, 70, 10), champ, 0.01) is True
    # worse coverage (.60 vs .80)
    assert _is_regression(rep(60, 59, 10), champ, 0.01) is True
    # worse gold accuracy (.80 vs 1.0)
    assert _is_regression(rep(80, 78, 8), champ, 0.01) is True
    # within tolerance on every axis → not a regression
    assert _is_regression(rep(79, 77, 10), champ, 0.01) is False


def test_data_signature_changes_with_new_rows():
    log = _fresh_log()
    _seed_clean(log, "atom_type", n_deals=10)
    s1 = data_signature(log, "atom_type")
    log.add_many(_separable_rows("atom_type", ["extra_deal"]))
    s2 = data_signature(log, "atom_type")
    assert s1 != s2


def test_fit_candidate_returns_trained_head():
    log = _fresh_log()
    _seed_clean(log, "atom_type")
    out = fit_candidate(log, "atom_type", _embed)
    assert out is not None
    head, n_train = out
    assert n_train > 0
    assert head.trained  # separable + enough per class


# ── end-to-end promotion lifecycle ───────────────────────────────────────────
def test_promote_first_then_skip_no_new_data(tmp_path):
    log = _fresh_log()
    _seed_clean(log, "atom_type")
    reg = HeadRegistry(str(tmp_path / "reg"))

    r1 = retrain_relation(log, "atom_type", reg, _embed, embed_model="emb-test")
    assert r1.action == "promote_first"
    assert r1.promoted is True
    assert reg.champion_version("atom_type") == r1.candidate_version
    assert r1.candidate["ready"] is True

    # No data change → must be a clean no-op (champion stays).
    r2 = retrain_relation(log, "atom_type", reg, _embed, embed_model="emb-test")
    assert r2.action == "skip_no_new_data"
    assert r2.promoted is False
    assert reg.champion_version("atom_type") == r1.candidate_version


def test_promote_better_on_new_clean_data(tmp_path):
    log = _fresh_log()
    _seed_clean(log, "atom_type", n_deals=60)
    reg = HeadRegistry(str(tmp_path / "reg"))

    r1 = retrain_relation(log, "atom_type", reg, _embed, embed_model="emb-test")
    assert r1.promoted is True
    first_version = r1.candidate_version

    # Add more clean deals → signature changes, candidate still clears bar and
    # does not regress → promoted (>= champion accuracy counts as "better").
    log.add_many(_separable_rows("atom_type", _deal_ids(20, prefix="more_")))
    r2 = retrain_relation(log, "atom_type", reg, _embed, embed_model="emb-test")
    assert r2.action in ("promote_better", "promote_stale_champion")
    assert r2.promoted is True
    assert reg.champion_version("atom_type") == r2.candidate_version
    assert reg.champion_version("atom_type") != first_version


def test_hold_not_ready_keeps_no_champion(tmp_path):
    log = _fresh_log()
    # 'NNN' rows are not separable by label → candidate cannot clear the bar.
    ids = _deal_ids(60)
    rows = []
    rng = np.random.default_rng(0)
    for di in ids:
        for k in range(6):
            label = "type_a" if rng.random() < 0.5 else "type_b"
            rows.append(TrainingRow(
                relation="messy", label=label,
                raw_text=f"NNN {di} {k}",
                masked_text=f"NNN noise {di} {k}",
                teacher=TEACHER_LLM, deal_id=di,
            ))
    log.add_many(rows)
    reg = HeadRegistry(str(tmp_path / "reg"))

    r = retrain_relation(log, "messy", reg, _embed, embed_model="emb-test")
    assert r.action == "hold_not_ready"
    assert r.promoted is False
    assert reg.champion_version("messy") is None  # bad candidate never serves
    # but the candidate IS registered for audit
    assert r.candidate_version in [m.version for m in reg.history("messy")]


def test_hold_champion_better_refuses_regression(tmp_path):
    """A wrong-labeled candidate (worse on unseen deals) must NOT replace a good
    champion. We promote a clean champion, then retrain on data whose new rows
    flip the mapping — the candidate regresses and is held."""
    log = _fresh_log()
    clean_ids = _seed_clean(log, "atom_type", n_deals=60)
    reg = HeadRegistry(str(tmp_path / "reg"))

    r1 = retrain_relation(log, "atom_type", reg, _embed, embed_model="emb-test")
    assert r1.promoted is True
    champ_version = r1.candidate_version

    # Flood TRAIN with swapped-label rows so the candidate learns a corrupted
    # boundary; holdout labels stay correct → candidate accuracy collapses.
    swap_ids = _deal_ids(120, prefix="swap_")
    swap_train = [d for d in swap_ids if assign_split(d) == "train"]
    log.add_many(_separable_rows("atom_type", swap_train, swap=True))

    r2 = retrain_relation(log, "atom_type", reg, _embed, embed_model="emb-test")
    assert r2.promoted is False
    assert r2.action in ("hold_champion_better", "hold_not_ready")
    # Champion is untouched — the good head still serves.
    assert reg.champion_version("atom_type") == champ_version
