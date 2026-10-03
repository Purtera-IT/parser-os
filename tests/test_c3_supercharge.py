"""ml/c3 v6: the long labels teach the heads; the heads run alone (no training).

* the run-time model holds no teacher and no language model;
* every teaching term reaches the heads and none reaches the teacher;
* masking keeps only the chosen lines' labels and WHYs;
* the label-efficiency harness runs both arms on the same lines.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3.brain import Brain  # noqa: E402
from ml.c3.data import IGNORE, DealExample, featurize  # noqa: E402
from ml.c3.efficiency import run_curve  # noqa: E402
from ml.c3.lm import LMBase, TinyCausalLM  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402
from ml.c3.supercharge import mask_labels, supercharge_loss, teacher_view  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "ml" / "c3" / "fixtures"
SMALL = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def deal():
    return DealExample.load(HERE / "synthetic_deal.json")


@pytest.fixture(scope="module")
def batch(schema, deal):
    return featurize(deal, schema)


def test_the_heads_run_without_any_teacher_or_language_model(schema, batch):
    m = C3Model(schema, SMALL)
    assert not any(isinstance(x, (LMBase, Brain)) for x in m.modules())
    with torch.no_grad():
        out = m(batch.inputs(), company="purtera")
    assert out.logits and all(torch.isfinite(v).all() for v in out.logits.values())


def test_teaching_reaches_the_heads_and_never_the_teacher(schema, batch):
    torch.manual_seed(0)
    m = C3Model(schema, SMALL)
    m.train()
    teacher = Brain(schema, TinyCausalLM(32, 1, 2, 1024), context_lines=2)
    view = teacher_view(teacher, batch, "purtera")
    assert teacher.training                                 # mode restored
    keep = {i for i, w in enumerate(batch.why) if w}
    sub = mask_labels(batch, set(list(keep)[:4]))
    view = teacher_view(teacher, sub, "purtera")
    out = m(sub.inputs(), company="purtera")
    parts = supercharge_loss(m, out, view)
    for k in ("teach_heads", "teach_geometry", "teach_unlabeled"):
        assert parts[k] > 0, k
    sum(parts.values()).backward()
    assert m.heads["content"].proto.weight.grad.abs().sum() > 0
    assert m.rationale[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_masking_keeps_only_the_chosen_lines(batch):
    keep = {i for i, x in enumerate(batch.labeled) if x}
    keep = set(sorted(keep)[:2])
    sub = mask_labels(batch, keep)
    for i in range(len(batch)):
        if i not in keep:
            assert sub.why[i] is None and not sub.labeled[i]
            assert all(col[i] == IGNORE for col in sub.targets.values())
    assert any(sub.targets[k][i] != IGNORE for k in sub.targets for i in keep)


def test_the_efficiency_harness_compares_matched_arms(schema, deal):
    rows = run_curve(deal, schema, budgets=(2, 3), seeds=(0,), steps=2, teacher_steps=1, cfg=SMALL)
    assert {(r.budget, r.arm) for r in rows} == {(2, "labels"), (2, "explained"),
                                                 (3, "labels"), (3, "explained")}
    for k in (2, 3):
        a, b = [r for r in rows if r.budget == k]
        assert a.keep == b.keep and len(a.keep) == k and a.n_eval == b.n_eval > 0
        assert 0.0 <= a.accuracy <= 1.0 and 0.0 <= b.accuracy <= 1.0
