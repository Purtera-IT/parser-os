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


def test_the_teacher_reads_what_each_why_rests_on(schema, batch):
    torch.manual_seed(0)
    teacher = Brain(schema, TinyCausalLM(32, 1, 2, 4096), context_lines=2)
    pointers, words = teacher.grounding(batch, window=4)
    for i in range(len(batch)):
        if batch.why[i]:
            assert torch.isclose(pointers[i].sum(), torch.tensor(1.0))
            assert words[i] is not None and torch.isclose(words[i].sum(), torch.tensor(1.0))
        else:
            assert pointers[i].sum() == 0 and words[i] is None
    # The hardware line's WHY reaches back to other lines (the SOW scope line
    # sits inside its window), not only to the line itself.
    hw = next(i for i, t in enumerate(batch.texts) if "XD65" in t)
    assert pointers[hw, :-1].sum() > 0


def test_pointers_and_words_reach_the_heads(schema, batch):
    torch.manual_seed(0)
    m = C3Model(schema, SMALL)
    m.train()
    teacher = Brain(schema, TinyCausalLM(32, 1, 2, 4096), context_lines=2)
    view = teacher_view(teacher, batch, "purtera")
    out = m(batch.inputs(), company="purtera")
    parts = supercharge_loss(m, out, view, texts=batch.texts)
    assert parts["teach_pointers"] > 0 and parts["teach_words"] > 0
    (parts["teach_pointers"] + parts["teach_words"]).backward()
    for p in (m.cite_q.weight, m.cite_k.weight, m.word_r.weight, m.word_w.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
    # Foresight: a line can only point at earlier lines.
    s = out.pointers[:, :-1]
    assert torch.isinf(s[torch.triu(torch.ones_like(s, dtype=torch.bool))]).all()


def test_parser_feedback_never_reaches_training(batch):
    from ml.c3.notes import drop_meta

    from ml.c3.notes import split_note

    note = ("Qty 4 displays = 4 units of mount work; the price does not scope the job.\n"
            "[purtera] keep: crew planning.\n"
            "[parser] SHOULD SPLIT: two quote lines in one atom.")
    why, policy = split_note(note)
    assert drop_meta(why) == why and "[parser]" not in why
    assert drop_meta(policy) == "keep: crew planning."
    assert drop_meta("Real reason.\n[parser] one atom per line (fixed).") == "Real reason."
    assert not any("[parser]" in (t or "").lower() or "qty column" in (t or "")
                   for t in batch.why + batch.policy_note)
