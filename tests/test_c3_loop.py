"""The base loop trains the locked training deals and scores the holdout once."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3.coverage import READY_FOR_TRAINING  # noqa: E402
from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.gates import train as release_train  # noqa: E402
from ml.c3.loop import (Budget, gold_weights, load_deal, lock_split, require_live_teacher,  # noqa: E402
                       run_base, source_deal_id, training_batches)
from ml.c3.schema import GROUP, PAIR, load_schema  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "ml" / "c3" / "fixtures" / "synthetic_deal.json"


def _batches(schema):
    deal = DealExample.load(FIXTURE)
    train = featurize(deal, schema)
    hold = featurize(replace(deal, deal_id="synthetic-holdout"), schema)
    return train, hold


def test_a_holdout_deal_cannot_also_train():
    schema = load_schema()
    train, hold = _batches(schema)
    lock_split([train], [hold])
    with pytest.raises(ValueError, match="Holdout deal is in the training set"):
        lock_split([train], [train])


def test_only_a_question_seen_many_times_may_fill_a_blank_line():
    from ml.c3.loop import _lesson, frequent_heads

    counts = {"col:label_type": 40, "jdg:rare": 1, "rel:answers": 30}
    assert frequent_heads(counts) == {"col:label_type"}
    spread, keys = _lesson("spread", counts)
    assert keys == {"col:label_type"}
    assert spread.unlabeled > 0 and spread.rewrites == 0.0 and spread.flips == 0.0
    blank, allowed = _lesson("teach", counts)
    assert blank.flips == 0.0 and allowed is None
    bounds, _ = _lesson("flips", counts)
    assert bounds.flips > 0 and bounds.rewrites == 0.0


def test_low_rank_slot_matches_a_bfloat16_base():
    from ml.c3.lm import LoRALinear

    base = torch.nn.Linear(4, 3, bias=False).bfloat16()
    slot = LoRALinear(base, rank=2)
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    y = slot(x)
    assert y.dtype == torch.bfloat16
    assert torch.allclose(y.float(), base(x).float(), atol=1e-2)


def test_low_rank_slot_matches_its_base_until_it_is_trained():
    from ml.c3.lm import LoRALinear

    base = torch.nn.Linear(4, 3, bias=False)
    slot = LoRALinear(base, rank=2)
    x = torch.randn(2, 4)
    assert torch.allclose(slot(x), base(x))


def test_sliced_teacher_pages_match_one_pass():
    schema = load_schema()
    batch = featurize(DealExample.load(FIXTURE), schema)
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM

    torch.manual_seed(0)
    one = Brain(schema, TinyCausalLM(32, 1, 2, 4096), context_lines=2)
    other = Brain(schema, TinyCausalLM(32, 1, 2, 4096), context_lines=2, page_batch=3)
    other.load_state_dict(one.state_dict())
    one.eval()
    other.eval()
    with torch.no_grad():
        left = one(batch)
        right = other(batch)
    for key, logits in left.logits.items():
        assert torch.allclose(logits, right.logits[key], atol=1e-5)


def test_the_frozen_teacher_keeps_dropout_off_and_checkpointing_on():
    from ml.c3.loop import _arm_teacher_memory, _steady_teacher

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout(0.5)
            self.config = type("Cfg", (), {"use_cache": True})()
            self.checkpointed = False
            self.inputs = False

        def gradient_checkpointing_enable(self):
            self.checkpointed = True

        def enable_input_require_grads(self):
            self.inputs = True

    class Teacher:
        def __init__(self):
            self.lm = type("Lm", (), {})()
            self.lm.model = Body()

    teacher = Teacher()
    _arm_teacher_memory(teacher)
    teacher.lm.model.eval()
    _steady_teacher(teacher)
    assert teacher.lm.model.checkpointed and teacher.lm.model.inputs
    assert teacher.lm.model.training
    assert not teacher.lm.model.drop.training
    assert teacher.lm.model.config.use_cache is False


def test_a_real_run_must_name_its_teacher():
    require_live_teacher("some-org/some-causal-lm", tiny=False)
    require_live_teacher("", tiny=True)
    with pytest.raises(ValueError, match="pretrained causal LM"):
        require_live_teacher("", tiny=False)


def test_labeling_export_and_joined_deal_both_load(tmp_path):
    joined = load_deal(FIXTURE)
    assert joined.deal_id == "synthetic-0001"
    export = tmp_path / "export.json"
    export.write_text(
        '{"deal_id":"blob-1","company":"purtera","company_policy":"Two techs over 55 inches.",'
        '"atoms":[{"label_key":"a","atom_id":"at-a",'
        '"text":"Install cameras."}],"labels":[{"label_key":"a","labeler":"expert@example.com",'
        '"label_type":"task","text":"Install cameras."}],"deal_answers":[]}',
        encoding="utf-8")
    deal = load_deal(export)
    assert deal.deal_id == "blob-1"
    assert deal.atoms[0].label["label_type"] == "task"
    assert deal.company_policy == "Two techs over 55 inches."


def test_a_long_deal_stays_one_source_and_cannot_leak_into_holdout():
    schema = load_schema()
    deal = DealExample.load(FIXTURE)
    parts, _notes = training_batches(deal, schema, max_atoms=4)
    assert len(parts) > 1
    assert {source_deal_id(part.deal_id) for part in parts} == {deal.deal_id}
    hold = featurize(replace(deal, deal_id="synthetic-holdout"), schema)
    lock_split(parts, [hold])
    with pytest.raises(ValueError, match="Holdout deal is in the training set"):
        lock_split(parts, [featurize(deal, schema)])


def test_card_stages_teach_pairs_before_the_whole_deal():
    _, card_sizes = gold_weights("cards")
    _, deal_sizes = gold_weights("deal")
    assert card_sizes == (PAIR, GROUP)
    assert deal_sizes is None
    line_w, _ = gold_weights("lines")
    assert line_w.relations == 0.0 and line_w.judgments == 0.0 and line_w.deal_numbers == 0.0
    open_w, _ = gold_weights("deal")
    assert open_w.relations > 0 and open_w.deal_numbers > 0


def test_base_loop_trains_only_the_training_deals_and_opens_the_link_gate(tmp_path):
    schema = load_schema()
    train, hold = _batches(schema)
    seen: list[str] = []
    import ml.c3.loop as loop

    real_c3 = loop.c3_loss
    real_brain = loop.brain_loss

    def watch_c3(model, batch, *args, **kwargs):
        seen.append(batch.deal_id)
        return real_c3(model, batch, *args, **kwargs)

    def watch_brain(brain, batch, *args, **kwargs):
        seen.append("teacher:" + batch.deal_id)
        return real_brain(brain, batch, *args, **kwargs)

    loop.c3_loss = watch_c3
    loop.brain_loss = watch_brain
    try:
        model, result = run_base(
            [train], [hold], schema,
            Budget(teacher=1, lines=1, links=1, cards=1, numbers=1, deal=1, open_links=1, teach=1),
            checkpoint=str(tmp_path / "base.pt"),
        )
    finally:
        loop.c3_loss = real_c3
        loop.brain_loss = real_brain
    assert seen
    assert all(deal_id.endswith(train.deal_id) or deal_id == train.deal_id for deal_id in seen)
    assert hold.deal_id not in seen
    assert result.holdout_ids == (hold.deal_id,)
    assert result.holdout_used_for_tuning is False
    assert result.unlabeled_enabled is False
    assert result.ready_for_training is False
    assert READY_FOR_TRAINING is False
    from ml.c3.loop import hold_line_encoder
    assert hold_line_encoder(model) is False
    assert model.link_mix.weight.requires_grad
    assert float(model.link_mix.weight.detach().abs().sum()) > 0
    assert result.heads_seen
    assert hold.deal_id in result.holdout_scores
    raw = (tmp_path / "base.pt").read_bytes()
    assert b"Bayview" not in raw
    with pytest.raises(RuntimeError, match="READY_FOR_TRAINING is false"):
        release_train()


def test_teacher_loss_in_pieces_covers_every_taught_line_card_and_flip_once():
    from ml.c3.brain import Brain, told_why
    from ml.c3.lm import TinyCausalLM
    from ml.c3.losses import supposed_twins, teacher_chunks, teacher_rows

    schema = load_schema()
    train, _ = _batches(schema)
    brain = Brain(schema, TinyCausalLM(32, 1, 2, 4096), page_batch=2, read_chunk=2)
    rows, _ = teacher_rows(train, told_why(train, schema))
    rows = list(range(len(train))) if rows is None else rows
    budget = 3000
    pieces = teacher_chunks(brain, train, 4, budget)
    assert sorted(r for p in pieces for r in p.rows) == sorted(rows)
    assert sum(len(p.cards) for p in pieces) == len(train.judged)
    assert sorted(f for p in pieces for f in p.flips) == list(range(len(supposed_twins(train)[0])))
    assert [p.extras for p in pieces].count(True) == 1 and pieces[0].extras
    for p in pieces:
        kinds = [bool(p.rows), bool(p.cards), bool(p.flips)]
        assert sum(kinds) <= 1                      # one kind of term per piece
        size = len(p.rows) + len(p.cards) + len(p.flips)
        assert size <= 4 and (p.tokens <= budget or size == 1)
    for kind in ("rows", "cards", "flips"):
        mine = [p for p in pieces if getattr(p, kind)]
        if mine:
            assert abs(sum(p.scale for p in mine) - 1.0) < 1e-9
    whole = teacher_chunks(brain, train, 0)
    assert len(whole) == 1 and whole[0].rows is None and whole[0].extras


def test_every_teacher_pass_stays_under_the_call_cap():
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM
    from ml.c3.losses import BrainWeights, brain_loss, teacher_chunks, teacher_context

    calls: list[tuple[int, int]] = []

    class Counting(TinyCausalLM):
        def forward(self, ids, mask, prefix=None, inputs_embeds=None):
            calls.append(tuple(ids.shape))
            return super().forward(ids, mask, prefix=prefix, inputs_embeds=inputs_embeds)

    schema = load_schema()
    train, _ = _batches(schema)
    cap = 2048
    lm = Counting(32, 1, 2, 8192)
    lm.EMBED_TOKENS = cap
    brain = Brain(schema, lm, page_batch=8, read_chunk=4, call_tokens=cap)
    context = teacher_context(brain, train)
    for part in teacher_chunks(brain, train, 4, 4000):
        loss, _ = brain_loss(brain, train, BrainWeights(), rows=part.rows, judged=part.cards,
                             flips=part.flips, extras=part.extras, context=context)
        if loss.requires_grad:
            loss.backward()
    assert calls
    assert all(n * t <= cap or n == 1 for n, t in calls), max(calls, key=lambda c: c[0] * c[1])
    assert any(n > 1 for n, _ in calls)


def test_capped_calls_read_the_same_as_one_call():
    import torch
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM

    schema = load_schema()
    train, _ = _batches(schema)
    torch.manual_seed(0)
    one = Brain(schema, TinyCausalLM(32, 1, 2, 4096))
    many = Brain(schema, TinyCausalLM(32, 1, 2, 4096), page_batch=3, call_tokens=1500)
    many.load_state_dict(one.state_dict())
    one.eval()
    many.eval()
    with torch.no_grad():
        a, b = one(train), many(train)
        assert max(float((a.logits[k] - b.logits[k]).abs().max()) for k in a.logits) < 1e-4
        assert abs(float(one.value_loss(train)) - float(many.value_loss(train))) < 1e-4
        assert abs(float(one.rationale_loss(train)) - float(many.rationale_loss(train))) < 1e-4


def test_base_loop_runs_with_the_teacher_loss_in_pieces(tmp_path):
    schema = load_schema()
    train, hold = _batches(schema)
    _, result = run_base(
        [train], [hold], schema,
        Budget(teacher=1, lines=1, links=1, cards=1, numbers=1, deal=1, open_links=1, teach=1),
        page_batch=2, read_chunk=2, teacher_rows_per_step=2, teacher_piece_tokens=3000,
        call_tokens=2048,
    )
    assert result.ready_for_training is False
    assert hold.deal_id in result.holdout_scores


def test_value_loss_skips_a_piece_with_no_value_lines_and_supervises_every_value_token():
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM

    schema = load_schema()
    train, _ = _batches(schema)
    brain = Brain(schema, TinyCausalLM(32, 1, 2, 4096))
    valued = {item.line for item in train.literals if str(item.text or "").strip()}
    bare = [i for i in range(len(train)) if i not in valued]
    assert valued and bare
    zero = brain.value_loss(train, rows=bare[:2], deal_level=False)
    assert float(zero) == 0.0 and not zero.requires_grad
    assert float(brain.value_loss(train, rows=sorted(valued & set(range(len(train))))).detach()) > 0

    ids, mask, start = brain.lm.tokenize_pairs(["Line one\nValue:", "x\nValue:"], [" 12 cameras", " abc"])
    assert start.tolist() == [1 + len("Line one\nValue:"), 1 + len("x\nValue:")]
    assert mask.sum(1).tolist() == [int(start[0]) + len(" 12 cameras"), int(start[1]) + len(" abc")]


def test_value_loss_counts_values_that_encode_to_nothing():
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM

    class Mute(TinyCausalLM):
        def encode(self, text, first):
            return super().encode(text, first) if first else []

    schema = load_schema()
    train, _ = _batches(schema)
    brain = Brain(schema, Mute(32, 1, 2, 4096))
    with pytest.raises(ValueError, match="encodes to no tokens"):
        brain.value_loss(train)


def test_student_memory_savers_give_the_same_loss_and_gradients(monkeypatch):
    import torch
    import ml.c3.lattice as lattice
    import ml.c3.model as model_mod
    from ml.c3.losses import c3_loss
    from ml.c3.model import C3Config, C3Model

    schema = load_schema()
    train, _ = _batches(schema)

    def run(light: bool):
        monkeypatch.setattr(model_mod, "CHECKPOINT_LINES", 1 if light else 10**9)
        monkeypatch.setattr(lattice, "CHECKPOINT_LINES", 1 if light else 10**9)
        monkeypatch.setattr(model_mod, "BOX_ROWS", 3)
        torch.manual_seed(0)
        m = C3Model(schema, replace(C3Config(), asked_layers=1))
        m.eval()                                     # no dropout; gradients still on
        loss, _ = c3_loss(m, train)
        loss.backward()
        return float(loss.detach()), {n: p.grad.clone() for n, p in m.named_parameters()
                                      if p.grad is not None}

    a_loss, a_grad = run(False)
    b_loss, b_grad = run(True)
    assert abs(a_loss - b_loss) < 1e-5
    assert a_grad.keys() == b_grad.keys()
    assert max(float((a_grad[k] - b_grad[k]).abs().max()) for k in a_grad) < 1e-4


@pytest.mark.parametrize("encoder_lr", [None, 2e-5])
def test_every_stage_steps_under_the_pods_bf16_autocast(monkeypatch, tmp_path, encoder_lr):
    """One real optimizer step of every stage, teacher and student, under
    bf16 autocast, the way the pod runs (A: frozen encoder; B: encoder lr).
    CPU autocast allows ops CUDA autocast refuses, so those are made to
    refuse here too."""
    import torch
    import torch.nn.functional as F
    import ml.c3.loop as loop

    real_bce = F.binary_cross_entropy

    def cuda_rules(*args, **kwargs):
        if torch.is_autocast_enabled("cpu"):
            raise RuntimeError("binary_cross_entropy is unsafe to autocast")
        return real_bce(*args, **kwargs)

    monkeypatch.setattr(F, "binary_cross_entropy", cuda_rules)
    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy", cuda_rules)
    monkeypatch.setattr(loop, "CPU_AUTOCAST", True)
    schema = load_schema()
    train, hold = _batches(schema)
    check = featurize(replace(DealExample.load(FIXTURE), deal_id="synthetic-validation"), schema)
    _, result = run_base(
        [train], [hold], schema,
        Budget(teacher=1, lines=1, links=1, cards=1, numbers=1, deal=1, open_links=1,
               bind=1, teach=1, flips=1, spread=1),
        checkpoint=str(tmp_path / "base.pt"), adamw=True, freeze_teacher=True,
        deal_graph=True, page_batch=2, read_chunk=2, call_tokens=2048,
        teacher_rows_per_step=4, teacher_piece_tokens=4000, encoder_lr=encoder_lr,
        teacher_caps=(5, 3, 2), view_ground_rows=3,
        validation=[check], score_holdout=False,
    )
    assert result.ready_for_training is False


def test_desc_once_per_part_gives_the_same_gradient_as_per_piece():
    import torch
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM
    from ml.c3.loop import _teacher_part
    from ml.c3.losses import BrainWeights, brain_loss, teacher_chunks, teacher_context

    schema = load_schema()
    train, _ = _batches(schema)
    torch.manual_seed(0)
    a = Brain(schema, TinyCausalLM(32, 1, 2, 4096), page_batch=2)
    b = Brain(schema, TinyCausalLM(32, 1, 2, 4096), page_batch=2)
    b.load_state_dict(a.state_dict())
    a.eval()
    b.eval()
    w = BrainWeights(consolidate=0.0, flip=0.0)
    _teacher_part(a, train, w, device="cpu", scale=1.0, size=4, budget=4000, caps=(0, 0, 0), rng=None)
    context = teacher_context(b, train)
    for part in teacher_chunks(b, train, 4, 4000):
        loss, _ = brain_loss(b, train, w, rows=part.rows, judged=part.cards, flips=part.flips,
                             extras=part.extras, context=context)
        (loss * part.scale).backward()
    ga = {n: p.grad for n, p in a.named_parameters() if p.grad is not None}
    gb = {n: p.grad for n, p in b.named_parameters() if p.grad is not None}
    assert ga.keys() == gb.keys() and ga
    assert max(float((ga[k] - gb[k]).abs().max()) for k in ga) < 1e-4


def test_teacher_caps_sample_a_fresh_bounded_slice_each_step():
    import random
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM
    from ml.c3.losses import teacher_chunks

    schema = load_schema()
    train, _ = _batches(schema)
    brain = Brain(schema, TinyCausalLM(32, 1, 2, 4096))
    rng = random.Random(0)
    first = teacher_chunks(brain, train, 4, 4000, caps=(3, 2, 1), rng=rng)
    second = teacher_chunks(brain, train, 4, 4000, caps=(3, 2, 1), rng=rng)
    for pieces in (first, second):
        assert sum(len(p.rows) for p in pieces) <= 3
        assert sum(len(p.cards) for p in pieces) <= 2
        assert sum(len(p.flips) for p in pieces) <= 1
    assert ([r for p in first for r in p.rows] != [r for p in second for r in p.rows]
            or len(train) <= 3)
