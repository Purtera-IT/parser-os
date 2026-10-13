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


def test_teacher_loss_in_pieces_covers_every_taught_line_and_card_once():
    from ml.c3.brain import Brain, told_why
    from ml.c3.lm import TinyCausalLM
    from ml.c3.losses import teacher_chunks, teacher_rows

    schema = load_schema()
    train, _ = _batches(schema)
    brain = Brain(schema, TinyCausalLM(32, 1, 2, 4096), page_batch=2, read_chunk=2)
    rows, _ = teacher_rows(train, told_why(train, schema))
    rows = list(range(len(train))) if rows is None else rows
    pieces = teacher_chunks(brain, train, 2)
    assert sorted(r for p in pieces for r in p[0]) == sorted(rows)
    assert sum(len(p[1]) for p in pieces) == len(train.judged)
    assert [p[2] for p in pieces].count(True) == 1
    assert all(len(p[0]) <= 2 and len(p[1]) <= 2 for p in pieces)
    assert teacher_chunks(brain, train, 0) == [(None, list(train.judged), True)]


def test_base_loop_runs_with_the_teacher_loss_in_pieces(tmp_path):
    schema = load_schema()
    train, hold = _batches(schema)
    _, result = run_base(
        [train], [hold], schema,
        Budget(teacher=1, lines=1, links=1, cards=1, numbers=1, deal=1, open_links=1, teach=1),
        page_batch=2, read_chunk=2, teacher_rows_per_step=2,
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
