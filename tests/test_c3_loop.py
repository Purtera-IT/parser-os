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
    # Queried line judgments (the Questions head) train with the cards too.
    assert card_sizes == (PAIR, GROUP, "line")
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
    from ml.c3.data import Judged

    schema = load_schema()
    train, hold = _batches(schema)
    check = featurize(replace(DealExample.load(FIXTURE), deal_id="synthetic-validation"), schema)
    # The fixture has no cards; the card paths (teacher, lessons, scoring) must run too.
    keys = [o.key for o in schema.opportunities if o.key.startswith("jdg:") and o.universal
            and not o.multiple]
    cards = [Judged(keys[0], "PAIR", (0, 1), 0, note="the pair agrees", query="Do these agree?",
                    subject_key="p"),
             Judged(keys[-1], "GROUP", (0, 1, 2), 0, note="the group is one ask", query="One ask?",
                    subject_key="g")]
    train, hold, check = (replace(b, judged=cards) for b in (train, hold, check))
    _, result = run_base(
        [train], [hold], schema,
        Budget(teacher=1, lines=1, links=1, cards=1, numbers=1, deal=1, open_links=1,
               bind=1, teach=1, flips=1, spread=1),
        checkpoint=str(tmp_path / "base.pt"), adamw=True, freeze_teacher=True,
        deal_graph=True, page_batch=2, read_chunk=2, call_tokens=2048,
        teacher_rows_per_step=4, teacher_piece_tokens=4000, encoder_lr=encoder_lr,
        teacher_caps=(5, 3, 2), view_ground_rows=3, view_parts=1,
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


def test_an_overlong_page_drops_context_first_and_an_overlong_card_is_left_out():
    from ml.c3.brain import Brain, teacher_cards
    from ml.c3.lm import TinyCausalLM
    from ml.c3.losses import teacher_chunks

    schema = load_schema()
    train, _ = _batches(schema)
    i = max(range(len(train)), key=lambda k: k if k >= 3 else -1)
    wide = Brain(schema, TinyCausalLM(32, 1, 2, 8192), context_lines=3)
    full = wide.page(train, i, ["a note"])
    need = len(train.texts[i]) + 200
    tight = Brain(schema, TinyCausalLM(32, 1, 2, need), context_lines=3)
    page = tight.page(train, i, ["a note"], "Answer:")
    assert train.texts[i] in page and tight.fits(page + "Answer:")
    assert len(page) < len(full)
    from ml.c3.data import Judged
    from ml.c3.model import C3Model
    from ml.c3.supercharge import supercharge_loss, teacher_view

    key = next(o.key for o in schema.opportunities if o.key.startswith("jdg:") and o.universal
               and not o.multiple)
    short = Judged(key, "PAIR", (0, 1), 0, note="short reason", subject_key="s")
    long_ = Judged(key, "PAIR", (0, 1), 0, note="x" * 9000, subject_key="l")
    carded = replace(train, judged=[short, long_])
    assert wide.card_fits(carded, short) and not wide.card_fits(carded, long_)
    assert teacher_cards(wide, carded) == [short]
    pieces = teacher_chunks(wide, carded, 4, 4000)
    assert [c for p in pieces for c in p.cards] == [short]
    view = teacher_view(wide, carded, grounding=False)
    assert len(view.card_subjects[key]) == 1
    model = C3Model(schema)
    out = model(carded.inputs())
    parts = supercharge_loss(model, out, view, texts=carded.texts, batch=carded)
    assert "teach_judgments" in parts


def test_budget_check_reads_every_teacher_page_of_a_part():
    from ml.c3.budget_check import check
    from ml.c3.lm import TinyCausalLM

    schema = load_schema()
    train, _ = _batches(schema)
    counts, bad = check([train], schema, TinyCausalLM(32, 1, 2, 8192))
    assert counts["parts"] == 1 and counts["pages"] >= 2 * len(train) and not bad
    tight_counts, tight_bad = check([train], schema, TinyCausalLM(32, 1, 2, 120))
    assert tight_counts["pages_fitted"] > 0 or tight_bad


def test_the_trained_teacher_is_saved_and_loads_back(tmp_path):
    import torch
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM
    from ml.c3.loop import load_teacher, save_teacher

    schema = load_schema()
    torch.manual_seed(0)
    a = Brain(schema, TinyCausalLM(32, 1, 2, 4096))
    path = str(tmp_path / "base.pt.teacher")
    save_teacher(path, a, "teacher")
    torch.manual_seed(1)
    b = Brain(schema, TinyCausalLM(32, 1, 2, 4096))
    assert load_teacher(path, b) == "teacher"
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), n


def test_a_run_saves_its_teacher_and_a_later_run_reuses_it(tmp_path):
    schema = load_schema()
    train, hold = _batches(schema)
    budget = Budget(teacher=1, lines=1, links=0, cards=0, numbers=0, deal=0, open_links=0, teach=1)
    run_base([train], [hold], schema, budget, checkpoint=str(tmp_path / "a.pt"))
    saved = tmp_path / "a.pt.teacher"
    assert saved.exists()
    _, result = run_base([train], [hold], schema, budget, checkpoint=str(tmp_path / "b.pt"),
                         teacher_from=str(saved))
    assert "teacher" not in result.stage_loss


def test_a_lesson_step_runs_the_trunk_and_the_descriptions_once():
    """With a trainable encoder every trunk pass and every describe() is held
    for backward, so the lesson must reuse the gold terms' pass."""
    from ml.c3.brain import Brain
    from ml.c3.data import Judged
    from ml.c3.lm import TinyCausalLM
    from ml.c3.loop import _student_step
    from ml.c3.losses import LossWeights
    from ml.c3.model import C3Model
    from ml.c3.supercharge import teacher_view

    schema = load_schema()
    train, _ = _batches(schema)
    keys = [o.key for o in schema.opportunities if o.key.startswith("jdg:") and o.universal
            and not o.multiple][:2]
    cards = [Judged(k, "PAIR", (0, 1), 0, note="short reason", subject_key=f"s{n}")
             for n, k in enumerate(keys)]
    carded = replace(train, judged=cards)
    view = teacher_view(Brain(schema, TinyCausalLM(32, 1, 2, 8192), context_lines=3),
                        carded, grounding=False)
    assert len(view.card_logits) == 2
    model = C3Model(schema)
    calls = {"describe": 0, "forward": 0}
    describe, forward = model.describe, model.forward

    def counted_describe():
        calls["describe"] += 1
        return describe()

    def counted_forward(*a, **k):
        calls["forward"] += 1
        return forward(*a, **k)

    model.describe, model.forward = counted_describe, counted_forward
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    # keyword_dropout (why_carry.py) is a second trunk pass by design; off here.
    loss = _student_step(model, opt, carded, LossWeights(keyword_dropout=0.0), None, object(), view, "cpu")
    assert torch.isfinite(loss)
    assert calls == {"describe": 1, "forward": 1}


def test_an_overlong_line_is_encoded_in_windows_with_no_token_dropped():
    from ml.c3.text import HFEncoder

    class Tok:
        cls_token_id, sep_token_id = 1, 2

        def __call__(self, texts, truncation=False):
            return {"input_ids": [[1] + [3 + len(w) for w in t.split()] + [2] for t in texts]}

        def pad(self, enc, return_tensors="pt"):
            rows = enc["input_ids"]
            width = max(len(r) for r in rows)

            class Enc(dict):
                def to(self, dev):
                    return self
            return Enc(input_ids=torch.tensor([r + [0] * (width - len(r)) for r in rows]),
                       attention_mask=torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]))

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(64, 4)

        def forward(self, input_ids, attention_mask):
            from types import SimpleNamespace
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    enc = HFEncoder.__new__(HFEncoder)
    torch.nn.Module.__init__(enc)
    enc.tok, enc.model, enc.max_length, enc.chunk_tokens, enc.cache = Tok(), Body(), 6, 32, {}
    long_ = " ".join(["a", "bb", "ccc", "dddd", "e", "ff", "ggg", "hhhh", "i"])
    wins = enc._windows(enc._all_ids([long_, "short one"]))
    assert all(len(w) <= 6 for _, w in wins) and [o for o, _ in wins].count(0) == 3
    assert [t for o, w in wins if o == 0 for t in w[1:-1]] == [3 + len(w) for w in long_.split()]
    vec = enc([long_, "short one"])
    every = torch.cat([enc.model.emb(torch.tensor(w)) for o, w in wins if o == 0])
    assert torch.allclose(vec[0], every.mean(0), atol=1e-6)
    assert torch.allclose(vec[1], enc.model.emb(torch.tensor([1, 8, 6, 2])).mean(0), atol=1e-6)


def test_budget_check_counts_lines_the_encoder_reads_in_windows():
    from collections import Counter

    from ml.c3.budget_check import check_encoder

    schema = load_schema()
    train, _ = _batches(schema)

    def tok(texts, truncation=False):
        return {"input_ids": [list(range(len(t.split()) + 2)) for t in texts]}

    counts = Counter()
    longest = max(len(t.split()) + 2 for t in train.texts)
    assert check_encoder(tok, train, longest, counts) == [] and counts["encoder_max_tokens"] == longest
    assert len(check_encoder(tok, train, longest - 1, counts)) >= 1


def test_a_checkpoint_saved_with_the_torch_version_loads_back(tmp_path):
    import torch
    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.model import C3Config, C3Model

    schema = load_schema()
    model = C3Model(schema, C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8,
                                    d_head=16, residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2))
    new = tmp_path / "new.pt"
    save_checkpoint(new, model, step=3, schema_sha256="abc")
    assert load_checkpoint(new, model, schema_sha256="abc") == 3
    # The live runs saved torch.__version__ as a TorchVersion, before the str().
    old = tmp_path / "old.pt"
    torch.save({"step": 5, "schema_sha256": "abc", "torch": torch.__version__,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, old)
    assert load_checkpoint(old, model, schema_sha256="abc") == 5


def test_a_checkpoint_with_the_hindsight_copy_loads_into_a_fresh_model(tmp_path):
    import copy
    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.model import C3Config, C3Model

    schema = load_schema()
    cfg = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8,
                   d_head=16, residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
    trained = C3Model(schema, cfg)
    # What hindsight_targets() leaves on a model that trained.
    trained.teacher = copy.deepcopy(trained.encoder)
    trained.teacher_proj = copy.deepcopy(trained.q_target)
    path = tmp_path / "run.pt"
    save_checkpoint(path, trained, step=7, schema_sha256="abc")
    fresh = C3Model(schema, cfg)
    assert load_checkpoint(path, fresh, schema_sha256="abc") == 7
    for (name, a), b in zip(trained.encoder.state_dict().items(), fresh.encoder.state_dict().values()):
        assert (a == b).all(), name


def test_type_compare_scores_base_and_parser_on_the_same_lines():
    from ml.c3.type_compare import compare

    read = {"lines": {
        "a": {"reads": {"col:label_type": {"answer": "task", "p": 0.95}}},
        "b": {"reads": {"col:label_type": {"answer": "task", "p": 0.6}}},
        "c": {"reads": {"col:label_type": {"answer": "rate_card", "p": 0.9}}},
    }}
    labels = [
        {"label_key": "a", "label_type": "task", "parser_type": "task"},
        {"label_key": "b", "label_type": "task", "parser_type": "scope_item"},
        {"label_key": "c", "label_type": "rate_card", "parser_type": "service_line"},
        {"label_key": "d", "label_type": "task", "parser_type": "task"},   # no read: left out
    ]
    out = compare(read, labels)
    assert out["lines"] == 3
    assert out["parser"]["task"] == {"labeled": 2, "said": 1, "right": 1, "recall": 0.5, "precision": 1.0}
    assert out["base"]["exact"] == 1.0
    assert out["parser"]["group"] == 1.0            # each miss stays inside its group
    assert out["base_when_sure"]["0.9"]["n"] == 2


def test_type_compare_can_score_the_best_named_type():
    from ml.c3.type_compare import compare

    read = {"lines": {"a": {"reads": {"col:label_type": {"answer": "_keep", "p": 0.6,
                                                         "typed": "rate_card", "typed_p": 0.3}}}}}
    labels = [{"label_key": "a", "label_type": "rate_card", "parser_type": "service_line"}]
    assert compare(read, labels)["base"]["exact"] == 0.0
    assert compare(read, labels, typed=True)["base"]["exact"] == 1.0


def test_type_compare_can_leave_out_a_document_kind():
    from ml.c3.type_compare import compare

    read = {"lines": {"a": {"doc_kind": "deal_kit", "reads": {"col:label_type": {"answer": "task", "p": 0.9}}},
                      "b": {"doc_kind": "email", "reads": {"col:label_type": {"answer": "task", "p": 0.9}}}}}
    labels = [{"label_key": k, "label_type": "task", "parser_type": "task"} for k in "ab"]
    assert compare(read, labels, skip_doc_kinds=("Deal_Kit",))["lines"] == 1


def test_type_compare_routes_unsure_lines_to_the_parser():
    from ml.c3.type_compare import compare

    read = {"lines": {"a": {"reads": {"col:label_type": {"answer": "rate_card", "p": 0.9}}},
                      "b": {"reads": {"col:label_type": {"answer": "_keep", "p": 0.3}}}}}
    labels = [{"label_key": "a", "label_type": "rate_card", "parser_type": "service_line"},
              {"label_key": "b", "label_type": "task", "parser_type": "task"}]
    routed = compare(read, labels)["routed"]["0.5"]
    assert routed["exact"] == 1.0 and routed["share_of_lines"] == 0.5


def test_stage_steps_override_named_stages_only():
    from ml.c3.loop import Budget, with_stage_steps

    b = with_stage_steps(Budget(), "lines=30, links=0,teach=0")
    assert (b.lines, b.links, b.teach, b.cards) == (30, 0, 0, Budget().cards)
    with pytest.raises(SystemExit):
        with_stage_steps(Budget(), "linez=3")


def test_no_teacher_page_is_over_the_window_and_what_is_left_out_is_counted(capsys):
    """Pilot C's bind died on a digest part whose whole-deal cards cite all
    8,192 lines (one card page: 337,557 Qwen tokens). Here a byte-level
    stand-in with a 1,024-token window plays the teacher: an over-long
    card, an over-long plain line, and a line that is short in characters
    but long in tokens (the old character shortcut said it fit). None of
    them reaches a forward; each is counted and printed; the part trains."""
    import math
    import torch
    from ml.c3.brain import TEACHER_LEFT_OUT, Brain, told_why
    from ml.c3.data import Judged
    from ml.c3.lm import TinyCausalLM
    from ml.c3.loop import _teacher_part
    from ml.c3.losses import BrainWeights, teacher_chunks, teacher_rows

    window = 1024
    widths: list[int] = []

    class Recording(TinyCausalLM):
        def forward(self, ids, mask, prefix=None, inputs_embeds=None):
            widths.append(int(ids.shape[1]))
            return super().forward(ids, mask, prefix=prefix, inputs_embeds=inputs_embeds)

    schema = load_schema()
    train, _ = _batches(schema)
    taught, _ = teacher_rows(train, told_why(train, schema))
    taught = list(range(len(train))) if taught is None else taught
    wide_chars, plain = taught[0], taught[1]
    texts = list(train.texts)
    texts[wide_chars] = "é" * 700          # 700 characters, 1,400 byte tokens
    texts[plain] = "x" * 3000
    key = next(o.key for o in schema.opportunities if o.key.startswith("jdg:") and o.universal
               and not o.multiple)
    everything = Judged(key, "GROUP", tuple(range(len(train))), 0, note="whole deal", subject_key="all")
    calm = tuple(i for i in range(len(train)) if i not in (wide_chars, plain))[:2]
    short = Judged(key, "PAIR", calm, 0, note="short", subject_key="s")
    part = replace(train, texts=texts, judged=[short, everything])

    torch.manual_seed(0)
    brain = Brain(schema, Recording(32, 1, 2, window), page_batch=2, read_chunk=2)
    limit = window - brain.FIT_MARGIN
    # The token count the old shortcut never asked for.
    assert len(texts[wide_chars]) <= limit < brain.lm.token_count(texts[wide_chars])
    assert not brain.fits(texts[wide_chars]) and not brain.fits(texts[plain])
    assert brain.lm.token_count(brain.card_page(part, everything, with_why=True)) > window

    before = dict(TEACHER_LEFT_OUT)
    pieces = teacher_chunks(brain, part, 4, 4000)
    rows = {r for p in pieces for r in p.rows}
    assert wide_chars not in rows and plain not in rows
    assert set(taught) - {wide_chars, plain} <= rows
    assert [c for p in pieces for c in p.cards] == [short]
    assert TEACHER_LEFT_OUT["rows"] - before.get("rows", 0) == 2
    assert TEACHER_LEFT_OUT["cards"] - before.get("cards", 0) == 1
    printed = capsys.readouterr().out
    assert f"teacher_left_out {part.deal_id} rows=2/" in printed
    assert f"teacher_left_out {part.deal_id} cards=1/2" in printed

    widths.clear()
    loss = _teacher_part(brain, part, BrainWeights(consolidate=0.0), device="cpu", scale=1.0,
                         size=4, budget=4000, caps=(0, 0, 0), rng=None, label="bind")
    assert math.isfinite(float(loss))
    assert widths and max(widths) <= window

    # The lesson view reads every line, so it stops before any forward, named.
    from ml.c3.supercharge import teacher_view

    widths.clear()
    with pytest.raises(ValueError, match="2 lines cannot fit the teacher's 1024-token window"):
        teacher_view(brain, part, grounding=False)
    assert widths == []


def test_a_non_finite_teacher_piece_names_its_part_piece_and_terms(monkeypatch):
    import torch
    import ml.c3.loop as loop
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM
    from ml.c3.losses import BrainWeights

    schema = load_schema()
    train, _ = _batches(schema)
    brain = Brain(schema, TinyCausalLM(32, 1, 2, 4096))
    real = loop.brain_loss

    def poisoned(*args, **kwargs):
        total, terms = real(*args, **kwargs)
        return total * float("nan"), {**terms, "distill": float("nan")}

    monkeypatch.setattr(loop, "brain_loss", poisoned)
    with pytest.raises(RuntimeError, match=r"Non-finite loss during bind: synthetic.* piece 1/\d+ .*"
                                           r"non-finite terms distill"):
        loop._teacher_part(brain, train, BrainWeights(consolidate=0.0, flip=0.0), device="cpu",
                           scale=1.0, size=4, budget=4000, caps=(0, 0, 0), rng=None, label="bind")
