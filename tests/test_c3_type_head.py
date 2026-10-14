"""Line type gets its own weights, its own loss share and its own checks."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3.data import IGNORE, DealExample, featurize  # noqa: E402
from ml.c3.gates import load_checkpoint, save_checkpoint  # noqa: E402
from ml.c3.loop import (Budget, gate_failures, parse_gates, run_base,  # noqa: E402
                       type_check)
from ml.c3.losses import LossWeights, head_losses  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "ml" / "c3" / "fixtures" / "synthetic_deal.json"
CFG = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
               residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
KEY = "col:label_type"


def _deal(parser_type: str | None = "task") -> DealExample:
    deal = copy.deepcopy(DealExample.load(FIXTURE))
    for a in deal.atoms:
        if a.label and parser_type:
            a.label["parser_type"] = parser_type
    return deal


def test_a_free_head_starts_at_zero_and_reads_like_the_described_head():
    schema = load_schema()
    torch.manual_seed(0)
    model = C3Model(schema, CFG).eval()
    batch = featurize(_deal(), schema)
    with torch.no_grad():
        before = model(batch.oneshot()).logits[KEY]
        model.add_free_head(KEY)
        after = model(batch.oneshot()).logits[KEY]
    assert torch.allclose(before, after)
    assert model.free_heads[KEY].weight.shape == (len(schema.by_key()[KEY].answers), CFG.d_head + CFG.d)


def test_a_free_head_moves_the_type_logits_once_it_has_weights():
    schema = load_schema()
    model = C3Model(schema, CFG).eval()
    batch = featurize(_deal(), schema)
    head = model.add_free_head(KEY)
    with torch.no_grad():
        before = model(batch.oneshot()).logits[KEY].clone()
        head.bias[3] = 5.0
        after = model(batch.oneshot()).logits[KEY]
    assert torch.allclose(after[:, 3] - before[:, 3], torch.full_like(after[:, 3], 5.0), atol=1e-4)


def test_checkpoints_load_with_or_without_a_free_head(tmp_path):
    schema = load_schema()
    plain = C3Model(schema, CFG)
    old = tmp_path / "old.pt"
    save_checkpoint(old, plain, step=2, schema_sha256="abc")
    # A run that adds a free head starts from a checkpoint that had none.
    widened = C3Model(schema, CFG)
    widened.add_free_head(KEY)
    assert load_checkpoint(old, widened, schema_sha256="abc") == 2
    assert float(widened.free_heads[KEY].weight.abs().sum()) == 0.0
    # A checkpoint with one builds it in a fresh model (predict).
    with torch.no_grad():
        widened.free_heads[KEY].bias.fill_(0.5)
    new = tmp_path / "new.pt"
    save_checkpoint(new, widened, step=4, schema_sha256="abc")
    fresh = C3Model(schema, CFG)
    assert load_checkpoint(new, fresh, schema_sha256="abc") == 4
    assert torch.allclose(fresh.free_heads[KEY].bias, torch.full_like(fresh.free_heads[KEY].bias, 0.5))


def test_a_head_weight_raises_that_questions_share_of_the_heads_loss():
    schema = load_schema()
    torch.manual_seed(0)
    model = C3Model(schema, CFG)
    batch = featurize(_deal(), schema)
    out = model(batch.inputs())
    same, _ = head_losses(model, out, batch, LossWeights(), "universal")
    one, _ = head_losses(model, out, batch, LossWeights(head={KEY: 1.0}), "universal")
    assert torch.allclose(same, one)
    only, _ = head_losses(model, out, batch, LossWeights(head={KEY: 1e6}), "universal")
    from ml.c3.losses import answer_loss, row_weights
    alone = answer_loss(out.logits[KEY], batch, KEY, row_weights(batch, out.r.device))
    assert torch.allclose(only, alone, rtol=1e-3)


def test_the_parsers_type_is_carried_but_never_an_input():
    schema = load_schema()
    batch = featurize(_deal("task"), schema)
    task = schema.by_key()[KEY].index("task")
    assert batch.parser_targets[KEY] == [task] * len(batch)
    assert "parser_targets" not in batch.inputs()
    assert featurize(_deal(None), schema).parser_targets == {}


def test_type_check_scores_the_routed_split():
    schema = load_schema()
    model = C3Model(schema, CFG)
    batch = featurize(_deal("task"), schema)
    row = type_check(model, [batch])
    labeled = sum(y != IGNORE for y in batch.targets[KEY])
    assert row["n"] == labeled
    # Nothing is sure at 1.01, so the routed read is the parser's alone.
    off = type_check(model, [batch], sure=1.01)
    assert off["routed_share"] == 0.0 and off["routed_exact"] == off["parser_exact"]
    assert -1.0 <= off["margin"] <= 1.0
    # A Deal Kit line is never scored.
    kit = replace(batch, doc_kind=["deal_kit"] * len(batch))
    assert type_check(model, [kit])["n"] == 0


def test_gates_read_their_stage_and_bar():
    gates = parse_gates(["lines@5.fit_exact>=0.5", "lines.valid_exact>=0.45"])
    assert gates == [("lines@5", "fit_exact", ">=", 0.5), ("lines", "valid_exact", ">=", 0.45)]
    assert gate_failures(gates, "lines@5", {"fit_exact": 0.6}) == []
    assert gate_failures(gates, "lines@5", {"fit_exact": 0.4})
    assert gate_failures(gates, "lines", {}) == ["valid_exact was not measured"]
    with pytest.raises(ValueError):
        parse_gates(["lines.fit_exact=0.5"])


def test_a_failed_gate_stops_the_run_and_the_checks_are_saved(tmp_path):
    import json

    schema = load_schema()
    train = featurize(_deal(), schema)
    hold = featurize(replace(_deal(), deal_id="synthetic-holdout"), schema)
    val = featurize(replace(_deal(), deal_id="synthetic-validation"), schema)
    budget = Budget(teacher=0, lines=3, links=2, cards=0, numbers=0, deal=0, open_links=0, teach=0)
    model, result = run_base(
        [train], [hold], schema, budget, checkpoint=str(tmp_path / "run.pt"),
        validation=[val], free_heads=(KEY,), head_weight={KEY: 20.0}, check_every=1,
        gates=parse_gates(["lines@2.fit_exact>=1.01"]), score_holdout=False)
    checks = json.loads((tmp_path / "run.pt.checks.json").read_text())
    assert set(checks) == {"start", "lines@1", "lines@2"}
    assert {"fit_exact", "valid_exact", "valid_routed_exact", "valid_focus_recall"} <= set(checks["lines@1"])
    assert "links" not in result.stage_loss and "lines" in result.stage_loss
    assert (tmp_path / "run.pt.lines@1").exists()
    assert float(model.free_heads[KEY].weight.abs().sum()) > 0


def test_free_heads_all_covers_every_universal_class_question():
    schema = load_schema()
    train = featurize(_deal(), schema)
    hold = featurize(replace(_deal(), deal_id="synthetic-holdout"), schema)
    budget = Budget(teacher=0, lines=1, links=0, cards=0, numbers=0, deal=0, open_links=0, teach=0)
    model, _ = run_base([train], [hold], schema, budget, free_heads=("all",), fit_parts=0,
                        score_holdout=False)
    assert KEY in model.free_heads and len(model.free_heads) > 10
    assert all(not k.startswith("rel:") for k in model.free_heads)


def test_head_fit_trains_free_heads_on_cached_rows_and_beats_the_start():
    from ml.c3.head_fit import _cache, fit_heads
    from ml.c3.schema import NUMBER, RELATION

    schema = load_schema()
    torch.manual_seed(0)
    model = C3Model(schema, CFG).eval()
    batch = featurize(_deal(), schema)
    keys = [o.key for o in schema.select(layer="universal")
            if o.kind not in (RELATION, NUMBER) and len(o.answers) > 1]
    tr = _cache(model, [batch], keys, "cpu")
    assert KEY in tr and len(tr[KEY]["y"]) == sum(y != IGNORE for y in batch.targets[KEY])
    keys = [k for k in keys if k in tr]
    for k in keys:
        model.add_free_head(k)
    best, state = fit_heads(model, tr, tr, keys, epochs=150, check_every=50, lr=5e-2, batch=64)
    assert state is not None and best > 0.9


def test_a_run_that_does_not_score_the_holdout_may_leave_its_files_out():
    from ml.c3.loop import lock_split

    schema = load_schema()
    train = featurize(_deal(), schema)
    budget = Budget(teacher=0, lines=1, links=0, cards=0, numbers=0, deal=0, open_links=0, teach=0)
    _, result = run_base([train], [], schema, budget, fit_parts=0, score_holdout=False)
    assert result.holdout_ids == () and result.holdout_scores == {}
    with pytest.raises(ValueError):
        lock_split([train], [])
    with pytest.raises(ValueError, match="fixed holdout"):
        lock_split([replace(train, deal_id="010038")], [], need_holdout=False)


class _Body(torch.nn.Module):
    """Stands in for HFEncoder: a hashing reader plus a 'downloaded' body."""

    pretrained_body = "model"

    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        from ml.c3.text import HashingEncoder
        self.reader = HashingEncoder(dim, buckets=64)
        torch.manual_seed(seed)
        self.model = torch.nn.Linear(dim, dim)

    def forward(self, texts):
        return self.model(self.reader(texts))


def test_a_slim_checkpoint_leaves_out_the_frozen_body_and_refills_it(tmp_path):
    from ml.c3.gates import slim_state

    schema = load_schema()
    model = C3Model(schema, CFG, text=_Body(CFG.d_text))
    model.add_free_head(KEY)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    slim, body = slim_state(state, C3Model(schema, CFG, text=_Body(CFG.d_text)))
    assert body["tensors"] == 2 and not any(".model." in k or k.startswith("text.model") for k in slim)
    path = tmp_path / "slim.pt"
    torch.save({"step": 3, "schema_sha256": "abc", "torch": "x", "state": slim, "pretrained_body": body}, path)
    reader = C3Model(schema, CFG, text=_Body(CFG.d_text))
    assert load_checkpoint(path, reader, schema_sha256="abc") == 3
    for k, v in model.state_dict().items():
        assert torch.equal(reader.state_dict()[k], v), k
    # A different download is refused, not silently used.
    with pytest.raises(RuntimeError, match="not the one"):
        load_checkpoint(path, C3Model(schema, CFG, text=_Body(CFG.d_text, seed=1)), schema_sha256="abc")


def test_a_trained_body_layer_stays_in_the_slim_checkpoint():
    from ml.c3.gates import slim_state

    schema = load_schema()
    model = C3Model(schema, CFG, text=_Body(CFG.d_text))
    with torch.no_grad():
        model.text.model.bias.add_(1.0)
    slim, body = slim_state(dict(model.state_dict()), C3Model(schema, CFG, text=_Body(CFG.d_text)))
    assert body["tensors"] == 1
    assert "text.model.bias" in slim and "text.model.weight" not in slim


def test_a_run_with_no_teacher_stage_never_builds_the_language_model():
    schema = load_schema()
    train = featurize(_deal(), schema)
    budget = Budget(teacher=0, lines=1, links=0, cards=0, numbers=0, deal=0, open_links=1, teach=0)

    def refuse():
        raise AssertionError("the teacher model was built")

    run_base([train], [], schema, budget, fit_parts=0, score_holdout=False, make_lm=refuse)


def test_warmup_starts_the_rate_small_and_reaches_it_after_the_passes():
    schema = load_schema()
    train = featurize(_deal(), schema)
    budget = Budget(teacher=0, lines=3, links=0, cards=0, numbers=0, deal=0, open_links=0, teach=0)
    model = C3Model(schema, CFG)
    seen = []
    orig = torch.optim.AdamW.step

    def spy(self, *a, **k):
        seen.append(self.param_groups[0]["lr"])
        return orig(self, *a, **k)

    torch.optim.AdamW.step = spy
    try:
        run_base([train], [], schema, budget, model=model, lr=1e-3, adamw=True, warmup_passes=2,
                 fit_parts=0, score_holdout=False)
    finally:
        torch.optim.AdamW.step = orig
    assert seen[0] == pytest.approx(5e-4) and seen[1] > seen[0]
