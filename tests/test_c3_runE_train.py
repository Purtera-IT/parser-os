"""Run E's training script: resume, checkpoint rotation, the loss ramp, the
layout lock's reused descriptions and the pick over the gated heads."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import ml.c3.loop as loop  # noqa: E402
from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.loop import Budget, run_base  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss  # noqa: E402
from ml.c3.loop import SMALL  # noqa: E402
from ml.c3.model import C3Model  # noqa: E402
from ml.c3.pick import best_at, composite_heads, judge, pick  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "ml" / "c3" / "fixtures" / "synthetic_deal.json"
BUDGET = Budget(teacher=0, lines=2, links=2, cards=0, numbers=0, deal=0, open_links=1, teach=0)


def _train(schema):
    deal = DealExample.load(FIXTURE)
    return featurize(deal, schema)


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_a_resumed_run_ends_where_an_unbroken_run_ends(tmp_path, monkeypatch):
    schema = load_schema()
    train = _train(schema)
    kw = dict(budget=BUDGET, check_every=1, score_holdout=False, seed=3)
    whole, _ = run_base([train], [], schema, checkpoint=str(tmp_path / "a.pt"), **kw)

    real = loop._progress
    calls = {"n": 0}

    def crash(stage, done, total, started):
        real(stage, done, total, started)
        calls["n"] += 1
        if calls["n"] == 4:  # lines 1, lines 2, links 1 saved; dies in links pass 2
            raise RuntimeError("pod lost")

    monkeypatch.setattr(loop, "_progress", crash)
    with pytest.raises(RuntimeError, match="pod lost"):
        run_base([train], [], schema, checkpoint=str(tmp_path / "b.pt"), **kw)
    monkeypatch.setattr(loop, "_progress", real)
    saved = torch.load(tmp_path / "b.pt.resume.pt", weights_only=False)
    assert (saved["stage"], saved["passed"], saved["completed"]) == ("links", 1, 3)
    resumed, _ = run_base([train], [], schema, checkpoint=str(tmp_path / "b.pt"), resume=True, **kw)
    a, b = _state(whole), _state(resumed)
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])][:5]
    checks_a = json.loads((tmp_path / "a.pt.checks.json").read_text())
    checks_b = json.loads((tmp_path / "b.pt.checks.json").read_text())
    assert list(checks_a) == list(checks_b)


def test_the_rotation_keeps_n_checkpoints_and_the_best(tmp_path):
    schema = load_schema()
    train = _train(schema)
    ck = tmp_path / "r.pt"
    run_base([train], [], schema, budget=replace(BUDGET, lines=4), checkpoint=str(ck), check_every=1,
             score_holdout=False, keep_checkpoints=2)
    files = sorted(p.name for p in tmp_path.glob("r.pt.*") if not p.name.endswith((".json", ".resume.pt", ".teacher")))
    checks = json.loads((tmp_path / "r.pt.checks.json").read_text())
    best = best_at(checks, {}, exists=lambda at: (tmp_path / f"r.pt.{at}").exists())
    assert len([f for f in files if f != f"r.pt.{best}"]) <= 2
    assert best is None or f"r.pt.{best}" in files
    assert ck.is_symlink() and ck.resolve().exists()
    assert len(checks) == 1 + 3 + 1 + 1 + 1 + 1  # start, lines@1-3, lines, links@1, links, open_links


def test_a_ramped_term_grows_from_zero_from_its_first_step(monkeypatch):
    schema = load_schema()
    train = _train(schema)
    seen = []
    real = loop._student_step

    def watch(model, opt, batch, weights, *a, **k):
        seen.append((weights.why_contrast, weights.relations))
        return real(model, opt, batch, weights, *a, **k)

    monkeypatch.setattr(loop, "_student_step", watch)
    run_base([train], [], schema, budget=replace(BUDGET, lines=3, links=2, open_links=0),
             score_holdout=False, fit_parts=0,
             loss_weight={"why_contrast": 0.3, "relations": 0.5}, ramp={"why_contrast": 2, "relations": 1})
    contrast = [round(c, 4) for c, _ in seen]
    assert contrast == [0.15, 0.3, 0.3, 0.3, 0.3]
    # relations is off in lines; its ramp starts when links turns it on.
    assert [r for _, r in seen] == [0.0, 0.0, 0.0, 0.5, 0.5]


def test_the_layout_lock_and_keyword_dropout_reuse_the_steps_descriptions(monkeypatch):
    schema = load_schema()
    batch = _train(schema)
    rows = [i for i, on in enumerate(batch.labeled) if on][:3]
    batch = replace(batch, layout_rewrites=[(i, "cell a\tcell b") for i in rows])
    model = C3Model(schema, SMALL)
    calls = {"n": 0}
    real = model.describe

    def counted():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(model, "describe", counted)
    _, parts = c3_loss(model, batch, LossWeights(layout_lock=0.5, keyword_dropout=0.3, keyword_dropout_p=1.0))
    assert "layout_lock" in parts and parts["layout_lock"] > 0
    assert calls["n"] == 1


def _checks():
    def row(adm, sup, typ, n=100, fit=0.7):
        return {"valid_exact": typ, "fit_exact": fit,
                "valid_col_admission": adm, "valid_col_admission_n": n,
                "valid_col_supplier": sup, "valid_col_supplier_n": n,
                "valid_col_label_type": typ, "valid_col_label_type_n": n,
                "valid_jdg_site": 0.0, "valid_jdg_site_n": 2}
    return {"start": row(0.60, 0.60, 0.60), "lines@2": row(0.80, 0.50, 0.70),  # supplier broke
            "lines": row(0.78, 0.66, 0.64), "links": row(0.79, 0.65, 0.66, fit=0.95)}  # over-fit


def test_the_pick_is_a_macro_over_the_gated_heads_held_to_the_live_floors(tmp_path):
    checks = _checks()
    floors = {"col:admission": {"exact": 0.77, "n": 3520}, "col:supplier": {"exact": 0.62, "n": 843},
              "col:label_type": {"exact": 0.63, "n": 1833}, "jdg:site": {"exact": 1.0, "n": 1}}
    heads = composite_heads(checks)
    assert heads == ["col:admission", "col:supplier", "col:label_type"]  # n=2 site cards: reported only
    ok, comp, bad = judge(checks["lines@2"], floors, heads, checks["start"])
    assert not ok and any("supplier" in b for b in bad)
    assert not judge(checks["links"], floors, heads, checks["start"])[0]  # the fit gap grew
    assert not judge(checks["start"], floors, heads, checks["start"])[0]  # run C re-read is below live
    assert best_at(checks, floors) == "lines"
    ck = tmp_path / "e.pt"
    (tmp_path / "e.pt.checks.json").write_text(json.dumps(checks))
    for at in ("lines@2", "lines", "links"):
        (tmp_path / f"e.pt.{at}").write_bytes(b"x")
    (tmp_path / "floors.json").write_text(json.dumps({"heads": floors}))
    out = pick(str(ck), str(tmp_path / "init.pt"), str(tmp_path / "floors.json"))
    assert out["at"] == "lines" and Path(f"{ck}.best").resolve() == (tmp_path / "e.pt.lines").resolve()
    (tmp_path / "e.pt.lines").unlink()
    assert pick(str(ck), str(tmp_path / "init.pt"), str(tmp_path / "floors.json"))["at"] == "start"


def test_an_opt_in_term_is_on_in_every_stage_and_ramps(monkeypatch):
    """layout_why is 0 by default (LossWeights), so gold_weights never turns it on: --loss-weight does, in every
    stage, and its ramp grows from 0 like the others."""
    schema = load_schema()
    train = _train(schema)
    seen = []
    real = loop._student_step

    def watch(model, opt, batch, weights, *a, **k):
        seen.append((weights.layout_why, weights.keyword_dropout_broad))
        return real(model, opt, batch, weights, *a, **k)

    monkeypatch.setattr(loop, "_student_step", watch)
    run_base([train], [], schema, budget=replace(BUDGET, lines=3, links=2, open_links=0),
             score_holdout=False, fit_parts=0,
             loss_weight={"layout_why": 0.3, "keyword_dropout_broad": 0.5}, ramp={"layout_why": 2})
    assert [round(w, 4) for w, _ in seen] == [0.15, 0.3, 0.3, 0.3, 0.3]
    assert {b for _, b in seen} == {0.5}
    seen.clear()
    run_base([train], [], schema, budget=replace(BUDGET, lines=2, links=0, open_links=0),
             score_holdout=False, fit_parts=0)
    assert {w for w, _ in seen} == {0.0}
