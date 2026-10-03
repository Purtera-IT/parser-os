"""ml/c3: the C3 model's contracts, checked on a synthetic deal (no training).

Each test pins one property the design docs claim, so a later change that
breaks it fails here rather than in a training run:

* inputs never carry labels; the foresight mask holds;
* universal outputs are identical whatever company is asking;
* the schema's text builds the heads: edit a description, the head changes;
  add an opportunity, no parameter is added;
* folds are folds; the WHY arithmetic parses into exact programs;
* every loss term produces a gradient.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3.data import IGNORE, DealExample, featurize  # noqa: E402
from ml.c3.folds import FoldStack  # noqa: E402
from ml.c3.losses import LossWeights, absence_targets, c3_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.notes import extract_programs, mask_verdict, split_note  # noqa: E402
from ml.c3.schema import POLICY_WORDS, RELATION, Answer, Opportunity, load_schema  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "ml" / "c3" / "fixtures" / "synthetic_deal.json"
SMALL = C3Config(d_text=64, d=64, n_layers=2, n_heads=4, d_r=32, d_q=16, d_head=32,
                 residual_dim=8, box_dim=8, n_policy_atoms=4, policy_rank=4)


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def deal():
    return DealExample.load(FIXTURE)


def _model(schema, **kw):
    torch.manual_seed(0)
    m = C3Model(schema, SMALL, **kw)
    m.eval()
    return m


def _universal(out):
    return {k: v for k, v in out.logits.items() if not k.startswith("read:co_")
            and k not in ("read:intake_gap", "read:needed_by", "read:scope_category",
                          "read:delivery_field", "read:source_of_truth", "read:co_stage_raw")}


# ---------------------------------------------------------------- schema

def test_schema_covers_every_trainable_reading_and_relation(schema):
    import json

    heads = json.loads((Path(__file__).resolve().parents[1] / "app/core/label_heads.json").read_text())
    types = json.loads((Path(__file__).resolve().parents[1] / "app/core/atom_types.json").read_text())
    layer = {r["key"]: r.get("layer") for r in types["reads"]}
    values = {r["key"]: r.get("values") for r in types["reads"]}
    want = set()
    for h in heads["heads"]:
        if h["layer"] not in ("universal", "company"):
            continue
        want |= {f"read:{r}" for r in h["reads"]
                 if layer.get(r) in ("universal", "company") and values.get(r) != "free text"}
        want |= {f"rel:{r}" for r in h["relations"]}
    assert want <= set(schema.by_key())


def test_universal_descriptions_carry_no_company_policy(schema):
    for o in schema.select(layer="universal"):
        for text in [o.description, *(a.description for a in o.answers)]:
            assert not any(w in text.lower() for w in POLICY_WORDS), (o.key, text)
        assert o.description.strip()


# ---------------------------------------------------------------- notes

def test_split_note_matches_the_ingest():
    try:
        from app.learning.human_labels import split_note as ingest_split
    except Exception:  # pragma: no cover - app deps missing locally
        pytest.skip("app.learning.human_labels not importable here")
    for note in ["[EXCLUDE_FROM_TRAINING: old manual Deal Kit]\nWHY line\n[purtera] ignore: x",
                 "Plain WHY only.", "WHY\n  [purtera] reject: seller promise", "",
                 "WHY\n[purtera] keep: x\n[parser] SHOULD SPLIT: two sites in one atom"]:
        assert split_note(note) == ingest_split(note)


def test_mask_verdict_hides_the_answer():
    assert "superseded" not in mask_verdict("Superseded by the v2 count").lower()
    assert "[MASK]" in mask_verdict("we reject it")


def test_programs_come_out_of_the_why_and_execute_exactly(deal):
    found = [p for a in deal.atoms for p in extract_programs(split_note(a.label["note"])[0])]
    assert {(p.operands, p.stated) for p in found} >= {((2.0, 8.0), 16.0), ((16.0, 95.0), 1520.0)}
    assert all(p.exact for p in found)
    assert extract_programs("3 x 4 = 13") == []          # wrong arithmetic is not a target


# ---------------------------------------------------------------- data

def test_inputs_carry_no_labels(schema, deal):
    batch = featurize(deal, schema)
    assert set(batch.inputs()) == {"texts", "numbers", "doc_kind", "role", "side", "times",
                                   "same_doc", "same_section", "context"}
    m = _model(schema)
    relabeled = copy.deepcopy(deal)
    for a in relabeled.atoms:
        a.label = dict(a.label, label_type="risk", note="something else entirely",
                       reads_set={k: v for k, v in a.label["reads_set"].items()
                                  if k in ("stage_std", "location_tier")})
    with torch.no_grad():
        a = m(batch.inputs(), company="purtera")
        b = m(featurize(relabeled, schema).inputs(), company="purtera")
    for k in a.logits:
        assert torch.equal(a.logits[k], b.logits[k]), k


def test_absence_targets_follow_the_timeline(schema, deal):
    batch = featurize(deal, schema)
    t = absence_targets(batch)
    sites = list(__import__("ml.c3.model", fromlist=["CLAIM_SLOTS"]).CLAIM_SLOTS).index("sites")
    # No site line until the CRM note (index 12): every earlier line is "filled later".
    assert t[:12, sites].tolist() == [1.0] * 12
    assert t[12:, sites].tolist() == [-1.0, -1.0]


# ---------------------------------------------------------------- model

def test_foresight_mask_hides_the_future(schema, deal):
    m = _model(schema)
    batch = featurize(deal, schema)
    later = copy.deepcopy(batch)
    later.texts[-1] = "A completely different last line about cranes and permits."
    with torch.no_grad():
        a, b = m(batch.inputs()), m(later.inputs())
    n = len(batch) - 1
    for name in ("z_c", "r", "q_mu"):
        assert torch.allclose(getattr(a, name)[:n], getattr(b, name)[:n], atol=1e-6), name
    assert not torch.allclose(a.r[n], b.r[n])
    for k in a.logits:
        assert torch.allclose(a.logits[k][:n], b.logits[k][:n], atol=1e-5), k


def test_universal_outputs_ignore_the_company(schema, deal):
    m = _model(schema, companies=("purtera", "other_co"))
    with torch.no_grad():
        m.alpha[1].normal_()
        inp = featurize(deal, schema).inputs()
        a, b = m(inp, company="purtera"), m(inp, company="other_co")
    for k, v in _universal(a).items():
        assert torch.equal(v, b.logits[k]), k
    assert not torch.equal(a.logits["read:co_action"], b.logits["read:co_action"])


def test_editing_a_description_edits_the_head(schema, deal):
    m = _model(schema)
    inp = featurize(deal, schema).inputs()
    with torch.no_grad():
        before = m(inp).logits["read:sow_coverage"]
    edited = copy.deepcopy(schema)
    i = next(j for j, o in enumerate(edited.opportunities) if o.key == "read:sow_coverage")
    o = edited.opportunities[i]
    answers = list(o.answers)
    answers[1] = Answer(answers[1].value, "The final signed document carries this line word for word.")
    edited.opportunities[i] = Opportunity(**{**o.__dict__, "answers": tuple(answers)})
    m.schema = edited
    with torch.no_grad():
        after = m(inp).logits["read:sow_coverage"]
    assert torch.equal(before[:, 0], after[:, 0]) and not torch.equal(before[:, 1], after[:, 1])


def test_a_new_opportunity_needs_no_new_parameters(schema, deal):
    m = _model(schema)
    n_params = sum(p.numel() for p in m.parameters())
    grown = copy.deepcopy(schema)
    grown.opportunities.append(Opportunity(
        key="read:ladder_height", field="ladder_height", source="read", head="content.claims",
        space="content", layer="universal", kind="class",
        description="Which ladder or lift does the work need?",
        answers=(Answer("_absent", "not stated"), Answer("step", "a step ladder"),
                 Answer("lift", "a scissor lift or boom lift"))))
    m.schema = grown
    with torch.no_grad():
        out = m(featurize(deal, grown).inputs())
    assert out.logits["read:ladder_height"].shape == (len(deal.atoms), 3)
    assert sum(p.numel() for p in m.parameters()) == n_params


def test_a_full_fold_identifies_mirror_images():
    torch.manual_seed(1)
    d = 8
    n = torch.nn.functional.normalize(torch.randn(d), dim=0)
    p = {"n": n.view(1, 1, d), "b": torch.zeros(1, 1), "sigma": torch.ones(1, 1),
         "gamma": torch.ones(1, d), "beta": torch.zeros(1, d)}
    x = torch.randn(5, d)
    mirror = x - 2 * (x @ n).unsqueeze(-1) * n
    fx, fm = FoldStack.apply(x, p, 0), FoldStack.apply(mirror, p, 0)
    assert torch.allclose(fx, fm, atol=1e-5)                 # both sides land on one sheet
    assert ((fx @ n) >= -1e-6).all()
    pos = x[(x @ n) > 0]
    assert torch.allclose(FoldStack.apply(pos, p, 0), pos)    # identity on the kept side


def test_a_new_company_fits_only_its_code(schema, deal):
    m = _model(schema)
    k = m.add_company("northstar_av", policy_text="Two technicians for anything over 70 inches.")
    assert m.alpha.shape[0] == 2 and k == 1 and m.alpha[1].abs().sum() > 0
    m.freeze_for_new_company()
    assert [n for n, p in m.named_parameters() if p.requires_grad] == ["alpha"]


def test_consequence_samples_have_the_engine_shape(schema, deal):
    m = _model(schema)
    with torch.no_grad():
        out = m(featurize(deal, schema).inputs())
        s = m.sample_consequence(out, k=7)
    assert s.shape == (7, len(deal.atoms), SMALL.d_q)


def test_every_loss_term_reaches_the_weights(schema, deal):
    m = _model(schema)
    m.train()
    batch = featurize(deal, schema)
    loss, parts = c3_loss(m, batch, LossWeights())
    for name in ("heads", "conduct", "relations", "governs", "why_align",
                 "why_sufficiency", "hindsight", "absence"):
        assert parts[name] > 0, name
    loss.backward()
    for name in ("heads.content.folds.normals.weight", "conduct_head.proto.weight",
                 "why_proj.weight", "q_head.weight", "atom_u", "relation_head.src.normals.weight"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name
    # Relations under a causal mask: an answer never points forward in time.
    with torch.no_grad():
        out = m(batch.inputs())
    s = out.relations["answers"]
    upper = torch.triu(torch.ones_like(s, dtype=torch.bool), 1)
    assert (s[upper] == float("-inf")).all()
    assert IGNORE in batch.targets["col:wants"]
