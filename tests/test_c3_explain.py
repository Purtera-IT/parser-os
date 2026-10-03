"""ml/c3 explanations: written reasons the model reads and applies (no training).

* a rule card changes predictions the moment it is added, with no new parameters;
* a company's rule card never moves a universal output;
* a line never reads its own WHY;
* written guidance changes the head it is written for, and company words are
  kept out of universal guidance;
* the explanation-only loss reaches the reader.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.explain import Explanation, ExplanationBank, load_rule_cards  # noqa: E402
from ml.c3.losses import c3_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "ml" / "c3" / "fixtures"
SMALL = C3Config(d_text=64, d=64, n_layers=2, n_heads=4, d_r=32, d_q=16, d_head=32,
                 residual_dim=8, box_dim=8, n_policy_atoms=4, policy_rank=4)


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    return featurize(DealExample.load(HERE / "synthetic_deal.json"), schema)


def _model(schema):
    torch.manual_seed(0)
    m = C3Model(schema, SMALL)
    m.eval()
    return m


def _margin(logits, j):
    others = torch.cat([logits[:, :j], logits[:, j + 1:]], -1)
    return logits[:, j] - torch.logsumexp(others, -1)


def test_rule_cards_load_and_are_checked_against_the_schema(schema, tmp_path):
    rules = load_rule_cards(HERE / "rule_cards.json", schema)
    assert {e.layer for e in rules.items} == {"universal", "company"}
    bad = tmp_path / "bad.json"
    bad.write_text('[{"id": "x", "layer": "universal", "text": "t", '
                   '"opportunity": "read:co_action", "answer": "reject"}]')
    with pytest.raises(ValueError):
        load_rule_cards(bad, schema)


def test_a_new_rule_card_moves_its_answer_with_no_retraining(schema, batch):
    m = _model(schema)
    n_params = sum(p.numel() for p in m.parameters())
    card = ExplanationBank([Explanation(
        "A promise from a seller to send a document creates no delivery work.",
        "company", company="purtera", conclusions=(("read:co_reason", "seller_promise"),))])
    with torch.no_grad():
        before = m(batch.inputs(), company="purtera").logits["read:co_reason"]
        after = m(batch.inputs(), company="purtera", bank=card).logits["read:co_reason"]
    j = [a.value for a in schema.by_key()["read:co_reason"].answers].index("seller_promise")
    assert (_margin(after, j) > _margin(before, j)).all()
    assert sum(p.numel() for p in m.parameters()) == n_params


def test_company_rules_never_reach_universal_heads(schema, batch):
    m = _model(schema)
    rules = load_rule_cards(HERE / "rule_cards.json", schema)
    company_only = ExplanationBank([e for e in rules.items if e.layer == "company"])
    with torch.no_grad():
        a = m(batch.inputs(), company="purtera")
        b = m(batch.inputs(), company="purtera", bank=company_only)
    universal = [o.key for o in schema.select(layer="universal") if o.key in a.logits]
    for k in universal:
        assert torch.equal(a.logits[k], b.logits[k]), k
    assert not torch.equal(a.logits["read:co_reason"], b.logits["read:co_reason"])
    # and a universal rule does move universal heads
    uni = ExplanationBank([e for e in rules.items if e.layer == "universal"])
    with torch.no_grad():
        c = m(batch.inputs(), company="purtera", bank=uni)
    assert not torch.equal(a.logits["read:superseded"], c.logits["read:superseded"])


def test_a_line_never_reads_its_own_why(schema, batch):
    m = _model(schema)
    bank = ExplanationBank.from_batch(batch, schema)
    with torch.no_grad():
        out = m(batch.inputs(), company="purtera", bank=bank)
    uni = bank.select("universal")
    att = out.attention["col:label_type"]                      # [N, M_universal + 1]
    for col, i in enumerate(uni):
        line = bank.items[i].line
        assert att[line, col] == 0
    assert torch.allclose(att.sum(-1), torch.ones(len(batch)))


def test_guidance_text_rebuilds_its_head(batch):
    plain = load_schema()
    guided = load_schema(guidance=HERE / "guidance.json")
    m = _model(plain)
    with torch.no_grad():
        a = m(batch.inputs()).logits["read:sow_coverage"]
        m.schema = guided
        b = m(batch.inputs()).logits["read:sow_coverage"]
        c = m(batch.inputs()).logits["read:superseded"]
        m.schema = plain
        d = m(batch.inputs()).logits["read:superseded"]
    assert not torch.equal(a, b)
    assert torch.equal(c, d)                                   # other heads untouched


def test_universal_guidance_drops_company_policy():
    s = load_schema(guidance={"read:sow_coverage": {
        "how_to_decide": "Compare with the final SOW. The Deal Kit always wins."}})
    o = s.by_key()["read:sow_coverage"]
    assert "Compare with the final SOW." in o.description and "Deal Kit" not in o.description
    assert s.scrubbed == [("read:sow_coverage", "The Deal Kit always wins.")]


def test_explanation_only_loss_reaches_the_reader(schema, batch):
    m = _model(schema)
    m.train()
    rules = load_rule_cards(HERE / "rule_cards.json", schema)
    loss, parts = c3_loss(m, batch, rules=rules)
    assert parts["explanation_only"] > 0
    loss.backward()
    for name in ("reader_universal.k_exp.weight", "reader_company.argue_e.weight",
                 "reader_universal.kappa"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name
