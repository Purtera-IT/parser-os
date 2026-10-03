"""ml/c3 v5: reasons compiled into operators, graded by outcomes (no training).

* with no explanations the operators change nothing;
* an overruling switches the older reason off inside its own region only;
* a line never applies its own WHY; a reason with zero reliability does nothing;
* closed deals grade the reasons that fired; twins differ only in rule text;
* the explanation-value ranking covers every line;
* every v5 loss reaches its weights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3.ask import explanation_value  # noqa: E402
from ml.c3.consequence import ReliabilityLedger, change_targets  # noqa: E402
from ml.c3.data import DealExample, featurize, realized_changes  # noqa: E402
from ml.c3.explain import Explanation, ExplanationBank, load_rule_cards  # noqa: E402
from ml.c3.losses import c3_loss  # noqa: E402
from ml.c3.model import CHANGE_SLOTS, C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402
from ml.c3.synthetic import COMPANY, role_twins, rule_following, threshold_twins  # noqa: E402

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


def _rule(text, ref, overrules=(), reliability=1.0, concl=()):
    return Explanation(text, "universal", conclusions=concl, ref=ref,
                       overrules=tuple(overrules), reliability=reliability)


def test_an_empty_bank_leaves_every_head_alone(schema, batch):
    m = _model(schema)
    with torch.no_grad():
        a = m(batch.inputs(), company="purtera")
        b = m(batch.inputs(), company="purtera", bank=ExplanationBank())
    for k in a.logits:
        assert torch.equal(a.logits[k], b.logits[k]), k


def test_overruling_switches_the_old_reason_off_only_where_the_new_one_applies(schema, batch):
    m = _model(schema)
    old = _rule("Every quantity line is current.", "old", concl=(("read:superseded", "false"),))
    new = _rule("A quantity stated before a site change is no longer current.", "new",
                overrules=["old"], concl=(("read:superseded", "true"),))
    with torch.no_grad():
        alone = m(batch.inputs(), bank=ExplanationBank([old])).gates["read:superseded"][:, 0]
        both = m(batch.inputs(), bank=ExplanationBank([old, new]))
        g_old, g_new = both.gates["read:superseded"][:, 0], both.gates["read:superseded"][:, 1]
    assert torch.allclose(g_old, alone * (1 - g_new), atol=1e-6)
    assert (g_old < alone).any() and (g_old > 0).all()      # weakened, never deleted


def test_own_why_and_zero_reliability_do_nothing(schema, batch):
    m = _model(schema)
    bank = ExplanationBank.from_batch(batch, schema)
    with torch.no_grad():
        out = m(batch.inputs(), bank=bank)
    g = out.gates["col:label_type"]
    for col, bi in enumerate(out.reason_index["universal"]):
        assert g[bank.items[bi].line, col] == 0
    dead = ExplanationBank([_rule("Any line about displays is a site.", "dead", reliability=0.0,
                                  concl=(("col:label_type", "physical_site"),))])
    with torch.no_grad():
        a = m(batch.inputs())
        b = m(batch.inputs(), bank=dead)
    assert torch.equal(b.gates["col:label_type"], torch.zeros_like(b.gates["col:label_type"]))
    # The vote path still reads it; the operator path must not move the lines.
    assert torch.allclose(b.logits["col:label_type"] - b.explained["col:label_type"],
                          a.logits["col:label_type"], atol=1e-5)


def test_closed_deals_grade_the_reasons_that_fired(schema, batch):
    assert realized_changes(batch.outcome) == {"hours": "up", "price": "up", "sites": "none"}
    t = change_targets(batch)
    assert t[2, CHANGE_SLOTS.index("crew")] == 2 and t[0, 0] == -100
    m = _model(schema)
    bank = load_rule_cards(HERE / "rule_cards.json", schema)
    with torch.no_grad():
        out = m(batch.inputs(), company="purtera", bank=bank)
    ledger = ReliabilityLedger(fire_threshold=0.0)
    graded = ledger.grade(out, batch, bank)
    assert graded and all(0.0 <= v <= 1.0 for v in graded.values())
    rebanked = ledger.apply(bank)
    for e in rebanked.items:
        assert e.reliability == ledger.reliability(e.ref)


def test_twins_differ_only_in_the_rule_text(schema):
    for a, b in (threshold_twins(3), role_twins(3)):
        assert [x.text for x in a.deal.atoms] == [x.text for x in b.deal.atoms]
        assert a.rules.items[0].text != b.rules.items[0].text
        assert a.truth != b.truth
        m = _model(schema)
        assert 0.0 <= rule_following(m, a, schema) <= 1.0
        assert a.deal.company == COMPANY


def test_explanation_value_ranks_every_line(schema, batch):
    m = _model(schema)
    ranked = explanation_value(m, batch, ExplanationBank.from_batch(batch, schema))
    assert sorted(j for j, _ in ranked) == list(range(len(batch)))
    vals = [v for _, v in ranked]
    assert vals == sorted(vals, reverse=True) and min(vals) >= 0


def test_every_v5_loss_reaches_its_weights(schema, batch):
    m = _model(schema)
    m.train()
    loss, parts = c3_loss(m, batch, rules=load_rule_cards(HERE / "rule_cards.json", schema))
    for name in ("changes", "claims", "rule_links", "why_echo"):
        assert parts[name] > 0, name
    loss.backward()
    for name in ("compiler_universal.region_n.weight", "compiler_company.region_n.weight",
                 "compiler_universal.claims.weight", "changes_head.weight", "r_to_text.weight",
                 "compiler_universal.move.weight"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name
