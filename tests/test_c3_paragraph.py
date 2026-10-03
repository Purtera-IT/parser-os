"""ml/c3 full paragraphs: every clause of a written explanation is used (no training).

* the clause splitter keeps every word and gives each clause its job;
* an "unless" clause can only shrink where a rule applies;
* deleting any condition, exception, cause or consequence clause changes the answers;
* each line reads the explanation's exact words, differently per line;
* arithmetic in a paragraph becomes executable programs;
* the clause-use loss reaches the clause weights.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3.clauses import split_clauses  # noqa: E402
from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.explain import Explanation, ExplanationBank, load_rule_cards  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "ml" / "c3" / "fixtures"
SMALL = C3Config(d_text=64, d=64, n_layers=2, n_heads=4, d_r=32, d_q=16, d_head=32,
                 residual_dim=8, box_dim=8, n_policy_atoms=4, policy_rank=4)

PARAGRAPH = ("Each clinic gets four 65-inch displays. Because a 65-inch panel is a two-person "
             "lift, every mount needs two technicians, so the crew doubles and install hours go "
             "up, unless the customer's own staff lift the panels. Matches the reseller quote: "
             "8 displays. 2 sites x 4 = 8 displays.")


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


def _bank(text, concl=(("read:superseded", "true"),)):
    return ExplanationBank([Explanation(text, "universal", conclusions=concl, ref="p")])


def test_every_word_survives_the_split_and_each_clause_has_a_job():
    words = lambda t: re.findall(r"[a-z0-9]+", t.lower())  # noqa: E731
    cs = split_clauses(PARAGRAPH)
    assert sum((words(c.text) for c in cs), []) == words(PARAGRAPH)
    roles = [c.role for c in cs]
    assert roles == ["statement", "cause", "consequence", "consequence", "exception",
                     "evidence", "quantity"]


def test_an_unless_clause_only_shrinks_where_the_rule_applies(schema, batch):
    m = _model(schema)
    base = "When an earlier line states a site count, a later site change replaces it."
    with torch.no_grad():
        a = m(batch.inputs(), bank=_bank(base)).gates["read:superseded"]
        b = m(batch.inputs(), bank=_bank(base + " unless the later line is only a question.")
              ).gates["read:superseded"]
    assert (b <= a + 1e-6).all() and (b < a - 1e-6).any()


def test_deleting_any_working_clause_changes_the_answers(schema, batch):
    m = _model(schema)
    cs = split_clauses(PARAGRAPH)
    with torch.no_grad():
        full = m(batch.inputs(), bank=_bank(PARAGRAPH))
        for i, c in enumerate(cs):
            if c.role == "statement":
                continue
            shorter = " ".join(x.text for j, x in enumerate(cs) if j != i)
            out = m(batch.inputs(), bank=_bank(shorter))
            moved = any(not torch.allclose(full.logits[k], out.logits[k], atol=1e-6)
                        for k in full.gates)
            assert moved, (c.role, c.text)


def test_each_line_reads_its_own_words_and_arithmetic_becomes_programs(schema, batch):
    m = _model(schema)
    with torch.no_grad():
        emb = m.text([PARAGRAPH])
        ops = m.compiler_universal.compile(emb, texts=[PARAGRAPH], encoder=m.text)
        x = torch.randn(5, SMALL.d_head)
        detail = m.compiler_universal.detail(x, ops)
    assert detail.shape == (5, 1, SMALL.d_head)
    assert not torch.allclose(detail[0, 0], detail[1, 0])
    progs = ops["programs"][0]
    assert [(p.operands, p.stated) for p in progs] == [((2.0, 4.0), 8.0)]
    assert [c.role for c in ops["clauses"][0]][-1] == "quantity"


def test_clause_use_loss_reaches_the_clause_weights(schema, batch):
    m = _model(schema)
    m.train()
    rules = load_rule_cards(HERE / "rule_cards.json", schema)
    rules = rules.extend([Explanation(PARAGRAPH, "universal", ref="para")])
    loss, parts = c3_loss(m, batch, LossWeights(clause_use=1.0), rules=rules)
    assert "clause_use" in parts
    loss.backward()
    for name in ("compiler_universal.cond_n.weight", "compiler_universal.clause_move.weight",
                 "compiler_universal.tok_v.weight", "compiler_universal.exc_n.weight"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name
