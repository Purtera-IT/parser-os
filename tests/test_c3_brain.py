"""ml/c3 v6: a language model reads the explanations (no training).

* every v6 loss term is finite and reaches the LM, the head readers and the
  paragraph memory;
* the WHY paragraph changes the answers, and every word of it reaches them;
* a paragraph written into weights acts without being on the page, and no
  paragraphs leave the LM bit-identical;
* company notes and company memory never move a universal answer;
* nothing on the v6 path imports the keyword clause splitter.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3.brain import Brain, notes_from_deal  # noqa: E402
from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.losses import BrainWeights, brain_loss  # noqa: E402
from ml.c3.lm import TinyCausalLM  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "ml" / "c3" / "fixtures"

PARAGRAPH = ("Each clinic gets four 65-inch displays. Because a 65-inch panel is a two-person "
             "lift, every mount needs two technicians, so the crew doubles and install hours go "
             "up, unless the customer's own staff lift the panels.")


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    return featurize(DealExample.load(HERE / "synthetic_deal.json"), schema)


def _brain(schema, d_graph=None):
    torch.manual_seed(0)
    b = Brain(schema, TinyCausalLM(dim=32, layers=1, heads=2, max_len=1024),
              d_graph=d_graph, context_lines=2)
    b.eval()
    return b


def test_every_brain_loss_reaches_the_weights(schema, batch):
    b = _brain(schema)
    b.train()
    loss, parts = brain_loss(b, batch, BrainWeights())
    for k in ("direct", "reasoned", "rationale", "distill", "consolidate"):
        assert k in parts and parts[k] == parts[k], k        # present and not NaN
    assert parts["rationale"] > 0 and parts["consolidate"] >= 0
    loss.backward()
    named = dict(b.named_parameters())
    for name in ("lm.tok.weight", "lm.blocks.0.q.base.weight", "reader_universal.q.weight",
                 "reader_company.ans.weight", "memory_universal.coef.weight",
                 "memory_company.coef.weight"):
        g = named[name].grad
        assert g is not None and g.abs().sum() > 0, name


def test_the_why_paragraph_changes_the_answers_and_every_word_reaches_them(schema, batch):
    b = _brain(schema)
    desc = {k: (e.detach(), a.detach()) for k, (e, a) in b.describe().items()}
    n = len(batch)
    with torch.no_grad():
        bare = b(batch, desc=desc)
        told = b(batch, why=[PARAGRAPH] * n, desc=desc)
    key = "col:label_type"
    assert not torch.allclose(bare.logits[key], told.logits[key])

    # Gradient of one head's answer w.r.t. every token of the paragraph:
    # nothing is cut away or ignored by construction.
    page = b.page(batch, 0) + "Why: "
    ids, mask = b.lm.tokenize([page + PARAGRAPH + "\nAnswer:"])
    start = int(b.lm.tokenize([page])[1].sum())
    end = int(b.lm.tokenize([page + PARAGRAPH])[1].sum())
    emb = b.lm.input_embeddings(ids).detach().requires_grad_(True)
    h, _ = b.lm(ids, mask, inputs_embeds=emb)
    opps = b.opps["universal"]
    logits, _, _ = b.reader_universal(h, mask, torch.stack([desc[o.key][0] for o in opps]),
                                      [desc[o.key][1] for o in opps])
    logits[[o.key for o in opps].index(key)].max().backward()
    per_token = emb.grad[0, start:end].abs().sum(-1)
    assert (per_token > 0).all()


def test_a_paragraph_in_weights_acts_off_the_page_and_none_changes_nothing(schema, batch):
    b = _brain(schema)
    desc = {k: (e.detach(), a.detach()) for k, (e, a) in b.describe().items()}
    with torch.no_grad():
        base = b(batch, desc=desc)
        empty = b(batch, desc=desc, deltas=b.memory_universal(b.lm, []))
        kept = b(batch, desc=desc, deltas=b.memory_universal(b.lm, [PARAGRAPH]))
        two = b.memory_universal(b.lm, [PARAGRAPH, "Sellers never own delivery tasks."])
    for k in base.logits:
        assert torch.equal(base.logits[k], empty.logits[k])
    assert not torch.allclose(base.logits["col:label_type"], kept.logits["col:label_type"])
    one = b.memory_universal(b.lm, [PARAGRAPH])
    assert two[0][0].shape[0] == 2 * one[0][0].shape[0]       # ranks stack: effects add


def test_company_text_never_moves_a_universal_answer(schema, batch):
    b = _brain(schema)
    desc = {k: (e.detach(), a.detach()) for k, (e, a) in b.describe().items()}
    n = len(batch)
    rule = "[purtera] reject any line a seller wrote about delivery tasks."
    with torch.no_grad():
        plain = b(batch, company="purtera", desc=desc)
        loud = b(batch, company="purtera", desc=desc, company_notes=[[rule]] * n,
                 company_deltas=b.memory_company(b.lm, [rule]))
    uni = [o.key for o in b.opps["universal"]]
    com = [o.key for o in b.opps["company"]]
    assert uni and com
    for k in uni:
        assert torch.equal(plain.logits[k], loud.logits[k]), k
    assert any(not torch.allclose(plain.logits[k], loud.logits[k]) for k in com)


def test_a_line_never_reads_its_own_why_in_its_notes(batch):
    uni, _ = notes_from_deal(batch, k=100)
    for i, ns in enumerate(uni):
        if batch.why[i]:
            assert batch.why[i] not in ns or batch.why.count(batch.why[i]) > 1


def test_the_deal_graph_reaches_the_brain(schema, batch):
    torch.manual_seed(0)
    cfg = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                   residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
    c3 = C3Model(schema, cfg).eval()
    b = _brain(schema, d_graph=cfg.d_r)
    with torch.no_grad():
        r = c3(batch.inputs()).r
        desc = b.describe()
        a = b(batch, graph=r, desc=desc).logits["col:label_type"]
        z = b(batch, graph=torch.zeros_like(r), desc=desc).logits["col:label_type"]
    assert not torch.allclose(a, z)


def test_the_v6_path_has_no_keyword_rules():
    for f in ("brain.py", "lm.py", "consolidate.py"):
        tree = ast.parse((ROOT / "ml" / "c3" / f).read_text())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not mods & {"clauses", "notes", "operators", "explain", "re"}, f
        assert not any(isinstance(n, ast.Import) and any(a.name == "re" for a in n.names)
                       for n in ast.walk(tree)), f
