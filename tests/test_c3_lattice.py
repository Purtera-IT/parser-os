"""Asked attention (ml/c3/lattice.py), on invented rows.

* each line gets one slot per question space; heads read their own slot;
* slots look only at lines that entered no later (foresight), and their
  attention is a citation over those lines;
* the valve: base answers are bit-identical whatever the company, and with
  or without a company at all; the company answers do change;
* what-if at run time (C3Model.suppose) moves only the supposed lines.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

ASKED = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2, asked_layers=2)


def _atom(k, text, t, doc="sow-1"):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": doc, "doc_kind": "sow",
            "entered_at": f"2026-05-01T10:{t:02d}:00Z", "speaker_role": "customer",
            "speaker_side": "customer"}


ATOMS = [_atom("k1", "Site list: 12 Oak Road, Springfield.", 0),
         _atom("k2", "Provider installs 4 wall mounts per room.", 1),
         _atom("k3", "Rooms: 101, 102, 103.", 2, doc="email-1"),
         _atom("k4", "Price is $920 per day.", 3, doc="email-1")]
LABELS = [{"label_key": "k1", "label_type": "physical_site", "note": "A street address on the list."},
          {"label_key": "k2", "label_type": "task", "supplier": "us", "note": "Our crew mounts them."}]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    return featurize(DealExample.from_training_blob({"labels": LABELS}, ATOMS, deal_id="lattice"), schema)


def test_slots_and_foresight_citations(schema, batch):
    torch.manual_seed(0)
    m = C3Model(schema, ASKED).eval()
    with torch.no_grad():
        out = m(batch.inputs())
    n, s = len(batch), len(m.lattice.spaces)
    assert out.slots.shape[:2] == (n, s) and out.citations.shape == (n, s, n)
    assert torch.allclose(out.citations.sum(-1), torch.ones(n, s))
    later = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    assert out.citations.transpose(0, 1)[:, later].abs().sum() == 0   # never cites a later line
    # The first line can only look at itself, so changing a later line leaves it alone.
    inp = batch.inputs()
    inp2 = {**inp, "texts": inp["texts"][:3] + ["Price is $1,500 per day."]}
    with torch.no_grad():
        out2 = m(inp2)
    assert torch.equal(out.slots[0], out2.slots[0]) and not torch.equal(out.slots[3], out2.slots[3])


def test_the_valve_keeps_base_answers_bit_identical(schema, batch):
    torch.manual_seed(0)
    m = C3Model(schema, ASKED, companies=("purtera", "other_co")).eval()
    with torch.no_grad():
        m.alpha[1].normal_()
        inp = batch.inputs()
        a, b, none = m(inp, company="purtera"), m(inp, company="other_co"), m(inp)
    uni = [o.key for o in schema.select(layer="universal") if o.key in a.logits]
    com = [o.key for o in schema.select(layer="company") if o.key in a.logits]
    assert uni and com
    for k in uni:
        assert torch.equal(a.logits[k], b.logits[k]) and torch.equal(a.logits[k], none.logits[k])
    assert torch.equal(a.slots, none.slots) and a.company_slot is not None and none.company_slot is None
    assert any(not torch.equal(a.logits[k], b.logits[k]) for k in com)


def test_trains_end_to_end_and_suppose_moves_only_the_supposed_line(schema, batch):
    torch.manual_seed(0)
    m = C3Model(schema, ASKED)
    loss, _ = c3_loss(m, batch, LossWeights(clause_use=0.0))
    loss.backward()
    assert m.lattice.layers[0].deal_attn.q.weight.grad is not None
    assert m.lattice.seed.weight.grad.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        out = m(batch.inputs())
    what_if = m.suppose(out, [1], ["The customer's own staff do the mounting."])
    k = "col:label_type"
    assert what_if[k].shape == (1, out.logits[k].shape[1])
    assert not torch.equal(what_if[k][0], out.logits[k][1])
