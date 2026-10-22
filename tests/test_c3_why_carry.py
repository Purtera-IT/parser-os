"""The WHY carries the decision, teacher-free (ml/c3/why_carry.py), on invented rows.

* surface cues (field labels, header words, "site", file extensions) are
  masked or paraphrased; numbers stay; a line with nothing to mask is left;
* identical WHYs are one column of the contrast, never each other's negative;
  the near-miss partner's WHY is a hard negative;
* a swapped WHY must flip the answer: only the heads learn from it (r, h and
  the encoder get nothing), and twins must hold;
* "the WHY alone" really is alone: the line's own h cannot move it;
* the parts are named LossWeights fields, on by default, and 0 turns each off.
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402
from ml.c3.why_carry import (  # noqa: E402
    _columns, _why_rows, keyword_dropout_loss, mask_surface, swap_pairs, why_alone_loss,
    why_contrast_loss, why_swap_loss)

SMALL = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
SITE_WHY = "One building on the customer's list of places to survey; work happens there."


def _atom(k, text, t):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": "sow-1", "doc_kind": "sow",
            "entered_at": f"2026-05-01T10:{t:02d}:00Z", "speaker_role": "customer",
            "speaker_side": "customer"}


ATOMS = [
    _atom("k1", "Provider installs 4 wall mounts for the 55 inch displays.", 0),
    _atom("k2", "Customer installs 4 wall mounts for the 55 inch displays.", 1),
    _atom("k3", "City: CONROE | Zip: 77304 | Floors: 2 | Total Hrs: 6.5", 2),
    _atom("k4", "City: TOMBALL | Zip: 77375 | Floors: 1 | Total Hrs: 4.0", 3),
    _atom("k5", "Provider installs 4 wall mounts for the 65 inch displays.", 4),
    _atom("k6", "Work window is weekdays after 6pm.", 5),
]
LABELS = [
    {"label_key": "k1", "label_type": "task", "supplier": "us",
     "note": "Our crew mounts four 55 inch displays, so the labor is ours."},
    {"label_key": "k2", "label_type": "dependency", "supplier": "customer",
     "note": "The customer's own staff mount them before we arrive, so our work waits on it."},
    {"label_key": "k3", "label_type": "physical_site", "note": SITE_WHY},
    {"label_key": "k4", "label_type": "physical_site", "note": SITE_WHY},
    {"label_key": "k5", "label_type": "task", "supplier": "us", "note": "Same job as the 55 inch, bigger screens."},
    {"label_key": "k6", "label_type": "site_access_window",
     "note": "When the crew may work. It would be a deadline if it named a finish date."},
]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    deal = DealExample.from_training_blob({"labels": LABELS}, ATOMS, deal_id="synthetic-why-carry")
    return featurize(deal, schema)


def test_surface_cues_are_masked_and_numbers_stay():
    rng = random.Random(0)
    row = "City: CONROE | Zip: 77304 | Total Hrs: 6.5"
    seen = {mask_surface(row, rng, word_drop=0.0) for _ in range(30)}
    assert all(t and "77304" in t and "6.5" in t and "CONROE" in t for t in seen)
    assert any("City:" not in t for t in seen) and len(seen) > 3
    t = mask_surface("Attached is the site list.csv", random.Random(1), word_drop=0.0)
    assert t is not None and ".csv" not in t and " site " not in f" {t} "
    assert mask_surface("1234 5678", random.Random(0)) is None


def test_identical_whys_share_one_column(batch):
    idx = _why_rows(batch)
    texts, col = _columns(batch, idx)
    assert len(texts) == len(idx) - 1          # k3 and k4 share the site WHY
    k3, k4 = idx.index(2), idx.index(3)
    assert col[k3] == col[k4]


def test_contrast_hard_negative_raises_the_loss(schema, batch):
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    out = model(batch.inputs())
    assert (0, 1, "col:label_type") in batch.near_misses
    soft = why_contrast_loss(model, out, batch, hard=1.0)
    hard = why_contrast_loss(model, out, batch, hard=8.0)
    assert torch.isfinite(soft) and hard > soft
    hard.backward()
    assert model.why_proj.weight.grad is not None


def test_swap_flips_and_holds_and_trains_only_the_heads(schema, batch):
    pairs = swap_pairs(batch, cap=8, rng=random.Random(0))
    assert (0, 1, "col:label_type", True) in pairs and (1, 0, "col:label_type", True) in pairs
    assert any(not flip for *_, flip in pairs)             # k1/k5 twins hold
    assert not [p for p in pairs if {p[0], p[1]} == {2, 3}]  # one shared WHY: nothing to swap
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    out = model(batch.inputs())
    loss = why_swap_loss(model, out, batch, model.describe(), cap=8, rng=random.Random(0))
    assert loss > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.heads.parameters())
    # The line state's own layers get nothing (the described heads may move
    # their description encodings, which a released encoder shares).
    trunk = [p for n, p in model.encoder.named_parameters() if not n.startswith("text.")]
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in trunk)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.rationale.parameters())
    assert model.why_proj.weight.grad is None


def test_why_alone_ignores_the_lines_own_text(schema, batch):
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    lin = model.add_free_head("col:label_type")
    torch.nn.init.normal_(lin.weight)
    model.eval()
    desc = model.describe()
    out = model(batch.inputs(), desc=desc)
    a = why_alone_loss(model, out, batch, desc)
    out.h = out.h + 5.0 * torch.randn_like(out.h)
    b = why_alone_loss(model, out, batch, desc)
    assert torch.allclose(a, b)


def test_keyword_dropout_and_parts_in_c3_loss(schema, batch):
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    out = model(batch.inputs())
    w = replace(LossWeights(), keyword_dropout_p=1.0)
    loss = keyword_dropout_loss(model, out, batch, w, rng=random.Random(0))
    assert torch.isfinite(loss) and loss > 0
    total, parts = c3_loss(model, batch, LossWeights(clause_use=0.0))
    assert {"why_contrast", "why_swap", "why_alone", "keyword_dropout"} <= set(parts)
    assert torch.isfinite(total)
    off = replace(LossWeights(clause_use=0.0), why_contrast=0.0, why_swap=0.0, why_alone=0.0,
                  keyword_dropout=0.0)
    _, parts = c3_loss(model, batch, off)
    assert not {"why_contrast", "why_swap", "why_alone", "keyword_dropout"} & set(parts)
