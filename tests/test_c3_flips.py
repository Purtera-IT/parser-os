"""Near-miss contrasts train ml/c3, on invented rows.

* "if X, this would be Y" in a WHY becomes a flip: Y is the registered answer
  the sentence names, on a head where it differs from the line's gold;
  "unless X" (or a Y that names nothing registered) means "not the gold type";
* lines that read almost alike but were labeled differently become
  near-miss pairs; alike lines with the same answer, and unlike lines, do not;
* the heads learn both (flip_loss, near_miss_loss), the teacher has a flip
  pass and teaches it (teach_flip), and the efficiency check scores pairs;
* the labels-only arm and lines masked out carry no flips.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.c3.brain import Brain  # noqa: E402
from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.efficiency import near_miss_accuracy  # noqa: E402
from ml.c3.lm import TinyCausalLM  # noqa: E402
from ml.c3.losses import BrainWeights, LossWeights, brain_loss, c3_loss, flip_loss, near_miss_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.notes import extract_flips  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402
from ml.c3.supercharge import mask_labels, supercharge_loss, teacher_view  # noqa: E402

SMALL = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)


def _atom(k, text, t):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": "sow-1", "doc_kind": "sow",
            "entered_at": f"2026-05-01T10:{t:02d}:00Z", "speaker_role": "customer",
            "speaker_side": "customer"}


ATOMS = [
    _atom("k1", "Provider installs 4 wall mounts for the 55 inch displays.", 0),
    _atom("k2", "Customer installs 4 wall mounts for the 55 inch displays.", 1),
    _atom("k3", "Site list: 12 Oak Road, Springfield.", 2),
    _atom("k4", "Work window is weekdays after 6pm.", 3),
    _atom("k5", "Provider installs 4 wall mounts for the 65 inch displays.", 4),
]
LABELS = [
    {"label_key": "k1", "label_type": "task", "supplier": "us",
     "note": "Our crew mounts four 55 inch displays. If the customer supplied the mounts, "
             "this would be a customer task, not ours."},
    {"label_key": "k2", "label_type": "dependency", "supplier": "customer",
     "note": "The customer's own staff do the mounting before we arrive, so our work waits on it."},
    {"label_key": "k3", "label_type": "physical_site",
     "note": "A street address on the customer's site list stays in scope unless a later dated "
             "correction removes it."},
    {"label_key": "k4", "label_type": "site_access_window",
     "note": "When the crew may work. It would be a deadline if it named a finish date."},
    {"label_key": "k5", "label_type": "task", "supplier": "us", "note": "Same job, larger displays."},
]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    deal = DealExample.from_training_blob({"labels": LABELS}, ATOMS, deal_id="synthetic-flips")
    return featurize(deal, schema)


def test_flip_sentences_are_found_and_others_are_not():
    f = extract_flips("Our crew mounts them. If the customer supplied the mounts, this would be "
                      "a customer task, not ours.")
    assert [(x.condition, x.then) for x in f] == [("the customer supplied the mounts", "a customer task")]
    assert extract_flips("It would be hardware if the line named a part number.")[0].then == "hardware"
    assert extract_flips("In scope unless a later correction removes it.")[0].then == ""
    assert extract_flips("If they want it, we quote it. The price is irrelevant.") == []
    # A phrase the WHY quotes from a document is not the labeler's contrast.
    assert extract_flips("'Unless separately agreed' leaves a door open, but nothing prices it.") == []
    assert extract_flips('The SOW says "unless otherwise agreed". It stays ours.') == []


def test_flips_resolve_to_registered_answers(schema, batch):
    by_key = schema.by_key()
    got = {(f.line, f.key, f.answer) for f in batch.flips}
    sup = by_key["col:supplier"]
    # "a customer task" names supplier=customer (gold: us).
    assert (0, "col:supplier", sup.index("customer")) in got
    # "unless ..." says only that the type would change.
    assert (2, "col:label_type", None) in got
    # "would be a deadline" names a registered type.
    assert (3, "col:label_type", by_key["col:label_type"].index("deadline")) in got
    assert not [f for f in batch.flips if f.line in (1, 4)]


def test_near_misses_pair_alike_lines_that_answer_differently(batch):
    # k1/k2 differ by one word and in type; k1/k5 are alike but answer the same.
    assert (0, 1, "col:label_type") in batch.near_misses
    assert not [p for p in batch.near_misses if {p[0], p[1]} == {0, 4}]
    assert not [p for p in batch.near_misses if 2 in p[:2] or 3 in p[:2]]


def test_heads_learn_flips_and_near_misses(schema, batch):
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    out = model(batch.inputs())
    flip, size, sub, move = flip_loss(model, out, batch, model.describe())
    assert flip > 0 and size > 0 and move.shape[0] == len(batch.flips)
    near = near_miss_loss(out, batch)
    assert near > 0
    (flip + near).backward()
    assert model.flip_proj.weight.grad is not None
    _, parts = c3_loss(model, batch, LossWeights(clause_use=0.0))
    assert parts["flips"] > 0 and parts["near_misses"] > 0


def test_teacher_flip_pass_and_teach_flip(schema, batch):
    torch.manual_seed(0)
    teacher = Brain(schema, TinyCausalLM(32, 1, 2, 1024), context_lines=2)
    _, parts = brain_loss(teacher, batch, BrainWeights(consolidate=0.0))
    assert parts["flip"] > 0
    view = teacher_view(teacher, batch, grounding=False)
    assert view.flipped is not None
    assert next(iter(view.flipped.values())).shape[0] == len(batch.flips)
    model = C3Model(schema, SMALL)
    out = model(batch.inputs())
    taught = supercharge_loss(model, out, view, texts=batch.texts, batch=batch)
    assert taught["teach_flip"] > 0


def test_masked_lines_carry_no_flips_or_pairs_and_pairs_are_scored(schema, batch):
    kept = mask_labels(batch, {0, 2})
    assert {f.line for f in kept.flips} <= {0, 2} and kept.near_misses == []
    model = C3Model(schema, SMALL)
    acc, n = near_miss_accuracy(model, batch, [0, 1, 2, 3, 4])
    assert n == len(batch.near_misses) and 0.0 <= acc <= 1.0
