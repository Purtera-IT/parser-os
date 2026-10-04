"""Near-miss contrasts train ml/c3 by comprehension, on invented rows.

* every sentence of a labeled line's WHY becomes a supposition; nothing
  looks for "if", "would" or "unless", so rewording a sentence never changes
  which sentences are supposed;
* the teacher reads each supposition and its answer teaches the heads,
  moved a little by the sentence (teach_flip);
* lines that read almost alike are near misses (different answers) or twins
  (same answers); both give the teacher practice at supposing, from labels;
* the teacher writes each labeled line again (same reason in other words,
  and changed as a WHY sentence says) and reads both; the heads, on the real
  rewritten text, hold on the first and follow the teacher on the second;
  a reworded line the teacher reads differently is dropped;
* the efficiency check scores held-out near misses and twins; the
  labels-only arm and lines masked out carry no suppositions.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.c3.brain import Brain  # noqa: E402
from ml.c3.data import DealExample, featurize, suppositions  # noqa: E402
from ml.c3.efficiency import near_miss_accuracy, twin_consistency  # noqa: E402
from ml.c3.lm import TinyCausalLM  # noqa: E402
from ml.c3.losses import BrainWeights, LossWeights, brain_loss, c3_loss, near_miss_loss, supposed_twins  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.notes import sentences  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402
from ml.c3.supercharge import (  # noqa: E402
    Rewrite, mask_labels, rewrite_lines, supercharge_loss, teach_flip, teach_rewrites, teacher_view)

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
    {"label_key": "k5", "label_type": "task", "supplier": "us", "note": "Same job, bigger."},
]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    deal = DealExample.from_training_blob({"labels": LABELS}, ATOMS, deal_id="synthetic-flips")
    return featurize(deal, schema)


def test_every_sentence_is_supposed_and_no_word_decides():
    why = "Our crew mounts them. If the customer supplied the mounts, this would be a customer task, not ours."
    assert sentences(why) == ["Our crew mounts them.",
                              "If the customer supplied the mounts, this would be a customer task, not ours."]
    # The same reasons in other words: the same sentences are supposed.
    reworded = "Our crew mounts them. Had the client brought the brackets, the job is theirs."
    assert len(suppositions([why])) == len(suppositions([reworded])) == 2
    # A sentence with no "if" is supposed too; the teacher decides what it changes.
    assert [f.condition for f in suppositions(["The price on this line is irrelevant here."])] == \
        ["The price on this line is irrelevant here."]
    assert suppositions(["Too short.", ""]) == []


def test_suppositions_come_from_labeled_lines(batch):
    by_line: dict[int, list[str]] = {}
    for f in batch.flips:
        by_line.setdefault(f.line, []).append(f.condition)
    assert len(by_line[0]) == 2 and by_line[0][1].startswith("If the customer supplied")
    assert by_line[2] and by_line[3]
    assert 4 not in by_line            # "Same job, bigger." is under four words


def test_look_alikes_split_into_near_misses_and_twins(batch):
    # k1/k2 differ by one word and in type; k1/k5 differ by one word, same answers.
    assert (0, 1, "col:label_type") in batch.near_misses
    assert (0, 4) in batch.twins
    assert not [p for p in batch.near_misses if {p[0], p[1]} == {0, 4}]
    assert not [p for p in batch.near_misses + batch.twins if 2 in p[:2] or 3 in p[:2]]


def test_teacher_practices_supposing_on_look_alikes(schema, batch):
    sup, gold = supposed_twins(batch)
    by_key = schema.by_key()
    # Line k1's page with k2's text supposed must answer k2's type, and back.
    assert (0, ATOMS[1]["text"]) in [(f.line, f.condition) for f in sup]
    n = [(f.line, f.condition) for f in sup].index((0, ATOMS[1]["text"]))
    assert gold[n] == ("col:label_type", by_key["col:label_type"].index("dependency"))
    torch.manual_seed(0)
    teacher = Brain(schema, TinyCausalLM(32, 1, 2, 1024), context_lines=2)
    _, parts = brain_loss(teacher, batch, BrainWeights(consolidate=0.0))
    assert parts["flip"] > 0


def test_heads_learn_from_the_teachers_reading(schema, batch):
    torch.manual_seed(0)
    teacher = Brain(schema, TinyCausalLM(32, 1, 2, 1024), context_lines=2)
    view = teacher_view(teacher, batch, grounding=False)
    assert view.flipped is not None
    assert next(iter(view.flipped.values())).shape[0] == len(batch.flips)
    model = C3Model(schema, SMALL)
    out = model(batch.inputs())
    kl, size = teach_flip(model, out, view, batch, 2.0)
    assert kl > 0 and size > 0
    (kl + size).backward()
    assert model.flip_proj.weight.grad is not None
    out = model(batch.inputs())
    taught = supercharge_loss(model, out, view, texts=batch.texts, batch=batch)
    assert taught["teach_flip"] > 0 and taught["teach_flip_size"] > 0
    assert near_miss_loss(out, batch) > 0
    _, parts = c3_loss(model, batch, LossWeights(clause_use=0.0))
    assert parts["near_misses"] > 0 and "flips" not in parts


def test_masked_lines_carry_nothing_and_pairs_are_scored(schema, batch):
    kept = mask_labels(batch, {0, 2})
    assert {f.line for f in kept.flips} <= {0, 2}
    assert kept.near_misses == [] and kept.twins == []
    model = C3Model(schema, SMALL)
    acc, n = near_miss_accuracy(model, batch, [0, 1, 2, 3, 4])
    assert n == len(batch.near_misses) and 0.0 <= acc <= 1.0
    tw, n_tw = twin_consistency(model, batch, [0, 1, 2, 3, 4])
    assert n_tw == len(batch.twins) == 1 and tw in (0.0, 1.0)


def test_meaning_lock_rewrites(schema, batch):
    torch.manual_seed(0)
    teacher = Brain(schema, TinyCausalLM(32, 1, 2, 1024), context_lines=2)
    written = rewrite_lines(teacher, batch, max_new=12)   # untrained: noise, but well formed
    assert all(isinstance(r, Rewrite) and batch.labeled[r.line] for r in written)
    rewrites = [Rewrite(0, "Provider fits 4 wall brackets for the 55 inch screens.", True),
                Rewrite(0, "Customer installs 4 wall mounts for the 55 inch displays.", False),
                Rewrite(3, "Work window is weekends before 8am.", True)]
    view = teacher_view(teacher, batch, grounding=False, rewrites=rewrites)
    assert any(not r.same for r in view.rewrites)          # a changed line is always read
    for n, r in enumerate(view.rewrites):
        if r.same:                                         # held to the original reading
            assert torch.allclose(view.rewritten["col:label_type"][n], view.told["col:label_type"][r.line])
    model = C3Model(schema, SMALL)
    loss = teach_rewrites(model, batch, view, 2.0)
    assert loss > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    out = model(batch.inputs())
    assert supercharge_loss(model, out, view, texts=batch.texts, batch=batch)["teach_rewrites"] > 0


def test_a_near_miss_link_adds_lines_alike_in_meaning_not_wording(schema):
    # k3 and k4 share almost no characters, so trigrams never pair them; a
    # labeler's near_miss link does, at the first answer that differs.
    link = {"from_key": "k4", "to_label_key": "k3", "relation": "near_miss",
            "labeler": "pm@example.com", "note": "Both are about the site, but one is a time window."}
    deal = DealExample.from_training_blob({"labels": LABELS, "links": [link]}, ATOMS, deal_id="synthetic-link")
    b = featurize(deal, schema)
    assert (2, 3, "col:label_type") in b.near_misses
    assert "rel:near_miss" in schema.by_key()


def test_a_draft_why_nobody_saved_counts_less(schema):
    import dataclasses

    from ml.c3.data import DRAFT_WHY_WEIGHT
    from ml.c3.losses import why_echo_loss

    drafted = [dict(lb, reads_set={"why_author": "machine_draft"}) if lb["label_key"] == "k1" else lb
               for lb in LABELS]
    b = featurize(DealExample.from_training_blob({"labels": drafted}, ATOMS, deal_id="synthetic-draft"), schema)
    assert 0 < DRAFT_WHY_WEIGHT < 1
    assert b.why_weights[0] == DRAFT_WHY_WEIGHT and b.why_weights[1:] == [1.0] * 4
    # The answers keep their full weight; only the WHY's terms scale.
    assert b.weights[0] == 1.0
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    out = model(b.inputs())
    full = why_echo_loss(model, out, dataclasses.replace(b, why_weights=[1.0] * len(b)))
    less = why_echo_loss(model, out, b)
    assert not torch.isclose(full, less)
