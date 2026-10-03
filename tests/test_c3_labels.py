"""ml/c3 uses all of the labeling (heads-readthrough.md), on invented rows.

* old manual Deal Kit rows never train, whichever way they are marked;
* Missed rows (text the parser skipped) become lines;
* weight_tier weights the loss; the old `rejected` column teaches co_action;
* per-field notes reach the teacher's reasoned pass for the right layer;
* hint_refs teach the pointer head; entity_keys pull lines together.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.c3.brain import company_field_notes, told_why  # noqa: E402
from ml.c3.data import IGNORE, DealExample, featurize  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss, entity_loss, pointer_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

SMALL = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)


def _atom(k, text, t, **kw):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text,
            "entered_at": f"2026-05-01T10:{t:02d}:00Z", "doc_kind": kw.get("doc_kind", "email"),
            "speaker_role": "customer", "speaker_side": "customer"}


ATOMS = [
    _atom("k1", "We need six panels hung at the north office.", 0),
    _atom("k2", "Old kit row: 6 displays, 2 techs, 1 day.", 1, doc_kind="deal_kit"),
    _atom("k3", "SOW: install six (6) displays at the north office.", 2, doc_kind="sow"),
    _atom("k4", "Hardware quote: panel, qty 6, $900 each.", 3, doc_kind="quote"),
    _atom("k5", "The north office loading dock closes at 3pm.", 4),
]

BLOB = {"labels": [
    {"label_key": "k1", "label_type": "_keep", "about": "deal", "weight_tier": "load_bearing",
     "entity_keys": ["site:north"], "note": "The job: 6 panels at one site. Sets quantity and site.",
     "reads_set": {"site": "north office", "address_note": "City-level only; no street given."}},
    {"label_key": "k2", "label_type": "_keep",
     "note": "[EXCLUDE_FROM_TRAINING: old manual Deal Kit] Old kit line.",
     "reads_set": {"equipment_qty": "6"}},
    {"label_key": "k3", "label_type": "_keep", "entity_keys": ["site:north"],
     "hint_refs": [{"atomId": "at-k1", "hint": "equipment_qty", "text": "six panels hung"}],
     "note": "The SOW carries the customer's six panels at the north office.",
     "reads_set": {"sow_coverage": "in_sow",
                   "sow_coverage_note": "Matches the customer's ask of six panels, same site.",
                   "scope_category_note": "Our display-install SOW group."}},
    {"label_key": "k4", "label_type": "_keep", "weight_tier": "slight", "rejected": "true",
     "note": "Reseller hardware line; qty 6 agrees with the SOW, price is not ours.",
     "reads_set": {"equipment_qty": "6"}},
    {"label_key": "k5", "label_type": "_keep", "weight_tier": "exclude",
     "reads_set": {"site": "north office"}},
    {"label_key": "k9", "origin": "labeler", "label_type": "_keep",
     "text": "Customer staff will move the old screens out first.",
     "entered_at": "2026-05-01T10:05:00Z", "doc_kind": "email",
     "note": "Missed by the parser: a customer-side step before install.",
     "reads_set": {"scope_side": "customer"}},
]}


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    deal = DealExample.from_training_blob(BLOB, ATOMS, deal_id="synthetic-labels")
    return featurize(deal, schema)


def _i(batch, word):
    return next(i for i, t in enumerate(batch.texts) if word in t)


def test_excluded_rows_never_train_but_stay_as_context(batch):
    for word in ("Old kit row", "loading dock"):
        i = _i(batch, word)
        assert not batch.labeled[i] and batch.why[i] is None
        assert all(col[i] == IGNORE for col in batch.targets.values())


def test_missed_rows_become_lines(batch):
    i = _i(batch, "move the old screens")
    assert batch.labeled[i] and batch.why[i]
    assert batch.targets["read:scope_side"][i] != IGNORE


def test_weight_tier_and_the_old_rejected_column(schema, batch):
    assert batch.weights[_i(batch, "six panels hung")] == 3.0
    assert batch.weights[_i(batch, "Hardware quote")] == 0.3
    co = schema.by_key()["read:co_action"]
    assert batch.targets["read:co_action"][_i(batch, "Hardware quote")] == co.index("reject")


def test_field_notes_reach_the_teacher_on_the_right_side(schema, batch):
    i = _i(batch, "SOW: install")
    assert batch.field_notes[i]["read:sow_coverage"].startswith("Matches")
    why = told_why(batch, schema)[i]
    assert "sow_coverage: Matches" in why and "SOW group" not in why
    assert any("SOW group" in t for t in company_field_notes(batch, schema)[i])
    j = _i(batch, "six panels hung")
    assert "address_level: City-level" in told_why(batch, schema)[j]


def test_hint_refs_and_entity_keys_teach_the_heads(schema, batch):
    i, j = _i(batch, "SOW: install"), _i(batch, "six panels hung")
    assert batch.hint_lines[i] == [j]
    torch.manual_seed(0)
    m = C3Model(schema, SMALL)
    m.train()
    out = m(batch.inputs())
    p, e = pointer_loss(out, batch), entity_loss(out, batch)
    assert p > 0 and e > 0
    (p + e).backward()
    assert m.cite_q.weight.grad.abs().sum() > 0 and m.rationale[0].weight.grad.abs().sum() > 0
    _, parts = c3_loss(C3Model(schema, SMALL), batch, LossWeights(clause_use=0.0))
    assert parts["pointers"] > 0 and parts["entities"] > 0
