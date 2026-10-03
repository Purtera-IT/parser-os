"""ml/c3 uses all of the labeling (heads-readthrough.md), on invented rows.

* old manual Deal Kit rows never train, whichever way they are marked;
* Missed rows (text the parser skipped) become lines;
* weight_tier weights the loss; a reject with no company decision teaches
  admission (not a fact); a company reject (co_action) trains only the
  Purtera layer, its ruled-out readings and why-not included;
* train_for on our own Deal Kit lines trains the Purtera layer, not the base;
* readings the labeler removed or ruled out are taught as absent, and the
  why-not reaches the teacher;
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
    _atom("k6", "Reseller line: mounts, qty 6, $40 each.", 6, doc_kind="quote"),
    _atom("k7", "Kit plan: 2 techs on site for 1 day.", 7, doc_kind="Deal Kit"),
    _atom("k8", "Thanks again, talk soon!", 8),
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
    {"label_key": "k4", "label_type": "bom_line", "weight_tier": "slight", "rejected": "true",
     "hints": ["table_column", "own_words"],
     "note": "Reseller hardware line; qty 6 agrees with the SOW, price is not ours.",
     "reads_set": {"equipment_qty": "6"}, "reads_shown": ["equipment_qty", "rate"],
     "rejected_reads": {"billing_type": "A unit price on a hardware line is not how the job bills."}},
    {"label_key": "k5", "label_type": "_keep", "weight_tier": "exclude",
     "reads_set": {"site": "north office"}},
    {"label_key": "k9", "origin": "labeler", "label_type": "_keep",
     "text": "Customer staff will move the old screens out first.",
     "entered_at": "2026-05-01T10:05:00Z", "doc_kind": "email",
     "note": "Missed by the parser: a customer-side step before install.",
     "reads_set": {"scope_side": "customer"}},
    {"label_key": "k6", "label_type": "bom_line", "rejected": "true",
     "note": "Mount line from the reseller; qty agrees with six panels. [purtera] Hardware prices are not ours.",
     "reads_set": {"equipment_qty": "6", "co_action": "reject", "co_reason": "hw_price_not_ours"},
     "reads_shown": ["equipment_qty", "rate"],
     "rejected_reads": {"billing_type": "We never bill from a reseller's unit price."}},
    {"label_key": "k7", "label_type": "_keep", "note": "Crew plan for the install: 2 techs, 1 day.",
     "reads_set": {"crew_size": "2", "train_for": ["delivery_parser"]}},
    {"label_key": "k8", "label_type": "small_talk", "reads_set": {"noise_class": "greeting_thanks"}},
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


def test_weight_tier_reject_and_known_negatives(schema, batch):
    by = schema.by_key()
    hw, ask = _i(batch, "Hardware quote"), _i(batch, "six panels hung")
    assert batch.weights[ask] == 3.0 and batch.weights[hw] == 0.3
    # The card's reject means "not a fact": admission drop, not a company reject.
    assert batch.targets["col:admission"][hw] == by["col:admission"].index("drop")
    assert batch.targets["read:co_action"][hw] == IGNORE
    assert batch.targets["col:admission"][_i(batch, "move the old screens")] == by["col:admission"].index("keep")
    # Removed by the labeler (rate) and ruled out (billing_type): known absent.
    assert batch.targets["read:rate"][hw] == by["read:rate"].index("_absent")
    assert batch.targets["read:billing_type"][hw] == by["read:billing_type"].index("_absent")
    assert batch.field_notes[hw]["read:billing_type"].startswith("not billing_type:")
    assert batch.targets["col:hints"][hw] == by["col:hints"].index("table_column")
    assert batch.targets["col:admission"][_i(batch, "Thanks again")] == by["col:admission"].index("drop")


def test_company_reject_stays_in_the_purtera_layer(schema, batch):
    by = schema.by_key()
    i = _i(batch, "Reseller line")
    # A real fact we drop on purpose: the base keeps it, co_action rejects it.
    assert batch.targets["col:admission"][i] == by["col:admission"].index("keep")
    assert batch.targets["read:co_action"][i] == by["read:co_action"].index("reject")
    # Base readings removed or ruled out under our reject are not taught absent,
    # and the why-not reaches only the company pass.
    assert batch.targets["read:rate"][i] == IGNORE
    assert batch.targets["read:billing_type"][i] == IGNORE
    assert "read:billing_type" not in batch.field_notes[i]
    assert "never bill" not in (told_why(batch, schema)[i] or "")
    assert "not billing_type: We never bill" in batch.policy_note[i]


def test_deal_kit_routing_trains_only_the_purtera_layer(schema, batch):
    by = schema.by_key()
    i = _i(batch, "Kit plan")
    assert by["read:co_deal_kit_route"].layer == "company" and by["read:train_for"].universal
    assert batch.targets["read:train_for"][i] == IGNORE
    assert batch.targets["read:co_deal_kit_route"][i] == by["read:co_deal_kit_route"].index("delivery_parser")
    assert batch.targets["read:crew_size"][i] != IGNORE


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
