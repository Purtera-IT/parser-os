"""A held reading is unknown, not absent, without hiding its source or other heads."""
import copy
from types import SimpleNamespace

import pytest
import torch

from ml.c3.brain import company_field_notes
from ml.c3.chunks import split_deal
from ml.c3.data import IGNORE, DealExample, featurize
from ml.c3.holds import mask_training_blob
from ml.c3.longdeal import plan_long_deal
from ml.c3.loop import accept_drafts
from ml.c3.losses import LossWeights, head_losses
from ml.c3.schema import load_schema
from ml.c3.why2 import card_rows

KEY = "read:trigger_event"
SENTINEL = "UNCERTAIN_FIELD_REASON_ONLY"


def payload(value="unreviewed event wording"):
    return {"deal_id": "synthetic-field-hold", "atoms": [
        {"label_key": f"k{i}", "atom_id": f"a{i}", "text": f"Synthetic decision {i}.",
         "entered_at": i + 1, "doc_id": "email", "doc_kind": "email"} for i in range(5)],
        "labels": [
            {"id": f"saved{i}", "label_key": f"k{i}", "labeler": "owner", "label_type": "decision",
             "note": "A general decision reason.", "reads_shown": ["trigger_event", "commitment"],
             "rejected_reads": {"trigger_event": SENTINEL} if i == 0 else {},
             "reads_set": {"trigger_event": value if i == 0 else True,
                           "trigger_event_note": SENTINEL if i == 0 else f"A verified event reason {i}.",
                           "co_action": "keep", "co_action_note": "Retain the commercial decision."}}
            for i in range(5)], "links": [], "judgments": [],
        "supervision_holds": {"version": 1, "groups": [
            {"group": "uncertain-read", "source_keys": ["k0"], "read_fields": ["trigger_event"]}]}}


def batch(blob, *, absent=False):
    deal = DealExample.from_training_blob(blob, blob["atoms"], deal_id=blob["deal_id"])
    return deal, featurize(deal, load_schema(), absent_is_negative=absent)


@pytest.mark.parametrize("value", [True, False, "_absent", "unreviewed event wording", None])
@pytest.mark.parametrize("absent", [False, True])
def test_only_held_reading_and_its_notes_are_unknown(value, absent):
    blob = payload(value)
    original = copy.deepcopy(blob)
    _, held = batch(blob, absent=absent)
    bare = copy.deepcopy(blob)
    bare.pop("supervision_holds")
    _, normal = batch(bare, absent=absent)
    assert blob == original
    assert held.targets[KEY][0] == IGNORE  # Even explicit _absent or a missing-as-negative run.
    assert held.targets[KEY][1:] == normal.targets[KEY][1:]
    assert held.targets["col:label_type"] == normal.targets["col:label_type"]
    assert held.targets["read:co_action"] == normal.targets["read:co_action"]
    assert held.targets["read:commitment"] == normal.targets["read:commitment"]
    assert held.why == normal.why and held.labeled == normal.labeled
    assert held.keys == normal.keys
    assert all(torch.equal(a, normal.inputs()[k]) if isinstance(a, torch.Tensor)
               else a == normal.inputs()[k] for k, a in held.inputs().items())
    assert KEY not in held.field_notes[0]
    rows = sum(card_rows(SimpleNamespace(schema=load_schema()), held), [])
    assert not any(i == 0 and k == KEY for i, k, t in rows)
    assert not any(SENTINEL in t for i, k, t in rows)  # No fallback into label_type's note loss.
    assert any(i == 0 and k == "read:co_action" for i, k, t in rows)
    assert not any(SENTINEL in t for notes in company_field_notes(held, load_schema()) for t in notes)


def test_target_ce_has_zero_gradient_only_at_the_held_field():
    schema = load_schema()
    _, held = batch(payload(True))
    keys = [KEY, "col:label_type", "read:co_action"]
    logits = {k: torch.zeros((5, len(schema.by_key()[k].answers)), requires_grad=True) for k in keys}
    out = SimpleNamespace(r=torch.zeros(5, 2), logits=logits, numbers={})
    model = SimpleNamespace(schema=schema)
    loss = sum(head_losses(model, out, held, LossWeights(), layer)[0]
               for layer in ("universal", "company"))
    loss.backward()
    assert torch.equal(logits[KEY].grad[0], torch.zeros_like(logits[KEY].grad[0]))
    assert logits[KEY].grad[1].abs().sum() > 0
    assert all(logits[k].grad[0].abs().sum() > 0 for k in keys[1:])


def test_hold_applies_to_saved_and_approved_draft_rows_without_source_edits():
    blob = payload(True)
    draft = copy.deepcopy(blob["labels"][0])
    draft.update(id="draft", labeler="bot (assistant)", supersedes="saved0")
    blob["labels"].append(draft)
    accept_drafts(blob)
    original = copy.deepcopy(blob)
    _, held = batch(blob)
    assert held.targets[KEY][0] == IGNORE
    assert blob == original


def test_field_and_relation_holds_coexist_and_survive_windows():
    blob = payload(True)
    blob["supervision_holds"]["groups"].append(
        {"group": "existing-relation", "source_keys": ["k1"], "relation": "same_as"})
    deal, _ = batch(blob)
    assert deal.held_relations == [("k1", "same_as")]
    assert deal.held_reads == [("k0", "trigger_event")]
    parts = plan_long_deal(deal, load_schema(), max_atoms=3, overlap=1).parts
    parts += [featurize(c.deal, load_schema()) for c in split_deal(deal, 3)]
    for part in parts:
        if "k0" in part.keys:
            i = part.keys.index("k0")
            assert part.targets[KEY][i] == IGNORE and KEY not in part.field_notes[i]
        assert "held_reads" not in part.inputs()


@pytest.mark.parametrize("extra", [
    {"read_fields": ["missing_read"]}, {"read_fields": []},
    {"read_fields": ["trigger_event"], "relation": "same_as"}])
def test_unknown_or_ambiguous_read_hold_fails_closed(extra):
    blob = payload()
    blob["supervision_holds"]["groups"][0].update(extra)
    with pytest.raises(ValueError, match="read-field"):
        mask_training_blob(blob, blob["atoms"])


@pytest.mark.parametrize("held", [False, True])
def test_field_note_alone_and_contrast_losses_have_no_held_note_gradient(monkeypatch, held):
    from ml.c3 import why2

    blob = payload(True)
    if not held:
        blob.pop("supervision_holds")
    _, b = batch(blob)
    rows = [row for row in sum(card_rows(SimpleNamespace(schema=load_schema()), b), []) if row[1] == KEY]
    # Keep the uncertain note's column in the embedding table. The real note
    # losses must leave it unused when held, rather than relabeling it as type WHY.
    notes = [SENTINEL] + [f"A verified event reason {i}." for i in range(1, 5)]
    col = [notes.index(t) for i, key, t in rows]
    torch.manual_seed(42)
    e = torch.randn(5, 3, requires_grad=True)
    r = torch.randn(5, 3, requires_grad=True)
    out = SimpleNamespace(r=r)
    monkeypatch.setattr(why2, "_stand_in", lambda model, out, lines, e, desc:
                        SimpleNamespace(logits={KEY: e}, numbers={}))
    model = SimpleNamespace(schema=load_schema())
    loss = why2.card_alone_loss(model, out, b, {}, rows, e, col)
    loss = loss + why2.card_contrast_loss(out, rows, e, col)
    loss.backward()
    assert bool(e.grad[0].abs().sum() == 0) == held
    assert bool(r.grad[0].abs().sum() == 0) == held
    assert e.grad[1:].abs().sum() > 0 and r.grad[1:].abs().sum() > 0
