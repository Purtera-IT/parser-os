"""The Run E SOW and site heads: read:visit_count, read:work_action,
read:sow_element, read:sow_transform and the jdg:sow_slot card.

They come from the registries (atom_types.json, label_heads.json,
judgment_heads.json), featurize like every other read, train through the
generic losses (a number head's presence and absence, multi-label sets), a
filled SOW slot points at the line that fills it, and a checkpoint saved
before they existed loads with --allow-new-heads.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from ml.c3.data import IGNORE, DealExample, featurize, field_note_targets, judgment_query
from ml.c3.gates import load_checkpoint, save_checkpoint
from ml.c3.loop import without_heads
from ml.c3.losses import judgment_cite_loss, judgment_loss
from ml.c3.model import C3Config, C3Model
from ml.c3.readiness import digest
from ml.c3.schema import CLASS, DEAL, NUMBER, POLICY_WORDS, load_schema, shape_digest

SMALL = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
NEW = "read:visit_count,read:work_action,read:sow_element,read:sow_transform,jdg:sow_slot,read:feeds_decision"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def test_heads_and_answers(schema):
    by = schema.by_key()
    v = by["read:visit_count"]
    assert v.kind == NUMBER and v.universal and [a.value for a in v.answers] == ["_absent", "present"]
    w = by["read:work_action"]
    assert w.kind == CLASS and w.multiple
    assert {"install", "decommission", "replace", "survey", "cabling", "return_visit"} <= {a.value for a in w.answers}
    e = by["read:sow_element"]
    assert e.multiple and {a.value for a in e.answers} >= {
        "_absent", "task", "deliverable", "assumption", "exclusion", "customer_responsibility",
        "acceptance", "fee", "site"}
    t = by["read:sow_transform"]
    assert not t.multiple and [a.value for a in t.answers] == [
        "_absent", "verbatim", "restated", "generalized", "inverted", "merged"]
    s = by["jdg:sow_slot"]
    assert s.size == DEAL and s.universal
    assert [a.value for a in s.answers] == ["_absent", "template", "filled", "house_default", "missing_chase"]
    notes = field_note_targets(schema)
    for f in ("visit_count", "work_action", "sow_element", "sow_transform"):
        assert notes[f"{f}_note"] == f"read:{f}"


def test_descriptions_are_layout_free_and_universal(schema):
    by = schema.by_key()
    layout = re.compile(r"\b(rows?|columns?|cells?|tabs?|sheets?|spreadsheet|port|sibling|indent\w*)\b", re.I)
    for key in NEW.split(","):
        o = by[key]
        for text in [o.description, *(a.description for a in o.answers)]:
            assert not layout.search(text), (key, text)
            assert not any(w in text.lower() for w in POLICY_WORDS), (key, text)
        # Every closed answer has its own description, not just its name.
        for a in o.answers[1:]:
            if o.kind == CLASS:
                assert len(a.description.split(". ", 1)[-1]) > 20, (key, a.value)


def _atom(k, text, t, doc="email-1", kind="email", section="body"):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": doc, "doc_kind": kind,
            "section": section, "entered_at": f"2026-05-01T10:{t:02d}:00Z",
            "speaker_role": "customer", "speaker_side": "customer"}


ATOMS = [
    _atom("k1", "Store 12, 4 Elm St: install the new pair and remove the old pair; 2 visits.", 0),
    _atom("k2", "Store 14, 9 Oak Ave, Springfield.", 1),
    _atom("k3", "All cable must be labeled at both ends.", 2),
    _atom("k4", "Can you also haul the old boxes to the depot?", 3),
    _atom("k5", "Signature: Jo, buyer.", 4),
]
PERSON = "pm@example.com"


def _label(k, **reads):
    return {"label_key": k, "labeler": PERSON, "label_type": "physical_site", "note": "A site.",
            "reads_set": reads}


BLOB = {
    "labels": [
        _label("k1", visit_count="2", visit_count_note="The line says 2 visits: install, then remove.",
               work_action=["install", "decommission", "return_visit"], sow_element=["site", "task"],
               sow_transform="generalized"),
        _label("k2", visit_count="_absent", work_action="_absent", sow_element=["site"], sow_transform="restated"),
        _label("k3", sow_element=["assumption"], sow_transform="verbatim",
               sow_element_note="A standard the work must meet becomes an assumption."),
        _label("k4", sow_element=["exclusion", "customer_responsibility"], sow_transform="inverted"),
        {**_label("k5"), "label_type": "_keep", "rejected": "true", "weight_tier": "exclude",
         "note": "EXCLUDE_FROM_TRAINING: test."},
    ],
    "judgments": [
        {"head": "sow_slot", "verdict": "filled", "labeler": PERSON, "target_key": "slot:sites:site_list",
         "target": {"query": "SOW section: sites. Slot: the site list.",
                    "source": [{"atomId": "at-k1"}, {"atomId": "at-k2"}]},
         "note": "Both stores are named before the SOW, so the site list is filled from them."},
        {"head": "sow_slot", "verdict": "missing_chase", "labeler": PERSON, "target_key": "slot:timeline:start_date",
         "target": {"query": "SOW section: timeline. Slot: start date."},
         "note": "No line gives a start date; ask for it."},
        {"head": "sow_slot", "verdict": "template", "labeler": PERSON, "target_key": "slot:acceptance:section_text",
         "target": {"query": "SOW section: acceptance. Slot: the standard wording."}},
        # A draft nobody accepted trains nothing.
        {"head": "sow_slot", "verdict": "house_default", "labeler": "Claude (assistant)",
         "target_key": "slot:contacts:provider_contact",
         "target": {"query": "SOW section: contacts. Slot: the provider's contact."}},
    ],
}


@pytest.fixture(scope="module")
def batch(schema):
    return featurize(DealExample.from_training_blob(BLOB, ATOMS, deal_id="synthetic-sow-heads"), schema)


def _i(batch, word):
    return next(i for i, t in enumerate(batch.texts) if word in t)


def test_visit_count_teaches_presence_absence_and_the_number(schema, batch):
    o = schema.by_key()["read:visit_count"]
    k1, k2, k3 = _i(batch, "Store 12"), _i(batch, "Store 14"), _i(batch, "labeled")
    col = batch.targets["read:visit_count"]
    assert col[k1] == o.index("present") and col[k2] == o.index("_absent") and col[k3] == IGNORE
    nums = batch.numbers_target["read:visit_count"]
    assert nums[k1] == 2.0 and nums[k2] is None
    assert "2 visits" in batch.field_notes[k1]["read:visit_count"]


def test_multi_label_sets_and_the_transform(schema, batch):
    by = schema.by_key()
    w, e, t = by["read:work_action"], by["read:sow_element"], by["read:sow_transform"]
    k1, k2, k4 = _i(batch, "Store 12"), _i(batch, "Store 14"), _i(batch, "haul")
    assert batch.multi_targets["read:work_action"][k1] == tuple(sorted(
        w.index(v) for v in ("install", "decommission", "return_visit")))
    assert batch.multi_targets["read:work_action"][k2] == (w.index("_absent"),)
    assert batch.multi_targets["read:sow_element"][k4] == tuple(sorted(
        e.index(v) for v in ("exclusion", "customer_responsibility")))
    assert batch.targets["read:sow_transform"][k4] == t.index("inverted")
    k5 = _i(batch, "Signature")
    assert batch.multi_targets["read:sow_element"][k5] is None      # set aside: nothing taught


def test_sow_slot_cards_carry_their_query_and_link(schema, batch):
    o = schema.by_key()["jdg:sow_slot"]
    cards = [j for j in batch.judged if j.key == "jdg:sow_slot"]
    assert len(cards) == 3                                          # the unaccepted draft is left out
    by_q = {j.query: j for j in cards}
    filled = by_q["SOW section: sites. Slot: the site list."]
    assert filled.answer == o.index("filled")
    assert set(filled.cites) == {_i(batch, "Store 12"), _i(batch, "Store 14")}
    chase = by_q["SOW section: timeline. Slot: start date."]
    assert chase.answer == o.index("missing_chase") and chase.cites == ()
    # A whole-deal card reads the deal's lines that are not set aside.
    assert _i(batch, "Signature") not in filled.lines and len(filled.lines) == len(batch) - 1
    assert judgment_query({"head": "sow_slot", "target": {"query": "q"}, "text": "not this"}) == "q"
    assert judgment_query({"head": "site_role", "target": {"query": "q"}}) == ""


def test_card_and_link_losses_train(schema, batch):
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    out = model(batch.inputs())
    cards = [j for j in batch.judged if j.key == "jdg:sow_slot" and j.cites]
    cite = judgment_cite_loss(model, out, "jdg:sow_slot", cards)
    assert torch.isfinite(cite) and cite > 0
    total = judgment_loss(model, out, batch)
    total.backward()
    assert model.judgment_query.weight.grad is not None and model.judgment_query.weight.grad.abs().sum() > 0
    # The pointer moves with the slot asked: two queries score the lines differently.
    a = model.judgment_pointer(out, "jdg:sow_slot", cards[0].lines, "SOW section: sites. Slot: the site list.")
    b = model.judgment_pointer(out, "jdg:sow_slot", cards[0].lines, "SOW section: fees. Slot: a price.")
    assert a.shape == (len(cards[0].lines),) and not torch.allclose(a, b)


def _schema_before_the_heads(tmp_path):
    """The registries as they were in the commit before visit_count was added."""
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout
    try:
        added = git("log", "--format=%H", "-S", '"key": "visit_count"', "--", "app/core/atom_types.json").split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no git history")
    ref = f"{added[-1]}^" if added else "HEAD"
    files = {"app/core/atom_types.json": "types", "app/core/label_heads.json": "heads",
             "ml/c3/judgment_heads.json": "judgments"}
    paths = {}
    for rel, name in files.items():
        text = git("show", f"{ref}:{rel}")
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(text, encoding="utf-8")
    return load_schema(paths["heads"], paths["types"], judgments_path=paths["judgments"])


def test_a_checkpoint_from_before_the_heads_loads_with_allow_new_heads(schema, tmp_path):
    old = _schema_before_the_heads(tmp_path)
    assert "read:visit_count" not in old.by_key()
    assert shape_digest(without_heads(schema, NEW)) == shape_digest(old)
    torch.manual_seed(0)
    before = C3Model(old, SMALL)
    ck = tmp_path / "old.pt"
    save_checkpoint(ck, before, step=7, schema_sha256=digest([asdict(o) for o in old.opportunities]),
                    shape_sha256=shape_digest(old))
    after = C3Model(schema, SMALL)
    step = load_checkpoint(ck, after, schema_sha256=digest([asdict(o) for o in schema.opportunities]),
                           allow_text_change=True, shape_sha256=shape_digest(without_heads(schema, NEW)))
    assert step == 7
    with pytest.raises(ValueError):
        load_checkpoint(ck, C3Model(schema, SMALL), schema_sha256=digest([asdict(o) for o in schema.opportunities]),
                        allow_text_change=True, shape_sha256=shape_digest(schema))


def test_run_e_script_allows_the_heads():
    text = (REPO / "ml/c3/pod/runE.sh").read_text(encoding="utf-8")
    m = re.search(r'NEW_HEADS:-([^}"\s]*)\}', text)
    assert m and set(NEW.split(",")) <= set(m.group(1).split(","))
    assert json.loads((REPO / "ml/c3/judgment_heads.json").read_text())["heads"]["sow_slot"]["size"] == "deal"
