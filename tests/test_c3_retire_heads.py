"""The heads that retire rule decisions (runE/regex_retirement.md "needs new head" rows): read:work_window,
read:access_need, read:job_detail, read:run_length, read:price_basis, read:party_role, deal:site_count and the
deal-size cards jdg:services (a set), jdg:end_customer (queried, links to the naming line), jdg:delivery_model and
jdg:exclusion_set (queried by the exclusion)."""
from __future__ import annotations

import re

import pytest
import torch

from ml.c3.data import IGNORE, DealExample, featurize, field_note_targets
from ml.c3.losses import judgment_loss
from ml.c3.model import C3Config, C3Model
from ml.c3.schema import BINARY, CLASS, DEAL, NUMBER, POLICY_WORDS, load_schema

SMALL = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
READS = {"read:work_window": (CLASS, True), "read:access_need": (CLASS, True), "read:job_detail": (BINARY, False),
         "read:run_length": (NUMBER, False), "read:price_basis": (CLASS, False), "read:party_role": (CLASS, True)}
CARDS = {"jdg:services": True, "jdg:end_customer": False, "jdg:delivery_model": False, "jdg:exclusion_set": False}
PERSON = "pm@example.com"


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def test_heads_kinds_and_layout_free_text(schema):
    by = schema.by_key()
    layout = re.compile(r"\b(rows?|columns?|cells?|tabs?|sheets?|spreadsheet|sibling|indent\w*)\b", re.I)
    for key, (kind, multi) in READS.items():
        o = by[key]
        assert (o.kind, o.multiple, o.universal) == (kind, multi, True), key
    for key, multi in CARDS.items():
        o = by[key]
        assert (o.size, o.multiple, o.universal) == (DEAL, multi, True), key
    s = by["deal:site_count"]
    assert s.kind == NUMBER and s.size == DEAL and [a.value for a in s.answers] == ["_absent", "present"]
    for key in [*READS, *CARDS, "deal:site_count"]:
        o = by[key]
        for text in [o.description, *(a.description for a in o.answers)]:
            assert not layout.search(text), (key, text)
            assert not any(w in text.lower() for w in POLICY_WORDS), (key, text)
    notes = field_note_targets(schema)
    for key in READS:
        assert notes[key.split(":")[1] + "_note"] == key
    assert {"lift", "ladder", "ground_level"} <= {a.value for a in by["read:access_need"].answers}
    assert {"business_hours", "after_hours", "weekend"} <= {a.value for a in by["read:work_window"].answers}


def _atom(k, text, t):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": "email-1", "doc_kind": "email",
            "section": "body", "entered_at": f"2026-05-01T10:{t:02d}:00Z", "speaker_role": "customer",
            "speaker_side": "customer"}


ATOMS = [
    _atom("k1", "All work after 6 pm or on weekends; a scissor lift is needed for the 35 ft ceiling.", 0),
    _atom("k2", "Each AP run will not exceed approximately 150 feet.", 1),
    _atom("k3", "We need a ballpark number by Friday for Acme Stores.", 2),
    _atom("k4", "Jo Smith, store manager, will meet the tech and sign off.", 3),
]
BLOB = {
    "labels": [
        {"label_key": "k1", "labeler": PERSON, "label_type": "constraint", "note": "Window and access.",
         "reads_set": {"work_window": ["after_hours", "weekend"], "access_need": ["lift"], "job_detail": "true",
                       "run_length": "_absent", "work_window_note": "after 6 pm and weekends"}},
        {"label_key": "k2", "labeler": PERSON, "label_type": "requirement", "note": "A run cap.",
         "reads_set": {"run_length": "150", "access_need": "_absent", "job_detail": True}},
        {"label_key": "k3", "labeler": PERSON, "label_type": "deal_metadata", "note": "Asks for a ROM.",
         "reads_set": {"price_basis": "rom_estimate", "job_detail": "false"}},
        {"label_key": "k4", "labeler": PERSON, "label_type": "stakeholder", "note": "Onsite contact.",
         "reads_set": {"party_role": ["onsite_contact", "decision_maker"]}},
    ],
    "judgments": [
        {"head": "services", "verdict": ["wireless", "low_voltage_cabling"], "labeler": PERSON,
         "target_key": "deal:services", "target": {}, "note": "APs and their runs."},
        {"head": "delivery_model", "verdict": "install_buildout", "labeler": PERSON,
         "target_key": "deal:delivery_model", "target": {}},
        {"head": "end_customer", "verdict": "named", "labeler": PERSON, "target_key": "deal:end_customer",
         "target": {"query": "Who the work is for.", "source": [{"atomId": "at-k3"}]}},
        {"head": "exclusion_set", "verdict": "excluded", "labeler": PERSON, "target_key": "excl:electrical_work",
         "target": {"query": "Standard exclusion: electrical work."}},
    ],
    "deal_answers": [{"labeler": PERSON, "declared_site_count": "12", "deal_id": "x"}],
}


@pytest.fixture(scope="module")
def batch(schema):
    return featurize(DealExample.from_training_blob(BLOB, ATOMS, deal_id="synthetic-retire"), schema)


def test_reads_featurize(schema, batch):
    by = schema.by_key()
    ww = by["read:work_window"]
    assert batch.multi_targets["read:work_window"][0] == tuple(sorted(ww.index(v) for v in ("after_hours", "weekend")))
    assert batch.numbers_target["read:run_length"][1] == 150.0
    assert batch.targets["read:run_length"][0] == by["read:run_length"].index("_absent")
    jd = by["read:job_detail"]
    assert [batch.targets["read:job_detail"][i] for i in range(3)] == [jd.index("true"), jd.index("true"), jd.index("false")]
    assert batch.targets["read:price_basis"][2] == by["read:price_basis"].index("rom_estimate")
    assert batch.multi_targets["read:party_role"][3] == tuple(sorted(
        by["read:party_role"].index(v) for v in ("onsite_contact", "decision_maker")))
    assert "6 pm" in batch.field_notes[0]["read:work_window"]
    assert batch.targets["read:price_basis"][0] == IGNORE


def test_deal_cards(schema, batch):
    by = schema.by_key()
    j = {x.key: x for x in batch.judged}
    s = by["jdg:services"]
    assert j["jdg:services"].answers == tuple(sorted(s.index(v) for v in ("wireless", "low_voltage_cabling")))
    assert j["jdg:delivery_model"].answer == by["jdg:delivery_model"].index("install_buildout")
    assert j["jdg:end_customer"].query and j["jdg:end_customer"].cites == (2,)
    assert j["jdg:exclusion_set"].query == "Standard exclusion: electrical work."
    assert j["deal:site_count"].answer == by["deal:site_count"].index("present")
    assert batch.deal_numbers["declared_site_count"] == 12.0
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    out = model(batch.inputs())
    loss = judgment_loss(model, out, batch)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()


def test_a_stated_absence_of_a_site_count(schema):
    blob = {**BLOB, "deal_answers": [{"labeler": PERSON, "declared_site_count": "_absent", "deal_id": "x"}]}
    b = featurize(DealExample.from_training_blob(blob, ATOMS, deal_id="synthetic-retire-2"), schema)
    j = {x.key: x for x in b.judged}
    assert j["deal:site_count"].answer == schema.by_key()["deal:site_count"].index("_absent")
    assert "declared_site_count" not in b.deal_numbers


def test_doc_kind_card_reads_its_document(schema):
    o = schema.by_key()["jdg:doc_kind"]
    assert o.size == "group" and o.universal
    assert {"request_form", "rfp", "sow_draft", "sow_signed", "deal_kit", "estimating_calc", "quote", "purchase_order",
            "site_list", "email", "install_guide"} <= {a.value for a in o.answers}
    for a in o.answers[1:]:
        assert len(a.description.split(". ", 1)[-1]) > 25, a.value      # says what the kind is for
    atoms = [*ATOMS, {**ATOMS[0], "label_key": "f1", "atom_id": "at-f1", "doc_id": "form-1", "doc_kind": "file",
                      "text": "Site address: 4 Elm St | Install date: June 3"}]
    blob = {"labels": [], "judgments": [
        {"head": "doc_kind", "verdict": "request_form", "labeler": PERSON, "target_key": "doc:form-1",
         "target": {"document": {"artifactId": "form-1"}}, "note": "The customer's intake form asking for the job."},
        {"head": "doc_kind", "verdict": "email", "labeler": PERSON, "target_key": "doc:email-1",
         "target": {"document": {"artifactId": "email-1"}}}]}
    b = featurize(DealExample.from_training_blob(blob, atoms, deal_id="synthetic-doc-kind"), schema)
    j = {x.answer: x for x in b.judged if x.key == "jdg:doc_kind"}
    form = j[o.index("request_form")]
    assert [b.texts[i] for i in form.lines] == ["Site address: 4 Elm St | Install date: June 3"]
    assert len(j[o.index("email")].lines) == len(ATOMS)
