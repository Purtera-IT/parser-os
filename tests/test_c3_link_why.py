"""A link's note shapes the relation decision, teacher-free (ml/c3/link_why.py), on invented rows.

* the saved note reaches ``Batch.edge_notes`` (universal part only) on the edge it explains;
* identical notes are one column, never each other's negative; near-miss and
  other-relation edges' notes are hard negatives and raise the contrast;
* the contrast reads the relation head's own pair representation and trains it;
* the note alone names its relation through the relation head, and fitting it
  moves the relation head's folds;
* both parts are LossWeights fields, off in the lines stage, on from links,
  encode the notes once and reuse the step's descriptions.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.link_why import (  # noqa: E402
    hard_columns, link_note_rows, link_why_alone_logits, link_why_alone_loss, link_why_contrast_loss,
    note_columns, note_embeddings)
from ml.c3.loop import gold_weights  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

SMALL = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
PM = "Pat Morgan"
SAME = "Both lines name the same install window."


def _atom(k, text, t):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": "email-1", "doc_kind": "email",
            "entered_at": f"2026-05-01T10:{t:02d}:00Z", "speaker_role": "customer", "speaker_side": "customer"}


ATOMS = [
    _atom("k1", "What kind of lock is on the IDF door?", 0),
    _atom("k2", "Install window is Saturdays.", 1),
    _atom("k3", "It is a maglock on the IDF door.", 2),
    _atom("k4", "Install window is weekday evenings.", 3),
    _atom("k5", "Saturday installs confirmed by the site lead.", 4),
    _atom("k6", "The window stays Saturdays per facilities.", 5),
]
LABELS = [{"label_key": f"k{n}", "label_type": t, "note": f"Line {n}."}
          for n, t in enumerate(("open_question", "schedule", "requirement", "schedule", "schedule",
                                 "schedule"), 1)]
LINKS = [
    {"from_head": "atom", "from_key": "k3", "to_label_key": "k1", "relation": "answers",
     "note": "The maglock line closes the lock-type question.", "labeler": PM},
    {"from_head": "atom", "from_key": "k4", "to_label_key": "k2", "relation": "contradicts",
     "note": "Weekday evenings replaces Saturdays; both cannot hold.\n[purtera] We quote evenings at night rates.",
     "labeler": PM},
    {"from_head": "atom", "from_key": "k5", "to_label_key": "k2", "relation": "supports", "note": SAME,
     "labeler": PM},
    {"from_head": "atom", "from_key": "k6", "to_label_key": "k2", "relation": "supports", "note": SAME,
     "labeler": PM},
]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    deal = DealExample.from_training_blob({"labels": LABELS, "links": LINKS}, ATOMS, deal_id="synthetic-link-why",
                                          serve_lines_only=False)
    return featurize(deal, schema)


def _i(batch, word):
    return next(i for i, t in enumerate(batch.texts) if word in t)


def test_the_note_rides_on_its_edge(batch):
    notes = {(i, j, rel): t for i, j, rel, t in batch.edge_notes}
    mag, q, eve, sat = (_i(batch, w) for w in ("maglock", "kind of lock", "evenings", "Install window is Sat"))
    assert notes[(mag, q, "answers")].startswith("The maglock line closes")
    assert "both cannot hold" in notes[(eve, sat, "contradicts")]
    assert "night rates" not in notes[(eve, sat, "contradicts")]        # the [purtera] part is the company's
    for i, j, rel, _ in batch.edge_notes:
        assert (i, j) in batch.edges[rel]


def test_identical_notes_share_a_column_and_hard_negatives(schema, batch):
    model = C3Model(schema, SMALL)
    rows = link_note_rows(model, batch)
    texts, col = note_columns(rows)
    assert len(rows) == 4 and len(texts) == 3                        # the two "same window" notes are one column
    sup = [m for m, r in enumerate(rows) if r[2] == "supports"]
    assert col[sup[0]] == col[sup[1]]
    hard = hard_columns(rows, col, batch)
    con = next(m for m, r in enumerate(rows) if r[2] == "contradicts")
    # contradicts shares its end (Saturdays) with the supports edges: their note is a hard negative.
    assert col[sup[0]] in hard[con]
    assert all(col[m] not in hard[m] for m in range(len(rows)))     # never its own note


def test_contrast_reads_the_relation_heads_pair_and_trains_it(schema, batch):
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    out = model(batch.inputs())
    assert out.rel_x is not None
    soft = link_why_contrast_loss(model, out, batch, hard=1.0)
    hard = link_why_contrast_loss(model, out, batch, hard=8.0)
    assert torch.isfinite(soft) and hard > soft
    hard.backward()
    assert model.relation_head.a.weight.grad is not None and model.relation_head.a.weight.grad.abs().sum() > 0
    assert model.judgment_note.weight.grad is not None


def test_the_note_alone_names_the_relation(schema, batch):
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    rows = link_note_rows(model, batch)
    texts, col = note_columns(rows)
    params = list(model.relation_head.parameters()) + list(model.judgment_note.parameters())
    opt = torch.optim.Adam(params, lr=3e-2)
    with torch.no_grad():
        out = model(batch.inputs())
    first = float(link_why_alone_loss(model, out, batch).detach())
    for _ in range(60):
        opt.zero_grad()
        loss = link_why_alone_loss(model, out, batch)
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.5 * first
    every = [o.field for o in schema.select(kind="relation")]
    pred = link_why_alone_logits(model, note_embeddings(model, texts)).argmax(-1).tolist()
    assert [every[pred[c]] for c in col] == [r[2] for r in rows]


def test_parts_named_off_in_lines_on_from_links_and_encode_once(schema, batch, monkeypatch):
    lines, _ = gold_weights("lines")
    links, _ = gold_weights("links")
    assert lines.link_why_contrast == 0 and lines.link_why_alone == 0
    assert (links.link_why_contrast, links.link_why_alone) == (0.3, 0.2)
    model = C3Model(schema, SMALL)
    calls = {"describe": 0, "aux": 0}
    real_d, real_a = model.describe, model.aux_text

    def describe():
        calls["describe"] += 1
        return real_d()

    def aux(texts):
        if any("maglock line closes" in t for t in texts):
            calls["aux"] += 1
        return real_a(texts)

    monkeypatch.setattr(model, "describe", describe)
    monkeypatch.setattr(model, "aux_text", aux)
    w = replace(LossWeights(clause_use=0.0, keyword_dropout=0.0), link_why_contrast=0.3, link_why_alone=0.2)
    _, parts = c3_loss(model, batch, w)
    assert parts["link_why_contrast"] > 0 and parts["link_why_alone"] > 0
    assert calls == {"describe": 1, "aux": 1}
    _, parts = c3_loss(model, batch, replace(w, link_why_contrast=0.0, link_why_alone=0.0))
    assert not {"link_why_contrast", "link_why_alone"} & set(parts)
