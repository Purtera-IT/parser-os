"""The labeling page's judgment tabs train ml/c3 (page-to-heads.md), on invented rows.

* each verdict finds its lines by atom id, or by text after a re-parse;
* one-line verdicts become that line's target; aliases fill an existing
  question (Dropped -> admission, BOM owner -> supplier) only where the card
  left it blank;
* verdicts about two lines, a document / table / sheet or the whole deal train
  pair, group and deal questions; project tier stays in the company layer;
* conflict and same-site verdicts also feed the link and same-entity terms;
* drafts written by a model never train.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.c3.data import IGNORE, DealExample, featurize  # noqa: E402
from ml.c3.losses import judgment_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.schema import DEAL, GROUP, LINE, PAIR, load_schema  # noqa: E402

SMALL = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)


def _atom(k, text, t, doc="email-1", section="body", kind="email"):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": doc, "doc_kind": kind,
            "section": section, "entered_at": f"2026-05-01T10:{t:02d}:00Z",
            "speaker_role": "customer", "speaker_side": "customer"}


ATOMS = [
    _atom("k1", "Install is at the Elm Street warehouse.", 0),
    _atom("k2", "Send the invoice to our head office on Main Avenue.", 1),
    _atom("k3", "Work window is Saturday only.", 2),
    _atom("k4", "Work window is weekdays after 6pm.", 3, doc="sow-1", kind="sow"),
    _atom("k5", "Mount 4 displays per room.", 4, doc="sheet-1", section="Scope", kind="spreadsheet"),
    _atom("k6", "Rooms: 101, 102, 103.", 5, doc="sheet-1", section="Scope", kind="spreadsheet"),
    _atom("k7", "Brackets, qty 4.", 6, doc="sheet-1", section="Parts", kind="spreadsheet"),
    _atom("k8", "Elm St. warehouse, dock door 3.", 7),
]
BLOB = {"labels": [{"label_key": "k7", "label_type": "bom_line", "about": "deal",
                    "note": "A parts line on the kit sheet.", "reads_set": {}}]}


def _j(head, verdict, target, text="", **kw):
    return {"head": head, "verdict": verdict, "target": target, "text": text,
            "labeler": "pm@example.com", "target_key": kw.pop("target_key", ""), **kw}


JUDGMENTS = [
    _j("conflict", "contradicts", {"a": {"atomId": "at-k3"}, "b": {"atomId": "at-k4"}},
       reason="different_window", note="Saturday only vs weekdays after 6pm: both cannot hold."),
    # A site pair after a re-parse: the ids changed, the text still matches.
    _j("site", "same_site", {"a": {"evidence": {"atomId": "old-1", "text": ATOMS[0]["text"]}},
                             "b": {"evidence": {"atomId": "old-2", "text": ATOMS[7]["text"]}}},
       target_key="pair:abc"),
    _j("site_role", "vendor_or_billing_address", {"site": {"evidence": {"atomId": "at-k2"}}}),
    _j("document_job", "this_deal", {"document": {"artifactId": "sow-1"}}),
    _j("sheet", "scope", {"a": {"atomId": "at-k5"}}),
    _j("billing_type", "per_site", {}, target_key="deal:billing_type"),
    _j("tier", "2", {}, target_key="deal:tier"),
    _j("suppression", "should_have_been_kept", {}, text=ATOMS[2]["text"], target_key="atom:gone"),
    _j("bom_owner", "customer_furnished", {"a": {"atomId": "at-k7"}}),
    _j("gap", "valid", {"source": {"atomId": "at-k6"}}, note="Room count drives crew size."),
    # Never trains: a draft for a person to accept, a verdict outside the set,
    # a subject that is not in this deal.
    {**_j("site_role", "job_site", {"site": {"evidence": {"atomId": "at-k1"}}}), "labeler": "Claude (assistant)"},
    _j("billing_type", "barter", {}),
    _j("conflict", "supports", {"a": {"atomId": "nope"}, "b": {"atomId": "at-k4"}}),
]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    deal = DealExample.from_training_blob({**BLOB, "judgments": JUDGMENTS}, ATOMS, deal_id="synthetic-judgments")
    return featurize(deal, schema)


def _i(batch, word):
    return next(i for i, t in enumerate(batch.texts) if word in t)


def test_judgment_questions_come_in_four_sizes(schema):
    by = schema.by_key()
    assert {by[k].size for k in ("jdg:conflict", "jdg:site")} == {PAIR}
    assert {by[k].size for k in ("jdg:document_job", "jdg:sheet", "jdg:roster")} == {GROUP}
    assert by["jdg:billing_type"].size == DEAL and by["jdg:site_role"].size == LINE
    assert by["jdg:tier"].layer == "company" and by["jdg:billing_type"].universal
    # Line-question code paths see only line questions.
    assert all(o.size == LINE for o in schema.select())


def test_vocabularies_match_the_parser_registry(schema):
    registry = pytest.importorskip("app.core.pm_feedback").HEAD_REGISTRY
    by = schema.by_key()
    for name, spec in registry.items():
        o = by.get(f"jdg:{name}")
        if o is not None and spec.candidates:
            assert {a.value for a in o.answers} - {"_absent"} == set(spec.candidates), name


def test_line_verdicts_and_aliases(schema, batch):
    by = schema.by_key()
    office, gap, kit, window = (_i(batch, w) for w in ("head office", "Rooms:", "Brackets", "Saturday"))
    assert batch.targets["jdg:site_role"][office] == by["jdg:site_role"].index("vendor_or_billing_address")
    assert batch.targets["jdg:site_role"][_i(batch, "Install is")] == IGNORE   # the draft did not train
    assert batch.targets["jdg:gap"][gap] == by["jdg:gap"].index("valid")
    assert batch.field_notes[gap]["jdg:gap"].startswith("Room count")
    # Dropped -> admission (found by its text), BOM owner -> supplier.
    assert batch.targets["col:admission"][window] == by["col:admission"].index("keep")
    assert batch.targets["col:supplier"][kit] == by["col:supplier"].index("customer")


def test_pair_group_and_deal_verdicts(schema, batch):
    by = schema.by_key()
    got = {j.key: j for j in batch.judged}
    sat, wkd = _i(batch, "Saturday"), _i(batch, "weekdays")
    assert set(got["jdg:conflict"].lines) == {sat, wkd}
    assert got["jdg:conflict"].note.startswith("different_window")
    assert set(got["jdg:site"].lines) == {_i(batch, "Install is"), _i(batch, "Elm St.")}
    assert set(got["jdg:document_job"].lines) == {wkd}
    assert set(got["jdg:sheet"].lines) == {_i(batch, "Mount 4"), _i(batch, "Rooms:")}
    assert len(got["jdg:billing_type"].lines) == len(batch)
    assert got["jdg:billing_type"].answer == by["jdg:billing_type"].index("per_site")
    assert got["jdg:tier"].answer == by["jdg:tier"].index("2")
    assert sum(j.key == "jdg:conflict" for j in batch.judged) == 1      # the unresolved one dropped
    assert "jdg:billing_type" in got and sum(j.key == "jdg:billing_type" for j in batch.judged) == 1
    # Feeds the link and same-entity terms too.
    assert (wkd, sat) in batch.edges["contradicts"]
    assert batch.entities[_i(batch, "Install is")][-1] == batch.entities[_i(batch, "Elm St.")][-1]


def test_heads_train_and_the_base_ignores_the_company(schema, batch):
    torch.manual_seed(0)
    m = C3Model(schema, SMALL, companies=("purtera", "other_co"))
    m.train()
    out = m(batch.inputs(), company="purtera")
    loss = judgment_loss(m, out, batch)
    assert loss > 0
    loss.backward()
    for p in (m.pair_in.weight, m.pool_in.weight, m.pool_in_company.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        m.alpha[1].normal_()
        inp = batch.inputs()
        a, b = m(inp, company="purtera"), m(inp, company="other_co")
        everyone = [tuple(range(len(batch)))]
        assert torch.equal(m.judge(a, "jdg:billing_type", everyone), m.judge(b, "jdg:billing_type", everyone))
        assert not torch.equal(m.judge(a, "jdg:tier", everyone), m.judge(b, "jdg:tier", everyone))
        with pytest.raises(ValueError):
            m.judge(m(inp), "jdg:tier", everyone)


def test_a_model_drafted_judgment_note_counts_draft_weight(schema):
    """why_author on a judgment row scales its note like an atom label's WHY;
    the verdict itself still trains in full."""
    from ml.c3.data import DRAFT_WHY_WEIGHT, why_weight

    pair = {"a": {"atomId": "at-k3"}, "b": {"atomId": "at-k4"}}
    rows = [_j("conflict", "contradicts", pair, note="Both windows cannot hold.", why_author="machine_draft")]
    b = featurize(DealExample.from_training_blob({**BLOB, "judgments": rows}, ATOMS, deal_id="synthetic-jdraft"), schema)
    (j,) = [x for x in b.judged if x.key == "jdg:conflict"]
    assert j.why_weight == DRAFT_WHY_WEIGHT
    assert j.answer == schema.by_key()["jdg:conflict"].index("contradicts")
    # A person's note, an accepted draft and a row from before the column count in full.
    for author in ("person", "accepted_draft", "edited_draft", None):
        assert why_weight({"why_author": author}) == 1.0
    assert why_weight({"reads_set": {"why_author": "machine_draft"}}) == DRAFT_WHY_WEIGHT
