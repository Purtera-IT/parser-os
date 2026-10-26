"""read:feeds_decision: which quote and SOW decisions a line is evidence for.

A multi-label universal read with a WHY field. It comes from the registries, featurizes like every other set-valued
read (an unlisted decision trains as a negative, `_absent` says the line feeds none, an unlabeled or set-aside line
teaches nothing) and its note trains the field's WHY.
"""
from __future__ import annotations

import re

import pytest

from ml.c3.data import DealExample, featurize, field_note_targets
from ml.c3.schema import CLASS, POLICY_WORDS, load_schema

DECISIONS = ["sites", "visits", "work_scope", "work_quantity", "labor_hours", "crew", "skill_level", "work_window",
             "site_access", "travel", "materials", "commercial", "schedule", "scope_boundary", "deliverables",
             "parties", "risk_flag"]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def test_head_answers_and_note(schema):
    o = schema.by_key()["read:feeds_decision"]
    assert o.kind == CLASS and o.multiple and o.universal
    assert [a.value for a in o.answers] == ["_absent", *DECISIONS]
    assert field_note_targets(schema)["feeds_decision_note"] == "read:feeds_decision"


def test_descriptions_are_layout_free_and_universal(schema):
    o = schema.by_key()["read:feeds_decision"]
    layout = re.compile(r"\b(rows?|columns?|cells?|tabs?|sheets?|spreadsheet|port|sibling|indent\w*)\b", re.I)
    for text in [o.description, *(a.description for a in o.answers)]:
        assert not any(w in text.lower() for w in POLICY_WORDS), text
    for a in o.answers[1:]:
        assert not layout.search(a.description), (a.value, a.description)
        assert len(a.description.split(". ", 1)[-1]) > 20, a.value


def _atom(k, text, t):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": "email-1", "doc_kind": "email",
            "section": "body", "entered_at": f"2026-05-01T10:{t:02d}:00Z",
            "speaker_role": "customer", "speaker_side": "customer"}


ATOMS = [
    _atom("k1", "Store 12, 4 Elm St: install the new pair, 2 visits, about 4 hours each.", 0),
    _atom("k2", "Thanks, forwarding to the team.", 1),
    _atom("k3", "Work only after 10 pm; you will need a lift for the 20 ft ceilings.", 2),
    _atom("k4", "All cable must be labeled at both ends.", 3),
    _atom("k5", "Signature: Jo, buyer.", 4),
]
PERSON = "pm@example.com"


def _label(k, t="work_scope_item", **reads):
    return {"label_key": k, "labeler": PERSON, "label_type": t, "note": "A line.", "reads_set": reads}


BLOB = {"labels": [
    _label("k1", feeds_decision=["sites", "visits", "work_scope", "labor_hours"],
           feeds_decision_note="Store 12 is a work site that gets 2 visits of about 4 hours for the install."),
    _label("k2", feeds_decision="_absent"),
    _label("k3", feeds_decision=["work_window", "site_access", "crew"]),
    _label("k4"),
    {**_label("k5", feeds_decision=["parties"]), "label_type": "_keep", "rejected": "true",
     "weight_tier": "exclude", "note": "EXCLUDE_FROM_TRAINING: test."},
]}


@pytest.fixture(scope="module")
def batch(schema):
    return featurize(DealExample.from_training_blob(BLOB, ATOMS, deal_id="synthetic-feeds-decision"), schema)


def _i(batch, word):
    return next(i for i, t in enumerate(batch.texts) if word in t)


def test_sets_absent_unlabeled_and_set_aside(schema, batch):
    o = schema.by_key()["read:feeds_decision"]
    col = batch.multi_targets["read:feeds_decision"]
    assert col[_i(batch, "Store 12")] == tuple(sorted(o.index(v) for v in ("sites", "visits", "work_scope",
                                                                             "labor_hours")))
    assert col[_i(batch, "Thanks")] == (o.index("_absent"),)
    assert col[_i(batch, "after 10 pm")] == tuple(sorted(o.index(v) for v in ("work_window", "site_access", "crew")))
    assert col[_i(batch, "labeled at both")] is None                # no answer saved: nothing taught
    assert col[_i(batch, "Signature")] is None                      # set aside: nothing taught
    assert "2 visits" in batch.field_notes[_i(batch, "Store 12")]["read:feeds_decision"]


def test_a_checkpoint_from_before_the_head_loads_with_allow_new_heads(schema, tmp_path):
    """A checkpoint saved before feeds_decision existed loads once the head is allowed as new."""
    import subprocess
    from dataclasses import asdict
    from pathlib import Path

    import torch

    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.loop import without_heads
    from ml.c3.model import C3Config, C3Model
    from ml.c3.readiness import digest
    from ml.c3.schema import shape_digest

    repo = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout
    try:
        added = git("log", "--format=%H", "-S", '"key": "feeds_decision"', "--", "app/core/atom_types.json").split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no git history")
    ref = f"{added[-1]}^" if added else "HEAD"
    paths = {}
    for rel, name in {"app/core/atom_types.json": "types", "app/core/label_heads.json": "heads",
                      "ml/c3/judgment_heads.json": "judgments"}.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(git("show", f"{ref}:{rel}"), encoding="utf-8")
    old = load_schema(paths["heads"], paths["types"], judgments_path=paths["judgments"])
    assert "read:feeds_decision" not in old.by_key()
    from ml.c3.loop import RESHAPED  # a head reshaped since (its answer list changed) is new to the old checkpoint
    later = ",".join(k for k in schema.by_key() if k not in old.by_key() or (
        k in RESHAPED and [a.value for a in old.by_key()[k].answers] != [a.value for a in schema.by_key()[k].answers]))
    assert "read:feeds_decision" in later.split(",")
    assert shape_digest(without_heads(schema, later)) == shape_digest(old)
    small = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                     residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
    torch.manual_seed(0)
    ck = tmp_path / "old.pt"
    save_checkpoint(ck, C3Model(old, small), step=3, schema_sha256=digest([asdict(o) for o in old.opportunities]),
                    shape_sha256=shape_digest(old))
    step = load_checkpoint(ck, C3Model(schema, small), schema_sha256=digest([asdict(o) for o in schema.opportunities]),
                           allow_text_change=True, shape_sha256=shape_digest(without_heads(schema, later)))
    assert step == 3


def test_the_serve_returns_a_set_valued_read(schema):
    """/v1/read carries a multi-label head as [answers, {answer: p}] (serving parity for routing)."""
    from ml.c3 import serve

    o = schema.by_key()["read:feeds_decision"]
    p = [0.01] * len(o.answers)
    p[o.index("sites")], p[o.index("visits")] = 0.9, 0.7

    def read(model, deal, s, **kw):
        return {"lines": {a.key: {"reads": {serve.TYPE_KEY: {"answer": "physical_site", "p": 0.8},
                                            "read:feeds_decision": {"answers": ["sites", "visits"], "p": p}}}
                          for a in deal.atoms}}
    base = serve.Base(load=lambda: ("m", schema, "cpu"), read=read)
    base.model = ("m", schema, "cpu")
    base._loaded.set()
    payload = {"deal_id": "d", "atoms": [{"key": "k1", "text": "Store 12, 4 Elm St: 2 visits.", "doc_id": "e1"}]}
    out = base.read(payload, 5.0, line_heads=("read:feeds_decision",))
    answers, probs = out["reads"]["k1"]["read:feeds_decision"]
    assert answers == ["sites", "visits"] and probs["sites"] == 0.9 and probs["_absent"] == 0.01
