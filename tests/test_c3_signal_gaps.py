"""The labeled signals the run E signal audit found training nothing, wired.

* said_by / said_to: column heads for the role and side (unknown teaches
  nothing); the Batch's sender inputs are the atoms' own with or without them,
  so the heads are labels, not leaked inputs; internal_only is a boolean head;
* jdg:rule: a rule card is a line card asked with its rule's question (two
  rules on one line are two cards), its reason a code head, its note a card WHY;
* a norm card's note is a field note of its reason head;
* the draft WHY weight is an absolute scale: a part of drafts only trains at
  DRAFT_WHY_WEIGHT of the same part saved, where a weight-normalized mean
  cancelled it.
"""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from ml.c3.data import DRAFT_WHY_WEIGHT, IGNORE, DealExample  # noqa: E402
from ml.c3.loop import SMALL, training_batches  # noqa: E402
from ml.c3.model import C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

PM = "Pat Morgan"


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def _atom(k, text, t, side="ours", role="internal"):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": "email-1", "doc_kind": "email",
            "entered_at": f"2026-05-01T10:{t:02d}:00Z", "speaker_role": role, "speaker_side": side,
            "speaker_org": "internal" if side == "ours" else "external"}


def _payload():
    atoms = [_atom("k1", "We will send two techs on Saturday.", 1),
             _atom("k2", "Please confirm the badge list.", 2, side="theirs", role="unknown"),
             _atom("k3", "Scope of Work", 3),
             _atom("k4", "Labor- Per Hour | $127.50 | Per Hour | 2", 4),
             _atom("k5", "Internal: margin looks thin.", 5)]
    labels = [
        {"label_key": "k1", "label_type": "schedule", "labeler": PM, "note": "The visit day.",
         "said_by": {"name": "A", "role": "provider", "side": "ours"},
         "said_to": [{"role": "customer", "side": "theirs"}, {"role": "unknown", "side": "ours"}],
         "internal_only": False},
        {"label_key": "k2", "label_type": "access", "labeler": PM, "note": "A badge list is needed.",
         "said_by": {"role": "unknown", "side": "theirs"}, "said_to": [], "internal_only": False},
        {"label_key": "k3", "label_type": "_keep", "labeler": PM, "note": "A heading."},
        {"label_key": "k4", "label_type": "labor", "labeler": PM, "note": "Two hours of labor."},
        {"label_key": "k5", "label_type": "pricing", "labeler": PM, "note": "Our own margin talk.",
         "said_by": {"role": "internal", "side": "ours"}, "internal_only": True},
    ]
    judgments = [
        {"head": "rule", "target_key": "rule_1", "labeler": PM, "text": "Scope of Work",
         "target": {"head": "rule", "rule": "section_title", "text": "Scope of Work", "fired": True},
         "verdict": "should_fire", "reason": "kept", "note": "A generic section heading, not the deal's title."},
        {"head": "rule", "target_key": "rule_2", "labeler": PM, "text": "Scope of Work",
         "target": {"head": "rule", "rule": "document_title_line", "text": "Scope of Work", "fired": True},
         "verdict": "should_not_fire", "reason": "overturned", "note": "A section heading names no document."},
        {"head": "norm", "target_key": "atom:at-k4", "labeler": PM, "text": "Labor- Per Hour | $127.50 | Per Hour | 2",
         "target": {"a": {"atomId": "at-k4", "text": "Labor- Per Hour | $127.50 | Per Hour | 2"}},
         "verdict": "2 hours", "reason": "wrong_shape", "note": "The labor row's value is its quantity in its unit."},
    ]
    return {"deal_id": "900002", "company": "purtera", "atoms": atoms, "labels": labels,
            "judgments": judgments, "links": [], "deal_answers": []}


def _batch(payload, schema):
    deal = DealExample.from_training_blob(payload, payload["atoms"], deal_id=payload["deal_id"])
    return training_batches(deal, schema, max_atoms=64)[0][0]


def _row(batch, key):
    return batch.texts.index({"k1": "We will send two techs on Saturday.", "k2": "Please confirm the badge list.",
                              "k3": "Scope of Work", "k4": "Labor- Per Hour | $127.50 | Per Hour | 2",
                              "k5": "Internal: margin looks thin."}[key])


def test_said_heads_are_labels_not_inputs(schema):
    b = _batch(_payload(), schema)
    by = schema.by_key()
    i1, i2, i5 = _row(b, "k1"), _row(b, "k2"), _row(b, "k5")
    role, side = by["col:said_by_role"], by["col:said_by_side"]
    assert b.targets["col:said_by_role"][i1] == role.index("provider")
    assert b.targets["col:said_by_role"][i2] == IGNORE                  # unknown teaches nothing
    assert b.targets["col:said_by_side"][i2] == side.index("theirs")
    to_role, to_side = by["col:said_to_role"], by["col:said_to_side"]
    assert b.multi_targets["col:said_to_role"][i1] == (to_role.index("customer"),)
    assert b.multi_targets["col:said_to_side"][i1] == tuple(sorted((to_side.index("ours"), to_side.index("theirs"))))
    io = by["col:internal_only"]
    assert b.targets["col:internal_only"][i5] == io.index(True)
    assert b.targets["col:internal_only"][i1] == io.index(False)
    # The sender inputs are the atoms' own, whatever the labels say.
    bare = copy.deepcopy(_payload())
    for r in bare["labels"]:
        for k in ("said_by", "said_to", "internal_only"):
            r.pop(k, None)
    b0 = _batch(bare, schema)
    assert (b0.role, b0.side, b0.org) == (b.role, b.side, b.org)
    assert b0.targets["col:said_by_side"][i2] == IGNORE
    from ml.c3.sender_labels import check

    c = check(_payload())
    assert c["said_by"] == 3 and c["consistent"] == 3 and c["refines"] == 1 and c["internal_only"] == 1


def test_rule_cards_are_queried_line_cards(schema):
    b = _batch(_payload(), schema)
    i3 = _row(b, "k3")
    rule = [j for j in b.judged if j.key == "jdg:rule"]
    assert len(rule) == 2 and all(j.lines == (i3,) for j in rule)
    by = schema.by_key()["jdg:rule"]
    queries = {j.query: j.answer for j in rule}
    assert len(queries) == 2
    title = next(q for q in queries if q.startswith("Parser rule document title line."))
    assert queries[title] == by.index("should_not_fire") and "fired on this line" in title
    assert all("Correct" not in q and "should" not in q.split(".")[0] for q in queries)
    assert any("generic section heading" in j.note for j in rule)
    reasons = [j for j in b.judged if j.key == "jdg:rule:reason"]
    assert sorted(j.answer for j in reasons) == sorted(
        schema.by_key()["jdg:rule:reason"].index(v) for v in ("kept", "overturned"))


def test_norm_note_is_a_field_note(schema):
    b = _batch(_payload(), schema)
    i4 = _row(b, "k4")
    assert "quantity in its unit" in b.field_notes[i4]["jdg:norm:reason"]
    gone = copy.deepcopy(_payload())
    gone["judgments"][2]["note"] = ""
    assert "jdg:norm:reason" not in _batch(gone, schema).field_notes[i4]


def test_draft_why_weight_is_absolute(schema):
    from ml.c3.losses import judgment_rationale_loss, why_mean
    from ml.c3.why_carry import why_alone_loss, why_contrast_loss

    w = torch.tensor([0.5, 0.5, 0.0])
    per = torch.tensor([2.0, 4.0, 100.0])
    assert float(why_mean(per, w)) == pytest.approx(1.5)            # (1 + 2) / 2, not 3
    b = _batch(_payload(), schema)
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    model.eval()
    draft = copy.copy(b)
    draft.why_weights = [DRAFT_WHY_WEIGHT * x for x in b.why_weights]
    draft.judged = [type(j)(**{**j.__dict__, "why_weight": DRAFT_WHY_WEIGHT * j.why_weight}) for j in b.judged]
    with torch.no_grad():
        out = model(b.inputs(), company=b.company or None, desc=model.describe())
        desc = model.describe()
        for fn in (lambda x: why_contrast_loss(model, out, x),
                   lambda x: why_alone_loss(model, out, x, desc),
                   lambda x: judgment_rationale_loss(model, out, x)):
            saved, drafted = float(fn(b)), float(fn(draft))
            assert saved > 0 and drafted == pytest.approx(DRAFT_WHY_WEIGHT * saved, rel=1e-4)


def test_bypass_reads_the_post_ramp_configuration(schema, tmp_path):
    """An r-only row still holding h (a ramp-off smoke, or mid-ramp) reads h;
    the check sets the share the run ends the ramp at, and the CLI says so."""
    import json

    from ml.c3 import smoke_checks as sc
    from ml.c3.free_rows import set_h_keep

    b = _batch(_payload(), schema)
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    model.add_free_head("col:label_type")
    row = model.add_free_head("r:col:label_type")
    with torch.no_grad():
        row.h_weight.normal_()
    set_h_keep(model, 1.0)
    assert sc.h_path_shares(model, b, ("col:label_type",))["col:label_type"] > 0.10
    set_h_keep(model, 0.0)
    assert sc.h_path_shares(model, b, ("col:label_type",))["col:label_type"] == 0.0
    f = tmp_path / "900002.json"
    f.write_text(json.dumps(_payload()))
    out = tmp_path / "gate.md"
    rc = sc.main(["--tiny", "--train", str(f), "--max-atoms", "64", "--free-head", "r:col:label_type",
                  "--free-head", "r:read:sow_section", "--free-h-out", "1", "--out", str(out)])
    text = out.read_text()
    assert "r-only free rows on col:label_type, read:sow_section, their h share 0" in text
    assert rc == 0 and "**GO**" in text


def test_run_script_passes_the_free_rows_to_the_gates():
    from pathlib import Path

    from ml.c3 import smoke_checks as sc
    from ml.c3.free_rows import R_ONLY_KEYS

    text = (Path(__file__).resolve().parents[1] / "ml/c3/pod/runE.sh").read_text(encoding="utf-8")
    assert text.count('"${why2_free[@]}"') >= 3          # the loop, the audit and the smoke checks
    assert "--free-h-out 1" in text
    by = load_schema().by_key()
    # Every critical line head with a free row reads r only.
    for k in sc.CRITICAL:
        o = by[k]
        if o.size == "line" and o.kind != "number":
            assert k in R_ONLY_KEYS, k
