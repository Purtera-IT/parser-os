"""Quote-only deal views (run E estimator files): draft approval, the hours
verdict's value / per / qty on the main model, and head_scale reaching the
number, relation and absence losses."""
import hashlib
import json

import pytest
import torch

from ml.c3.data import IGNORE, featurize
from ml.c3.loop import SMALL, load_deal, read_draft_approvals, without_heads
from ml.c3.losses import LossWeights, absence_targets, c3_loss
from ml.c3.model import C3Model
from ml.c3.schema import load_schema, shape_digest
from ml.c3.values import hours_per

QUOTE = ("read:co_hours_estimate", "read:labor_hours", "read:crew_size", "read:equipment_qty",
         "jdg:hours:reason", "jdg:hours:per", "jdg:hours:qty", "deal:primary_service")
TEXTS = ["Hello team, a small job for you.", "Mount four TVs in the lobby and two conference rooms.",
         "Site is open 8 to 5.", "Thanks, Pat"]


def _blob(schema, verdict="hours=2;per=display;qty=4;role=tech;basis=estimate", scale=None):
    atoms = [{"label_key": f"lbl_{i:020d}", "atom_id": f"atm_{i:016d}", "text": t,
              "entered_at": f"2026-01-0{i + 1}T10:00:00Z", "doc_id": "note.txt", "doc_kind": "",
              "section": "", "order": i, "speaker_role": "unknown", "speaker_side": "unknown"}
             for i, t in enumerate(TEXTS)]
    card = {"id": "c1", "deal_id": "900001", "head": "hours", "target_key": f"atom:{atoms[1]['atom_id']}",
            "text": TEXTS[1], "parser_value": None, "verdict": verdict, "note": "", "reason": "",
            "labeler": "claude-code (assistant)", "why_author": "machine_draft", "purpose": "train", "fields": {},
            "target": {"a": {"text": TEXTS[1], "atomId": atoms[1]["atom_id"], "atomType": "task",
                             "filename": "note.txt", "section": []},
                       "head": "hours", "text": TEXTS[1], "targetKey": f"atom:{atoms[1]['atom_id']}"}}
    keys = [o.key for o in schema.opportunities]
    return {"deal_id": "900001@pre_quote", "company": "purtera", "atoms": atoms, "labels": [],
            "judgments": [card], "links": [], "deal_answers": [], "missed_history": [], "view": "pre_quote",
            "head_scale": scale if scale is not None else {k: 0.0 for k in keys if k not in QUOTE}}


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def _write(tmp_path, blob, name="900001@pre_quote.json"):
    p = tmp_path / name
    p.write_text(json.dumps(blob))
    return p, {name: hashlib.sha256(p.read_bytes()).hexdigest()}


def test_drafts_skipped_without_approval_and_train_with_it(tmp_path, schema):
    p, approved = _write(tmp_path, _blob(schema))
    off = featurize(load_deal(p), schema)
    assert all(v == IGNORE for v in off.targets["read:co_hours_estimate"])
    assert all(v == IGNORE for v in off.targets["jdg:hours:per"])
    manifest = tmp_path / "approved.json"
    manifest.write_text(json.dumps({"files": [{"file": p.name, "sha256": approved[p.name]}]}))
    on = featurize(load_deal(p, read_draft_approvals(manifest)), schema)
    by = schema.by_key()
    assert on.targets["read:co_hours_estimate"][1] == by["read:co_hours_estimate"].index("present")
    assert on.numbers_target["read:co_hours_estimate"][1] == 2.0
    assert on.targets["jdg:hours:per"][1] == by["jdg:hours:per"].index("device")
    assert on.numbers_target["jdg:hours:qty"][1] == 4.0
    # Another file is not approved by this manifest, and a changed file is refused.
    q, _ = _write(tmp_path, _blob(schema), "900002@pre_quote.json")
    assert all(v == IGNORE for v in featurize(load_deal(q, read_draft_approvals(manifest)), schema)
               .targets["read:co_hours_estimate"])
    p.write_text(json.dumps(_blob(schema, verdict="hours=3;per=site;qty=1;role=tech;basis=estimate")))
    with pytest.raises(ValueError, match="changed since"):
        load_deal(p, read_draft_approvals(manifest))


def test_hours_per_folds_spellings():
    assert hours_per("displays") == "device" and hours_per("location") == "site"
    assert hours_per("job") == "job" and hours_per("school") == "site" and hours_per("weird") is None


def _grads(schema, blob, tmp_path):
    p, approved = _write(tmp_path, blob)
    batch = featurize(load_deal(p, approved), schema)
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    keep = {}
    total, logs = c3_loss(model, batch, LossWeights(), keep=keep)
    out = keep["out"]
    watch = {**{f"logits:{k}": t for k, t in out.logits.items()},
             **{f"number:{k}": t for k, t in out.numbers.items()},
             **{f"rel:{k}": t for k, t in out.relations.items()},
             **{f"rel_none:{k}": t for k, t in out.link_none.items()}}
    for t in watch.values():
        t.retain_grad()
    total.backward()
    return batch, logs, {k for k, t in watch.items() if t.grad is not None and t.grad.abs().sum() > 0}


def test_quote_numbers_train_and_masked_heads_get_nothing(tmp_path, schema):
    batch, logs, taught = _grads(schema, _blob(schema), tmp_path)
    assert {"number:read:co_hours_estimate", "logits:jdg:hours:per", "number:jdg:hours:qty"} <= taught
    masked = {k for k, v in batch.head_scale.items() if not v}
    assert not {n for n in taught if n.split(":", 1)[1] in masked or n.startswith("rel")}
    assert logs["relations"] == 0 and logs["absence"] == 0
    assert (absence_targets(batch) == -1).all()


def test_masked_number_head_gets_no_value_loss(tmp_path, schema):
    keys = [o.key for o in schema.opportunities]
    scale = {k: 0.0 for k in keys if k not in QUOTE or k == "read:co_hours_estimate"}
    _, _, taught = _grads(schema, _blob(schema, scale=scale), tmp_path)
    assert "number:read:co_hours_estimate" not in taught and "number:jdg:hours:qty" in taught


def test_new_heads_keep_the_old_shape(schema):
    old = without_heads(schema, "jdg:hours:per,jdg:hours:qty")
    assert [o.key for o in old.opportunities] == [o.key for o in schema.opportunities
                                                  if o.key not in ("jdg:hours:per", "jdg:hours:qty")]
    assert shape_digest(old) != shape_digest(schema)
    with pytest.raises(SystemExit):
        without_heads(schema, "jdg:hours:nope")
