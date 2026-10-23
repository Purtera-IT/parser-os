"""NEW_HEADS from the schema diff (ml/c3/new_heads.py, pod/runE.sh).

* a head added anywhere is listed; a removed key, a changed answer list or a
  reorder is not a head addition and fails loudly, naming the difference;
* the saved run C shape (ml/c3/runE/init_shape.json) plus the derived heads is
  this schema: the list loads INIT (shape digest of the schema without them);
* every head added since the live checkpoint is derived, the new label and card
  heads included, and runE.sh has no hand-kept default left to drift;
* a checkpoint saved under another shape is refused.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ml.c3 import new_heads as nh
from ml.c3.gates import schema_digests_at
from ml.c3.loop import without_heads
from ml.c3.schema import load_schema, shape_digest, shape_of

REPO = Path(__file__).resolve().parents[1]
SHAPE = REPO / "ml/c3/runE/init_shape.json"
OLD = [("a", ["x", "y"]), ("b", ["p", "q"]), ("c", ["1", "2"])]


def test_diff_lists_additions_and_refuses_the_rest():
    added, problems = nh.diff([("a", ["x", "y"]), ("n", ["z"]), ("b", ["p", "q"]), ("c", ["1", "2"]),
                               ("m", ["w"])], OLD)
    assert added == ["n", "m"] and not problems
    _, problems = nh.diff([("a", ["x", "y"]), ("c", ["1", "2"])], OLD)
    assert problems == ["b: removed (INIT has it, this schema does not)"]
    _, problems = nh.diff([("a", ["x", "y", "z"]), ("b", ["p", "q"]), ("c", ["1", "2"])], OLD)
    assert problems == ["a: answers added z"]
    _, problems = nh.diff([("a", ["y", "x"]), ("b", ["p", "q"]), ("c", ["1", "2"])], OLD)
    assert problems == ["a: answers reordered"]
    _, problems = nh.diff([("b", ["p", "q"]), ("a", ["x", "y"]), ("c", ["1", "2"])], OLD)
    assert problems and problems[0].startswith("keys reordered")


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def test_derived_heads_load_the_live_checkpoint_shape(schema):
    blob = json.loads(SHAPE.read_text())
    old = [(k, v) for k, v in blob["shape"]]
    assert nh._digest([list(x) for x in old]) == blob["shape_sha256"]
    added = nh.derive(schema, old)
    assert shape_digest(without_heads(schema, ",".join(added))) == blob["shape_sha256"]
    for k in ("read:feeds_decision", "read:visit_count", "read:work_action", "read:sow_element",
              "read:sow_transform", "jdg:sow_slot", "jdg:hours:per", "jdg:hours:qty",
              "jdg:rule", "col:said_by_role", "col:said_by_side", "col:said_to_side", "col:internal_only"):
        assert k in added, k
    # Everything this schema has beyond INIT is in the list.
    assert set(added) == {k for k, _ in shape_of(schema)} - {k for k, _ in old}
    assert schema_digests_at(str(SHAPE)) == (blob["schema_sha256"], blob["shape_sha256"])


def test_a_non_addition_fails_loudly(schema):
    blob = json.loads(SHAPE.read_text())
    old = [(k, list(v)) for k, v in blob["shape"]]
    old[0] = (old[0][0], old[0][1] + ["an_answer_init_had"])
    with pytest.raises(SystemExit, match="answers removed an_answer_init_had"):
        nh.derive(schema, old)


def test_checkpoint_must_be_the_shape(tmp_path):
    import torch

    blob = json.loads(SHAPE.read_text())
    old = [(k, v) for k, v in blob["shape"]]
    ok, bad, pre = tmp_path / "ok.pt", tmp_path / "bad.pt", tmp_path / "pre.pt"
    torch.save({"schema_sha256": "s", "shape_sha256": blob["shape_sha256"], "state": {}}, ok)
    torch.save({"schema_sha256": "s", "shape_sha256": "0" * 64, "state": {}}, bad)
    torch.save({"schema_sha256": blob["schema_sha256"], "state": {}}, pre)
    assert nh.checkpoint_matches(str(ok), old) == ""
    assert "not" in nh.checkpoint_matches(str(bad), old)
    assert nh.checkpoint_matches(str(pre), old, schema_sha256=blob["schema_sha256"]) == ""
    assert nh.checkpoint_matches(str(pre), old)
    assert nh.main(["--shape", str(SHAPE), "--checkpoint", str(ok)]) == 0
    with pytest.raises(SystemExit, match="INIT is not"):
        nh.main(["--shape", str(SHAPE), "--checkpoint", str(bad)])


def test_run_script_derives_the_list():
    text = (REPO / "ml/c3/pod/runE.sh").read_text(encoding="utf-8")
    assert not re.search(r"NEW_HEADS:-[a-z]", text)       # no hand-kept default to drift
    assert 'python -m ml.c3.new_heads "${shape_src[@]}" --checkpoint "$INIT"' in text
    assert '--allow-new-heads "$new_heads"' in text
    assert text.count('--init-schema-ref "${INIT_SCHEMA_REF:-$INIT_SHAPE}"') == 2
