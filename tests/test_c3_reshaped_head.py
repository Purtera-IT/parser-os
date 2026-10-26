"""read:requirement_kind became a 9-value class head (owner, round-2 approval 2026-10-26). To a checkpoint
saved before (presence: _absent | present) it is a new head: new_heads derives it, the shape check reads it at
its old answers, and its old free-head rows are dropped so it starts from its description."""
from dataclasses import asdict, replace

import pytest
import torch

from ml.c3 import new_heads as nh
from ml.c3.gates import load_checkpoint, save_checkpoint
from ml.c3.loop import RESHAPED, SMALL, without_heads
from ml.c3.model import C3Model
from ml.c3.readiness import digest
from ml.c3.schema import CLASS, PRESENCE, Answer, load_schema, shape_digest, shape_of

KEY = "read:requirement_kind"
NINE = ["access", "schedule", "personnel", "safety", "access_equipment", "tech_toolkit", "technical_spec",
        "documentation", "commercial"]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def old_schema(schema):
    """The schema as a checkpoint saved before the change had it."""
    return replace(schema, opportunities=[
        replace(o, kind=PRESENCE, answers=(Answer("_absent", "not said"), Answer("present", "said")))
        if o.key == KEY else o for o in schema.opportunities])


def test_requirement_kind_is_a_nine_value_class_head(schema):
    o = schema.by_key()[KEY]
    assert o.kind == CLASS and [a.value for a in o.answers] == ["_absent", *NINE]
    assert o.index("tech_toolkit") is not None and o.index("hardware") is None


def test_the_reshaped_head_is_derived_as_new_and_reads_at_its_old_answers(schema):
    old = shape_of(old_schema(schema))
    assert RESHAPED[KEY] == ("_absent", "present")
    assert nh.derive(schema, old) == [KEY]
    assert shape_digest(without_heads(schema, KEY)) == nh._digest([list(x) for x in old])
    # Any other answer-list change is still refused.
    other = [(k, (v + ["extra"] if k == "read:billing_type" else v)) for k, v in old]
    with pytest.raises(SystemExit):
        nh.derive(schema, other)


def test_an_old_checkpoint_loads_and_the_head_starts_from_its_description(schema, tmp_path):
    torch.manual_seed(0)
    before = old_schema(schema)
    m0 = C3Model(before, SMALL)
    lin = m0.add_free_head(KEY)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    path = tmp_path / "old.pt"
    save_checkpoint(path, m0, step=3, schema_sha256=digest([asdict(o) for o in before.opportunities]),
                    shape_sha256=shape_digest(before))
    m1 = C3Model(schema, SMALL)
    step = load_checkpoint(path, m1, schema_sha256=digest([asdict(o) for o in schema.opportunities]),
                           allow_text_change=True, shape_sha256=shape_digest(without_heads(schema, KEY)))
    assert step == 3
    w = m1.add_free_head(KEY).weight
    assert tuple(w.shape)[0] == 1 + len(NINE) and float(w.detach().abs().sum()) == 0.0
    # Every other weight came from the checkpoint.
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0 if not k.startswith(f"free_heads.{KEY}."))
