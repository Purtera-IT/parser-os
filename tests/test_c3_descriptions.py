"""The question and answer texts the described heads are built from (Run E audit)."""
import json

from ml.c3.schema import TYPES_PATH, load_schema, scrub


def _by_key():
    # The new texts themselves, without the description gate's pins (runE/text_pins.json).
    import os
    old, os.environ["C3_TEXT_PINS"] = os.environ.get("C3_TEXT_PINS"), "0"
    try:
        return load_schema().by_key()
    finally:
        if old is None:
            del os.environ["C3_TEXT_PINS"]
        else:
            os.environ["C3_TEXT_PINS"] = old


def test_the_supplier_question_reaches_the_model():
    # atom_types.json keeps the supplier question under "suppliers_doc".
    doc = scrub(json.loads(TYPES_PATH.read_text(encoding="utf-8"))["suppliers_doc"], True)
    assert doc and doc in _by_key()["col:supplier"].description


def test_closed_answers_do_not_open_with_the_shared_subject():
    for o in _by_key().values():
        if o.kind != "class" or len(o.answers) < 3:
            continue
        opened = [a for a in o.answers[1:] if a.description.lower().startswith(("why we reject", "primary service:",
                                                                                 "where the deal was"))]
        assert not opened, (o.key, opened[0].description)


def test_the_noise_type_does_not_say_keep():
    by_key = _by_key()
    noise = {a.value: a.description for a in by_key["col:label_type"].answers}["_keep"]
    assert noise.startswith("noise: ") and "keep" not in noise.split(".")[0].lower()
    # Where signatures and headings go is guidance, not embedded text (schema.EMBED_CONTRAST).
    from ml.c3.schema import type_contrast
    assert any("stakeholder" in c for c in type_contrast()["_keep"])
    assert any("list header" in c for c in type_contrast()["_keep"])


def test_primary_service_answers_say_what_the_work_is():
    answers = {a.value: a.description for a in _by_key()["deal:primary_service"].answers}
    assert answers["imac"].startswith("imac. Install, move, add or change")
    assert all(len(d.split()) >= 6 for v, d in answers.items() if v != "_absent")


def test_admission_drop_covers_background_facts():
    answers = {a.value: a.description for a in _by_key()["col:admission"].answers}
    assert "background, not an input" in answers["drop"] and "copy or repeat" in answers["drop"]


def test_rulings_the_skill_makes_are_in_the_text():
    by_key = _by_key()
    types = {a.value: a.description for a in by_key["col:label_type"].answers}
    # The contrast rulings are guidance (type_contrast), not the embedded answer text (pilot A M2).
    from ml.c3.schema import type_contrast
    contrast = type_contrast()
    assert "Payment timing is a payment term." in contrast["contract_term"]
    assert any(c.startswith("Unlike a dependency, nobody is asked") for c in contrast["assumption"])
    assert "Payment timing" not in types["contract_term"] and "Unlike" not in types["assumption"]
    assert "seller" in types["task"]
    owner = {a.value: a.description for a in by_key["read:task_owner_role"].answers}
    assert "sellers do not get tasks" not in by_key["read:task_owner_role"].description
    assert "owns the step" in owner["seller"]
    assert "signature" in {a.value: a.description for a in by_key["read:co_reason"].answers}["contact_from_crm"]
    same_as = by_key["rel:same_as"].description
    assert "From a copy to the learned line" in same_as and "background, not an input" in same_as
    assert "price attaches" in {a.value: a.description for a in by_key["jdg:task_tier"].answers}["parent"]


def _tiny(schema):
    from ml.c3.model import C3Config, C3Model

    return C3Model(schema, C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8,
                                    d_head=16, residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2))


def _text_changed(schema, key="col:supplier"):
    from dataclasses import replace

    from ml.c3.schema import Answer, Schema

    opps = []
    for o in schema.opportunities:
        if o.key == key:
            o = replace(o, description=o.description + " Reworded.",
                        answers=tuple(Answer(a.value, a.description + " Reworded.") for a in o.answers))
        opps.append(o)
    return Schema(opportunities=opps)


def _value_added(schema, key="col:supplier"):
    from dataclasses import replace

    from ml.c3.schema import Answer, Schema

    return Schema(opportunities=[replace(o, answers=o.answers + (Answer("abstain", "abstain."),))
                                 if o.key == key else o for o in schema.opportunities])


def _digest(schema):
    from dataclasses import asdict

    from ml.c3.readiness import digest

    return digest([asdict(o) for o in schema.opportunities])


def test_init_from_accepts_a_checkpoint_whose_schema_changed_only_text(tmp_path):
    import pytest

    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.schema import shape_digest

    old = load_schema()
    new = _text_changed(old)
    assert _digest(new) != _digest(old) and shape_digest(new) == shape_digest(old)
    path = tmp_path / "runC.pt"
    model = _tiny(old)
    save_checkpoint(path, model, step=9, schema_sha256=_digest(old), shape_sha256=shape_digest(old))
    # Serving and every other reader stay strict.
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path, model, schema_sha256=_digest(new))
    assert load_checkpoint(path, model, schema_sha256=_digest(new), allow_text_change=True,
                           shape_sha256=shape_digest(new)) == 9

    # A checkpoint saved before shape digests: matched through its commit's digests.
    legacy = tmp_path / "legacy.pt"
    save_checkpoint(legacy, model, step=4, schema_sha256=_digest(old))
    with pytest.raises(ValueError, match="no shape digest"):
        load_checkpoint(legacy, model, schema_sha256=_digest(new), allow_text_change=True,
                        shape_sha256=shape_digest(new))
    assert load_checkpoint(legacy, model, schema_sha256=_digest(new), allow_text_change=True,
                           shape_sha256=shape_digest(new),
                           known_shapes={_digest(old): shape_digest(old)}) == 4


def test_init_from_refuses_a_checkpoint_whose_answer_values_changed(tmp_path):
    import pytest

    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.schema import shape_digest

    old = load_schema()
    new = _value_added(old)
    assert shape_digest(new) != shape_digest(old)
    path = tmp_path / "runC.pt"
    model = _tiny(old)
    save_checkpoint(path, model, step=9, schema_sha256=_digest(old), shape_sha256=shape_digest(old))
    with pytest.raises(ValueError, match="more than text"):
        load_checkpoint(path, model, schema_sha256=_digest(new), allow_text_change=True,
                        shape_sha256=shape_digest(new))
    # A renamed or reordered opportunity is refused the same way.
    moved = type(old)(opportunities=list(reversed(old.opportunities)))
    with pytest.raises(ValueError, match="more than text"):
        load_checkpoint(path, model, schema_sha256=_digest(moved), allow_text_change=True,
                        shape_sha256=shape_digest(moved))
    legacy = tmp_path / "legacy.pt"
    save_checkpoint(legacy, model, step=4, schema_sha256=_digest(old))
    with pytest.raises(ValueError, match="more than text"):
        load_checkpoint(legacy, model, schema_sha256=_digest(new), allow_text_change=True,
                        shape_sha256=shape_digest(new), known_shapes={_digest(old): shape_digest(old)})


def test_schema_digests_at_a_commit_match_the_schema_it_loads():
    import subprocess

    from ml.c3.gates import schema_digests_at
    from ml.c3.schema import REPO_ROOT, shape_digest

    dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "ml/c3/schema.py",
                            "ml/c3/values.py", "app/core", "ml/c3/judgment_heads.json"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        import pytest
        pytest.skip("schema files differ from HEAD")
    full, shape = schema_digests_at("HEAD")
    assert (full, shape) == (_digest(load_schema()), shape_digest(load_schema()))
