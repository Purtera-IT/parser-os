"""The question and answer texts the described heads are built from (Run E audit)."""
import json

from ml.c3.schema import TYPES_PATH, load_schema, scrub


def _by_key():
    return load_schema().by_key()


def test_the_supplier_question_reaches_the_model():
    # atom_types.json keeps the supplier question under "suppliers_doc".
    doc = scrub(json.loads(TYPES_PATH.read_text(encoding="utf-8"))["suppliers_doc"], True)
    assert doc and doc in _by_key()["col:supplier"].description


def test_closed_answers_do_not_open_with_the_shared_subject():
    for o in load_schema().opportunities:
        if o.kind != "class" or len(o.answers) < 3:
            continue
        opened = [a for a in o.answers[1:] if a.description.lower().startswith(("why we reject", "primary service:",
                                                                                 "where the deal was"))]
        assert not opened, (o.key, opened[0].description)


def test_the_noise_type_does_not_say_keep():
    by_key = _by_key()
    noise = {a.value: a.description for a in by_key["col:label_type"].answers}["_keep"]
    assert noise.startswith("noise: ") and "keep" not in noise.split(".")[0].lower()
    assert "stakeholder" in noise and "list header" in noise


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
    assert "Payment timing is a payment term" in types["contract_term"]
    assert "Unlike a dependency, nobody is asked" in types["assumption"]
    assert "seller" in types["task"]
    owner = {a.value: a.description for a in by_key["read:task_owner_role"].answers}
    assert "sellers do not get tasks" not in by_key["read:task_owner_role"].description
    assert "owns the step" in owner["seller"]
    assert "signature" in {a.value: a.description for a in by_key["read:co_reason"].answers}["contact_from_crm"]
    same_as = by_key["rel:same_as"].description
    assert "From a copy to the learned line" in same_as and "background, not an input" in same_as
    assert "price attaches" in {a.value: a.description for a in by_key["jdg:task_tier"].answers}["parent"]
