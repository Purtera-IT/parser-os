"""The question and answer texts the described heads are built from (Run E audit)."""
import json

from ml.c3.schema import TYPES_PATH, load_schema, scrub


def _by_key():
    return load_schema().by_key()


def test_the_supplier_question_reaches_the_model():
    # atom_types.json keeps the supplier question under "suppliers_doc".
    doc = scrub(json.loads(TYPES_PATH.read_text(encoding="utf-8"))["suppliers_doc"], True)
    assert doc and doc in _by_key()["col:supplier"].description
