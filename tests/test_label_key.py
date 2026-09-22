"""label_key must survive a re-parse and match the JS labeler byte for byte."""
from __future__ import annotations

from app.core.label_key import label_key

DEAL = "1bf0c10e-e840-4a1f-b526-d8f417181ada"

# Shared vector: Platform-infra azure-function-api/shared/atom-labeling.test.js
# asserts the SAME value. Change both or neither.
SHARED_VECTOR = ((DEAL, "Notes.pdf", 0, "  Mount   110 TVs "), "lbl_7ea92c3fa76a1113c188")


def test_shared_vector_with_the_js_labeler():
    args, expected = SHARED_VECTOR
    assert label_key(*args) == expected


def test_whitespace_and_case_do_not_move_the_key():
    assert label_key(DEAL, "Notes.pdf", 0, "Mount 110 TVs") == label_key(DEAL, "notes.PDF", 0, "mount\n110  tvs")


def test_page_file_and_deal_do():
    base = label_key(DEAL, "Notes.pdf", 0, "Mount 110 TVs")
    assert base != label_key(DEAL, "Notes.pdf", 1, "Mount 110 TVs")
    assert base != label_key(DEAL, "SOW.pdf", 0, "Mount 110 TVs")
    assert base != label_key("other", "Notes.pdf", 0, "Mount 110 TVs")
    assert label_key(DEAL, "Notes.pdf", None, "x") != label_key(DEAL, "Notes.pdf", 0, "x")
