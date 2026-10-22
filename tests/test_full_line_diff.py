"""full_line_diff: every line compared, a one-answer jump flagged as collapse."""
from ml.c3.full_line_diff import compare


def test_unchanged_is_clean():
    r = {"types": {"a": ["task", 0.9], "b": ["physical_site", 0.99]}, "reads": {"a": {"read:co_action": ["keep", 0.9]}}}
    c = compare(r, r, ["read:co_action"], [])
    assert c["type"]["changed"] == 0 and not c["type"]["collapse"]
    assert c["read:co_action"]["changed"] == 0


def test_collapse_to_one_answer_is_flagged():
    old = {"types": {str(i): [t, 0.9] for i, t in enumerate(["task", "physical_site", "contact", "scope_item"] * 5)}}
    new = {"types": {str(i): ["scope_item", 0.99] for i in range(20)}}
    c = compare(old, new, [], [])
    assert c["type"]["changed"] == 15
    assert c["type"]["collapse"] and c["type"]["max_shift_answer"] == "scope_item"


def test_documents_compared():
    old = {"types": {}, "documents": {"d1": {"jdg:document_job": ["this_deal", 0.9]}}}
    new = {"types": {}, "documents": {"d1": {"jdg:document_job": ["other_job", 0.9]}}}
    c = compare(old, new, [], ["jdg:document_job"])
    assert c["jdg:document_job"]["changed"] == 1
