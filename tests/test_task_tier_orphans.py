"""A child needs a parent: a document whose task lines are all children proposes nothing (010283)."""
from types import SimpleNamespace

from app.core.schemas import AtomType
from app.core.task_tier_classifier import classify_task_tiers, promote_orphan_children


def _t(text, artifact_id, depth=None, tier="child"):
    v = {"task_tier": tier, "is_quote_line": tier == "parent"}
    if depth is not None:
        v["depth"] = depth
    return SimpleNamespace(atom_type=AtomType.task, raw_text=text, value=v, artifact_id=artifact_id, review_flags=["task_tier_" + tier], source_refs=[])


def test_a_document_of_only_children_promotes_its_shallowest_lines():
    lines = [_t("Cable / device deinstallation", "note", 1), _t("Site surveys", "note", 1), _t("Troubleshooting of fiber & copper", "note", 1),
             _t("verify LED after reseat", "note", 2)]
    assert promote_orphan_children(lines) == 3
    assert [l.value["task_tier"] for l in lines] == ["parent", "parent", "parent", "child"]
    assert lines[0].value["is_quote_line"] is True and lines[0].value["tier_promoted"] == "orphan"
    assert "task_tier_parent" in lines[0].review_flags and "task_tier_child" not in lines[0].review_flags


def test_a_document_with_a_parent_or_a_single_line_is_left_alone():
    runbook = [_t("Install the AP", "sow", tier="parent"), _t("Connect the cable", "sow", 2), _t("Verify the LED", "sow", 2)]
    assert promote_orphan_children(runbook) == 0
    assert [l.value["task_tier"] for l in runbook] == ["parent", "child", "child"]
    lone = [_t("This scope assumes the work of one technician", "email")]
    assert promote_orphan_children(lone) == 0 and lone[0].value["task_tier"] == "child"
    other = [_t("Confirm the count", "a"), _t("Verify the count", "b")]  # two documents, one child each
    assert promote_orphan_children(other) == 0


def test_classify_task_tiers_runs_the_promotion():
    lines = [SimpleNamespace(atom_type=AtomType.task, raw_text="Verify the LED", value={"depth": 1}, artifact_id="n", review_flags=[], source_refs=[]),
             SimpleNamespace(atom_type=AtomType.task, raw_text="Confirm connectivity", value={"depth": 1}, artifact_id="n", review_flags=[], source_refs=[])]
    atoms, changed = classify_task_tiers(lines)
    assert changed >= 2 and all(a.value["task_tier"] == "parent" for a in atoms)
