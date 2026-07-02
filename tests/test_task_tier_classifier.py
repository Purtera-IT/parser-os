"""Tests for parent vs child task tier classification."""

from __future__ import annotations

from types import SimpleNamespace

from app.core.task_tier_classifier import (
    classify_task_tiers,
    infer_task_tier,
    is_quote_line_task_atom,
)


def _task(text: str, **value: object) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"atm_{hash(text) & 0xFFFF:04x}",
        atom_type=SimpleNamespace(value="task"),
        raw_text=text,
        normalized_text=text,
        value=dict(value),
        source_refs=[],
        review_flags=[],
        entity_keys=["site:test_site"],
    )


class TestInferTaskTier:
    def test_step_header_is_parent_quote_line(self) -> None:
        tier, quote = infer_task_tier(
            text="Step 1: Unbox and verify kiosk parts",
            structured={"kind": "paragraph"},
        )
        assert tier == "parent"
        assert quote is True

    def test_runbook_imperative_is_child(self) -> None:
        tier, quote = infer_task_tier(
            text="Verify power and network LEDs are illuminated.",
            structured={"kind": "bullet", "depth": 1},
        )
        assert tier == "child"
        assert quote is False

    def test_deliverable_scope_is_parent(self) -> None:
        tier, quote = infer_task_tier(
            text="Develop schedule based on stakeholder requirements and dependencies",
            structured={},
        )
        assert tier == "parent"
        assert quote is True

    def test_confirm_checklist_is_child(self) -> None:
        tier, quote = infer_task_tier(
            text="Confirm plenum, conduit, raceway, support, or firestop requirements.",
            structured={},
        )
        assert tier == "child"
        assert quote is False

    def test_vendor_scope_statement_is_parent(self) -> None:
        text = (
            "PurTera will connect the provided router to power and connect the Ethernet "
            "cable between the router and the kiosk at the site."
        )
        tier, quote = infer_task_tier(text=text, structured={"kind": "paragraph"})
        assert tier == "parent"
        assert quote is True


class TestClassifyTaskTiers:
    def test_stamps_value_and_parent_hint(self) -> None:
        parent = _task("Step 2: Position the base plate", kind="paragraph")
        child = _task("Place the base plate flat on the floor.", kind="bullet", depth=1)
        _, changed = classify_task_tiers([parent, child])
        assert changed == 2
        assert parent.value["task_tier"] == "parent"
        assert parent.value["is_quote_line"] is True
        assert child.value["task_tier"] == "child"
        assert child.value["is_quote_line"] is False
        assert child.value.get("parent_task_id") == parent.id
        assert child.value.get("parent_task_hint") == "Position the base plate"

    def test_is_quote_line_task_atom(self) -> None:
        parent = _task("Validate deliverables with Customer")
        child = _task("Locate the network closet, IDF, MDF, or switch location serving the APs.")
        classify_task_tiers([parent, child])
        assert is_quote_line_task_atom(parent) is True
        assert is_quote_line_task_atom(child) is False
