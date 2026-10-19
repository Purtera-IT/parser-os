"""jdg:sow_section: the SOW placement rubric applied through decide()."""
from pathlib import Path

import pytest

from app.core import decide as decide_mod
from app.core import semantic_role
from app.core import sow_rubric_judge as J
from app.core.decide import Decision


@pytest.fixture
def calls(monkeypatch):
    seen = []
    answers = {}

    def _clf(text, candidates, *, instruction, context="", timeout=None, model=None):
        seen.append({"text": text, "candidates": candidates, "instruction": instruction, "context": context,
                     "model": model})
        kind = "owner" if candidates == J.OWNERS else "section"
        return answers.get((kind, text), (None, 0.0))

    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    prev = decide_mod.get_store()
    decide_mod.set_store(None)
    yield seen, answers
    decide_mod.set_store(prev)


def test_rubric_text_is_the_doc():
    doc = Path(__file__).resolve().parents[1] / "docs" / "C3_RUBRIC_SOW_SECTION.md"
    assert doc.read_text() == J.RUBRIC


def test_section_asked_with_rubric_and_named_model(calls, monkeypatch):
    seen, answers = calls
    monkeypatch.delenv("SOWSMITH_SOW_JUDGE_MODEL", raising=False)
    answers[("section", "Mount the 65 inch display")] = ("scope_of_work", 0.9)
    v = J.judge("Mount the 65 inch display", job="av_conference_room", under="Scope", deal_id="010120")
    assert (v.section, v.owner, v.source) == ("scope_of_work", "technician", "llm")
    assert len(seen) == 1  # owner follows from scope_of_work, no second call
    assert seen[0]["model"] == "ollama:qwen3:32b"
    assert seen[0]["candidates"] == J.SECTIONS
    assert "first test that matches wins" in seen[0]["instruction"]
    assert "Job family: av_conference_room" in seen[0]["context"]


def test_owner_asked_only_where_the_labels_split(calls):
    seen, answers = calls
    t = "CDW will ship the switches to each store"
    answers[("section", t)] = ("customer_responsibilities", 0.8)
    answers[("owner", t)] = ("partner", 0.7)
    v = J.judge(t)
    assert (v.section, v.owner) == ("customer_responsibilities", "partner")
    assert len(seen) == 2 and "customer_responsibilities" in seen[1]["instruction"]


def test_undecided_model_changes_nothing(calls):
    v = J.judge("anything at all")
    assert v.section is None and v.owner is None and v.source == "fallback"
    assert J.judge("   ").section is None


def test_teacher_rows_never_decide(calls, monkeypatch):
    seen, _ = calls
    asked = {}

    class Store:
        def resolve(self, *, relation, exclude_created_by=(), **_):
            asked["exclude"] = exclude_created_by
            return Decision(verdict="scope_of_work", confidence=0.99, source="store", correction_id="teacher_abc")

        def few_shot(self, **_):
            return []

    decide_mod.set_store(Store())
    v = J.judge("Install the AP")
    assert asked["exclude"] == ("teacher",)
    assert v.section is None and v.source == "teacher"


def test_a_pm_correction_decides_without_the_model(calls):
    seen, _ = calls

    class Store:
        def resolve(self, *, relation, **_):
            if relation == J.RELATION:
                return Decision(verdict="out_of_scope", confidence=0.95, source="store", correction_id="corr_pm")
            return None

        def few_shot(self, **_):
            return []

    decide_mod.set_store(Store())
    v = J.judge("No cabling")
    assert v.section == "out_of_scope" and v.source == "store"
    assert all(c["candidates"] == J.OWNERS for c in seen)  # only the owner question reached the model


def test_model_env_default_keeps_decide_route(monkeypatch):
    monkeypatch.setenv("SOWSMITH_SOW_JUDGE_MODEL", "default")
    assert J.judge_model() is None
    monkeypatch.setenv("SOWSMITH_SOW_JUDGE_MODEL", "ollama:qwen3:14b")
    assert J.judge_model() == "ollama:qwen3:14b"
