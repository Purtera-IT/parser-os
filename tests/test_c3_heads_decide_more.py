"""Heads decide, the LLM does not (2026-10-28): typing's untaken sure types, the site catalog,
entity-name dedup, site role and geo mentions skip the LLM when the C3 heads are on; the brief
says per part who decided and why no head did; c3_share counts who decided what."""
from __future__ import annotations

import pytest

import app.core.c3_brief as brief
import app.core.c3_type_base as c3
import app.core.semantic_role as semantic_role
import app.core.typed_atom_classifier as tac
from app.core import c3_share
from app.core.site_geo_fallback import suppress_vendor_sites
from tests.test_c3_type_base import _atoms, _clean, llm  # noqa: F401  (fixtures)
from tests.test_vendor_site_suppression import _site


def test_heads_decide_follows_the_type_switch_and_the_restore_list(monkeypatch):
    monkeypatch.delenv("SOWSMITH_C3_TYPE", raising=False)
    assert not c3.heads_decide("site_role")
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "on")
    assert c3.heads_decide("site_role")
    monkeypatch.setenv("SOWSMITH_LLM_WITH_HEADS", "site_role,entity_dedup")
    assert not c3.heads_decide("site_role") and c3.heads_decide("site_catalog")
    monkeypatch.setenv("SOWSMITH_LLM_WITH_HEADS", "all")
    assert not c3.heads_decide("typing_untaken")


def test_a_sure_type_outside_the_llm_taxonomy_is_assigned_not_sent(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "risk,assumption")
    atoms = _atoms()
    answers = {"a_task": ("risk", 0.95), "a_keep": ("assumption", 0.9)}
    monkeypatch.setattr(c3, "classify", lambda a, shas=None: {id(x): answers[x.id] for x in a if x.id in answers})
    tac.classify_atoms(atoms)
    by = {a.id: a for a in atoms}
    assert str(getattr(by["a_task"].atom_type, "value", by["a_task"].atom_type)) == "risk"
    assert "a_task" not in llm and "a_keep" not in llm
    assert c3.status()["typing"]["sure_not_taken"] == {}


def test_the_old_path_comes_back_with_the_restore_switch(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "risk")
    monkeypatch.setenv("SOWSMITH_LLM_WITH_HEADS", "typing_untaken")
    atoms = _atoms()
    monkeypatch.setattr(c3, "classify", lambda a, shas=None: {id(x): ("risk", 0.95) for x in a if x.id == "a_task"})
    tac.classify_atoms(atoms)
    assert "a_task" in llm


def test_site_role_takes_the_heads_reading_at_any_p_and_never_calls_the_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "on")
    calls = []
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: calls.append(a) or None)
    vendor = _site("atm_v", raw_text="Acme LLC, 1 Main St, Alpharetta GA 30009")
    job = _site("atm_j", raw_text="Store 12, 40 Elm St, Dayton OH 45402")
    other = _site("atm_x", raw_text="Plant 3, 9 Oak Rd, Akron OH 44301")
    monkeypatch.setattr(c3, "_DEAL", {"reads": {"atm_v": {"jdg:site_role": ["vendor_or_billing_address", 0.7]},
                                                "atm_j": {"jdg:site_role": ["job_site", 0.6]}}})
    kept, dropped = suppress_vendor_sites([vendor, job, other], project_id="p")
    assert not calls
    assert dropped == 0 and len(kept) == 3  # a head's vendor reading proposes, it never deletes
    assert "vendor_address_suspected" in vendor.review_flags
    assert vendor.value["vendor_role_proposal"]["source"] == "c3"
    assert "vendor_address_suspected" not in job.review_flags + other.review_flags


def test_brief_parts_say_a_head_not_served_in_the_view_fell_back(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_BRIEF_TRUST", "read:co_action,read:sow_section")
    reads = {"atm_1": {"read:co_action": ["keep", 0.99]}}
    out = brief._part("read:sow_section", False, False, "planning", True)
    assert out["by"] == "rule" and "planning view" in out["why"]
    answered = brief._answered(reads, [], {})
    assert answered["read:co_action"] and not answered["read:sow_section"]
    assert brief._part("read:needed_by", False, True, "planning", True)["why"].startswith("registry status")


def test_a_set_valued_reading_reads_as_its_surest_answer():
    assert brief._answer([["project_manager", "atlas"], {"project_manager": 0.8, "atlas": 0.6}]) == ("project_manager", 0.8)
    assert brief._answer([[], {"project_manager": 0.2}]) == ("_absent", 0.0)
    assert brief._answer(["keep", 0.9]) == ("keep", 0.9)


def test_c3_share_counts_heads_rules_and_llm_from_a_result():
    atoms = [{"atom_type": "task", "value": {"c3_label": {"type": "task", "act": "authority"}}},
             {"atom_type": "scope_item", "value": {"c3_label": {"type": "_keep", "act": "keep"}}},
             {"atom_type": "risk", "value": {"c3_label": {"type": "task", "act": "assign"}}},
             {"atom_type": "task", "value": {}},
             {"atom_type": "task", "value": {}}]
    b = {"decided_by": {"keep": {"head:keep": 3, "rule:never": 1},
                        "parts": {"read:co_action": {"by": "head"}, "read:sow_section": {"by": "rule"}}},
         "sections": {"scope_of_work": [{"section_by": "base:read:sow_section"}, {"section_by": "rule:type"}]},
         "quote": {"lines": [{"by": {"techs": "heads:read:crew_size", "hours": "fallback:family_median"}}],
                   "billing": {"by": "none"}}}
    caps = {"deflect_counts": {"c3": {"typing": {"lines": 3, "taken": 2}}},
            "calls_by_stage": {"c3_brief": {"llm_calls": 3}, "entity_resolution": {"llm_calls": 40}}}
    s = c3_share.summarize(atoms, b, caps)
    assert s["line_type"] == {"head": 2, "override": 1, "rule": 1, "llm": 1, "share_head": 0.4}
    assert s["keep"]["head"] == 3 and s["section"]["head"] == 1 and s["brief_part"]["rule"] == 1
    assert s["quote"] == {"head": 1, "rule": 1, "share_head": 0.5}
    assert s["llm_decision_calls"] == {"entity_resolution": 40}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("SOWSMITH_LLM_WITH_HEADS", raising=False)
    semantic_role.reset_reachability()
    yield
