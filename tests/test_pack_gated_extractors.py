"""v57 pack-gated extractor selection: a domain pack's llm_extractors
allow-list narrows which of the ~28 multi_entity_llm extractors run,
cutting the per-document fan-out that dominates enrich_entities cost.

Tested at the pure-helper level so it needs no LLM / network.
"""

from __future__ import annotations

from app.core.multi_entity_llm import _gate_extractor_keys

ALL = [
    "customer", "stakeholders", "milestones", "requirements",
    "site_clusters", "quantities", "data_flow_steps", "system_mappings",
    "integration_checkpoints", "compliance_classifications",
]


def test_empty_allow_list_runs_everything():
    assert _gate_extractor_keys(ALL, []) == ALL
    assert _gate_extractor_keys(ALL, None) == ALL


def test_allow_list_narrows_to_subset_plus_customer():
    out = _gate_extractor_keys(ALL, ["requirements", "site_clusters", "quantities"])
    assert out == ["customer", "requirements", "site_clusters", "quantities"]
    # the irrelevant enterprise-IT extractors are dropped
    assert "data_flow_steps" not in out
    assert "system_mappings" not in out


def test_customer_always_included_even_if_omitted():
    out = _gate_extractor_keys(ALL, ["requirements"])
    assert "customer" in out


def test_order_is_preserved():
    out = _gate_extractor_keys(ALL, ["quantities", "milestones"])
    # order follows the input call_keys, not the allow-list
    assert out == ["customer", "milestones", "quantities"]


def test_unknown_keys_ignored_not_crash():
    out = _gate_extractor_keys(ALL, ["requirements", "totally_made_up_key"])
    assert out == ["customer", "requirements"]


def test_allow_list_matching_nothing_falls_back_to_all():
    # never silently extract nothing
    out = _gate_extractor_keys(ALL, ["nonexistent_a", "nonexistent_b"])
    assert out == ALL
