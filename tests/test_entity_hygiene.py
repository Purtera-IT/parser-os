"""Entity hygiene regression tests.

Closes the ``site:product_or_framework`` blocker from the corpus
review. After PR3 these keys are dropped before they reach
``site_reality`` clustering.
"""
from __future__ import annotations

from app.core.entity_hygiene import filter_entity_keys_for_atom
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def make_atom(text: str) -> EvidenceAtom:
    return EvidenceAtom(
        id="atm_test",
        project_id="p",
        artifact_id="a",
        atom_type=AtomType.scope_item,
        raw_text=text,
        normalized_text=text.lower(),
        value={"text": text},
        entity_keys=[],
        source_refs=[
            SourceRef(
                id="src_test",
                artifact_id="a",
                artifact_type=ArtifactType.txt,
                filename="x.txt",
                locator={"line": 1},
                extraction_method="test",
                parser_version="test",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def test_product_terms_do_not_become_sites():
    atom = make_atom("Install Belden Cat6 CMP to IDF patch panels.")
    keys = filter_entity_keys_for_atom(
        atom, ["site:belden_cat6_cmp", "vendor:belden"]
    )
    assert "site:belden_cat6_cmp" not in keys
    assert "vendor:belden" in keys


def test_real_site_survives():
    atom = make_atom(
        "Banks High School / District Core at 13050 NW Main St requires "
        "after-hours MDF access."
    )
    keys = filter_entity_keys_for_atom(atom, ["site:banks_high_school"])
    assert "site:banks_high_school" in keys


def test_framework_does_not_become_site():
    atom = make_atom("Operations follow CISA vulnerability playbook for triage.")
    keys = filter_entity_keys_for_atom(atom, ["site:cisa_vulnerability_playbook"])
    assert "site:cisa_vulnerability_playbook" not in keys


def test_saas_tool_does_not_become_site():
    atom = make_atom("Tickets logged in ServiceNow with PagerDuty alerts.")
    keys = filter_entity_keys_for_atom(
        atom, ["site:servicenow", "site:pagerduty"]
    )
    assert "site:servicenow" not in keys
    assert "site:pagerduty" not in keys


def test_non_site_keys_pass_through_untouched():
    atom = make_atom("ServiceNow integration noted.")
    keys = filter_entity_keys_for_atom(
        atom,
        [
            "vendor:servicenow",
            "device:ip_camera",
            "part_number:cw9166i_b",
            "standard:nfpa_72",
        ],
    )
    assert "vendor:servicenow" in keys
    assert "device:ip_camera" in keys
    assert "part_number:cw9166i_b" in keys
    assert "standard:nfpa_72" in keys


def test_address_with_some_product_words_still_survives():
    atom = make_atom(
        "Cisco Meraki AP refresh at the building at 13050 NW Main St."
    )
    keys = filter_entity_keys_for_atom(atom, ["site:13050_nw_main_st"])
    assert "site:13050_nw_main_st" in keys
