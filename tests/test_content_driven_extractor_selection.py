"""v57 content-driven extractor selection: which of the ~28
multi_entity_llm extractors run is derived per-deal from the deal's own
content (filename / atom-type / body-keyword signal), NOT from a per-pack
allow-list. An extractor the whole deal is silent for is SKIPPED (zero LLM
calls) rather than fanned out across every document.

This is the universality-preserving lever: an enterprise-IT deal that
ships integration / data-flow content keeps those extractors; an AV deal
with none of that content drops them. Nothing is gated on the routed pack.

Pure-helper level — no LLM / network.
"""

from __future__ import annotations

from app.core.multi_entity_llm import _active_extractor_keys, _extractor_has_signal

ALL = [
    "customer", "stakeholders", "milestones", "requirements",
    "site_clusters", "quantities", "data_flow_steps", "system_mappings",
    "integration_checkpoints", "compliance_classifications",
]


def _doc(filename="", headings=(), bodies=()):
    return {"filename": filename, "headings": list(headings), "bodies": list(bodies)}


def test_zero_signal_extractor_is_dropped():
    # An AV-flavored deal: cameras + a site. No integration / data-flow /
    # mapping / classification content anywhere.
    by_artifact = {
        "a1": _doc(
            filename="site_survey.pdf",
            bodies=["Install 12 cameras at Main Campus building."],
        ),
    }
    atom_index = {"a1": {"physical_site", "bom_line"}}
    out = _active_extractor_keys(ALL, by_artifact, atom_index)
    # Enterprise-IT-only extractors have no signal here -> dropped.
    assert "data_flow_steps" not in out
    assert "system_mappings" not in out
    assert "integration_checkpoints" not in out
    # The AV-relevant ones survive.
    assert "site_clusters" in out
    assert "quantities" in out


def test_enterprise_it_deal_keeps_its_extractors():
    # The SAME extractor set, but an integration-heavy deal: those
    # extractors must NOT be starved.
    by_artifact = {
        "a1": _doc(
            filename="integration_spec.pdf",
            bodies=[
                "Data flow: export records from HubSpot CRM to the data lake.",
                "Field mapping from source to target system.",
                "Integration test checkpoint IC-3 before go-live.",
            ],
        ),
    }
    atom_index = {"a1": {"system_mapping", "integration_checkpoint"}}
    out = _active_extractor_keys(ALL, by_artifact, atom_index)
    assert "data_flow_steps" in out
    assert "system_mappings" in out
    assert "integration_checkpoints" in out


def test_customer_anchor_always_runs():
    # A deal with essentially no signal for anything still keeps customer.
    by_artifact = {"a1": _doc(filename="random.bin", bodies=["zzz"])}
    out = _active_extractor_keys(["customer", "data_flow_steps"], by_artifact, {"a1": set()})
    assert "customer" in out


def test_disable_flag_runs_everything(monkeypatch):
    monkeypatch.setenv("SOWSMITH_RELEVANCE_GATE_DISABLE", "1")
    by_artifact = {"a1": _doc(filename="x.txt", bodies=["nothing relevant"])}
    out = _active_extractor_keys(ALL, by_artifact, {"a1": set()})
    assert out == ALL


def test_order_is_preserved():
    by_artifact = {
        "a1": _doc(
            filename="integration_spec.pdf",
            bodies=["data flow export; field mapping source target; checkpoint IC-1"],
        ),
    }
    out = _active_extractor_keys(
        ["customer", "data_flow_steps", "system_mappings"], by_artifact, {"a1": set()}
    )
    assert out == ["customer", "data_flow_steps", "system_mappings"]


def test_never_returns_empty():
    # Pathological: an extractor list where nothing matches and no anchor.
    by_artifact = {"a1": _doc(filename="x", bodies=["zzz"])}
    out = _active_extractor_keys(["data_flow_steps"], by_artifact, {"a1": set()})
    # Falls back to the input rather than silently extracting nothing.
    assert out == ["data_flow_steps"]


def test_weak_signal_extractor_never_dropped():
    # 'assumptions' has no atom_types and only filename+text_kw; but the
    # default-True branch in _doc_is_relevant_for means filename-less docs
    # still need a body-keyword hit. Verify an extractor with ONLY
    # filename signal (no atom_types, no text_kw) always passes.
    # 'penalties' has empty atom_types but real text_kw, so use a key with
    # no signal entry at all -> default True everywhere.
    by_artifact = {"a1": _doc(filename="x", bodies=["zzz"])}
    assert _extractor_has_signal("totally_unclassified_key", by_artifact, {"a1": set()})


def test_body_keyword_deep_in_doc_still_keeps_extractor():
    # Regression: the relevance gate used to scan only the first 5 atom
    # bodies, so a 'shall' clause buried past atom #5 in a long doc made
    # the requirements extractor get dropped despite real requirements
    # being present. The full-text haystack must catch it regardless of
    # position. (Found by the cross-deal selection harness on a 130-atom
    # doc where requirement clauses sat deep in the body.)
    bodies = [f"line {i} of boilerplate header text" for i in range(40)]
    bodies.append("The contractor shall provide redundant power to each rack.")
    by_artifact = {"a1": _doc(filename="notes.txt", bodies=bodies)}
    out = _active_extractor_keys(["customer", "requirements"], by_artifact, {"a1": set()})
    assert "requirements" in out


def test_signatory_keyword_anywhere_keeps_extractor():
    # Same shape for signatures sitting at the bottom of a long doc.
    bodies = ["intro"] * 30 + ["Authorized signature: ____  Approved by: J. Doe"]
    by_artifact = {"a1": _doc(filename="contract.pdf", bodies=bodies)}
    out = _active_extractor_keys(["customer", "signatories"], by_artifact, {"a1": set()})
    assert "signatories" in out


def test_atom_type_signal_alone_keeps_extractor():
    # No filename / body-keyword hit, but the typed classifier already
    # emitted a compliance_classification atom -> keep that extractor.
    by_artifact = {"a1": _doc(filename="notes.txt", bodies=["misc text"])}
    atom_index = {"a1": {"compliance_classification"}}
    out = _active_extractor_keys(["compliance_classifications"], by_artifact, atom_index)
    assert out == ["compliance_classifications"]
