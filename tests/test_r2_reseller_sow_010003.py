"""Round-2 fixes from re-parsing a reseller display-install deal (010003):
a reseller SOW PDF (BOM table, PMO steps, exclusions, page footers), a
Gantt workbook, e-sign notices and seller emails with long quoted history.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EdgeType,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _mk(aid: str, atype: AtomType, text: str, keys: list[str]) -> EvidenceAtom:
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id="sow", atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value={"text": text}, entity_keys=keys,
        source_refs=[SourceRef(id="s" + aid, artifact_id="sow", artifact_type=ArtifactType.pdf,
                               filename="sow.pdf", extraction_method="x", parser_version="x", locator={})],
        authority_class=AuthorityClass.contractual_scope, confidence=0.85,
        review_status=ReviewStatus.needs_review, parser_version="x",
    )


# ── 3: one generic exclusion clause must not conflict with every scope line ──

def test_generic_exclusion_clause_does_not_conflict_with_every_scope_line() -> None:
    from app.core.graph_builder import build_edges

    clauses = [
        "Electrical work of any kind",
        "Drywall cutting or patching",
        "Furniture movement",
        "Troubleshooting or remediation of Customer network issues",
    ]
    # Every line of the SOW carries the document's one site.
    atoms = [_mk(f"ex{i}", AtomType.exclusion, t, ["site:nyc_office"]) for i, t in enumerate(clauses)]
    atoms += [_mk(f"sc{i:02d}", AtomType.scope_item, f"Mount display {i} on the wall",
                  ["site:nyc_office", "device:display"]) for i in range(37)]
    edges = build_edges("p", atoms, [])
    assert not [e for e in edges if e.edge_type == EdgeType.excludes]


def test_one_clause_is_capped_and_a_specific_exclusion_still_conflicts() -> None:
    from app.core.graph_builder import _MAX_EXCLUDES_PER_CLAUSE, build_edges

    ex = _mk("ex", AtomType.exclusion, "Display mounting above 12 ft is not included",
             ["site:nyc_office", "device:display"])
    scope = [_mk(f"sc{i:02d}", AtomType.scope_item, f"Mount display {i}", ["site:nyc_office", "device:display"])
             for i in range(30)]
    edges = [e for e in build_edges("p", [ex] + scope, []) if e.edge_type == EdgeType.excludes]
    assert 1 <= len(edges) <= _MAX_EXCLUDES_PER_CLAUSE

    # A whole-site exclusion names the site and still reaches its lines.
    site_ex = _mk("ex_site", AtomType.exclusion, "NYC Office is removed from scope", ["site:nyc_office"])
    edges = [e for e in build_edges("p", [site_ex] + scope[:3], []) if e.edge_type == EdgeType.excludes]
    assert len(edges) == 3
