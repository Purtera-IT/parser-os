"""Graded site-readiness maturity model (Gap E).

A site we've positively located (a physical_site atom anchors it) but
have no other detail on must read as "located, details pending" (amber)
— not collapse to readiness 0.0, identical to a site we know nothing
about. These tests pin the maturity bands and the anchor floor.
"""

from __future__ import annotations

from app.core.orbitbrief_core import build_site_readiness, _SITE_ANCHOR_FLOOR
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(atom_type, text, *, value=None, entity_keys=None, rid="atm_x"):
    return EvidenceAtom(
        id=rid,
        project_id="p",
        artifact_id="art_x",
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=value if value is not None else {},
        entity_keys=entity_keys or [],
        source_refs=[
            SourceRef(
                id="src_1",
                artifact_id="art_x",
                artifact_type=ArtifactType.txt,
                filename="f.txt",
                locator={},
                extraction_method="test",
                parser_version="t",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.8,
        confidence_raw=0.8,
        calibrated_confidence=0.8,
        review_status=ReviewStatus.needs_review,
        review_flags=[],
        parser_version="t",
    )


def test_anchored_only_site_is_amber_not_zero():
    """A bare geo-anchored site floors at the anchor value and reads
    amber/anchored — not red/0.0."""
    atoms = [
        _atom(AtomType.physical_site, "Santa Fe site",
              value={"id": "santa_fe_87506"}, rid="atm_s1"),
    ]
    out = build_site_readiness(atoms=atoms, edges=[])
    assert out["site_count"] == 1
    site = out["sites"][0]
    assert site["site"] == "site:santa_fe_87506"
    assert site["anchored"] is True
    assert site["readiness"] == _SITE_ANCHOR_FLOOR
    assert site["maturity"] == "anchored"
    assert site["band"] == "amber"
    assert out["avg_readiness"] == _SITE_ANCHOR_FLOOR
    assert out["anchored_count"] == 1
    assert out["maturity_breakdown"]["anchored"] == 1


def test_anchored_site_with_signals_progresses_band():
    """Adding devices + scope to an anchored site moves it past the
    floor into a higher maturity stage."""
    key = "site:santa_fe_87506"
    atoms = [
        _atom(AtomType.physical_site, "Santa Fe site",
              value={"id": "santa_fe_87506"}, rid="atm_s1"),
        _atom(AtomType.scope_item, "Replace 110 displays",
              entity_keys=[key, "device:display"], rid="atm_sc1"),
    ]
    out = build_site_readiness(atoms=atoms, edges=[])
    site = out["sites"][0]
    assert site["anchored"] is True
    assert site["readiness"] > _SITE_ANCHOR_FLOOR
    assert site["signal_count"] >= 2
    assert site["maturity"] in ("planning", "ready")
    assert site["band"] in ("amber", "green")


def test_fully_detailed_site_is_green():
    key = "site:santa_fe_87506"
    atoms = [
        _atom(AtomType.physical_site, "Santa Fe site",
              value={"id": "santa_fe_87506"}, rid="atm_s1"),
        _atom(AtomType.scope_item, "Replace displays",
              entity_keys=[key, "device:display", "stakeholder:site_lead",
                           "money:budget", "milestone:cutover"], rid="atm_sc1"),
        _atom(AtomType.constraint, "Work after hours only",
              entity_keys=[key], rid="atm_c1"),
    ]
    out = build_site_readiness(atoms=atoms, edges=[])
    site = out["sites"][0]
    assert site["readiness"] >= 0.75
    assert site["maturity"] == "ready"
    assert site["band"] == "green"


def test_no_sites_returns_empty_breakdown():
    out = build_site_readiness(atoms=[], edges=[])
    assert out["site_count"] == 0
    assert out["anchored_count"] == 0
    assert out["avg_readiness"] == 0.0
