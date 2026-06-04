"""Semantic address-role gate: a vendor's letterhead / billing address must
not be minted as a phantom job site.

The Yonah deal's SOW carries the service provider's footer
("PurTera LLC, 11720 Amber Park Dr #350, Alpharetta, GA 30009 | DCW"). A
city/state/zip regex turned that into ``site:alpharetta_30009`` — a job site
that does not exist — competing with the real site in Santa Fe, NM. The fix
classifies each site's address ROLE with a small local LLM and drops vendor
addresses, but only ever safely (never the only site, never on a guess).
"""

from __future__ import annotations

import app.core.semantic_role as semantic_role
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.site_geo_fallback import geo_fallback_sites, suppress_vendor_sites


def _site(atom_id: str, *, raw_text: str, address: str = "") -> EvidenceAtom:
    src = SourceRef(
        id=f"src_{atom_id}",
        artifact_id="art",
        artifact_type=ArtifactType.docx,
        filename="sow.docx",
        locator={"extraction": "test"},
        extraction_method="test",
        parser_version="test",
    )
    return EvidenceAtom(
        id=atom_id,
        project_id="p",
        artifact_id="art",
        atom_type=AtomType.physical_site,
        raw_text=raw_text,
        normalized_text=raw_text.lower(),
        value={"kind": "physical_site", "address": address or raw_text},
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def _text_atom(atom_id: str, raw_text: str) -> EvidenceAtom:
    """A plain prose atom (NOT a physical_site) carrying a City/ST/ZIP anchor —
    the raw material geo_fallback mints sites from."""
    src = SourceRef(
        id=f"src_{atom_id}",
        artifact_id="art",
        artifact_type=ArtifactType.docx,
        filename="sow.docx",
        locator={"extraction": "test"},
        extraction_method="test",
        parser_version="test",
    )
    return EvidenceAtom(
        id=atom_id,
        project_id="p",
        artifact_id="art",
        atom_type=AtomType.scope_item,
        raw_text=raw_text,
        normalized_text=raw_text.lower(),
        value={"text": raw_text},
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


_VENDOR = _site(
    "atm_vendor",
    raw_text="PurTera LLC 11720 Amber Park Dr #350, Alpharetta, GA 30009 | DCW",
    address="11720 Amber Park Dr #350, Alpharetta, GA 30009",
)
_JOBSITE = _site("atm_job", raw_text="Santa Fe, NM 87506", address="Santa Fe, NM 87506")


def _fake_classifier(vendor_substr: str, *, match_context: bool = False):
    """Fake role classifier. By default keys on the address ``text``; with
    ``match_context=True`` it keys on the ``context`` arg — mirroring how a
    real vendor signal (company name / letterhead) lives in the originating
    source text, not the terse minted address."""
    def _clf(text, candidates, *, instruction, context="", timeout=None, model=None):
        haystack = context if match_context else text
        if vendor_substr in haystack:
            return ("vendor_or_billing_address", 0.92)
        return ("job_site", 0.9)
    return _clf


def setup_function(_):
    semantic_role.reset_reachability()


def test_vendor_address_dropped_realsite_kept(monkeypatch):
    monkeypatch.setattr(semantic_role, "classify_role", _fake_classifier("Alpharetta"))
    kept, dropped = suppress_vendor_sites([_VENDOR, _JOBSITE], project_id="p")
    assert dropped == 1
    kept_ids = {a.id for a in kept}
    assert "atm_vendor" not in kept_ids
    assert "atm_job" in kept_ids


def test_geo_minted_vendor_site_dropped_via_source_context(monkeypatch):
    # Real-world flow (the Yonah bug): the vendor letterhead and the genuine
    # site both arrive only as bare "City, ST ZIP" text inside prose atoms.
    # geo_fallback mints BOTH as physical_site atoms; the discriminating signal
    # (the company name) survives only in source_context. Suppression must read
    # that context to drop the letterhead site and keep the real one.
    monkeypatch.setattr(
        semantic_role, "classify_role", _fake_classifier("PurTera", match_context=True)
    )
    vendor_line = _text_atom(
        "atm_v", "PurTera LLC 11720 Amber Park Dr #350, Alpharetta, GA 30009 | DCW"
    )
    site_line = _text_atom("atm_s", "Project location Santa Fe, NM 87506")

    minted = geo_fallback_sites([vendor_line, site_line], project_id="p")
    assert len(minted) == 2  # both anchors became physical_site atoms
    alpha = next(m for m in minted if m.value.get("zip") == "30009")
    assert "PurTera" in alpha.value.get("source_context", "")

    kept, dropped = suppress_vendor_sites(minted, project_id="p")
    assert dropped == 1
    kept_zips = {a.value.get("zip") for a in kept}
    assert "30009" not in kept_zips  # vendor letterhead dropped
    assert "87506" in kept_zips  # real site kept


def test_dropped_atom_records_decision_provenance(monkeypatch):
    # A demoted site must carry WHY it was demoted (invariant I): which tier
    # decided and, when a learned correction drove it, that correction's id —
    # so a PM can trace the suppression with no keyword list involved.
    monkeypatch.setattr(semantic_role, "classify_role", _fake_classifier("Alpharetta"))
    kept, dropped = suppress_vendor_sites([_VENDOR, _JOBSITE], project_id="p")
    assert dropped == 1
    prov = _VENDOR.value.get("_decision")
    assert prov is not None
    # LLM-driven here (no store wired) → source llm, no correction_id, conf set.
    assert prov["source"] == "llm"
    assert prov["correction_id"] is None
    assert prov["confidence"] >= 0.6
    # The kept real site is never stamped.
    assert _JOBSITE.value.get("_decision") is None


def test_no_op_when_only_one_site(monkeypatch):
    # Even if the classifier flags it vendor, never strip the deal's only
    # locational anchor.
    monkeypatch.setattr(semantic_role, "classify_role", _fake_classifier("Alpharetta"))
    kept, dropped = suppress_vendor_sites([_VENDOR], project_id="p")
    assert dropped == 0
    assert len(kept) == 1


def test_no_op_when_all_sites_flagged_vendor(monkeypatch):
    # If the model would drop every site, keep them all (never zero sites).
    def _all_vendor(text, candidates, *, instruction, context="", timeout=None, model=None):
        return ("vendor_or_billing_address", 0.95)
    monkeypatch.setattr(semantic_role, "classify_role", _all_vendor)
    kept, dropped = suppress_vendor_sites([_VENDOR, _JOBSITE], project_id="p")
    assert dropped == 0
    assert len(kept) == 2


def test_low_confidence_not_dropped(monkeypatch):
    def _weak(text, candidates, *, instruction, context="", timeout=None, model=None):
        return ("vendor_or_billing_address", 0.4)  # below threshold
    monkeypatch.setattr(semantic_role, "classify_role", _weak)
    kept, dropped = suppress_vendor_sites([_VENDOR, _JOBSITE], project_id="p")
    assert dropped == 0
    assert len(kept) == 2


def test_llm_unreachable_is_safe_noop(monkeypatch):
    # When the LLM is disabled/unreachable, classify_role returns (None, 0.0)
    # and NOTHING is dropped — no site removed on a guess.
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    semantic_role.reset_reachability()
    kept, dropped = suppress_vendor_sites([_VENDOR, _JOBSITE], project_id="p")
    assert dropped == 0
    assert len(kept) == 2


def test_classify_role_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    semantic_role.reset_reachability()
    role, conf = semantic_role.classify_role(
        "11720 Amber Park Dr, Alpharetta GA",
        ["job_site", "vendor_or_billing_address"],
        instruction="x",
    )
    assert role is None
    assert conf == 0.0


def test_classify_role_parses_model_json():
    role, conf = semantic_role._parse_response(
        '{"role": "vendor_or_billing_address", "confidence": 0.88}',
        ["job_site", "vendor_or_billing_address"],
    )
    assert role == "vendor_or_billing_address"
    assert conf == 0.88


def test_classify_role_rejects_offmenu_and_unknown():
    allowed = ["job_site", "vendor_or_billing_address"]
    assert semantic_role._parse_response('{"role": "unknown"}', allowed) == (None, 0.0)
    assert semantic_role._parse_response('{"role": "warehouse"}', allowed) == (None, 0.0)
    assert semantic_role._parse_response("not json", allowed) == (None, 0.0)
