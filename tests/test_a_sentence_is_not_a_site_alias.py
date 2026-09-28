"""A sentence that names the building is not a second copy of the address.

`_dedupe_physical_site_atoms` treats a physical_site atom with no structured
address as a "name-only prose guess" -- an alias -- and removes it once a
location-backed site exists. That is right for "the new office location" and
wrong for a sentence carrying a fact, and on live 010180 it deleted two:

    "Lease has been signed for 7 Penn"   15 copies in, 0 out -- the deal's
                                          only milestone
    "7 Penn building in NYC"             30 copies in, 0 out

Neither stage was wrong alone. `quoted_history_dedup` dropped the quoted
echoes and correctly kept the authored original; the site pass then removed
that original as an alias. Together they deleted the fact, and because the
losers were merged for metadata the deal still "knew" the site -- so nothing
looked broken.

The trade this pins is the one the module already states elsewhere: never
trade a sentence for a type. Such an atom keeps its words and loses only its
claim to BE a site declaration.
"""
from __future__ import annotations

from app.core.schemas import (ArtifactType, AtomType, AuthorityClass, EvidenceAtom,
                              ReviewStatus, SourceRef)
from app.core.semantic_dedup import _dedupe_physical_site_atoms


def _site(aid: str, text: str, value: dict) -> EvidenceAtom:
    return EvidenceAtom(
        id=aid, project_id="d1", artifact_id="art_1", atom_type=AtomType.physical_site,
        raw_text=text, normalized_text=text.lower(), value=value, entity_keys=[],
        source_refs=[SourceRef(id=f"src_{aid}", artifact_id="art_1",
                               artifact_type=ArtifactType.email, filename="m.eml",
                               locator={}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t")


#: The real site: an address, with structure behind it.
CANONICAL = _site("atm_addr", "Address: 7 Penn Plaza West 31st Street, Floor 12 Suite 1200",
                  {"id": "7_penn_plaza", "site_id": "7_penn_plaza",
                   "street_address": "7 Penn Plaza West 31st Street",
                   "facility_name": "7 Penn Plaza"})


def _run(*extra):
    out = _dedupe_physical_site_atoms([CANONICAL, *extra])
    return {a.raw_text: a for a in out}


def test_a_milestone_that_names_the_site_is_not_deleted():
    lease = _site("atm_lease", "Lease has been signed for 7 Penn",
                  {"id": "7_penn", "site_id": "7_penn"})
    kept = _run(lease)
    assert "Lease has been signed for 7 Penn" in kept, \
        "the deal's only milestone was deleted as a site alias"


def test_it_keeps_the_words_and_loses_the_claim():
    """Surviving as a physical_site would be its own bug: the deal would show
    two sites. The point is the sentence, not the type."""
    lease = _site("atm_lease", "Lease has been signed for 7 Penn",
                  {"id": "7_penn", "site_id": "7_penn"})
    a = _run(lease)["Lease has been signed for 7 Penn"]
    assert a.atom_type is AtomType.deal_metadata
    assert "kept_over_site_dedup" in a.review_flags
    assert not (a.value or {}).get("site_id"), "still claims to be a site"


def test_a_genuine_name_only_alias_still_collapses():
    """The behaviour being narrowed, not removed. "7 Penn Plaza" alone adds no
    words to the address atom -- it is the same telling, and two site rows for
    one building is what this pass exists to prevent."""
    alias = _site("atm_alias", "7 Penn Plaza", {"id": "7_penn", "site_id": "7_penn"})
    assert "7 Penn Plaza" not in _run(alias)


def test_the_canonical_site_is_never_the_one_dropped():
    lease = _site("atm_lease", "Lease has been signed for 7 Penn",
                  {"id": "7_penn", "site_id": "7_penn"})
    kept = _run(lease)
    addr = kept.get("Address: 7 Penn Plaza West 31st Street, Floor 12 Suite 1200")
    assert addr is not None and addr.atom_type is AtomType.physical_site


def test_a_short_mention_is_not_promoted_by_accident():
    """The guard asks for words of its own AND a verb. A fragment has neither,
    so this must not become a licence to keep every stray mention."""
    frag = _site("atm_frag", "7 Penn", {"id": "7_penn", "site_id": "7_penn"})
    assert "7 Penn" not in _run(frag)
