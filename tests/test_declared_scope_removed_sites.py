"""The site-count gap counts the sites a later document took back out.

Deal 000132 (compile 5181b802): the documents declare six locations; a later
HubSpot note reads "Need Troy and Wilmington sites removed."; four sites are
live. The gap still said "declare 6 locations; 5 identified" -- the removed
sites were in the declaration and in the found list alike.
"""
from __future__ import annotations

from app.core.declared_scope import declared_scope_questions
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

CITIES = ["Delphos, OH", "Hudson, WI", "Plymouth, MI", "Troy, MI", "Tupelo, MS", "Wilmington, DE"]


def _atom(aid, atype, text, *, artifact="sow", keys=(), value=None, meta=None):
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id=artifact, atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value=value or {"text": text, **(meta or {})},
        entity_keys=list(keys),
        source_refs=[SourceRef(id=stable_id("src", aid), artifact_id=artifact,
                               artifact_type=ArtifactType.txt, filename=f"{artifact}.txt",
                               locator={}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.contractual_scope, confidence=0.9,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t",
    )


def _site(i, city):
    name = city.split(",")[0]
    slug = city.lower().replace(", ", "_").replace(" ", "_")
    return _atom(f"s{i}", AtomType.physical_site, f"facility: {city}",
                 keys=[f"site:{slug}"],
                 value={"kind": "physical_site", "id": slug.upper(), "site_id": slug.upper(), "name": name})


def _deal(listed, *, removal=True, removal_date="2026-07-01T00:00:00Z"):
    atoms = [
        _atom("meta_sow", AtomType.deal_metadata, "note_id=1", artifact="sow",
              value={"kind": "hubspot_note_meta", "date": "2026-05-01T00:00:00Z"}),
        _atom("decl", AtomType.scope_item, "Support is required at 6 locations: " + "; ".join(CITIES) + ".",
              artifact="sow"),
    ]
    atoms += [_site(i, c) for i, c in enumerate(listed)]
    if removal:
        atoms += [
            _atom("meta_rm", AtomType.deal_metadata, "note_id=2", artifact="note_rm",
                  value={"kind": "hubspot_note_meta", "date": removal_date}),
            _atom("rm", AtomType.deal_metadata, "Need Troy and Wilmington sites removed.", artifact="note_rm"),
        ]
    return atoms


def _gap(atoms):
    return [a for a in declared_scope_questions(project_id="p", atoms=atoms)
            if a.value["declared_scope"]["kind"] == "site_count_gap"]


def test_removed_sites_leave_both_the_declaration_and_the_found_list():
    # Five identified (Troy among them), two removed: four live, four remain.
    listed = ["Delphos, OH", "Hudson, WI", "Plymouth, MI", "Troy, MI", "Tupelo, MS"]
    assert _gap(_deal(listed)) == [], "4 declared sites remain and 4 live sites are identified"


def test_a_real_gap_after_removal_is_reported_with_the_removal():
    listed = ["Delphos, OH", "Hudson, WI", "Troy, MI"]
    (q,) = _gap(_deal(listed))
    ds = q.value["declared_scope"]
    assert ds["declared_count"] == 6
    assert ds["removed_count"] == 2 and ds["removed_sites"] == ["troy", "wilmington"]
    assert ds["live_declared_count"] == 4
    assert ds["found_count"] == 2, "Troy is listed but removed"
    assert ds["removing_atom_ids"] == ["rm"]
    assert "declare 6 locations; 2 later removed (Troy, Wilmington), 4 remain; 2 identified" in q.raw_text


def test_without_a_removal_the_gap_is_unchanged():
    listed = ["Delphos, OH", "Hudson, WI", "Plymouth, MI", "Troy, MI", "Tupelo, MS"]
    (q,) = _gap(_deal(listed, removal=False))
    assert "declare 6 locations; 5 identified" in q.raw_text


def test_a_removal_before_the_declaration_does_not_count():
    listed = ["Delphos, OH", "Hudson, WI", "Plymouth, MI", "Troy, MI", "Tupelo, MS"]
    (q,) = _gap(_deal(listed, removal_date="2026-04-01T00:00:00Z"))
    assert q.value["declared_scope"]["removed_count"] == 0


def test_removing_equipment_at_a_site_is_not_removing_the_site():
    listed = ["Delphos, OH", "Hudson, WI", "Plymouth, MI", "Troy, MI", "Tupelo, MS"]
    atoms = _deal(listed, removal=False) + [
        _atom("eq", AtomType.scope_item, "Remove the old access points at the Troy office.", artifact="note_rm"),
    ]
    (q,) = _gap(atoms)
    assert q.value["declared_scope"]["removed_count"] == 0
