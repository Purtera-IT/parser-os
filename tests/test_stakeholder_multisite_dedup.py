"""A person common to many single-site documents keeps every site they touch.

Deal 010215: Bernard Donnelly is the backup contact named in all ten Marion
County SOWs. Collapsing his ten atoms to one BEFORE any of them could be
tagged with a school left him attributed to whichever single document his
surviving instance happened to come from -- correct name, email, phone,
attached to one school out of ten.
"""

from __future__ import annotations

from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus
from app.core.semantic_dedup import dedupe_stakeholder_atoms, semantic_dedup_atoms
from app.core.site_provenance_join import join_atoms_to_document_site


def _stakeholder(artifact_id, name, email, role="Backup Contact"):
    return EvidenceAtom(
        id=f"atm_{artifact_id}_{role}",
        project_id="p",
        artifact_id=artifact_id,
        atom_type=AtomType.stakeholder,
        raw_text=f"{role}: {name}",
        normalized_text=name.lower(),
        value={"role": role, "kind": "person", "name": name, "email": email},
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        entity_keys=[],
        parser_version="test",
    )


def _site(artifact_id, school):
    return EvidenceAtom(
        id=f"atm_site_{artifact_id}",
        project_id="p",
        artifact_id=artifact_id,
        atom_type=AtomType.physical_site,
        raw_text=school,
        normalized_text=school.lower(),
        value={"kind": "physical_site", "id": school, "site_id": school, "name": school},
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.95,
        review_status=ReviewStatus.auto_accepted,
        entity_keys=[f"site:{school.lower()}"],
        parser_version="test",
    )


def test_semantic_dedup_leaves_stakeholder_atoms_uncollapsed():
    """The exemption: collapsing here would happen before any site: key
    exists, so it must not happen here at all."""
    atoms = [_stakeholder(f"doc{i}", "Bernard Donnelly", "bernie@sodexo.com") for i in range(3)]
    out = semantic_dedup_atoms(atoms)
    assert len(out) == 3


def test_a_shared_backup_contact_accumulates_every_site():
    schools = ["johnakin", "palmetto", "marion_high"]
    atoms = []
    for s in schools:
        atoms.append(_site(f"doc_{s}", s))
        atoms.append(_stakeholder(f"doc_{s}", "Bernard Donnelly", "bernie@sodexo.com"))

    atoms = semantic_dedup_atoms(atoms)  # stakeholder untouched here
    join_atoms_to_document_site(atoms)   # each of the 3 gets its OWN site: key
    atoms = dedupe_stakeholder_atoms(atoms)  # now they collapse

    people = [a for a in atoms if a.atom_type == AtomType.stakeholder]
    assert len(people) == 1, "the three Bernard Donnelly atoms must collapse to one"
    site_keys = {k for k in people[0].entity_keys if k.startswith("site:")}
    assert site_keys == {f"site:{s}" for s in schools}, (
        "the survivor must carry ALL three sites, not just one"
    )


def test_a_distinct_on_site_contact_per_school_is_unaffected():
    """Different people at different schools must never merge into one."""
    atoms = [
        _site("doc_a", "johnakin"),
        _stakeholder("doc_a", "Romell Shird", "romell@sodexo.com", role="On Site Contact"),
        _site("doc_b", "palmetto"),
        _stakeholder("doc_b", "Jamonica Turner", "jamonica@sodexo.com", role="On Site Contact"),
    ]
    atoms = semantic_dedup_atoms(atoms)
    join_atoms_to_document_site(atoms)
    atoms = dedupe_stakeholder_atoms(atoms)
    people = [a for a in atoms if a.atom_type == AtomType.stakeholder]
    assert len(people) == 2
    assert {p.value["name"] for p in people} == {"Romell Shird", "Jamonica Turner"}
